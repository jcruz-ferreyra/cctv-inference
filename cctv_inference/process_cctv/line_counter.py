from collections import Counter, defaultdict
import json
import logging
from typing import Any, Dict, List

import numpy as np
import supervision as sv

from .types import CCTVProcessingContext

logger = logging.getLogger(__name__)


def _restart_line_counters(ctx):

    for line_counter in ctx.line_counters["line_counters"]:
        line_counter["line_zone"]._in_count_per_class = Counter()
        line_counter["line_zone"]._out_count_per_class = Counter()


def _calculate_prevalent_gender(
    tracker_history: Dict[int, List], direction: str
) -> Dict[int, Any]:
    """
    Calculate the most prevalent gender for each tracker with temporal weighting.

    Weights predictions based on direction:
    - "in": Earlier predictions weighted more (entering scene, closer to camera)
    - "out": Later predictions weighted more (exiting scene, closer to camera)

    TODO: Replace temporal weighting with bounding box size-based weighting
          for more accurate confidence based on actual distance to camera.

    Args:
        tracker_history: Dict mapping tracker_id to list of gender predictions
        direction: "in" or "out" to determine weighting strategy

    Returns:
        Dict mapping tracker_id to single prevalent gender prediction
    """
    prevalent_genders = {}

    for tracker_id, predictions in tracker_history.items():
        if not predictions:
            logger.warning(f"No predictions for tracker {tracker_id}, skipping")
            continue

        # Single prediction - no weighting needed
        if len(predictions) == 1:
            prevalent_genders[tracker_id] = predictions[0]
            continue

        # Create weights based on direction
        n = len(predictions)
        if direction == "in":
            # Earlier predictions weighted more: [n, n-1, ..., 2, 1]
            weights = np.arange(n, 0, -1)
        elif direction == "out":
            # Later predictions weighted more: [1, 2, ..., n-1, n]
            weights = np.arange(1, n + 1)
        else:
            # Fallback: equal weights
            weights = np.ones(n)

        # Count weighted votes for each class
        vote_counts = defaultdict(float)
        for pred, weight in zip(predictions, weights):
            vote_counts[pred] += weight

        # Get class with highest weighted vote
        prevalent_gender = max(vote_counts.items(), key=lambda x: x[1])[0]
        prevalent_genders[tracker_id] = prevalent_gender

        logger.debug(
            f"Tracker {tracker_id}: {len(predictions)} predictions, "
            f"direction={direction}, prevalent={prevalent_gender}, "
            f"votes={dict(vote_counts)}"
        )

    return prevalent_genders


def _get_counts_results(ctx):
    """Get counting results aggregated by line counter, vehicle, lane, and gender."""
    # Accumulate using nested dict
    counts = defaultdict(lambda: defaultdict(int))

    for idx, line_counter in enumerate(ctx.line_counters["line_counters"], 1):
        lc_id = line_counter.get("count_id", "missing_id")
        lc_direction = line_counter.get("direction", "in")
        lane = line_counter.get("lane", False)

        if lc_direction == "in":
            lc_results = line_counter["line_zone"].in_count_per_class
        elif lc_direction == "out":
            lc_results = line_counter["line_zone"].out_count_per_class
        else:
            continue

        for class_id, count in lc_results.items():
            vehicle = ctx.detection["class_label"].get(class_id, f"unknown_{class_id}")

            if vehicle == "bicycle":
                crossed_bicycles = line_counter["crossed_bicycles"]

                crossed_bicycles_history = {
                    k: v
                    for k, v in ctx.classification["tracker_gender_history"].items()
                    if k in crossed_bicycles
                }

                crossed_bicycles_gender = _calculate_prevalent_gender(
                    crossed_bicycles_history, lc_direction
                )

                # Count by unique gender
                gender_breakdown = {}
                for gender in set(crossed_bicycles_gender.values()):
                    current_gender_tracks = [
                        k for k, v in crossed_bicycles_gender.items() if v == gender
                    ]
                    gender_count = len(current_gender_tracks)
                    counts[lc_id][(vehicle, lane, gender)] += gender_count
                    gender_breakdown[gender] = gender_count

            else:
                counts[lc_id][(vehicle, lane, None)] += count

    result = []
    for count_id, vehicle_lane_counts in counts.items():
        for (vehicle, lane, gender), count in vehicle_lane_counts.items():
            result.append(
                {
                    "count_id": count_id,
                    "vehicle": vehicle,
                    "lane": lane,
                    "gender": gender,
                    "count": count,
                }
            )

    return result


def _initialize_counters(ctx: CCTVProcessingContext) -> None:
    """Load line counter configuration from JSON file and create LineZone objects."""
    try:
        with open(ctx.line_counters_path, "r") as f:
            line_counters = json.load(f)
        logger.info(f"Successfully loaded {len(line_counters)} line configurations")
    except FileNotFoundError:
        logger.error(f"Line counter file not found at {ctx.line_counters_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON line counter file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading line counter file: {e}")
        raise

    # Map anchor string to supervision Position
    anchor_mapping = {
        "center": sv.Position.CENTER,
        "bottom": sv.Position.BOTTOM_CENTER,
        "top": sv.Position.TOP_CENTER,
    }

    for line_counter in line_counters:
        # Get and validate anchor
        anchor = line_counter.get("anchor", "center")
        if anchor not in anchor_mapping:
            raise ValueError(f"Invalid anchor: {anchor}. Supported: {list(anchor_mapping.keys())}")
        triggering_anchors = [anchor_mapping[anchor]]

        # Get and validate direction
        direction = line_counter.get("direction", "in")
        if direction not in ["in", "out"]:
            raise ValueError(f"Invalid direction: {direction}. Supported: ['in', 'out']")

        # Validate classes exist
        if "labels" in line_counter and line_counter["labels"]:
            line_counter["classes"] = [
                k for k, v in ctx.detection["class_label"].items() if v in line_counter["labels"]
            ]
            if not line_counter["classes"]:
                raise ValueError(
                    f"No matching classes found for labels {line_counter['labels']} in line {line_counter.get('id')}"
                )
        else:
            line_counter["classes"] = list(ctx.detection["class_label"].keys())

        lc_count_id = line_counter.get("count_id", "missing_id")
        logger.debug(
            f"Line '{lc_count_id}': mapped labels {line_counter.get('labels', 'all')} to class IDs {line_counter['classes']}"
        )

        # Create LineZone with proper start and end points
        line_counter["line_zone"] = sv.LineZone(
            start=sv.Point(*line_counter["coords"][0]),
            end=sv.Point(*line_counter["coords"][1]),
            triggering_anchors=triggering_anchors,
        )

        # Initialize set to store tracks that crossed the line.
        line_counter["crossed"] = set()
        line_counter["crossed_bicycles"] = set()

        logger.debug(f"Created LineZone: anchor={anchor}, direction={direction}")

    ctx.line_counters["line_counters"] = line_counters

    logger.info(f"Created {len(line_counters)} line counter zones")
