from collections import defaultdict
import logging

import cv2
import numpy as np
import supervision as sv

from .line_counter import _get_counts_results
from .types import CCTVProcessingContext

logger = logging.getLogger(__name__)


def _check_if_should_draw(video_output_info, feature):
    feature_info = video_output_info.get(feature)
    return feature_info and feature_info.get("draw", False)


def _initialize_annotators(ctx: CCTVProcessingContext) -> None:
    """Initialize supervision annotators for video output based on configuration."""
    # Box annotator
    if _check_if_should_draw(ctx.video_output, "bboxes"):
        box_params = ctx.video_output["bboxes"].get("params", {})
        ctx.annotators["bboxes"] = sv.BoxAnnotator(**box_params)

    # Label annotator
    if _check_if_should_draw(ctx.video_output, "labels"):
        label_params = ctx.video_output["labels"].get("params", {})
        ctx.annotators["labels"] = sv.LabelAnnotator(**label_params)

    # Counter display flag (will be drawn using cv2 in main loop)
    logger.debug("Checking if should draw for line counters")
    if _check_if_should_draw(ctx.video_output, "line_counters"):
        counter_params = ctx.video_output["line_counters"].get("params", {})

        for line_counter in ctx.line_counters["line_counters"]:
            custom_text = " | ".join([label[:3] for label in line_counter["labels"]])

            if line_counter["direction"] == "in":
                line_zone_annotator = sv.LineZoneAnnotator(
                    custom_in_text=f"{custom_text}",
                    display_out_count=False,
                    text_orient_to_line=True,
                    **counter_params,
                )
            elif line_counter["direction"] == "out":
                line_zone_annotator = sv.LineZoneAnnotator(
                    custom_out_text=f"{custom_text}",
                    display_in_count=False,
                    text_orient_to_line=True,
                    **counter_params,
                )

            line_counter["line_zone_annotator"] = line_zone_annotator


def _format_gender_counts(gender_counts: dict) -> str:
    """Format gender counts into a readable string."""
    parts = []
    for gender, count in gender_counts.items():
        if gender is None:
            parts.append(f"{count}")
        elif isinstance(gender, str):
            parts.append(f"{gender[0]}: {count}")  # First letter only
        else:
            parts.append(f"{gender}: {count}")
    return " | ".join(parts)


def _get_counts_results_txt(results):
    """Format counts results into readable text with optional lane discrimination."""
    # Group by count_id, then by vehicle
    grouped = defaultdict(lambda: defaultdict(dict))

    for idx, entry in enumerate(results):
        count_id = entry["count_id"]
        vehicle = entry["vehicle"]
        lane = entry["lane"]
        gender = entry["gender"]
        count = entry["count"]

        if lane not in grouped[count_id][vehicle]:
            grouped[count_id][vehicle][lane] = {}

        grouped[count_id][vehicle][lane][gender] = count

    # Build text for each count_id
    output = {}
    for count_id, vehicles in grouped.items():
        lines = [f"{count_id}:"]

        for vehicle, lane_gender_counts in vehicles.items():
            # Check if we have both in-lane and out-lane counts
            has_both = True in lane_gender_counts and False in lane_gender_counts

            if has_both:
                # Show lane discrimination
                lines.append(f"  {vehicle}:")

                if True in lane_gender_counts:
                    gender_counts = lane_gender_counts[True]
                    count_str = _format_gender_counts(gender_counts)
                    lines.append(f"    in lane: {count_str}")

                if False in lane_gender_counts:
                    gender_counts = lane_gender_counts[False]
                    count_str = _format_gender_counts(gender_counts)
                    lines.append(f"    out lane: {count_str}")
            else:
                # Get the single lane's gender counts
                single_lane = list(lane_gender_counts.values())[0]
                count_str = _format_gender_counts(single_lane)
                lines.append(f"  {vehicle}: {count_str}")

        formatted_output = "\n".join(lines)
        output[count_id] = formatted_output

    return output


def _annotate_results(frame: np.ndarray, results_txt: dict, params: dict = None) -> np.ndarray:
    """Annotate frame with count results."""
    if not results_txt:
        return frame

    # Default parameters
    params = params or {}
    position = params.get("position", "top-center")
    text_scale = params.get("text_scale", 0.4)
    text_padding = params.get("text_padding", 3)

    # Drawing constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_height_multiplier = 1.5

    border_offset = 80
    box_spacing = 20

    frame_h, frame_w = frame.shape[:2]

    # Calculate text boxes for each count_id
    boxes = []
    for count_id, text in results_txt.items():
        lines = text.split("\n")

        # Calculate box dimensions
        max_width = 0
        total_height = 0
        line_sizes = []

        for line in lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, font, text_scale, thickness)
            line_sizes.append((text_w, text_h, baseline))
            max_width = max(max_width, text_w)
            total_height += int(text_h * line_height_multiplier)

        box_w = max_width + 2 * text_padding
        box_h = total_height + 2 * text_padding

        boxes.append(
            {
                "count_id": count_id,
                "text": text,
                "lines": lines,
                "line_sizes": line_sizes,
                "width": box_w,
                "height": box_h,
            }
        )

    # Calculate total width needed for all boxes
    total_width = sum(box["width"] for box in boxes) + box_spacing * (len(boxes) - 1)

    # Calculate starting x position based on alignment
    if position in ["top-left", "bottom-left"]:
        start_x = border_offset
    elif position in ["top-right", "bottom-right"]:
        start_x = frame_w - border_offset - total_width
    else:  # center
        start_x = (frame_w - total_width) // 2

    # Calculate y position
    if position.startswith("top"):
        start_y = border_offset
    else:  # bottom
        max_height = max(box["height"] for box in boxes)
        start_y = frame_h - border_offset - max_height

    # Draw each box
    current_x = start_x
    for box in boxes:
        # Draw background rectangle
        x1 = current_x
        y1 = start_y
        x2 = x1 + box["width"]
        y2 = y1 + box["height"]

        # White background
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # Draw text lines
        text_y = y1 + text_padding
        for i, (line, (text_w, text_h, baseline)) in enumerate(
            zip(box["lines"], box["line_sizes"])
        ):
            text_y += text_h
            text_x = x1 + text_padding

            cv2.putText(
                frame,
                line,
                (text_x, text_y),
                font,
                text_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            text_y += int((text_h * line_height_multiplier) - text_h)

        # Move to next box position
        current_x += box["width"] + box_spacing

    return frame


def _annotate_frame(ctx, frame, detections, classifications):
    """Annotate frame with bounding boxes and labels including gender for cyclists."""
    annotated_frame = frame.copy()

    if ctx.annotators.get("bboxes"):
        annotated_frame = ctx.annotators["bboxes"].annotate(
            scene=annotated_frame, detections=detections
        )

    if ctx.annotators.get("labels"):
        labels = []
        for _, _, confidence, class_id, tracker_id, _ in detections:
            label = ctx.detection["class_label"][class_id]

            if label == "bicycle" and len(classifications) > 0:
                # Find this tracker_id in classifications
                mask = classifications.tracker_id == tracker_id

                if np.any(mask):
                    # Get prediction (returns array, take first element)
                    gender_pred = classifications.predictions[mask][0]

                    # Get gender label
                    if classifications.classes:
                        gender_label = classifications.classes[gender_pred]
                        label = f"bicycle + {gender_label}"
                    else:
                        label = f"bicycle + class_{gender_pred}"

            labels.append(f"#{tracker_id} {label} {confidence:0.2f}")

        annotated_frame = ctx.annotators["labels"].annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

    if ctx.video_output["line_counters"].get("draw"):
        for line_counter in ctx.line_counters["line_counters"]:

            annotator = line_counter["line_zone_annotator"]
            annotated_frame = annotator.annotate(
                frame=annotated_frame, line_counter=line_counter["line_zone"]
            )

    if _check_if_should_draw(ctx.video_output, "results"):
        results = _get_counts_results(ctx)
        results_txt = _get_counts_results_txt(results)

        result_params = ctx.video_output["results"].get("params", {})
        annotated_frame = _annotate_results(annotated_frame, results_txt, result_params)

    return annotated_frame
