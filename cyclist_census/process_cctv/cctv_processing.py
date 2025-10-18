from collections import Counter, deque
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
from typing import Dict

import cv2
import numpy as np
import supervision as sv

# import trackers
from tqdm import tqdm

from cyclist_census.utils import (
    Classifications,
    get_classifications_from_folder,
    get_classifications_from_images,
    get_ultralytics_detections,
)

from .frame_annotation import _annotate_frame, _initialize_annotators
from .frame_management import FrameCounter, SinkManager
from .line_counter import (
    _get_counts_results,
    _initialize_counters,
    _restart_line_counters,
)
from .model_loading import _initialize_models
from .types import CCTVProcessingContext

logger = logging.getLogger(__name__)


def _copy_video_to_colab(ctx: CCTVProcessingContext) -> Path:
    """Set up local Colab storage and copy video from Drive."""

    colab_data_dir = Path("/content/data")
    logger.info("Setting up Colab environment")
    logger.info(f"  Remote data dir (Drive): {ctx.data_dir}")
    logger.info(f"  Local data dir (Colab): {colab_data_dir}")

    # Create local directory structure
    colab_input_dir = colab_data_dir / ctx.input_folder
    colab_input_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created input directory: {colab_input_dir}")

    colab_output_dir = colab_data_dir / ctx.output_folder
    colab_output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {colab_output_dir}")

    # Copy video from Drive to local storage
    remote_video = ctx.video_path
    colab_video = colab_data_dir / ctx.input_folder / ctx.video_name

    logger.info("Copying video from Drive to Colab storage")
    logger.info(f"  Source: {remote_video}")
    logger.info(f"  Destination: {colab_video}")

    try:
        colab_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_video, colab_video)
        video_size_mb = colab_video.stat().st_size / (1024 * 1024)
        logger.info(f"Video copied successfully ({video_size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"Failed to copy video from Drive: {e}")
        raise

    # Validate video exists
    if not colab_video.exists():
        raise FileNotFoundError(f"Video copy failed: {colab_video}")

    # Create output subdirectories
    video_stem = Path(ctx.video_name).stem
    video_output_dir = colab_output_dir / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Create crops directory if classification is enabled and batch mode
    if ctx.classification.get("enabled", False) and not ctx.classification.get("online", False):
        crops_dir = video_output_dir / f"{video_stem}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created crops directory: {crops_dir}")

    logger.info("Colab environment setup complete")

    ctx.remote_data_dir = ctx.data_dir
    ctx.data_dir = colab_data_dir


def _initialize_tracker(ctx: CCTVProcessingContext) -> None:
    """Initialize the object tracker based on context settings."""
    tracker_type = ctx.tracking["tracker_type"].lower()
    logger.info(f"Initializing tracker: {tracker_type}")

    if tracker_type == "bytetrack":
        tracker_params = ctx.tracking.get("tracker_params", {})
        video_info = sv.VideoInfo.from_video_path(ctx.video_path)
        tracker_params["frame_rate"] = video_info.fps

        logger.debug(f"ByteTrack parameters: {tracker_params}")
        ctx.tracker = sv.ByteTrack(**tracker_params) if tracker_params else sv.ByteTrack()
    elif tracker_type == "botsort":
        raise NotImplementedError(f"Tracker type '{tracker_type}' not yet implemented")
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}. " f"Supported: 'bytetrack'")

    # Initialize class history tracking for smoothing
    ctx.tracking["tracker_class_history"] = {}
    if not ctx.tracking.get("class_history_length"):
        ctx.tracking["class_history_length"] = 7

    logger.info(f"Tracker {tracker_type} initialized successfully")


def _get_output_video_info(ctx: CCTVProcessingContext) -> sv.VideoInfo:
    """Create output video info adjusted for inference interval and resolution."""
    # Get info from input video
    input_info = sv.VideoInfo.from_video_path(ctx.video_path)

    # Calculate output FPS based on inference interval
    inference_interval = ctx.frame_processing.get("inference_interval", 1)
    output_fps = input_info.fps / inference_interval

    # Calculate output resolution based on ratio
    resolution_ratio = ctx.video_output.get("resolution_ratio", 1.0)
    output_width = int(input_info.width * resolution_ratio)
    output_height = int(input_info.height * resolution_ratio)

    # Create new VideoInfo with adjusted parameters
    output_info = sv.VideoInfo(
        width=output_width,
        height=output_height,
        fps=output_fps,
        total_frames=input_info.total_frames // inference_interval,  # Also adjust frame count
    )

    logger.info(f"Input video: {input_info.width}x{input_info.height} @ {input_info.fps:.1f}fps")
    logger.info(f"Output video: {output_width}x{output_height} @ {output_fps:.1f}fps")
    logger.info(f"Inference interval: every {inference_interval} frame(s)")

    return output_info


def _process_frame(ctx, frame):
    detections = get_ultralytics_detections(
        frame,
        ctx.detection_model,
        ctx.detection["model_params"],
        ctx.detection["class_confidence"],
    )

    # Update tracker
    detections = ctx.tracker.update_with_detections(detections)

    # Skip if no detections or no tracker IDs
    if len(detections) == 0 or detections.tracker_id is None:
        return detections

    # Update class history and compute smoothed classes
    smoothed_class_ids = []
    class_history = ctx.tracking["tracker_class_history"]
    history_length = ctx.tracking["class_history_length"]

    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
        # Initialize deque for new tracker
        if tracker_id not in class_history:
            class_history[tracker_id] = deque(maxlen=history_length)

        # Add current class to history
        class_history[tracker_id].append(class_id)

        # Get majority class from history
        class_counts = Counter(class_history[tracker_id])
        majority_class = class_counts.most_common(1)[0][0]
        smoothed_class_ids.append(majority_class)

    # Assign smoothed class IDs back to detections
    detections.class_id = np.array(smoothed_class_ids)

    return detections


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize frame to specified dimensions.

    Args:
        frame: Input frame
        width: Target width
        height: Target height

    Returns:
        Resized frame
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def _save_results_as_csv(ctx: CCTVProcessingContext, results: list, partition_id: int) -> None:
    """Save count results as CSV file."""
    import pandas as pd

    if not results:
        logger.warning("No results to save as CSV")
        return

    partition_minutes = ctx.frame_processing.get("partition_minutes")
    if partition_minutes:
        csv_path = (
            ctx.counts_csv_path.parent / f"{ctx.counts_csv_path.stem}_part_{partition_id:04d}.csv"
        )
    else:
        csv_path = ctx.counts_csv_path

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to CSV: {ctx.counts_csv_path}")


def _save_results_as_json(ctx: CCTVProcessingContext, results: list, partition_id: int) -> None:
    """Save count results as JSON file."""
    if not results:
        logger.warning("No results to save as JSON")
        return

    partition_minutes = ctx.frame_processing.get("partition_minutes")
    if partition_minutes:
        json_path = (
            ctx.counts_json_path.parent
            / f"{ctx.counts_json_path.stem}_part_{partition_id:04d}.json"
        )
    else:
        json_path = ctx.counts_json_path

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to JSON: {ctx.counts_json_path}")


def _save_counts(
    ctx: CCTVProcessingContext, start_dt: datetime, end_dt: datetime, partition_id: int
) -> None:
    """Save count results in configured output formats."""
    results = _get_counts_results(ctx)

    # Add temporal information to each result
    for result in results:
        result["start_time"] = start_dt.isoformat()
        result["end_time"] = end_dt.isoformat()

    formats = ctx.output.get("formats", ["csv"])
    if not isinstance(formats, list):
        formats = [formats]

    for fmt in formats:
        if fmt == "csv":
            _save_results_as_csv(ctx, results, partition_id)
        elif fmt == "json":
            _save_results_as_json(ctx, results, partition_id)
        else:
            logger.warning(f"Unsupported output format: {fmt}. Skipping.")

    logger.info(f"Counts saved in {len(formats)} format(s)")


def _update_line_counters(ctx, detections):
    for line_counter in ctx.line_counters["line_counters"]:
        # Get the classes the current line counts
        mask = np.isin(detections.class_id, line_counter["classes"])
        line_detections = detections[mask]

        # Filter out tracks that already crossed
        if len(line_detections) > 0 and line_detections.tracker_id is not None:
            mask = ~np.isin(line_detections.tracker_id, list(line_counter["crossed"]))
            line_detections = line_detections[mask]

        crossed_in, crossed_out = line_counter["line_zone"].trigger(detections=line_detections)

        # Add newly crossed tracker_ids to the set
        if len(line_detections) > 0 and line_detections.tracker_id is not None:
            direction = line_counter["direction"]

            if direction == "in":
                newly_crossed_mask = crossed_in
            elif direction == "out":
                newly_crossed_mask = crossed_out
            else:
                raise ValueError(f"Invalid direction: {direction}")

            newly_crossed_detections = line_detections[newly_crossed_mask]
            line_counter["crossed"].update(newly_crossed_detections.tracker_id)

            bicycle_class_id = ctx.detection["label_class"]["bicycle"]
            if bicycle_class_id in line_counter["classes"]:
                mask = newly_crossed_detections.class_id == bicycle_class_id
                newly_crossed_bicycles = newly_crossed_detections[mask]

                line_counter["crossed_bicycles"].update(newly_crossed_bicycles.tracker_id)


def _initialize_frame_counter(ctx, video_info, total_frames):
    start_from_partition = ctx.frame_processing["start_from_partition"]

    frame_counter = FrameCounter(
        fps=video_info.fps,
        start_dt=ctx.video_start_datetime,
        total_frames=total_frames,
        partition_minutes=ctx.frame_processing["partition_minutes"],
        inference_interval=ctx.frame_processing["inference_interval"],
        start_from_partition=start_from_partition,
    )

    starting_frame = frame_counter.get_starting_frame()
    logger.info("Frame counter initialized:")
    logger.info(f"  Starting partition: {start_from_partition}")
    logger.info(f"  Starting frame: {starting_frame}")
    logger.info(f"  Inference interval: {ctx.frame_processing['inference_interval']}")
    logger.info(f"  Partition minutes: {ctx.frame_processing['partition_minutes']}")

    return frame_counter


def _initialize_sink_manager(ctx, output_info):
    sink_manager = SinkManager(
        target_path=ctx.output_video_path,
        video_info=output_info,
        save_video=ctx.output["save_video"],
        save_single=ctx.video_output["save_single"],
        start_from_partition=ctx.frame_processing["start_from_partition"],
    )

    if ctx.output["save_video"]:
        logger.info(f"Video output enabled: {ctx.output_video_path}")
        if ctx.video_output["save_single"]:
            logger.info("  Mode: Single continuous file")
        else:
            logger.info("  Mode: Partitioned files")
    else:
        logger.info("Video output disabled")

    return sink_manager


def _extract_cyclists(ctx, frame, detections):
    """Extract cyclist crops by matching bicycle and person detections."""
    # Get bicycle detections
    mask = np.isin(detections.class_id, ctx.detection["label_class"]["bicycle"])
    bicycle_detections = detections[mask]

    if len(bicycle_detections) == 0:
        return {}

    mask = np.isin(detections.class_id, ctx.detection["label_class"]["person"])
    person_detections = detections[mask]

    if len(person_detections) == 0:
        return {}

    # Calculate IoU between all bicycles and persons
    bicycle_boxes = bicycle_detections.xyxy
    person_boxes = person_detections.xyxy
    iou_matrix = sv.box_iou_batch(bicycle_boxes, person_boxes)

    # For each bicycle, find the person with highest IoU
    max_iou_per_bicycle = np.max(iou_matrix, axis=1)
    best_person_idx = np.argmax(iou_matrix, axis=1)

    # Define padding offsets (in pixels)
    padding = 4

    # Minimum IoU threshold to consider a match
    iou_threshold = 0.05

    crops = {}
    frame_height, frame_width = frame.shape[:2]

    for bicycle_idx, (person_idx, iou) in enumerate(zip(best_person_idx, max_iou_per_bicycle)):
        # Skip if IoU is too low (no meaningful overlap)
        if iou < iou_threshold:
            logger.debug(f"Skipping bicycle with low IoU: {iou:.3f}")
            continue

        # Get bicycle tracker_id
        tracker_id = bicycle_detections.tracker_id[bicycle_idx]

        if tracker_id is None:
            logger.debug("Skipping detection without tracker_id")
            continue

        # Get person bounding box
        x1, y1, x2, y2 = person_boxes[person_idx]

        # Add padding with boundary checks
        x1_padded = max(0, int(x1 - padding))
        y1_padded = max(0, int(y1 - padding * 2))  # double padding for top
        x2_padded = min(frame_width, int(x2 + padding))
        y2_padded = min(frame_height, int(y2 + padding))

        # Validate crop dimensions
        if x2_padded <= x1_padded or y2_padded <= y1_padded:
            logger.warning(f"Invalid crop dimensions for tracker {tracker_id}, skipping")
            continue

        # Crop from frame
        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]

        # Store with tracker_id
        crops[int(tracker_id)] = crop

        logger.debug(
            f"Extracted cyclist crop: tracker_id={tracker_id}, "
            f"size={crop.shape[1]}x{crop.shape[0]}, IoU={iou:.3f}"
        )

    logger.debug(
        f"Extracted {len(crops)} cyclist crops from {len(bicycle_detections)} bicycle detections"
    )

    return crops


def _process_cyclists(ctx, cyclists):
    if len(cyclists) == 0:
        logger.debug("No cyclists to process")
        return Classifications.empty()

    if ctx.classification.get("online"):
        logger.debug("Classification method set to online. Classifying...")
        classifications = get_classifications_from_images(
            cyclists,
            model=ctx.classification_model,
            classes=ctx.classification.get("labels"),
            threshold=ctx.classification.get("threshold"),
            input_size=ctx.classification.get("input_size"),
        )

        gender_history = ctx.classification["tracker_gender_history"]
        for _, tracker_id, prediction, _ in classifications:
            if tracker_id not in gender_history:
                gender_history[tracker_id] = []

            if classifications.classes:
                gender_history[tracker_id].append(classifications.classes[prediction])
            else:
                gender_history[tracker_id].append(prediction)

        logger.debug(f"len classifications object: {len(classifications)}")
        return classifications

    return Classifications.empty()


def _save_cyclists(
    ctx: CCTVProcessingContext, cyclists: Dict[int, np.ndarray], frame_number: int
) -> None:
    """Save cyclist crops to disk with tracker_id and frame number in filename."""

    if not cyclists:
        return

    for tracker_id, crop in cyclists.items():
        filename = f"{tracker_id}_frame{frame_number:06d}.jpg"
        filepath = ctx.crops_dir / filename

        try:
            cv2.imwrite(str(filepath), crop)
            logger.debug(f"Saved crop: {filename}")
        except Exception as e:
            logger.error(f"Failed to save crop {filename}: {e}")

    logger.debug(f"Saved {len(cyclists)} cyclist crops for frame {frame_number}")


def _process_video(ctx: CCTVProcessingContext) -> None:
    """Process video with detection, tracking, and counting."""
    # TODO: Ddd the classification part.
    logger.info("Starting video processing")

    # Get input and output video info
    input_info = sv.VideoInfo.from_video_path(ctx.video_path)
    output_info = _get_output_video_info(ctx)

    # Calculate total inference frames
    inference_limit = ctx.frame_processing.get("inference_limit")
    if inference_limit:
        total_frames = inference_limit
    else:
        total_frames = input_info.total_frames

    # Get input video info
    frames_generator = sv.get_video_frames_generator(ctx.video_path)

    # Create video counter and sink manager
    counter = _initialize_frame_counter(ctx, input_info, total_frames)
    sink = _initialize_sink_manager(ctx, output_info)

    logger.info("Beginning frame processing loop")

    processed_frames = 0

    for frame in tqdm(frames_generator, total=total_frames):
        should_process = counter.increment()

        if not should_process:
            continue

        processed_frames += 1

        detections = _process_frame(ctx, frame)

        if ctx.classification.get("enabled"):
            cyclists = _extract_cyclists(ctx, frame, detections)

            logger.debug(f"Length of extracted cyclist dictionary: {len(cyclists)}")

            if ctx.output.get("keep_crops"):
                _save_cyclists(ctx, cyclists, counter.total_frames)

            classifications = _process_cyclists(ctx, cyclists)
        else:
            classifications = Classifications.empty()

        _update_line_counters(ctx, detections)

        if ctx.output.get("save_video"):
            annotated_frame = _annotate_frame(ctx, frame, detections, classifications)
            annotated_frame = _resize_frame(annotated_frame, output_info.width, output_info.height)
            sink.write_frame(annotated_frame)

        if counter.is_last_video_frame():
            break

        if counter.is_last_partition_frame():
            start_dt, end_dt = counter.get_partition_start_end()

            if ctx.output.get("save_counts", True):
                _save_counts(ctx, start_dt, end_dt, counter.partition_index)

            if counter.is_last_video_frame():
                break

            logger.info(
                f"Partition {counter.partition_index} with start time {start_dt} and end time {end_dt} ended. "
                "Starting new partition."
            )

            if not ctx.output.get("save_single"):
                sink.start_new_partition()

            _restart_line_counters(ctx)

    start_dt, end_dt = counter.get_partition_start_end()
    if ctx.output.get("save_counts", True):
        _save_counts(ctx, start_dt, end_dt, counter.partition_index)

    sink.close()


def process_cctv(ctx: CCTVProcessingContext):

    if ctx.system["environment"] == "colab":
        _copy_video_to_colab(ctx)

    _initialize_models(ctx)

    _initialize_tracker(ctx)

    _initialize_counters(ctx)

    if ctx.output["save_video"]:
        _initialize_annotators(ctx)

    _process_video(ctx)
