from datetime import datetime, timedelta
import logging
import os

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


class FrameCounter:
    """
    Tracks frame counts, controls which frames to process, and manages interval completions.
    Treats "no partitioning" as a single partition encompassing the entire video.
    """

    def __init__(
        self,
        fps: float,
        start_dt: datetime,
        total_frames: int,
        partition_minutes: int = 0,
        inference_interval: int = 1,
        start_from_partition: int = 0,
    ):
        """
        Args:
            fps: Frames per second of the video
            start_dt: Video start datetime
            total_frames: Total number of frames in the video
            partition_minutes: Minutes per partition interval (0 or None means no partitions)
            inference_interval: Process every Nth frame (1 = process all frames, 2 = every other, etc.)
            start_from_partition: Partition index to resume from (0-indexed, 0 = beginning)
        """
        self.fps = fps
        self.start_dt = start_dt
        self.total_frames_expected = total_frames
        self.partition_minutes = partition_minutes
        self.inference_interval = inference_interval
        self.start_from_partition = start_from_partition

        # Calculate frames per partition
        if partition_minutes and partition_minutes > 0:
            self.frames_per_partition = int(partition_minutes * 60 * fps)
            if self.frames_per_partition <= 1:
                raise ValueError(
                    f"Provided partition minutes {partition_minutes} would generate single frame partitions. "
                    f"Try increasing the partition minutes value."
                )
        else:
            self.frames_per_partition = total_frames
            self.start_from_partition = 0

        # Counters
        self._total_frames = 0
        self._partition_frames = 0
        self._partition_index = 0
        self._is_last_partition_frame = False
        self._is_last_video_frame = False

    def get_starting_frame(self) -> int:
        """
        Calculate the frame number to start processing from based on start_from_partition.

        Returns:
            int: Frame number (0-indexed) to begin processing
        """
        return self.start_from_partition * self.frames_per_partition

    def increment(self) -> bool:
        """
        Increment frame counter. Call this at the start of each loop iteration.

        Returns:
            bool: True if this frame should be processed (based on inference_interval)
        """
        # Update partition tracking variables if previous frame was the last of its partition
        # Warning: This logic will fail if we create single frame partitions (non-sense)
        if self._is_last_partition_frame:
            self._partition_index += 1
            self._partition_frames = 0  # assign 1 for the current frame and return
            self._is_last_partition_frame = False

            logger.debug(
                f"Starting partition {self._partition_index} at frame {self._total_frames}"
            )

        self._total_frames += 1
        self._partition_frames += 1

        logger.debug(
            f"Frame {self._total_frames}: Partition {self._partition_index}, "
            f"Partition Frame {self._partition_frames}/{self.frames_per_partition}"
        )

        # Check if frame is past starting point
        has_passed_starting_point = self._total_frames > self.get_starting_frame()
        logger.debug(
            f"Has passed starting point ({self.get_starting_frame()}): {has_passed_starting_point}"
        )

        # Check if is the last frame in the video based on expected frames. Process if it is
        self._is_last_video_frame = self._total_frames >= self.total_frames_expected
        logger.debug(f"Is last video frame: {self._is_last_video_frame}")

        # Check if is the last frame in the partition based on partition count. Process if it is
        self._is_last_partition_frame = self._partition_frames >= self.frames_per_partition
        logger.debug(f"Is last partition frame: {self._is_last_partition_frame}")

        is_first_partition_frame = self._partition_frames == 1
        logger.debug(f"Is first partition frame: {is_first_partition_frame}")

        if has_passed_starting_point:
            matches_interval = (self._partition_frames - 1) % self.inference_interval == 0
            logger.debug(f"Matches inference interval: {matches_interval}")

            should_process = (
                is_first_partition_frame
                | self._is_last_partition_frame
                | self._is_last_video_frame
                | matches_interval
            )
            logger.debug(f"Should process frame: {should_process}")

            return should_process

        return False

    def is_last_partition_frame(self) -> bool:
        """
        Check if a partition interval just completed.
        Call this after increment() to know when to save counts and rotate video.
        Only returns True for partitions that were actually processed (>= start_from_partition).

        Returns:
            bool: True if partition just finished on the last increment() call
        """
        return self._is_last_partition_frame

    def is_last_video_frame(self) -> bool:
        """
        Check if video has ended (reached total_frames_expected).

        Returns:
            bool: True if video processing is complete
        """
        return self._is_last_video_frame

    def get_partition_start_end(self) -> tuple[datetime, datetime]:
        """
        Get start and end datetime for the current partition.

        Returns:
            tuple: (start_dt, end_dt) for the partition that just finished
        """
        if self.partition_minutes and self.partition_minutes > 0:
            # Calculate partition start
            partition_start = self.start_dt + timedelta(
                minutes=self._partition_index * self.partition_minutes
            )

            # Calculate partition end (either full partition or remaining video time)
            full_partition_end = partition_start + timedelta(minutes=self.partition_minutes)
            video_end = self.start_dt + timedelta(seconds=self.total_frames_expected / self.fps)

            partition_end = min(full_partition_end, video_end)
        else:
            # Single partition (entire video)
            partition_start = self.start_dt
            partition_end = self.start_dt + timedelta(
                seconds=self.total_frames_expected / self.fps
            )

        return partition_start, partition_end

    @property
    def total_frames(self) -> int:
        """Total frames seen (including skipped frames)."""
        return self._total_frames

    @property
    def interval_frames(self) -> int:
        """Frames in current partition interval."""
        return self._partition_frames

    @property
    def partition_index(self) -> int:
        """Current partition index."""
        return self._partition_index


class SinkManager:
    """
    Manages video file creation and writing.
    Call start_new_partition() manually to rotate to a new video file.
    """

    def __init__(
        self,
        video_info: sv.VideoInfo,
        target_path: str,
        codec: str = "mp4v",
        save_video: bool = True,
        save_single: bool = True,
        start_from_partition: int = 0,
    ):
        """
        Args:
            video_info: VideoInfo with resolution, fps, etc.
            target_path: Base path for output files (extension will be removed if present)
            codec: FOURCC codec (default: "mp4v")
            save_video: Whether to save video at all
            save_single: If True, save as single file. If False, use partitioned mode
        """
        self.video_info = video_info
        # Remove extension if present
        self.target_path = os.path.splitext(target_path)[0]
        self.__codec = codec
        self.save_video = save_video
        self.save_single = save_single

        self._partition_index = start_from_partition
        self.__current_sink = None

        # Determine mode
        if not self.save_video:
            self._mode = "none"
        elif self.save_single:
            self._mode = "single"
        else:
            self._mode = "partitioned"

        # Initialize first sink if needed
        if self._mode == "single":
            self.__start_single_sink()
        elif self._mode == "partitioned":
            self.__start_partition()

    def __start_single_sink(self):
        """Start the single video sink."""
        path = f"{self.target_path}.mp4"
        self.__current_sink = sv.VideoSink(path, self.video_info, self.__codec)
        self.__current_sink.__enter__()

    def __start_partition(self):
        """Start a new partition sink."""
        partition_path = f"{self.target_path}_part_{self._partition_index:04d}.mp4"
        self.__current_sink = sv.VideoSink(partition_path, self.video_info, self.__codec)
        self.__current_sink.__enter__()
        self._partition_index += 1

    def start_new_partition(self):
        """
        Close current partition and start a new one.
        Only works in partitioned mode. Call this when your interval completes.
        """
        if self._mode != "partitioned":
            return

        # Close current sink
        if self.__current_sink:
            self.__current_sink.__exit__(None, None, None)

        # Start new partition
        self.__start_partition()

    def write_frame(self, frame: np.ndarray):
        """Write frame to current sink (no-op if not saving)."""
        if self.__current_sink:
            self.__current_sink.write_frame(frame)

    def close(self):
        """Close the current sink and clean up resources."""
        if self.__current_sink:
            self.__current_sink.__exit__(None, None, None)
            self.__current_sink = None
