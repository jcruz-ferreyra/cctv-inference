from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CCTVProcessingContext:
    """Context for CCTV video processing pipeline with detection, tracking, and classification."""

    # Top-level paths (from .env)
    data_dir: Path
    models_dir: Path

    # Top-level configuration
    input_folder: Path
    output_folder: Path
    video_name: str

    # Configuration sections (match YAML structure)
    frame_processing: Dict[str, Any]
    detection: Dict[str, Any]
    tracking: Dict[str, Any]
    classification: Dict[str, Any]
    line_counters: Dict[str, Any]
    output: Dict[str, Any]
    video_output: Dict[str, Any]
    system: Dict[str, Any]

    # Runtime objects (initialized during processing)
    colab_data_dir: Path = Path("/content/data")

    detection_model: Optional[Any] = None
    classification_model: Optional[Any] = None
    tracker: Optional[Any] = None

    annotators: Dict[str, Any] = field(default_factory=dict)

    @property
    def local_video_path(self) -> Path:
        """Full path to input video file."""
        return self.data_dir / self.input_folder / self.video_name

    @property
    def colab_video_path(self) -> Path:
        """Full path to input video file in Colab environment."""
        return self.colab_data_dir / self.input_folder / self.video_name

    @property
    def output_dir(self) -> Path:
        """Output directory for this specific video."""
        video_path = Path(self.video_name)
        parent_dir = video_path.parent

        # If video_name has no directory (parent is "."), don't append anything
        if parent_dir == Path("."):
            return self.data_dir / self.output_folder
        else:
            return self.data_dir / self.output_folder / parent_dir

    @property
    def crops_dir(self) -> Path:
        """Directory for temporary cyclist crops."""
        video_stem = Path(self.video_name).stem
        return self.output_dir / f"{video_stem}_crops"

    @property
    def counts_json_path(self) -> Path:
        """Path to counts JSON file."""
        video_stem = Path(self.video_name).stem
        return self.output_dir / f"{video_stem}_counts.json"

    @property
    def counts_csv_path(self) -> Path:
        """Path to counts CSV file."""
        video_stem = Path(self.video_name).stem
        return self.output_dir / f"{video_stem}_counts.csv"

    @property
    def output_video_path(self) -> Path:
        """Path to output annotated video."""
        video_stem = Path(self.video_name).stem
        return self.output_dir / f"{video_stem}_output.mp4"

    @property
    def detection_model_path(self) -> Path:
        """Full path to detection model weights."""
        return self.models_dir / self.detection["model_weights"]

    @property
    def classification_model_path(self) -> Optional[Path]:
        """Full path to classification model weights if classification enabled."""
        if self.classification.get("enabled", False):
            return self.models_dir / self.classification["model_weights"]
        return None

    @property
    def line_counters_path(self) -> Path:
        """Full path to line counter file (in input folder with videos)."""
        counter_file = self.line_counters["source"]
        return self.data_dir / self.input_folder / counter_file

    @property
    def video_start_datetime(self) -> datetime:
        """Parse video start time as datetime object."""
        start_time = self.frame_processing["video_start_time"]
        return datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    @property
    def video_end_datetime(self) -> Optional[datetime]:
        """Parse video end time as datetime object if provided."""
        end_time = self.frame_processing.get("video_end_time")
        if end_time:
            return datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        return None

    def __post_init__(self):
        """Validate paths and configuration."""
        # Validate input video exists
        if not self.local_video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.local_video_path}")

        # Validate detection model weights exist
        if not self.detection_model_path.exists():
            raise FileNotFoundError(
                f"Detection model weights not found:  {self.detection_model_path}"
            )

        # Validate classification model weights if classification enabled
        if self.classification.get("enabled", False):
            if not self.classification_model_path:
                raise ValueError("Classification enabled but model_weights not provided")
            if not self.classification_model_path.exists():
                raise FileNotFoundError(
                    f"Classification model weights not found: {self.classification_model_path}"
                )

        # Validate line counter file exists
        if not self.line_counters_path.exists():
            raise FileNotFoundError(f"Line counter file not found: {self.line_counters_path}")

        # Validate configuration values
        if self.frame_processing["inference_interval"] < 1:
            raise ValueError("inference_interval must be >= 1")

        if self.frame_processing["frame_batch_size"] < 1:
            raise ValueError("frame_batch_size must be >= 1")

        if self.frame_processing["partition_minutes"] <= 0:
            self.frame_processing["start_from_partition"] = 0
            self.video_output["save_single"] = True

        # Validate datetime format
        try:
            _ = self.video_start_datetime
            if self.frame_processing.get("video_end_time"):
                _ = self.video_end_datetime
        except ValueError as e:
            raise ValueError(f"Invalid datetime format (expected YYYY-MM-DD HH:MM:SS): {e}")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        offline = not self.classification.get("online", False)
        online_but_keep_crops = self.output.get("keep_crops", False)
        if self.classification.get("enabled", False) and (offline or online_but_keep_crops):
            self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Create the label_class dictionary
        self.detection["label_class"] = {v: k for k, v in self.detection["class_label"].items()}

        # Define classification model input size
        if self.classification.get("enabled"):

            architecture = self.classification.get("model_architecture")
            if not architecture:
                raise ValueError("Classification architecture must be specified")

            # EfficientNet specific sizes
            efficientnet_sizes = {
                "efficientnet_b0": 224,
                "efficientnet_b1": 240,
                "efficientnet_b2": 260,
                "efficientnet_b3": 300,
                "efficientnet_b4": 380,
                "efficientnet_b5": 456,
            }

            if architecture in efficientnet_sizes:
                self.classification["input_size"] = efficientnet_sizes[architecture]
            elif "resnet" in architecture.lower():
                self.classification["input_size"] = 224
            else:
                raise ValueError(f"Unknown classification architecture '{architecture}'")

        # Create classification history
        if self.classification.get("enabled"):
            self.classification["tracker_gender_history"] = {}
