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

    # Detection and count parameters (required)
    detection: Dict[str, Any]
    line_counters: Dict[str, Any]

    # System paramters (environment definition required)
    system: Dict[str, Any]

    # Configuration sections (match YAML structure)
    frame_processing: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    video_output: Optional[Dict[str, Any]] = None
    
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

        # Validate line counter file exists
        if not self.line_counters_path.exists():
            raise FileNotFoundError(f"Line counter file not found: {self.line_counters_path}")

        # Fill default values for optional sections
        self.frame_processing = _fill_frame_processing_with_defaults(self.frame_processing)
        self.detection = _fill_detection_with_defaults(self.detection)
        self.tracking = _fill_tracking_with_defaults(self.tracking)
        self.classification = _fill_classification_with_defaults(self.classification)
        self.output = _fill_output_with_defaults(self.output)
        self.video_output = _fill_video_output_with_defaults(self.video_output, self.output)

        # Validate frame processing values
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

        # Validate classification model parameters if enabled
        if self.classification.get("enabled", False):
            if not self.classification_model_path:
                raise ValueError("Classification enabled but model_weights not provided")
            if not self.classification_model_path.exists():
                raise FileNotFoundError(
                    f"Classification model weights not found: {self.classification_model_path}"
                )

            architecture = self.classification.get("model_architecture")
            if not architecture:
                raise ValueError("Classification architecture must be specified")

            # Get model input size based on provided architecture
            self.classification["input_size"] = _get_classification_model_input_size(architecture)

        # Create classification history
        if self.classification.get("enabled"):
            self.classification["tracker_gender_history"] = {}


def _get_classification_model_input_size(architecture: str):

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
        return efficientnet_sizes[architecture]
    elif "resnet" in architecture.lower():
        return 224
    else:
        raise ValueError(f"Unknown classification architecture '{architecture}'")


def _fill_frame_processing_with_defaults(frame_processing: Optional[Dict]) -> Dict:
    """Fill missing frame processing parameters with default values."""
    defaults = {
        "inference_interval": 1,
        "partition_minutes": 0,
        "frame_batch_size": 32,
        "video_start_time": "1900-01-01 00:00:00",
        "video_end_time": None,
        "start_from_partition": 0,
        "inference_limit": 0,
    }

    # If no frame_processing provided at all, use all defaults
    if frame_processing is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **frame_processing}


def _fill_detection_with_defaults(detection: Dict) -> Dict:
    """Fill missing detection parameters with default values."""

    # Required fields validation (should exist before calling this)
    required = ["model_architecture", "model_weights", "class_label"]
    missing = [key for key in required if key not in detection]
    if missing:
        raise ValueError(f"Missing required detection fields: {missing}")

    # Simple defaults
    defaults = {
        "batch_size": 16,
    }

    # Merge simple defaults
    detection = {**defaults, **detection}

    # Fill model_params based on architecture FIRST
    if "model_params" not in detection or detection["model_params"] is None:
        detection["model_params"] = {}

    model_arch = detection["model_architecture"].lower()

    if "yolo" in model_arch:
        model_params_defaults = {
            "imgsz": 640,
            "verbose": False,
            "conf": 0.25,
            "iou": 0.6,
            "agnostic_nms": False,
        }
    else:
        raise NotImplementedError(
            f"Model architecture '{detection['model_architecture']}' not implemented. "
            f"Only YOLO models are currently supported."
        )

    detection["model_params"] = {**model_params_defaults, **detection["model_params"]}

    # Fill category_classes from class_label if not provided
    if "category_classes" not in detection or detection["category_classes"] is None:
        # Create {label: [class_id]} mapping
        detection["category_classes"] = {
            label: [class_id] for class_id, label in detection["class_label"].items()
        }

    # Fill category_confidence from category_classes if not provided
    # Use the conf from model_params (now guaranteed to exist)
    if "category_confidence" not in detection or detection["category_confidence"] is None:
        conf_threshold = detection["model_params"]["conf"]
        detection["category_confidence"] = {
            category: conf_threshold for category in detection["category_classes"].keys()
        }

    return detection


def _fill_tracking_with_defaults(tracking: Optional[Dict]) -> Dict:
    """Fill missing tracking parameters with default values."""

    defaults = {
        "tracker_type": "bytetrack",
        "class_history_length": 5,
    }

    # If no tracking provided at all, use all defaults
    if tracking is None:
        tracking = {}

    # Merge simple defaults
    tracking = {**defaults, **tracking}

    # Validate tracker_type
    tracker_type = tracking["tracker_type"].lower()
    if tracker_type != "bytetrack":
        raise NotImplementedError(
            f"Tracker type '{tracking['tracker_type']}' not implemented. "
            f"Only ByteTrack is currently supported."
        )

    # Fill tracker_params based on tracker type
    if "tracker_params" not in tracking or tracking["tracker_params"] is None:
        tracking["tracker_params"] = {}

    if tracker_type == "bytetrack":
        tracker_params_defaults = {
            "minimum_matching_threshold": 0.75,
        }

    tracking["tracker_params"] = {**tracker_params_defaults, **tracking["tracker_params"]}

    return tracking


def _fill_classification_with_defaults(classification: Optional[Dict]) -> Dict:
    """Fill missing classification parameters with default values."""

    # If no classification provided at all, default to disabled
    if classification is None:
        return {"enabled": False}

    # If classification exists but enabled not specified, default to disabled
    if "enabled" not in classification:
        classification["enabled"] = False

    # If disabled, return as-is (no other fields needed)
    if not classification["enabled"]:
        return classification

    # Classification is enabled - validate required fields
    required = ["model_architecture", "model_weights", "labels"]
    missing = [key for key in required if key not in classification]
    if missing:
        raise ValueError(
            f"Classification is enabled but missing required fields: {missing}. "
            f"Required: model_architecture, model_weights, labels"
        )

    # Fill defaults for optional fields when enabled
    defaults = {
        "online": True,
        "model_params": {},
        "threshold": 0.5,
        "batch_size": 64,
    }

    # Merge defaults with user values
    classification = {**defaults, **classification}

    return classification


def _fill_output_with_defaults(output: Optional[Dict]) -> Dict:
    """Fill missing output parameters with default values."""

    defaults = {
        "save_video": False,
        "keep_crops": False,
        "save_counts": True,
        "formats": ["json", "csv"],
    }

    # If no output provided at all, use all defaults
    if output is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **output}


def _fill_video_output_with_defaults(video_output: Optional[Dict], output: Dict) -> Dict:
    """Fill missing video output parameters with default values."""

    # If save_video is False, video_output not needed
    if not output.get("save_video", False):
        return {} if video_output is None else video_output

    # save_video is True - provide full defaults
    defaults = {
        "resolution_ratio": 1.0,
        "save_single": False,
        "bboxes": {
            "draw": True,
            "params": {
                "thickness": 4,
            },
        },
        "labels": {
            "draw": True,
            "params": {
                "text_scale": 1,
                "text_padding": 3,
            },
        },
        "line_counters": {
            "draw": True,
            "params": {
                "text_thickness": 1,
                "text_scale": 0.8,
                "text_offset": 1,
                "text_padding": 10,
            },
        },
        "results": {
            "draw": False,
            "params": {
                "position": "top-right",
                "text_scale": 1,
                "text_padding": 3,
            },
        },
        "codec": "mp4",
    }

    # If no video_output provided, use all defaults
    if video_output is None:
        return defaults

    # Deep merge for nested structures
    merged = {**defaults}

    for key, value in video_output.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Merge nested dict (e.g., bboxes, labels, etc.)
            merged[key] = {**merged[key], **value}

            # Handle params sub-dict if present
            if "params" in merged[key] and "params" in value:
                merged[key]["params"] = {**merged[key]["params"], **value["params"]}
        else:
            # Direct override for non-dict values
            merged[key] = value

    return merged


def _fill_system_with_defaults(system: Dict) -> Dict:
    """Fill missing system parameters with default values."""

    # Validate required field
    if "environment" not in system:
        raise ValueError(
            "Missing required system field: 'environment'. " "Must be 'local' or 'colab'."
        )

    # Validate environment value
    valid_environments = ["local", "colab"]
    if system["environment"] not in valid_environments:
        raise ValueError(
            f"Invalid environment: '{system['environment']}'. "
            f"Must be one of: {valid_environments}"
        )

    # Fill optional fields with defaults
    defaults = {
        "num_workers": 4,
        "mixed_precision": True,
    }

    # Merge: defaults first, then override with user values
    return {**defaults, **system}
