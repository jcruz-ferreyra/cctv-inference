from pathlib import Path

from cyclist_census.config import DRIVE_MODELS_DIR as MODELS_DIR
from cyclist_census.config import LOCAL_DATA_DIR as DATA_DIR
from cyclist_census.utils import check_missing_keys, load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from cyclist_census.process_cctv import CCTVProcessingContext, process_cctv

logger.info("Starting CCTV processing pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

# Validate required top-level keys
required_keys = [
    "input_folder",
    "output_folder",
    "video_name",
    "frame_processing",
    "detection",
    "tracking",
    "classification",
    "line_counters",
    "output",
    "video_output",
    "system",
]
check_missing_keys(required_keys, script_config)

# Parse top-level configuration
INPUT_FOLDER = Path(script_config["input_folder"])
OUTPUT_FOLDER = Path(script_config["output_folder"])
VIDEO_NAME = script_config["video_name"]

# Parse configuration sections
FRAME_PROCESSING = script_config["frame_processing"]
DETECTION = script_config["detection"].copy()
TRACKING = script_config["tracking"]
CLASSIFICATION = script_config["classification"]
LINE_COUNTERS = script_config["line_counters"]
OUTPUT = script_config["output"]
VIDEO_OUTPUT = script_config["video_output"]
SYSTEM = script_config["system"]

DETECTION["class_confidence"] = [
    (DETECTION["category_classes"][k], v) for k, v in DETECTION["category_confidence"].items()
]

logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"Processing video: {VIDEO_NAME}")

# Create processing context
context = CCTVProcessingContext(
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    video_name=VIDEO_NAME,
    frame_processing=FRAME_PROCESSING,
    detection=DETECTION,
    tracking=TRACKING,
    classification=CLASSIFICATION,
    line_counters=LINE_COUNTERS,
    output=OUTPUT,
    video_output=VIDEO_OUTPUT,
    system=SYSTEM,
)

# Task main function
process_cctv(context)
