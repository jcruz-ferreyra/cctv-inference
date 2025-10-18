import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the HOME and DATA paths from the .env file and convert to a Path object
LOCAL_DIR = Path(os.getenv("LOCAL_DIR")).resolve()

DATA_FOLDER = Path(os.getenv("DATA_FOLDER"))
MODELS_FOLDER = Path(os.getenv("MODELS_FOLDER"))

LOCAL_DATA_DIR = LOCAL_DIR / DATA_FOLDER
LOCAL_MODELS_DIR = LOCAL_DIR / MODELS_FOLDER

for x in ["LOCAL_DIR"]:
    path = locals()[x]
    if not path.exists():
        raise ValueError(f"{x} path '{path}' from .env does not exist.")

# Prepare Drive paths, and check correct access to them
DRIVE_DIR = os.getenv("DRIVE_DIR")

if DRIVE_DIR is not None and Path(DRIVE_DIR).exists():
    DRIVE_DIR = Path(DRIVE_DIR)

    DRIVE_DATA_DIR = DRIVE_DIR / DATA_FOLDER
    DRIVE_MODELS_DIR = DRIVE_DIR / MODELS_FOLDER

else:
    DRIVE_DATA_DIR = None
    DRIVE_MODELS_DIR = None
