# utils/yaml_config.py
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file from the calling script's directory."""
    try:
        with open(config_path, "r") as f:
            script_config = yaml.safe_load(f)
        return script_config
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading config: {e}")
        raise


def check_missing_keys(required_keys, script_config):
    missing_keys = [key for key in required_keys if key not in script_config]
    if missing_keys:
        logger.error(f"Missing required config keys: {missing_keys}")
        raise ValueError(f"Missing required config keys: {missing_keys}")
