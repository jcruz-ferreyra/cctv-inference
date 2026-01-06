# deduplicate_frames/__init__.py
from .cctv_processing import process_cctv
from .types import CCTVProcessingContext

__all__ = [
    # Frames processing
    "process_cctv",
    # Data types
    "CCTVProcessingContext",
]