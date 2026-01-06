import logging

import torch
import torch.nn as nn

from .types import CCTVProcessingContext

logger = logging.getLogger(__name__)


def _initialize_yolo_model(ctx: CCTVProcessingContext) -> None:
    """Load YOLO detection model from weights."""
    from ultralytics import YOLO

    try:
        model = YOLO(str(ctx.detection_model_path))
        ctx.detection_model = model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"YOLO model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def _initialize_rfdetr_model(ctx: CCTVProcessingContext) -> None:
    """Load RFDETR detection model from weights."""
    from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall

    # Determine which RFDETR variant to load
    architecture = ctx.detection["model_architecture"].lower()

    model_mapping = {
        "rfdetr_base": RFDETRBase,
        "rfdetr_nano": RFDETRNano,
        "rfdetr_small": RFDETRSmall,
        "rfdetr_medium": RFDETRMedium,
    }

    if architecture not in model_mapping:
        raise ValueError(
            f"Unknown RFDETR variant: {architecture}. " f"Supported: {list(model_mapping.keys())}"
        )

    try:
        model_class = model_mapping[architecture]
        model = model_class(pretrain_weights=str(ctx.detection_model_path))
        model.optimize_for_inference()

        ctx.detection_model = model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"RFDETR model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load RFDETR model: {e}")
        raise


def _initialize_detection_model(ctx: CCTVProcessingContext) -> None:
    """Initialize detection model based on architecture type."""
    architecture = ctx.detection["model_architecture"].lower()

    logger.info(f"Loading detection model: {architecture}")
    logger.info(f"Loading custom checkpoint from: {ctx.detection_model_path}")

    if "yolo" in architecture:
        _initialize_yolo_model(ctx)
    elif "rfdetr" in architecture:
        _initialize_rfdetr_model(ctx)
    else:
        raise ValueError(
            f"Unsupported detection architecture: {architecture}. " f"Supported: 'yolo', 'rfdetr'"
        )


def _initialize_classification_model(ctx: CCTVProcessingContext) -> nn.Module:
    """Create model with trial-specific parameters."""
    import timm
    import torch

    architecture = ctx.classification["model_architecture"].lower()

    logger.info(f"Loading detection model: {architecture}")

    # Create base model
    model = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=len(ctx.classification["labels"]),
        drop_rate=0,
    )

    # Load custom checkpoint
    logger.info(f"Loading custom checkpoint from: {ctx.classification_model_path}")

    try:
        # Use weights_only=False for trusted checkpoints
        checkpoint_data = torch.load(ctx.classification_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        logger.info("Custom checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint {ctx.classification_model_path}: {e}")
        raise

    # Move to device and set to eval mode
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    ctx.classification_model = model

    logger.info(f"CNN model loaded successfully on {device}")


def _initialize_models(ctx: CCTVProcessingContext) -> None:
    """Initialize detection and classification models."""
    logger.info("Initializing models")

    # Initialize detection model (required)
    _initialize_detection_model(ctx)
    logger.info(f"Detection model loaded: {ctx.detection['model_architecture']}")

    # Initialize classification model (optional)
    if ctx.classification.get("enabled"):
        _initialize_classification_model(ctx)
        logger.info(f"Classification model loaded: {ctx.classification['model_architecture']}")
    else:
        logger.info("Classification disabled, skipping classification model")

    logger.info("Model initialization complete")
