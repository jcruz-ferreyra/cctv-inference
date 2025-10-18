import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import cv2
import numpy as np
from rfdetr import RFDETRBase
import supervision as sv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from ultralytics import YOLO

from .classification import Classifications

logger = logging.getLogger(__name__)


def get_ultralytics_detections(
    frame: np.ndarray,
    model: YOLO,
    model_params: Dict[str, Any],
    class_confidence: List[Tuple[List[int], float]] = None,
    bgr: bool = False,
) -> sv.Detections:
    """
    Runs object detection on a single frame and filters out low-confidence detections
    for specified class groups.

    Args:
        frame (np.ndarray): Input image/frame as a NumPy array (e.g., BGR or RGB).
        model (YOLO): An Ultralytics YOLO model or compatible callable model.
        model_params (Dict[str, Any]): Parameters to pass into the model's forward call (e.g., conf, iou, imgsz).
        class_confidence (List[Tuple[List[int], float]], optional):
            List of (class_ids, threshold) pairs. Detections belonging to any class in class_ids
            below the given threshold will be filtered out.
        bgr (bool): Whether the input frame is in BGR format. If True, converts to RGB for inference.

    Returns:
        sv.Detections: Filtered detections compatible with the Supervision library.
    """
    # Create a copy to avoid modifying the original frame
    frame_copy = frame.copy()

    # Convert BGR to RGB if needed
    if bgr:
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

    results = model(frame_copy, **model_params)[0]
    detections = sv.Detections.from_ultralytics(results)

    if class_confidence:
        for class_ids, threshold in class_confidence:
            is_in_classes = np.isin(detections.class_id, class_ids)
            is_below_threshold = detections.confidence < threshold
            detections = detections[~(is_in_classes & is_below_threshold)]

    return detections


def get_roboflow_detections(
    frame: np.ndarray,
    model: RFDETRBase,
    class_confidence: List[Tuple[List[int], float]] = None,
    bgr: bool = False,
) -> sv.Detections:
    """
    Runs object detection on a single frame and filters out low-confidence detections
    for specified class groups.

    Args:
        frame (np.ndarray): Input image/frame as a NumPy array (e.g., BGR or RGB).
        model (YOLO): An Ultralytics YOLO model or compatible callable model.
        model_params (Dict[str, Any]): Parameters to pass into the model's forward call (e.g., conf, iou, imgsz).
        class_confidence (List[Tuple[List[int], float]], optional):
            List of (class_ids, threshold) pairs. Detections belonging to any class in class_ids
            below the given threshold will be filtered out.
        bgr (bool): Whether the input frame is in BGR format. If True, converts to RGB for inference.

    Returns:
        sv.Detections: Filtered detections compatible with the Supervision library.
    """
    # Create a copy to avoid modifying the original frame
    frame_copy = frame.copy()

    # Convert BGR to RGB if needed
    if bgr:
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

    detections = model.predict(frame_copy, threshold=0.25)

    if class_confidence:
        for class_ids, threshold in class_confidence:
            is_in_classes = np.isin(detections.class_id, class_ids)
            is_below_threshold = detections.confidence < threshold
            detections = detections[~(is_in_classes & is_below_threshold)]

    return detections


def _preprocess_images(images: List[np.ndarray], input_size: int) -> torch.Tensor:
    """
    Preprocess list of BGR images to a batched tensor ready for model inference.

    Args:
        images: List of BGR numpy arrays (OpenCV format)
        input_size: Target size for resizing (e.g., 224)

    Returns:
        Batched tensor of shape (N, 3, input_size, input_size)
    """
    # Create transforms pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # ImageNet stats
            ),
        ]
    )

    tensors = []
    for img in images:
        # Convert BGR (OpenCV) to RGB (PIL/PyTorch convention)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # Apply transforms
        tensor = transform(pil_img)
        tensors.append(tensor)

    # Stack into batch (N, 3, H, W)
    batch = torch.stack(tensors)

    logger.debug(f"Preprocessed batch shape: {batch.shape}")

    return batch


def _predict_images(batch: torch.Tensor, model: nn.Module) -> np.ndarray:
    """
    Run model inference on image batch and return class probabilities.

    Args:
        batch: Batched tensor of images (N, 3, H, W)
        model: Classification model (already in eval mode)
        device: Device to run inference on

    Returns:
        Numpy array of probabilities:
            - Shape (N, C) for multi-class (C classes)
            - Can return (N,) for binary if only positive class proba needed
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    with torch.no_grad():
        # Move batch to device
        batch = batch.to(device)

        # Forward pass
        outputs = model(batch)

        # Apply softmax to get probabilities
        probas = torch.softmax(outputs, dim=1)

        # Move to CPU and convert to numpy
        probas_np = probas.cpu().numpy()

    logger.debug(f"Predictions shape: {probas_np.shape}")

    return probas_np


def get_classifications_from_images(
    crops: Dict[str, np.ndarray],
    model: nn.Module,
    classes: Optional[List[str]] = None,
    threshold: float = 0.5,
    input_size: int = 224,
) -> Classifications:
    """
    Classify cyclist crops and return Classifications object.

    Args:
        crops: Dictionary mapping tracker_id (str) to image (BGR numpy array)
        model: Classification model (already loaded and in eval mode)
        classes: List of class names (e.g., ["female", "male"])
        threshold: Classification threshold for binary classification (default: 0.5)
        input_size: Input image size for model (default: 224)

    Returns:
        Classifications object containing tracker_ids, probabilities, and predictions
    """
    logger.debug(f"Classifying {len(crops)} cyclist crops")

    # Handle empty input
    if not crops:
        logger.debug("No crops to classify, returning empty Classifications")
        return Classifications.empty()

    # Extract tracker IDs and images maintaining order
    images = list(crops.values())

    logger.debug(f"Preprocessing {len(images)} images to tensors")
    # Preprocess images to tensor batch
    batch = _preprocess_images(images, input_size)

    logger.debug(f"Running inference on batch of {len(batch)} images")
    # Run inference
    probas = _predict_images(batch, model)

    logger.debug(f"Creating Classifications object with {len(probas)} results")
    # Create Classifications object
    classifications = Classifications(
        probas=probas,
        tracker_id=np.array(list(crops.keys())),
        classes=classes,
        threshold=threshold,
    )

    # Log summary
    predictions = classifications.predictions
    if classes and len(classes) == 2:
        counts = {classes[i]: np.sum(predictions == i) for i in range(2)}
        logger.debug(f"Classification results: {counts}")

    return classifications


def get_classifications_from_folder(images_directory: str, model: nn.Module) -> List[str]:
    pass
