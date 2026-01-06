from dataclasses import dataclass, field
from typing import Iterator, Tuple, Optional, List
import numpy as np


@dataclass
class Classifications:
    """
    Container for classification results.

    Attributes:
        probas: Array of class probabilities - either (N,) for binary or (N, C) for multi-class
        tracker_id: Optional array of tracker IDs (N,)
        classes: Optional list of class names
        threshold: Threshold for binary classification (default: 0.5)
        image_paths: Optional array of image paths (N,)

    Example:
        >>> # Binary classification
        >>> classifications = Classifications(
        ...     probas=np.array([[0.15, 0.85], [0.92, 0.08], [0.23, 0.77]]),
        ...     tracker_id=np.array([1, 2, 3]),
        ...     classes=["female", "male"],
        ...     threshold=0.5
        ... )
        >>> for tracker_id, pred, proba in classifications:
        ...     print(f"Tracker {tracker_id}: class {pred} with proba {proba}")
    """

    probas: np.ndarray
    tracker_id: Optional[np.ndarray] = None
    classes: Optional[List[str]] = None
    threshold: float = 0.5
    image_paths: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate and initialize default values for optional attributes."""
        n = len(self.probas)

        # Initialize tracker_id if not provided
        if self.tracker_id is None:
            self.tracker_id = np.array([None] * n, dtype=object)
        elif len(self.tracker_id) != n:
            raise ValueError(
                f"tracker_id length {len(self.tracker_id)} doesn't match " f"probas length {n}"
            )

        # Initialize image_paths if not provided
        if self.image_paths is None:
            self.image_paths = np.array([None] * n, dtype=object)
        elif len(self.image_paths) != n:
            raise ValueError(
                f"image_paths length {len(self.image_paths)} doesn't match " f"probas length {n}"
            )

        # Validate probas shape
        if self.probas.ndim not in [1, 2]:
            raise ValueError(f"probas must be 1D or 2D array, got shape {self.probas.shape}")

        # Validate threshold
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")

        # Validate classes if provided
        if self.classes is not None:
            if self.probas.ndim == 2:
                if len(self.classes) != self.probas.shape[1]:
                    raise ValueError(
                        f"classes length {len(self.classes)} doesn't match "
                        f"probas columns {self.probas.shape[1]}"
                    )
            elif self.probas.ndim == 1:
                if len(self.classes) != 2:
                    raise ValueError(
                        f"For 1D probas, classes must have exactly 2 elements for binary classification, "
                        f"got {len(self.classes)}"
                    )

    @property
    def predictions(self) -> np.ndarray:
        """
        Calculate predictions from probabilities using threshold.

        For binary classification (2 classes):
            - Uses threshold on positive class (class 1) probability
        For multi-class:
            - Uses argmax (threshold ignored)

        Returns:
            Array of predicted class indices
        """
        if self.probas.ndim == 1:
            # Binary classification, probas are for positive class
            return (self.probas >= self.threshold).astype(int)
        elif self.probas.shape[1] == 2:
            # Binary classification with 2-column probabilities
            return (self.probas[:, 1] >= self.threshold).astype(int)
        else:
            # Multi-class classification
            return np.argmax(self.probas, axis=1)

    def __len__(self) -> int:
        """Return the number of classifications."""
        return len(self.probas)

    def __iter__(self) -> Iterator[Tuple[Optional[int], Optional[int], int, np.ndarray]]:
        """
        Iterate over classifications yielding (tracker_id, prediction, probas) tuples.

        Yields:
            Tuple of (tracker_id, prediction, probas) for each classification
        """
        predictions = self.predictions

        for i in range(len(self)):
            yield (
                self.image_paths[i],
                self.tracker_id[i],
                predictions[i],
                self.probas[i] if self.probas.ndim == 1 else self.probas[i, :],
            )

    def __eq__(self, other: "Classifications") -> bool:
        """Check equality with another Classifications object."""
        if not isinstance(other, Classifications):
            return False

        return all(
            [
                np.array_equal(self.tracker_id, other.tracker_id),
                np.array_equal(self.probas, other.probas),
                self.threshold == other.threshold,
                self.classes == other.classes,
                np.array_equal(self.image_paths, other.image_paths),
            ]
        )

    def __getitem__(self, index) -> "Classifications":
        """
        Get a subset of classifications using boolean or integer indexing.

        Args:
            index: Boolean mask, integer array, slice, or single integer

        Returns:
            New Classifications object with selected items
        """
        if isinstance(index, np.ndarray) and index.dtype == bool:
            # Boolean mask
            return Classifications(
                probas=self.probas[index] if self.probas.ndim == 1 else self.probas[index, :],
                tracker_id=self.tracker_id[index],
                classes=self.classes,
                threshold=self.threshold,
                image_paths=self.image_paths[index],
            )
        else:
            # Integer index, slice, or array
            return Classifications(
                probas=(
                    self.probas[index]
                    if self.probas.ndim == 1
                    else (
                        self.probas[index]
                        if isinstance(index, (int, np.integer))
                        else self.probas[index, :]
                    )
                ),
                tracker_id=self.tracker_id[index],
                classes=self.classes,
                threshold=self.threshold,
                image_paths=self.image_paths[index],
            )

    @classmethod
    def empty(cls) -> "Classifications":
        """
        Create an empty Classifications object.

        Returns:
            Empty Classifications with no entries
        """
        return cls(
            probas=np.array([], dtype=float),
            tracker_id=np.array([], dtype=object),
            image_paths=np.array([], dtype=object),
        )

    @classmethod
    def merge(cls, classifications_list: List["Classifications"]) -> "Classifications":
        """
        Merge multiple Classifications objects into one.
        All Classifications must have same classes, threshold, and probas dimensionality.

        Args:
            classifications_list: List of Classifications objects to merge

        Returns:
            Single merged Classifications object
        """
        if not classifications_list:
            return cls.empty()

        if len(classifications_list) == 1:
            return classifications_list[0]

        # Validate compatibility
        first = classifications_list[0]
        for c in classifications_list[1:]:
            if c.probas.ndim != first.probas.ndim:
                raise ValueError("Cannot merge Classifications with different probas dimensions")
            if c.classes != first.classes:
                raise ValueError("Cannot merge Classifications with different classes")
            if c.threshold != first.threshold:
                raise ValueError("Cannot merge Classifications with different thresholds")

        return cls(
            probas=np.concatenate([c.probas for c in classifications_list]),
            tracker_id=np.concatenate([c.tracker_id for c in classifications_list]),
            classes=first.classes,
            threshold=first.threshold,
            image_paths=np.concatenate([c.image_paths for c in classifications_list]),
        )
