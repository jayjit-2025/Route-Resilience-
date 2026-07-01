"""
Abstract base class for road segmentation models.

Provides shared implementation scaffolding for all segmentation model
implementations, partially satisfying the RoadSegmentationModel protocol
defined in core.interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseSegmentationModel(ABC):
    """
    Abstract base class with shared implementation for segmentation models.

    Subclasses must implement `load_weights()` and `predict()`.
    `predict_batch()` has a default implementation that falls back to
    calling `predict()` for each image individually; override for optimised
    batched inference.

    Device selection logic handles GPU → CPU fallback automatically when a
    device is not explicitly supplied by the caller.
    """

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load_weights(self, weights_path: str) -> None:
        """
        Load pretrained weights from *weights_path*.

        Args:
            weights_path: Absolute or relative path to the model weights file
                          (.pth / .pt / .h5 etc.).

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the weights format is incompatible with the model.
        """
        ...

    @abstractmethod
    def predict(self, image: np.ndarray, device=None) -> np.ndarray:
        """
        Predict a binary road mask for a single preprocessed image.

        Args:
            image: Preprocessed satellite image array (H, W, C) or (C, H, W),
                   normalised to [0, 1].
            device: Optional PyTorch device.  When *None* the device returned
                    by :meth:`_get_device` is used.

        Returns:
            Binary road mask (H, W) with values in {0, 1}.

        Raises:
            ValueError: If image dimensions are invalid.
            RuntimeError: If model weights have not been loaded yet.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers — available to all subclasses
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        images: list[np.ndarray],
        device=None,
    ) -> list[np.ndarray]:
        """
        Predict road masks for a batch of images.

        Default implementation delegates to :meth:`predict` for each image.
        Override in subclasses to leverage batched GPU inference for better
        throughput.

        Args:
            images: List of preprocessed image arrays.
            device: Optional PyTorch device forwarded to each :meth:`predict`
                    call.

        Returns:
            List of binary road masks in the same order as *images*.
        """
        return [self.predict(img, device) for img in images]

    def _get_device(self):
        """
        Return the best available compute device.

        Priority: CUDA GPU → CPU.  Returns *None* if PyTorch is not installed
        (allowing non-PyTorch subclasses to handle device selection themselves).

        Returns:
            ``torch.device`` instance or *None*.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        except ImportError:
            return None
