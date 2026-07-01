"""
DeepLabV3+ model wrapper for binary road segmentation.

Uses a ResNet50 backbone with pretrained ImageNet weights from torchvision.
The final classifier head is replaced with a single-channel output for
binary road / non-road prediction.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from segmentation.base import BaseSegmentationModel
from segmentation.model_registry import register_model
from utils.errors import ModelWeightsNotFoundError


@register_model("deeplabv3+")
class DeepLabV3PlusModel(BaseSegmentationModel):
    """
    DeepLabV3+ with ResNet50 backbone for road segmentation.

    Wraps the torchvision ``deeplabv3_resnet50`` model and replaces both the
    main classifier head and the auxiliary classifier head with a single
    1×1 convolution that outputs ``num_classes`` channels.

    Args:
        num_classes: Number of output channels.  Use ``1`` for binary
                     road / non-road segmentation (default).
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """
        Construct DeepLabV3+ from torchvision with pretrained ImageNet weights.

        Both the main and auxiliary classifier heads are replaced to match
        ``self.num_classes``.

        Returns:
            Configured ``nn.Module`` (not yet moved to a device).
        """
        from torchvision.models.segmentation import deeplabv3_resnet50

        model = deeplabv3_resnet50(weights="DEFAULT")

        # Replace main classifier head
        model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=self.num_classes,
            kernel_size=1,
        )

        # Replace auxiliary classifier head
        model.aux_classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=self.num_classes,
            kernel_size=1,
        )

        return model

    # ------------------------------------------------------------------
    # BaseSegmentationModel interface
    # ------------------------------------------------------------------

    def load_weights(self, weights_path: str) -> None:
        """
        Load model weights and prepare for inference.

        If *weights_path* points to an existing file, its state-dict is loaded
        on top of the pretrained backbone (``strict=False`` so that the
        replaced heads do not raise a mismatch error).  When the path is empty
        or the file does not exist the pretrained ImageNet backbone weights are
        kept as-is.

        Args:
            weights_path: Path to a ``.pth`` / ``.pt`` checkpoint file, or an
                          empty string to fall back to the pretrained backbone.

        Raises:
            ModelWeightsNotFoundError: Never raised here — missing custom
                weights silently fall back to pretrained weights.  The error
                class is available for callers that need stricter behaviour.
        """
        device = self._get_device()
        self._model = self._build_model()

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            self._model.load_state_dict(state_dict, strict=False)

        self._model = self._model.to(device)
        self._model.eval()
        self._device = device
        self._is_loaded = True

    def predict(self, image: np.ndarray, device=None) -> np.ndarray:
        """
        Predict a binary road mask for a single preprocessed image.

        The model is auto-loaded with pretrained backbone weights when
        :meth:`load_weights` has not been called explicitly.

        Args:
            image: Preprocessed satellite image array of shape ``(C, H, W)``
                   in ``float32`` with values in ``[0, 1]``.  This matches
                   the output of ``PreprocessingPipeline.preprocess_for_model``.
            device: Optional PyTorch device.  Falls back to the device chosen
                    during weight loading, or the best available device.

        Returns:
            Binary road mask of shape ``(H, W)`` with ``uint8`` values in
            ``{0, 1}``.

        Raises:
            ValueError: If *image* does not have exactly 3 dimensions.
        """
        if image.ndim != 3:
            raise ValueError(
                f"Expected image with shape (C, H, W), got shape {image.shape}."
            )

        # Lazy initialisation with pretrained weights
        if self._model is None:
            self.load_weights("")

        dev = device or self._device or self._get_device()

        # (C, H, W) → (1, C, H, W)
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(dev)

        with torch.no_grad():
            output = self._model(tensor)["out"]  # (1, num_classes, H, W)

        # Sigmoid + threshold for binary segmentation
        if self.num_classes == 1:
            prob = torch.sigmoid(output[0, 0]).cpu().numpy()
        else:
            prob = torch.softmax(output[0], dim=0)[1].cpu().numpy()

        binary_mask = (prob > 0.5).astype(np.uint8)
        return binary_mask

    def predict_batch(self, images: list[np.ndarray], device=None) -> list[np.ndarray]:
        """
        Optimized batch prediction using a single forward pass.

        Stacks all images into a single batched tensor ``(N, C, H, W)`` and
        runs one forward pass through the model, which is significantly faster
        than calling :meth:`predict` in a loop for large batches.

        Args:
            images: List of preprocessed satellite image arrays, each with
                    shape ``(C, H, W)`` in ``float32`` with values in ``[0, 1]``.
            device: Optional PyTorch device for inference.

        Returns:
            List of binary road masks, each with shape ``(H, W)`` and
            ``uint8`` values in ``{0, 1}``.  Empty list if *images* is empty.
        """
        if not images:
            return []

        # Lazy initialisation with pretrained weights
        if self._model is None:
            self.load_weights("")

        dev = device or self._device or self._get_device()

        # Stack to batch tensor: (N, C, H, W)
        batch = torch.stack([torch.from_numpy(img).float() for img in images]).to(dev)

        with torch.no_grad():
            output = self._model(batch)["out"]  # (N, num_classes, H, W)

        masks = []
        for i in range(len(images)):
            if self.num_classes == 1:
                prob = torch.sigmoid(output[i, 0]).cpu().numpy()
            else:
                prob = torch.softmax(output[i], dim=0)[1].cpu().numpy()
            masks.append((prob > 0.5).astype(np.uint8))
        return masks
