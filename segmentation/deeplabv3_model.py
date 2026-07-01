"""
DeepLabV3+ model wrapper for binary road segmentation.

Uses a ResNet50 backbone with pretrained ImageNet weights from torchvision.
The final classifier head is replaced with a single-channel output for
binary road / non-road prediction.
"""

from __future__ import annotations

import os

import numpy as np

# Optional torch — not available on Streamlit Cloud
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None

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

    def _build_model(self):
        """Construct DeepLabV3+ from torchvision. Returns None if torch unavailable."""
        if not _TORCH_AVAILABLE:
            return None
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(weights="DEFAULT")
        model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
        return model

    # ------------------------------------------------------------------
    # BaseSegmentationModel interface
    # ------------------------------------------------------------------

    def load_weights(self, weights_path: str) -> None:
        """Load model weights. Skips silently if torch not available."""
        if not _TORCH_AVAILABLE:
            self._is_loaded = False
            return
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
        """Predict binary road mask. Falls back to grayscale threshold if torch unavailable."""
        if not _TORCH_AVAILABLE:
            # Fallback: simple grayscale threshold
            import cv2
            if image.ndim == 3 and image.shape[0] == 3:
                img_hwc = np.transpose(image, (1, 2, 0))
            else:
                img_hwc = image
            img_u8 = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) if img_u8.ndim == 3 else img_u8
            return (gray > 127).astype(np.uint8)

        if image.ndim != 3:
            raise ValueError(f"Expected image with shape (C, H, W), got shape {image.shape}.")
        if self._model is None:
            self.load_weights("")
        dev = device or self._device or self._get_device()
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            output = self._model(tensor)["out"]
        if self.num_classes == 1:
            prob = torch.sigmoid(output[0, 0]).cpu().numpy()
        else:
            prob = torch.softmax(output[0], dim=0)[1].cpu().numpy()
        return (prob > 0.5).astype(np.uint8)

    def predict_batch(self, images: list[np.ndarray], device=None) -> list[np.ndarray]:
        """Batch prediction. Falls back to per-image predict if torch unavailable."""
        if not images:
            return []
        if not _TORCH_AVAILABLE:
            return [self.predict(img, device) for img in images]
        if self._model is None:
            self.load_weights("")
        dev   = device or self._device or self._get_device()
        batch = torch.stack([torch.from_numpy(img).float() for img in images]).to(dev)
        with torch.no_grad():
            output = self._model(batch)["out"]
        masks = []
        for i in range(len(images)):
            prob = torch.sigmoid(output[i, 0]).cpu().numpy() if self.num_classes == 1 \
                   else torch.softmax(output[i], dim=0)[1].cpu().numpy()
            masks.append((prob > 0.5).astype(np.uint8))
        return masks
