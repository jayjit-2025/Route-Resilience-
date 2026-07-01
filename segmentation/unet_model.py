"""
U-Net model wrapper for binary road segmentation (fallback model).

Attempts to use ``segmentation_models_pytorch`` (smp) for a full-featured
U-Net with an ImageNet-pretrained encoder.  When smp is not installed, a
lightweight custom U-Net is built purely from standard PyTorch modules.

Both variants expose the same :class:`BaseSegmentationModel` interface as
:class:`DeepLabV3PlusModel`.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.base import BaseSegmentationModel
from segmentation.model_registry import register_model
from utils.errors import ModelWeightsNotFoundError


# ---------------------------------------------------------------------------
# Minimal custom U-Net — used only when smp is unavailable
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two successive 3×3 convolutions each followed by BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _MinimalUNet(nn.Module):
    """
    Lightweight U-Net encoder-decoder.

    Encoder: 4 down-sampling stages with channel sizes 64 → 128 → 256 → 512.
    Bottleneck: 1024 channels.
    Decoder: mirrors the encoder using bilinear up-sampling + skip connections.

    Args:
        in_channels: Number of input image channels (default 3 for RGB).
        out_channels: Number of output segmentation channels (default 1).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        # Encoder
        self.enc1 = _DoubleConv(in_channels, 64)
        self.enc2 = _DoubleConv(64, 128)
        self.enc3 = _DoubleConv(128, 256)
        self.enc4 = _DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = _DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(1024, 512)  # 512 (up) + 512 (skip)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(512, 256)   # 256 + 256

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(256, 128)   # 128 + 128

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(128, 64)    # 64 + 64

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

@register_model("unet")
class UNetModel(BaseSegmentationModel):
    """
    U-Net model for binary road segmentation.

    Prefers ``segmentation_models_pytorch`` (smp) when available because it
    ships an ImageNet-pretrained ResNet34 encoder.  Falls back to a minimal
    custom U-Net when smp is not installed.

    Args:
        num_classes: Number of output channels.  Use ``1`` for binary
                     road / non-road segmentation (default).
        encoder_name: Encoder backbone name forwarded to smp (e.g.
                      ``"resnet34"``).  Ignored when smp is unavailable.
    """

    def __init__(
        self,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """
        Construct U-Net architecture.

        Tries ``segmentation_models_pytorch`` first; builds the minimal custom
        implementation when smp is not installed.

        Returns:
            Configured ``nn.Module`` (not yet moved to a device).
        """
        try:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=self.num_classes,
                activation=None,  # raw logits — sigmoid applied in predict()
            )
            return model

        except ImportError:
            # Fall back to the minimal built-in U-Net
            return _MinimalUNet(in_channels=3, out_channels=self.num_classes)

    # ------------------------------------------------------------------
    # BaseSegmentationModel interface
    # ------------------------------------------------------------------

    def load_weights(self, weights_path: str) -> None:
        """
        Load model weights and prepare for inference.

        If *weights_path* points to an existing file its state-dict is applied
        (``strict=False``).  Otherwise the pretrained encoder weights (when
        using smp) or random weights (minimal U-Net) are retained.

        Args:
            weights_path: Path to a ``.pth`` / ``.pt`` checkpoint, or an
                          empty string to use pretrained / default weights.
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

        Auto-loads the model with default weights when :meth:`load_weights`
        has not been called.

        Args:
            image: Preprocessed satellite image array of shape ``(C, H, W)``
                   in ``float32`` with values in ``[0, 1]``.
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

        # Lazy initialisation
        if self._model is None:
            self.load_weights("")

        dev = device or self._device or self._get_device()

        # (C, H, W) → (1, C, H, W)
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(dev)

        with torch.no_grad():
            output = self._model(tensor)  # (1, num_classes, H, W)

        # Sigmoid + threshold
        if self.num_classes == 1:
            prob = torch.sigmoid(output[0, 0]).cpu().numpy()
        else:
            prob = torch.softmax(output[0], dim=0)[1].cpu().numpy()

        binary_mask = (prob > 0.5).astype(np.uint8)
        return binary_mask
