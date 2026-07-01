"""U-Net model wrapper for binary road segmentation (fallback model)."""

from __future__ import annotations

import os
import numpy as np

# Optional torch — graceful fallback for cloud deployment
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from segmentation.base import BaseSegmentationModel
from segmentation.model_registry import register_model
from utils.errors import ModelWeightsNotFoundError


# Only define nn.Module subclasses when torch is available
if _TORCH_AVAILABLE:
    class _DoubleConv(nn.Module):
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
        def forward(self, x):
            return self.block(x)

    class _MinimalUNet(nn.Module):
        def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
            super().__init__()
            self.enc1 = _DoubleConv(in_channels, 64)
            self.enc2 = _DoubleConv(64, 128)
            self.enc3 = _DoubleConv(128, 256)
            self.enc4 = _DoubleConv(256, 512)
            self.bottleneck = _DoubleConv(512, 1024)
            self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.dec4 = _DoubleConv(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec3 = _DoubleConv(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = _DoubleConv(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = _DoubleConv(128, 64)
            self.final = nn.Conv2d(64, out_channels, kernel_size=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b  = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.final(d1)


@register_model("unet")
class UNetModel(BaseSegmentationModel):
    """U-Net for binary road segmentation."""

    def __init__(self, num_classes: int = 1, encoder_name: str = "resnet34") -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.encoder_name  = encoder_name

    def _build_model(self):
        if not _TORCH_AVAILABLE:
            return None
        try:
            import segmentation_models_pytorch as smp
            return smp.Unet(encoder_name=self.encoder_name,
                            encoder_weights="imagenet",
                            in_channels=3, classes=self.num_classes,
                            activation=None)
        except ImportError:
            return _MinimalUNet(in_channels=3, out_channels=self.num_classes)

    def load_weights(self, weights_path: str) -> None:
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
        if not _TORCH_AVAILABLE:
            import cv2
            img_hwc = np.transpose(image, (1, 2, 0)) if image.ndim == 3 else image
            img_u8  = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8)
            gray    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) if img_u8.ndim == 3 else img_u8
            return (gray > 127).astype(np.uint8)

        if image.ndim != 3:
            raise ValueError(f"Expected (C,H,W) image, got {image.shape}.")
        if self._model is None:
            self.load_weights("")
        dev    = device or self._device or self._get_device()
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            output = self._model(tensor)
        prob = torch.sigmoid(output[0, 0]).cpu().numpy() if self.num_classes == 1 \
               else torch.softmax(output[0], dim=0)[1].cpu().numpy()
        return (prob > 0.5).astype(np.uint8)
