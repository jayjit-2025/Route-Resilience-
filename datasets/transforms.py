"""Image augmentation and preprocessing transforms for road datasets."""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

# ImageNet statistics used by DeepLabV3+
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Compose:
    """Apply a sequence of transforms to (image, mask) pairs."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return np.fliplr(image).copy(), np.fliplr(mask).copy()
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return np.flipud(image).copy(), np.flipud(mask).copy()
        return image, mask


class RandomCrop:
    """Randomly crop both image and mask to crop_size."""

    def __init__(self, crop_size: tuple[int, int]) -> None:
        self.crop_h, self.crop_w = crop_size

    def __call__(self, image, mask):
        h, w = image.shape[:2]
        if h <= self.crop_h or w <= self.crop_w:
            return image, mask
        top  = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        return (image[top:top+self.crop_h, left:left+self.crop_w],
                mask[top:top+self.crop_h, left:left+self.crop_w])


class RandomRotation:
    """Rotate image and mask by a random angle."""

    def __init__(self, degrees: float = 15.0) -> None:
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        h, w  = image.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        mask  = cv2.warpAffine(mask,  M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT)
        return image, (mask > 0).astype(np.uint8)


class ColorJitter:
    """Random brightness/contrast/saturation jitter on image only."""

    def __init__(
        self,
        brightness: float = 0.3,
        contrast:   float = 0.3,
        saturation: float = 0.2,
    ) -> None:
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation

    def __call__(self, image, mask):
        img = image.astype(np.float32)
        # Brightness
        img += random.uniform(-self.brightness * 255, self.brightness * 255)
        # Contrast
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        img    = (img - 128) * factor + 128
        img    = np.clip(img, 0, 255).astype(np.uint8)
        return img, mask


class SimulateCloudOcclusion:
    """Randomly paste semi-transparent white patches to simulate clouds."""

    def __init__(self, p: float = 0.3, max_patches: int = 3) -> None:
        self.p           = p
        self.max_patches = max_patches

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask
        img = image.copy()
        h, w = img.shape[:2]
        n = random.randint(1, self.max_patches)
        for _ in range(n):
            ph = random.randint(h // 8, h // 3)
            pw = random.randint(w // 8, w // 3)
            top  = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            alpha = random.uniform(0.4, 0.85)
            patch = np.full((ph, pw, 3), 230, dtype=np.uint8)
            img[top:top+ph, left:left+pw] = (
                alpha * patch + (1 - alpha) * img[top:top+ph, left:left+pw]
            ).astype(np.uint8)
        return img, mask


class SimulateShadowOcclusion:
    """Add random dark polygon to simulate tree/building shadows."""

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask
        img = image.copy()
        h, w = img.shape[:2]
        # Random quadrilateral
        pts = np.array([
            [random.randint(0, w // 2), random.randint(0, h // 2)],
            [random.randint(w // 2, w), random.randint(0, h // 2)],
            [random.randint(w // 2, w), random.randint(h // 2, h)],
            [random.randint(0, w // 2), random.randint(h // 2, h)],
        ], dtype=np.int32)
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(shadow_mask, [pts], 255)
        alpha = random.uniform(0.3, 0.6)
        img[shadow_mask > 0] = (img[shadow_mask > 0] * (1 - alpha)).astype(np.uint8)
        return img, mask


class Normalize:
    """Normalize image to ImageNet statistics and return float32 (H,W,3)."""

    def __call__(self, image, mask):
        img = image.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return img, mask.astype(np.float32)


class ToChannelFirst:
    """Convert (H,W,C) → (C,H,W) for PyTorch."""

    def __call__(self, image, mask):
        return np.transpose(image, (2, 0, 1)), mask


def build_transforms(
    split: str = "train",
    target_size: tuple[int, int] = (512, 512),
    augment: bool = True,
) -> Compose:
    """Build the full transform pipeline for a given split.

    Args:
        split: ``"train"`` or ``"val"`` / ``"test"``.
        target_size: Final spatial dimensions (H, W).
        augment: Apply augmentations (only effective for train split).

    Returns:
        :class:`Compose` transform pipeline.
    """
    transforms = []

    if split == "train" and augment:
        transforms += [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomRotation(degrees=10.0),
            RandomCrop(crop_size=(min(target_size[0], 448), min(target_size[1], 448))),
            ColorJitter(brightness=0.3, contrast=0.3),
            SimulateCloudOcclusion(p=0.25),
            SimulateShadowOcclusion(p=0.25),
        ]

    transforms += [Normalize(), ToChannelFirst()]
    return Compose(transforms)
