"""Image preprocessing pipeline for satellite imagery.

Provides functions to prepare raw satellite images for segmentation model
inference and to post-process model output masks back to original resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.data_models import GeoMetadata
from preprocessing.geospatial_handler import load_image


def preprocess_for_model(
    image: np.ndarray,
    target_size: tuple[int, int] = (512, 512),
    normalize_range: tuple[float, float] = (0.0, 1.0),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Prepare a satellite image for segmentation model inference.

    Processing steps:
    1. Resize to ``target_size`` (H, W) using bilinear interpolation.
    2. Convert pixel values to ``float32``.
    3. Normalise pixels to the ``[0, 1]`` range (min-max over uint8 255 scale).
    4. Apply ImageNet channel-wise mean/std normalisation.
    5. Reorder dimensions from (H, W, C) to (C, H, W) for PyTorch compatibility.

    Args:
        image: Input RGB image as ``(H, W, 3)`` uint8 or float32 array.
        target_size: Output spatial dimensions as ``(height, width)``.
        normalize_range: Target pixel value range after step 3 (default [0, 1]).
            Currently the function always normalises to [0, 1] before applying
            ImageNet statistics; this parameter is reserved for future use and
            must be ``(0.0, 1.0)``.
        mean: Per-channel mean for ImageNet normalisation (R, G, B).
        std: Per-channel standard deviation for ImageNet normalisation (R, G, B).

    Returns:
        Preprocessed image as a ``(3, H, W)`` float32 array ready for model
        inference.

    Raises:
        ValueError: If ``image`` does not have 3 channels or is not 2-D/3-D.
    """
    if image.ndim == 2:
        # Grayscale — replicate to 3 channels
        image = np.stack([image, image, image], axis=2)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a (H, W, 3) image but received shape {image.shape}."
        )

    # Step 1: Resize to target_size=(H, W) — cv2 takes (width, height)
    h, w = target_size
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    # Step 2: Convert to float32
    arr = resized.astype(np.float32)

    # Step 3: Normalise pixels to [0, 1]
    if arr.max() > 1.0:
        arr /= 255.0

    # Step 4: ImageNet mean/std normalisation per channel
    mean_arr = np.array(mean, dtype=np.float32)   # shape (3,)
    std_arr = np.array(std, dtype=np.float32)      # shape (3,)
    arr = (arr - mean_arr) / std_arr               # broadcast over H, W

    # Step 5: (H, W, C) → (C, H, W)
    arr = np.transpose(arr, (2, 0, 1))

    return arr


def postprocess_mask(
    mask: np.ndarray,
    original_size: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Post-process a segmentation model output mask.

    Processing steps:
    1. Apply sigmoid activation if values are outside ``[0, 1]``.
    2. Threshold to a binary mask (values become 0 or 1).
    3. Resize back to ``original_size`` using nearest-neighbour interpolation
       to preserve sharp binary boundaries.
    4. Return as a ``uint8`` array with values in ``{0, 1}``.

    Args:
        mask: Model output mask.  Can be:
            * ``(H, W)`` — single-channel 2-D array.
            * ``(1, H, W)`` — channel-first single-channel array.
            Values may be raw logits (any range) or probabilities ``[0, 1]``.
        original_size: Target spatial dimensions as ``(height, width)``.
        threshold: Probability threshold for binarisation (default 0.5).

    Returns:
        Binary uint8 mask shaped ``(H, W)`` with values in ``{0, 1}``.
    """
    arr = mask.astype(np.float32)

    # Squeeze channel dimension if present: (1, H, W) → (H, W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Step 1: Apply sigmoid if values lie outside [0, 1]
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))

    # Step 2: Threshold to binary {0, 1}
    binary = (arr >= threshold).astype(np.float32)

    # Step 3: Resize back to original_size=(H, W)
    orig_h, orig_w = original_size
    resized = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Step 4: Return as uint8
    return resized.astype(np.uint8)


def load_and_preprocess(
    image_path: str | Path,
    target_size: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray, Optional[GeoMetadata]]:
    """Full pipeline: load an image from disk and preprocess it for inference.

    Convenience wrapper that combines :func:`~preprocessing.geospatial_handler.load_image`
    and :func:`preprocess_for_model` into a single call.

    Args:
        image_path: Path to the input image (GeoTIFF, PNG, or JPG).
        target_size: Model input spatial dimensions as ``(height, width)``.

    Returns:
        A three-element tuple:

        * **preprocessed_tensor** – ``(3, H, W)`` float32 array ready for
          model inference (channel-first, ImageNet-normalised).
        * **original_image** – ``(H, W, 3)`` uint8 RGB array as loaded from
          disk, before any resizing or normalisation.
        * **geo_metadata** – :class:`~core.data_models.GeoMetadata` instance
          if the file is a GeoTIFF, otherwise ``None``.

    Raises:
        InvalidImageFormatError: If the file extension is not supported.
        ImageLoadError: If the file cannot be read or is corrupt.
        ValueError: If the loaded image does not have 3 channels.
    """
    original_image, geo_metadata = load_image(image_path)
    preprocessed_tensor = preprocess_for_model(original_image, target_size=target_size)
    return preprocessed_tensor, original_image, geo_metadata
