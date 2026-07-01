"""SpaceNet Roads dataset loader.

Expected directory layout (SpaceNet v2/v3/v5 format):
    <root>/
        images/          # GeoTIFF or PNG satellite tiles
        masks/           # Binary road mask PNG (same basename)

SpaceNet download: https://spacenet.ai/roads/
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets.base import BaseRoadDataset

logger = logging.getLogger(__name__)

_IMAGE_EXTS  = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
_MASK_SUFFIX = "_mask"   # some SpaceNet releases append this


class SpaceNetDataset(BaseRoadDataset):
    """SpaceNet Roads dataset loader.

    Pairs images in ``<root>/images/`` with masks in ``<root>/masks/``
    by matching basenames (ignoring extension).
    """

    NAME = "spacenet"

    def _discover_pairs(self) -> list[dict]:
        images_dir = self.root / "images"
        masks_dir  = self.root / "masks"

        if not images_dir.exists():
            logger.warning("SpaceNet images dir not found: %s", images_dir)
            return []

        pairs = []
        mask_index: dict[str, Path] = {}

        if masks_dir.exists():
            for mp in masks_dir.iterdir():
                if mp.suffix.lower() in _IMAGE_EXTS:
                    # Strip optional _mask suffix
                    stem = mp.stem.replace(_MASK_SUFFIX, "")
                    mask_index[stem] = mp

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            stem      = img_path.stem
            mask_path = mask_index.get(stem) or mask_index.get(stem + _MASK_SUFFIX)
            pairs.append({
                "image_path": img_path,
                "mask_path":  mask_path,
                "source":     "spacenet",
            })

        logger.info("SpaceNet: found %d image–mask pairs in %s", len(pairs), self.root)
        return pairs
