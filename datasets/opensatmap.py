"""OpenSatMap dataset loader.

Expected directory layout:
    <root>/
        images/          # PNG/JPEG satellite tiles
        labels/          # PNG road masks (same basename)

OpenSatMap: https://opensatmap.github.io/
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets.base import BaseRoadDataset

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class OpenSatMapDataset(BaseRoadDataset):
    """OpenSatMap dataset loader.

    Matches images in ``<root>/images/`` with labels in ``<root>/labels/``
    by identical basename.
    """

    NAME = "opensatmap"

    def _discover_pairs(self) -> list[dict]:
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"

        if not images_dir.exists():
            # Try flat layout
            images_dir = self.root
            labels_dir = self.root

        if not images_dir.exists():
            logger.warning("OpenSatMap root not found: %s", self.root)
            return []

        # Build label index
        label_index: dict[str, Path] = {}
        for lp in labels_dir.rglob("*"):
            if lp.suffix.lower() in _IMAGE_EXTS:
                label_index[lp.stem] = lp

        pairs = []
        for img in sorted(images_dir.iterdir()):
            if img.suffix.lower() not in _IMAGE_EXTS:
                continue
            label_path = label_index.get(img.stem)
            pairs.append({
                "image_path": img,
                "mask_path":  label_path,
                "source":     "opensatmap",
            })

        logger.info("OpenSatMap: found %d pairs in %s", len(pairs), self.root)
        return pairs
