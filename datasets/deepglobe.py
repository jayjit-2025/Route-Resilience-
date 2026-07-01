"""DeepGlobe Road Extraction dataset loader.

Expected directory layout:
    <root>/
        <id>_sat.jpg      # satellite image
        <id>_mask.png     # binary road mask (white=road)

DeepGlobe download: https://competitions.codalab.org/competitions/18467
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets.base import BaseRoadDataset

logger = logging.getLogger(__name__)


class DeepGlobeDataset(BaseRoadDataset):
    """DeepGlobe Road Extraction dataset.

    Pairs ``<id>_sat.jpg`` with ``<id>_mask.png`` by shared ID prefix.
    """

    NAME = "deepglobe"

    def _discover_pairs(self) -> list[dict]:
        if not self.root.exists():
            logger.warning("DeepGlobe root not found: %s", self.root)
            return []

        # Build mask index: id → mask path
        mask_index: dict[str, Path] = {}
        for p in self.root.rglob("*_mask.png"):
            img_id = p.stem.replace("_mask", "")
            mask_index[img_id] = p

        pairs = []
        for p in sorted(self.root.rglob("*_sat.jpg")):
            img_id    = p.stem.replace("_sat", "")
            mask_path = mask_index.get(img_id)
            pairs.append({
                "image_path": p,
                "mask_path":  mask_path,
                "source":     "deepglobe",
                "image_id":   img_id,
            })

        logger.info("DeepGlobe: found %d pairs in %s", len(pairs), self.root)
        return pairs
