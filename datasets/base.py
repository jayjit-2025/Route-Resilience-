"""Base dataset class shared by all road segmentation datasets."""

from __future__ import annotations

import hashlib
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/datasets")


class RoadDatasetSample:
    """A single dataset sample: image + mask + metadata."""

    __slots__ = ("image", "mask", "metadata")

    def __init__(
        self,
        image: np.ndarray,      # (H, W, 3) uint8 RGB
        mask: np.ndarray,       # (H, W) uint8 binary {0,1}
        metadata: dict,
    ) -> None:
        self.image    = image
        self.mask     = mask
        self.metadata = metadata


class BaseRoadDataset(ABC):
    """Abstract base class for all road dataset loaders.

    Subclasses must implement:
      - ``_discover_pairs() -> list[dict]``  — returns list of
        {"image_path": Path, "mask_path": Path | None, **extra}

    Caching: serialised samples are stored in ``.cache/datasets/<key>.pkl``
    and loaded on subsequent runs to avoid re-reading raw files.
    """

    NAME: str = "base"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        target_size: tuple[int, int] = (512, 512),
        use_cache: bool = True,
    ) -> None:
        self.root        = Path(root)
        self.split       = split
        self.target_size = target_size
        self.use_cache   = use_cache
        self._pairs: list[dict] = []

        if not self.root.exists():
            logger.warning("Dataset root does not exist: %s", self.root)
            return

        cache_key = self._cache_key()
        cached    = self._load_cache(cache_key) if use_cache else None

        if cached is not None:
            self._pairs = cached
            logger.info("[%s] Loaded %d pairs from cache.", self.NAME, len(self._pairs))
        else:
            self._pairs = self._discover_pairs()
            logger.info("[%s] Discovered %d pairs.", self.NAME, len(self._pairs))
            if use_cache and self._pairs:
                self._save_cache(cache_key, self._pairs)

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _discover_pairs(self) -> list[dict]:
        """Scan the dataset root and return list of image/mask path dicts."""
        ...

    # ── Public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> RoadDatasetSample:
        pair = self._pairs[idx]
        image = self._load_image(Path(pair["image_path"]))
        mask  = self._load_mask(pair.get("mask_path"))
        meta  = {k: v for k, v in pair.items()
                 if k not in ("image_path", "mask_path")}
        meta.update({"dataset": self.NAME, "split": self.split,
                     "index": idx, "image_path": str(pair["image_path"])})
        return RoadDatasetSample(image=image, mask=mask, metadata=meta)

    @property
    def num_images(self) -> int:
        return len(self._pairs)

    @property
    def resolution(self) -> tuple[int, int]:
        return self.target_size

    # ── Loading helpers ───────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and resize a satellite image to (H, W, 3) uint8 RGB."""
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self.target_size[1], self.target_size[0]),
                          interpolation=cv2.INTER_LINEAR)

    def _load_mask(self, path: Optional[str | Path]) -> np.ndarray:
        """Load and resize road mask to (H, W) binary uint8."""
        if path is None:
            return np.zeros(self.target_size, dtype=np.uint8)
        p = Path(path)
        if not p.exists():
            return np.zeros(self.target_size, dtype=np.uint8)
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return np.zeros(self.target_size, dtype=np.uint8)
        resized = cv2.resize(gray, (self.target_size[1], self.target_size[0]),
                             interpolation=cv2.INTER_NEAREST)
        return (resized > 127).astype(np.uint8)

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self) -> str:
        raw = f"{self.NAME}_{self.root}_{self.split}_{self.target_size}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[list[dict]]:
        cache_file = CACHE_DIR / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Cache read failed: %s", e)
        return None

    def _save_cache(self, key: str, data: list[dict]) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning("Cache write failed: %s", e)
