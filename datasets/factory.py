"""DatasetFactory — unified entry point for all dataset configurations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets.base import BaseRoadDataset
from datasets.torch_dataset import RoadSegmentationDataset, CombinedDataset
from datasets.transforms import build_transforms

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset selection and paths."""

    # Which dataset(s) to use
    active: str = "spacenet"          # "spacenet" | "deepglobe" | "opensatmap" | "osm" | "combined"

    # Root paths for each dataset (set to None if not available)
    spacenet_root:   Optional[str] = None
    deepglobe_root:  Optional[str] = None
    opensatmap_root: Optional[str] = None
    osm_root:        Optional[str] = None

    # Training settings
    target_size:  tuple[int, int] = (512, 512)
    batch_size:   int  = 4
    num_workers:  int  = 0
    use_cache:    bool = True
    augment:      bool = True

    # Splits
    train_split: str = "train"
    val_split:   str = "val"


class DatasetFactory:
    """Creates dataset instances based on DatasetConfig.

    Usage::

        cfg = DatasetConfig(active="deepglobe",
                            deepglobe_root="datasets/deepglobe")
        factory = DatasetFactory(cfg)

        train_loader = factory.get_dataloader("train")
        val_loader   = factory.get_dataloader("val")
        info         = factory.get_info()
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._datasets: dict[str, BaseRoadDataset] = {}

    def _build_dataset(self, name: str, split: str) -> Optional[BaseRoadDataset]:
        """Instantiate a single named dataset."""
        cfg = self.config

        try:
            if name == "spacenet" and cfg.spacenet_root:
                from datasets.spacenet import SpaceNetDataset
                return SpaceNetDataset(cfg.spacenet_root, split,
                                       cfg.target_size, cfg.use_cache)

            elif name == "deepglobe" and cfg.deepglobe_root:
                from datasets.deepglobe import DeepGlobeDataset
                return DeepGlobeDataset(cfg.deepglobe_root, split,
                                        cfg.target_size, cfg.use_cache)

            elif name == "opensatmap" and cfg.opensatmap_root:
                from datasets.opensatmap import OpenSatMapDataset
                return OpenSatMapDataset(cfg.opensatmap_root, split,
                                          cfg.target_size, cfg.use_cache)

            elif name == "osm" and cfg.osm_root:
                from datasets.osm_dataset import OSMDataset
                return OSMDataset(cfg.osm_root, split,
                                  cfg.target_size, cfg.use_cache)

        except Exception as e:
            logger.warning("Failed to build dataset '%s': %s", name, e)

        return None

    def get_dataset(self, split: str = "train") -> Optional[RoadSegmentationDataset]:
        """Return a RoadSegmentationDataset for the configured split."""
        cfg = self.config
        transform = build_transforms(
            split=split,
            target_size=cfg.target_size,
            augment=(cfg.augment and split == "train"),
        )

        if cfg.active == "combined":
            raw_datasets = []
            for name in ["spacenet", "deepglobe", "opensatmap", "osm"]:
                ds = self._build_dataset(name, split)
                if ds is not None and len(ds) > 0:
                    raw_datasets.append(ds)
            if not raw_datasets:
                logger.warning("No datasets available for 'combined' mode.")
                return None
            combined = CombinedDataset(raw_datasets)
            return RoadSegmentationDataset(combined, transform)
        else:
            ds = self._build_dataset(cfg.active, split)
            if ds is None:
                logger.warning("Dataset '%s' not available (check root path).", cfg.active)
                return None
            return RoadSegmentationDataset(ds, transform)

    def get_dataloader(self, split: str = "train"):
        """Return a PyTorch DataLoader for the given split."""
        ds = self.get_dataset(split)
        if ds is None:
            return None
        return ds.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
        )

    def get_info(self) -> dict:
        """Return summary information about the active dataset configuration."""
        cfg   = self.config
        train = self.get_dataset("train")
        val   = self.get_dataset("val")

        return {
            "active_dataset": cfg.active,
            "target_size":    cfg.target_size,
            "batch_size":     cfg.batch_size,
            "augmentation":   cfg.augment,
            "use_cache":      cfg.use_cache,
            "train_images":   len(train) if train else 0,
            "val_images":     len(val)   if val   else 0,
            "spacenet_root":  cfg.spacenet_root   or "not configured",
            "deepglobe_root": cfg.deepglobe_root  or "not configured",
            "opensatmap_root":cfg.opensatmap_root or "not configured",
            "osm_root":       cfg.osm_root        or "not configured",
        }
