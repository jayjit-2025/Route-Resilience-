"""PyTorch Dataset and DataLoader wrappers for all road datasets."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from datasets.base import BaseRoadDataset, RoadDatasetSample
from datasets.transforms import Compose, build_transforms

logger = logging.getLogger(__name__)


class RoadSegmentationDataset:
    """PyTorch-compatible Dataset wrapper around any BaseRoadDataset.

    Applies transforms and returns (image_tensor, mask_tensor, metadata).
    Works without torch installed (returns numpy arrays in that case).
    """

    def __init__(
        self,
        dataset: BaseRoadDataset,
        transform: Optional[Compose] = None,
    ) -> None:
        self.dataset   = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample: RoadDatasetSample = self.dataset[idx]
        image = sample.image.copy()
        mask  = sample.mask.copy()

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        try:
            import torch
            image_t = torch.from_numpy(image).float()
            mask_t  = torch.from_numpy(mask).float()
            return image_t, mask_t, sample.metadata
        except ImportError:
            return image, mask, sample.metadata

    def get_dataloader(
        self,
        batch_size:  int  = 4,
        shuffle:     bool = True,
        num_workers: int  = 0,
    ):
        """Return a PyTorch DataLoader. Requires torch."""
        try:
            from torch.utils.data import DataLoader
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=False,
            )
        except ImportError:
            raise RuntimeError("PyTorch is required for DataLoader. "
                               "Install with: pip install torch")


class CombinedDataset:
    """Combines multiple BaseRoadDataset instances into one."""

    def __init__(self, datasets: list[BaseRoadDataset]) -> None:
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._offsets = []
        offset = 0
        for l in self._lengths:
            self._offsets.append(offset)
            offset += l

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, idx: int) -> RoadDatasetSample:
        # Find which sub-dataset this index belongs to
        for i, (offset, length) in enumerate(zip(self._offsets, self._lengths)):
            if idx < offset + length:
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range for CombinedDataset")

    @property
    def num_images(self) -> int:
        return len(self)
