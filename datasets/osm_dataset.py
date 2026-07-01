"""OpenStreetMap vector-based ground truth dataset.

Uses OSMnx to fetch road networks for georeferenced satellite tiles,
rasterizes them into binary masks, and serves image-mask pairs.

For validation: pairs any georeferenced image with an OSM-derived mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from datasets.base import BaseRoadDataset, CACHE_DIR

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class OSMDataset(BaseRoadDataset):
    """Dataset that auto-generates road masks from OpenStreetMap.

    Given a folder of georeferenced satellite images (GeoTIFF),
    it fetches road vectors from OSM and rasterizes them as masks.

    Requires: osmnx, rasterio, shapely
    Falls back to empty mask if image is not georeferenced.
    """

    NAME = "osm"

    def _discover_pairs(self) -> list[dict]:
        if not self.root.exists():
            logger.warning("OSM dataset root not found: %s", self.root)
            return []

        pairs = []
        for img_path in sorted(self.root.rglob("*")):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            pairs.append({
                "image_path": img_path,
                "mask_path":  None,   # generated on-the-fly
                "source":     "osm",
                "georef":     img_path.suffix.lower() in {".tif", ".tiff"},
            })

        logger.info("OSM dataset: found %d images in %s", len(pairs), self.root)
        return pairs

    def __getitem__(self, idx: int):
        pair = self._pairs[idx]
        image = self._load_image(Path(pair["image_path"]))

        # Try to generate OSM mask for georeferenced images
        mask = self._generate_osm_mask(Path(pair["image_path"]))

        meta = {
            "dataset":    self.NAME,
            "split":      self.split,
            "index":      idx,
            "image_path": str(pair["image_path"]),
            "osm_derived": True,
        }
        from datasets.base import RoadDatasetSample
        return RoadDatasetSample(image=image, mask=mask, metadata=meta)

    def _generate_osm_mask(self, image_path: Path) -> np.ndarray:
        """Fetch OSM roads and rasterize to binary mask at target_size."""
        cache_path = CACHE_DIR / "osm_masks" / f"{image_path.stem}.png"
        if cache_path.exists():
            return self._load_mask(cache_path)

        try:
            import rasterio
            import osmnx as ox
            from shapely.geometry import box

            with rasterio.open(image_path) as src:
                if src.crs is None:
                    return np.zeros(self.target_size, dtype=np.uint8)
                bounds = src.bounds
                # Reproject to WGS84 if necessary
                from rasterio.warp import transform_bounds
                west, south, east, north = transform_bounds(
                    src.crs, "EPSG:4326",
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )

            # Fetch road network from OSM
            G = ox.graph_from_bbox(
                north=north, south=south, east=east, west=west,
                network_type="drive", retain_all=True
            )
            gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

            # Rasterize roads onto blank mask
            h, w = self.target_size
            mask = np.zeros((h, w), dtype=np.uint8)

            # Scale factor: geo_coords → pixel_coords
            lon_scale = w / (east - west)
            lat_scale = h / (north - south)

            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    x1 = int((coords[i][0] - west) * lon_scale)
                    y1 = int((north - coords[i][1]) * lat_scale)
                    x2 = int((coords[i+1][0] - west) * lon_scale)
                    y2 = int((north - coords[i+1][1]) * lat_scale)
                    cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=3)

            # Cache the result
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_path), mask * 255)
            return mask

        except Exception as e:
            logger.warning("OSM mask generation failed for %s: %s", image_path.name, e)
            return np.zeros(self.target_size, dtype=np.uint8)
