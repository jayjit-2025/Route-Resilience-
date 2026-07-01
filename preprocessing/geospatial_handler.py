"""Geospatial handler for raster I/O.

Provides functions for loading satellite imagery (GeoTIFF, PNG, JPG) and
converting between pixel coordinates and geographic lat/lon coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from core.data_models import GeoMetadata
from utils.errors import ImageLoadError, InvalidImageFormatError

# Optional rasterio import — graceful fallback for cloud deployment
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform as rasterio_transform
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

# Supported file extensions
_GEOTIFF_EXTS = {".tif", ".tiff"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg"}
_ALL_SUPPORTED = _GEOTIFF_EXTS | _RASTER_EXTS


def load_image(image_path: str | Path) -> tuple[np.ndarray, Optional[GeoMetadata]]:
    """Load a satellite image from a GeoTIFF, PNG, or JPG file."""
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix not in _ALL_SUPPORTED:
        raise InvalidImageFormatError(
            f"Unsupported image format '{suffix}'. "
            f"Supported formats: {sorted(_ALL_SUPPORTED)}"
        )

    if suffix in _GEOTIFF_EXTS and _RASTERIO_AVAILABLE:
        return _load_geotiff(path)
    else:
        return _load_raster_cv2(path)


def _load_geotiff(path: Path) -> tuple[np.ndarray, GeoMetadata]:
    """Load a GeoTIFF using rasterio and return (array, GeoMetadata)."""
    try:
        with rasterio.open(path) as src:
            data = src.read()
            crs = src.crs
            transform = src.transform
            bounds = src.bounds
            shape = (src.height, src.width)

        image = _bands_to_hwc(data)
        image = _to_3channel(image)

        geo_meta = GeoMetadata(
            crs=crs,
            transform=transform,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            shape=shape,
        )
        return image, geo_meta

    except Exception as exc:
        raise ImageLoadError(f"Cannot open GeoTIFF '{path}': {exc}") from exc


def _load_raster_cv2(path: Path) -> tuple[np.ndarray, None]:
    """Load a PNG/JPG using OpenCV and return (array, None)."""
    try:
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise ImageLoadError(f"OpenCV failed to read '{path}': {exc}") from exc

    if bgr is None:
        raise ImageLoadError(f"Cannot open image '{path}'.")

    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 4:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb.astype(np.uint8), None


def _bands_to_hwc(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data[:, :, np.newaxis]
    return np.transpose(data, (1, 2, 0))


def _to_3channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    bands = image.shape[2]
    if bands == 1:
        return np.concatenate([image, image, image], axis=2)
    if bands == 3:
        return image
    if bands == 4:
        return image[:, :, :3]
    raise ImageLoadError(f"Cannot convert {bands}-band image to 3-channel RGB.")


def pixel_to_latlon(pixel_x: int, pixel_y: int, geo_meta: GeoMetadata) -> tuple[float, float]:
    """Convert pixel coordinates to lat/lon."""
    if not _RASTERIO_AVAILABLE:
        return (float(pixel_y), float(pixel_x))
    x_geo, y_geo = geo_meta.transform * (pixel_x, pixel_y)
    wgs84 = CRS.from_epsg(4326)
    if geo_meta.crs == wgs84:
        return float(y_geo), float(x_geo)
    xs, ys = rasterio_transform(geo_meta.crs, wgs84, [x_geo], [y_geo])
    return float(ys[0]), float(xs[0])


def latlon_to_pixel(lat: float, lon: float, geo_meta: GeoMetadata) -> tuple[int, int]:
    """Convert lat/lon to pixel coordinates."""
    if not _RASTERIO_AVAILABLE:
        return (int(lon), int(lat))
    wgs84 = CRS.from_epsg(4326)
    if geo_meta.crs == wgs84:
        x_geo, y_geo = lon, lat
    else:
        xs, ys = rasterio_transform(wgs84, geo_meta.crs, [lon], [lat])
        x_geo, y_geo = xs[0], ys[0]
    col, row = ~geo_meta.transform * (x_geo, y_geo)
    return int(col), int(row)


def get_image_bounds_latlon(geo_meta: GeoMetadata) -> tuple[float, float, float, float]:
    """Return (min_lat, min_lon, max_lat, max_lon)."""
    if not _RASTERIO_AVAILABLE:
        minx, miny, maxx, maxy = geo_meta.bounds
        return float(miny), float(minx), float(maxy), float(maxx)
    minx, miny, maxx, maxy = geo_meta.bounds
    wgs84 = CRS.from_epsg(4326)
    if geo_meta.crs == wgs84:
        return float(miny), float(minx), float(maxy), float(maxx)
    corners_x = [minx, maxx, minx, maxx]
    corners_y = [miny, miny, maxy, maxy]
    lons, lats = rasterio_transform(geo_meta.crs, wgs84, corners_x, corners_y)
    return float(min(lats)), float(min(lons)), float(max(lats)), float(max(lons))


def load_image(image_path: str | Path) -> tuple[np.ndarray, Optional[GeoMetadata]]:
    """Load a satellite image from a GeoTIFF, PNG, or JPG file.

    For GeoTIFF files rasterio is used and CRS/transform/bounds are captured
    in a ``GeoMetadata`` object.  For PNG/JPG files OpenCV is used and ``None``
    is returned for the metadata.

    The returned array is always shaped ``(H, W, 3)`` in RGB order with dtype
    ``uint8`` or ``float32``.  Single-band and four-band inputs are converted
    to three-channel RGB automatically.

    Args:
        image_path: Path to the image file.

    Returns:
        A ``(image_array, geo_metadata)`` tuple.

    Raises:
        InvalidImageFormatError: If the file extension is not supported.
        ImageLoadError: If the file cannot be read or is corrupt.
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix not in _ALL_SUPPORTED:
        raise InvalidImageFormatError(
            f"Unsupported image format '{suffix}'. "
            f"Supported formats: {sorted(_ALL_SUPPORTED)}"
        )

    if suffix in _GEOTIFF_EXTS:
        return _load_geotiff(path)
    else:
        return _load_raster_cv2(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_geotiff(path: Path) -> tuple[np.ndarray, GeoMetadata]:
    """Load a GeoTIFF using rasterio and return (array, GeoMetadata)."""
    try:
        with rasterio.open(path) as src:
            # Read all bands; shape is (bands, H, W)
            data = src.read()
            crs: CRS = src.crs
            transform = src.transform
            bounds = src.bounds  # BoundingBox(left, bottom, right, top)
            shape = (src.height, src.width)

        # Convert bands → (H, W, C)
        image = _bands_to_hwc(data)
        image = _to_3channel(image)

        geo_meta = GeoMetadata(
            crs=crs,
            transform=transform,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            shape=shape,
        )
        return image, geo_meta

    except rasterio.errors.RasterioIOError as exc:
        raise ImageLoadError(f"Cannot open GeoTIFF '{path}': {exc}") from exc
    except Exception as exc:
        raise ImageLoadError(f"Unexpected error loading GeoTIFF '{path}': {exc}") from exc


def _load_raster_cv2(path: Path) -> tuple[np.ndarray, None]:
    """Load a PNG/JPG using OpenCV and return (array, None)."""
    try:
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise ImageLoadError(f"OpenCV failed to read '{path}': {exc}") from exc

    if bgr is None:
        raise ImageLoadError(
            f"Cannot open image '{path}'. File may be corrupt or unreadable."
        )

    # cv2 loads as BGR; handle grayscale vs colour vs BGRA
    if bgr.ndim == 2:
        # Grayscale → RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 4:
        # BGRA → BGR first, then to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb.astype(np.uint8), None


def _bands_to_hwc(data: np.ndarray) -> np.ndarray:
    """Convert rasterio (C, H, W) array to (H, W, C)."""
    if data.ndim == 2:
        # Single-band without channel dim
        return data[:, :, np.newaxis]
    # (C, H, W) → (H, W, C)
    return np.transpose(data, (1, 2, 0))


def _to_3channel(image: np.ndarray) -> np.ndarray:
    """Ensure the image has exactly 3 channels (RGB).

    - 1-band  → replicate to 3 channels
    - 3-band  → assumed RGB, returned as-is
    - 4-band  → drop the alpha channel (keep first 3)
    - Other   → raise ImageLoadError
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    bands = image.shape[2]
    if bands == 1:
        return np.concatenate([image, image, image], axis=2)
    if bands == 3:
        return image
    if bands == 4:
        return image[:, :, :3]

    raise ImageLoadError(
        f"Cannot convert {bands}-band image to 3-channel RGB."
    )


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def pixel_to_latlon(pixel_x: int, pixel_y: int, geo_meta: GeoMetadata) -> tuple[float, float]:
    """Convert pixel coordinates to geographic (lat, lon).

    Uses the affine transform stored in ``geo_meta`` to map a pixel position
    to the image CRS, then reprojects to WGS84 (EPSG:4326) if needed.

    Args:
        pixel_x: Column index (x) of the pixel.
        pixel_y: Row index (y) of the pixel.
        geo_meta: Geospatial metadata containing CRS and affine transform.

    Returns:
        A ``(latitude, longitude)`` tuple in decimal degrees.
    """
    # Affine: (x_geo, y_geo) = transform * (col, row)
    x_geo, y_geo = geo_meta.transform * (pixel_x, pixel_y)

    wgs84 = CRS.from_epsg(4326)
    if geo_meta.crs == wgs84:
        lon, lat = x_geo, y_geo
    else:
        xs, ys = rasterio_transform(geo_meta.crs, wgs84, [x_geo], [y_geo])
        lon, lat = xs[0], ys[0]

    return float(lat), float(lon)


def latlon_to_pixel(lat: float, lon: float, geo_meta: GeoMetadata) -> tuple[int, int]:
    """Convert geographic (lat, lon) to pixel coordinates.

    Reprojects to the image CRS if necessary, then applies the inverse of the
    affine transform to obtain column/row indices.

    Args:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).
        geo_meta: Geospatial metadata containing CRS and affine transform.

    Returns:
        A ``(pixel_x, pixel_y)`` tuple (column, row) as integers.
    """
    wgs84 = CRS.from_epsg(4326)
    if geo_meta.crs == wgs84:
        x_geo, y_geo = lon, lat
    else:
        xs, ys = rasterio_transform(wgs84, geo_meta.crs, [lon], [lat])
        x_geo, y_geo = xs[0], ys[0]

    # Inverse affine: (col, row) = ~transform * (x_geo, y_geo)
    col, row = ~geo_meta.transform * (x_geo, y_geo)
    return int(col), int(row)


def get_image_bounds_latlon(geo_meta: GeoMetadata) -> tuple[float, float, float, float]:
    """Return the image extent in WGS84 geographic coordinates.

    Args:
        geo_meta: Geospatial metadata with CRS and bounds in native CRS.

    Returns:
        A ``(min_lat, min_lon, max_lat, max_lon)`` tuple in decimal degrees.
    """
    minx, miny, maxx, maxy = geo_meta.bounds
    wgs84 = CRS.from_epsg(4326)

    if geo_meta.crs == wgs84:
        min_lon, min_lat = minx, miny
        max_lon, max_lat = maxx, maxy
    else:
        corners_x = [minx, maxx, minx, maxx]
        corners_y = [miny, miny, maxy, maxy]
        lons, lats = rasterio_transform(geo_meta.crs, wgs84, corners_x, corners_y)
        min_lat, max_lat = float(min(lats)), float(max(lats))
        min_lon, max_lon = float(min(lons)), float(max(lons))

    return float(min_lat), float(min_lon), float(max_lat), float(max_lon)
