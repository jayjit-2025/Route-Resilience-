"""
Skeleton extraction from binary road masks.

Converts binary road masks into 1-pixel-wide centerlines and detects
topological features (junctions and endpoints) for graph construction.
"""

import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve


def extract_skeleton(road_mask: np.ndarray) -> np.ndarray:
    """
    Extract 1-pixel-wide skeleton from binary road mask.

    Args:
        road_mask: (H, W) uint8 binary mask (0=background, 1=road)

    Returns:
        (H, W) uint8 skeleton with 1-pixel-wide roads
    """
    if road_mask.max() == 0:
        return np.zeros_like(road_mask)

    # Convert to bool directly (> 0) so uint8 masks with values {0, 1}
    # are handled correctly.  img_as_bool expects uint8 in [0, 255].
    bool_mask = road_mask > 0
    skeleton = skeletonize(bool_mask)
    return skeleton.astype(np.uint8)


def detect_junctions_endpoints(
    skeleton: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Find junction pixels (3+ neighbors) and endpoints (1 neighbor) in skeleton.

    Args:
        skeleton: (H, W) binary skeleton (uint8 or bool)

    Returns:
        (junction_coords, endpoint_coords) — lists of (row, col) tuples
    """
    # 3×3 kernel that sums the 8-connected neighborhood, excluding the center
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    skel_u8 = skeleton.astype(np.uint8)
    neighbor_count = convolve(skel_u8, kernel, mode="constant", cval=0)
    # Only count neighbors for pixels that are actually on the skeleton
    neighbor_count = neighbor_count * skel_u8

    junctions = list(zip(*np.where(neighbor_count >= 3)))
    endpoints = list(zip(*np.where(neighbor_count == 1)))

    return (
        [(int(r), int(c)) for r, c in junctions],
        [(int(r), int(c)) for r, c in endpoints],
    )


def get_skeleton_stats(skeleton: np.ndarray) -> dict:
    """
    Return basic statistics about a skeleton image.

    Args:
        skeleton: (H, W) binary skeleton

    Returns:
        Dict with keys: total_pixels, junction_count, endpoint_count
    """
    junctions, endpoints = detect_junctions_endpoints(skeleton)
    return {
        "total_pixels": int(skeleton.sum()),
        "junction_count": len(junctions),
        "endpoint_count": len(endpoints),
    }
