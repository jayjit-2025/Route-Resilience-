"""Road graph builder: converts skeleton + junction/endpoint coordinates into a NetworkX graph.

Strategy:
1. Detect junction and endpoint pixels — these become graph nodes.
2. BFS from each node along skeleton pixels to discover connected nodes and
   accumulate the Euclidean path length as the edge weight.
3. Optionally attach lat/lon attributes to every node using the affine transform
   stored in ``GeoMetadata``.

Node attributes: pixel_x (int), pixel_y (int), node_type ('junction'|'endpoint'),
                 lat (float|None), lon (float|None)
Edge attributes: length_pixels (float), length_meters (float|None)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import math
import numpy as np
import networkx as nx

from core.data_models import GeoMetadata
from graph.skeleton_extractor import detect_junctions_endpoints


def build_graph_from_skeleton(
    skeleton: np.ndarray,
    geo_metadata: Optional[GeoMetadata] = None,
) -> nx.Graph:
    """Convert a skeleton image to a NetworkX Graph.

    Nodes are placed at every junction pixel (≥3 skeleton neighbours) and every
    endpoint pixel (1 skeleton neighbour).  Edges are discovered by BFS along
    the skeleton starting from each node; the edge weight is the cumulative
    Euclidean distance traversed in pixel space.

    If *geo_metadata* is supplied, each node additionally receives ``lat`` and
    ``lon`` attributes computed from the pixel position via the stored affine
    transform, and each edge receives a ``length_meters`` attribute computed
    from the geodesic distance between its two endpoint nodes.

    Args:
        skeleton:     (H, W) binary array (uint8 or bool).  Non-zero pixels are
                      considered part of the skeleton.
        geo_metadata: Optional geospatial reference used to convert pixel
                      coordinates to lat/lon.

    Returns:
        ``nx.Graph`` with node and edge attributes as described above.
    """
    from preprocessing.geospatial_handler import pixel_to_latlon

    G: nx.Graph = nx.Graph()

    junctions, endpoints = detect_junctions_endpoints(skeleton)
    all_nodes = junctions + endpoints

    if not all_nodes:
        return G

    # ------------------------------------------------------------------
    # Build node index
    # ------------------------------------------------------------------
    junction_set = set(map(tuple, junctions))
    node_map: dict[tuple[int, int], int] = {}  # (row, col) → node id

    for idx, (row, col) in enumerate(all_nodes):
        node_type = "junction" if (row, col) in junction_set else "endpoint"

        lat: Optional[float] = None
        lon: Optional[float] = None
        if geo_metadata is not None:
            try:
                lat, lon = pixel_to_latlon(col, row, geo_metadata)
            except Exception:
                pass  # Non-georeferenced images: leave lat/lon as None

        G.add_node(
            idx,
            pixel_x=col,
            pixel_y=row,
            node_type=node_type,
            lat=lat,
            lon=lon,
        )
        node_map[(row, col)] = idx

    # ------------------------------------------------------------------
    # BFS edge discovery
    # ------------------------------------------------------------------
    # Each BFS starts from a node pixel and walks skeleton pixels to find
    # the nearest neighbouring nodes along each branch.  A pixel already
    # belonging to a *different* node halts that branch immediately
    # (the edge is recorded), while the pixel of the *starting* node is
    # considered already visited so we don't immediately stop at source.

    skel_bool: np.ndarray = skeleton > 0
    node_set: set[tuple[int, int]] = set(node_map.keys())
    h, w = skeleton.shape

    for start_rc in all_nodes:
        start_id = node_map[start_rc]

        # visited tracks pixels explored in this BFS (prevents looping)
        visited: set[tuple[int, int]] = {start_rc}
        # queue items: (position, accumulated_distance_from_start)
        queue: deque[tuple[tuple[int, int], float]] = deque()
        queue.append((start_rc, 0.0))

        while queue:
            (r, c), dist = queue.popleft()

            # Explore 8-connected neighbours
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = r + dr, c + dc

                    # Bounds check
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue

                    # Only traverse skeleton pixels
                    if not skel_bool[nr, nc]:
                        continue

                    if (nr, nc) in visited:
                        continue

                    step = math.sqrt(dr * dr + dc * dc)  # 1.0 or √2
                    new_dist = dist + step

                    if (nr, nc) in node_set:
                        # Reached a neighbouring node — add edge (once)
                        end_id = node_map[(nr, nc)]
                        if not G.has_edge(start_id, end_id):
                            G.add_edge(
                                start_id,
                                end_id,
                                length_pixels=new_dist,
                                length_meters=_edge_length_meters(
                                    G, start_id, end_id, geo_metadata
                                ),
                            )
                        # Do NOT continue BFS past this node
                    else:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), new_dist))

    return G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _edge_length_meters(
    G: nx.Graph,
    u: int,
    v: int,
    geo_metadata: Optional[GeoMetadata],
) -> Optional[float]:
    """Compute real-world edge length in metres if geo_metadata is available."""
    if geo_metadata is None:
        return None

    u_data = G.nodes[u]
    v_data = G.nodes[v]

    lat_u, lon_u = u_data.get("lat"), u_data.get("lon")
    lat_v, lon_v = v_data.get("lat"), v_data.get("lon")

    if None in (lat_u, lon_u, lat_v, lon_v):
        return None

    return _haversine_meters(lat_u, lon_u, lat_v, lon_v)
