"""MST-based road graph healing algorithm.

Reconnects disconnected road network components by finding the nearest
node pairs across component boundaries and adding healing edges using
a Minimum Spanning Tree approach.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def heal_graph(
    G: nx.Graph,
    max_distance_pixels: float = 100.0,
) -> nx.Graph:
    """Reconnect disconnected road graph components using MST healing.

    Uses a KD-tree for fast nearest-neighbour search between components,
    making it viable even for graphs with thousands of components.

    Args:
        G: Input road network graph (may be disconnected).
            Nodes must have ``pixel_x`` and ``pixel_y`` attributes.
        max_distance_pixels: Maximum pixel distance for a healing edge.

    Returns:
        A new graph with healing edges added (attribute ``healed=True``).
    """
    if G.number_of_nodes() == 0:
        return G.copy()

    components = list(nx.connected_components(G))
    if len(components) == 1:
        logger.debug("Graph already fully connected — no healing needed.")
        return G.copy()

    logger.info(
        "Healing graph: %d nodes, %d components",
        G.number_of_nodes(), len(components),
    )

    healed = G.copy()

    # Build arrays of node positions and their component labels
    node_ids = np.array(list(G.nodes()))
    coords = np.array([
        [G.nodes[n]["pixel_x"], G.nodes[n]["pixel_y"]] for n in node_ids
    ], dtype=float)

    comp_label = np.zeros(len(node_ids), dtype=int)
    for ci, comp in enumerate(components):
        for n in comp:
            idx = np.searchsorted(node_ids, n)
            comp_label[idx] = ci

    # Use KD-tree to find nearest inter-component pairs efficiently
    from scipy.spatial import KDTree
    tree = KDTree(coords)

    # Query each node for its nearest neighbours (up to 10)
    k_neighbours = min(10, len(node_ids))
    dists, indices = tree.query(coords, k=k_neighbours)

    # Build meta-graph of components with cheapest connections
    meta = nx.Graph()
    meta.add_nodes_from(range(len(components)))

    best: dict[tuple[int,int], tuple[float, int, int]] = {}

    for i, (row_dists, row_idx) in enumerate(zip(dists, indices)):
        ci = comp_label[i]
        for dist, j in zip(row_dists[1:], row_idx[1:]):  # skip self
            cj = comp_label[j]
            if ci == cj:
                continue
            if dist > max_distance_pixels:
                continue
            key = (min(ci, cj), max(ci, cj))
            if key not in best or dist < best[key][0]:
                best[key] = (dist, int(node_ids[i]), int(node_ids[j]))
            meta.add_edge(ci, cj, weight=dist)

    # MST of meta-graph gives minimum healing edges
    try:
        mst_edges = list(nx.minimum_spanning_edges(meta, data=True))
    except nx.NetworkXError:
        mst_edges = []

    added = 0
    for ci, cj, _ in mst_edges:
        key = (min(ci, cj), max(ci, cj))
        if key not in best:
            continue
        dist, u, v = best[key]
        if not healed.has_edge(u, v):
            healed.add_edge(u, v, length_pixels=dist, length_meters=None, healed=True)
            added += 1

    logger.info("Added %d healing edges.", added)
    return healed


def compute_connectivity_ratio(G: nx.Graph) -> float:
    """Return the fraction of nodes in the largest connected component."""
    if G.number_of_nodes() == 0:
        return 0.0
    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components)
    return largest / G.number_of_nodes()
