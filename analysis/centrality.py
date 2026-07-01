"""Centrality analysis for road network graphs.

Computes betweenness centrality to identify critical "gatekeeper" nodes —
intersections whose removal would most severely fragment the road network.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np

from core.data_models import CentralityResult

logger = logging.getLogger(__name__)


def compute_centrality(
    G: nx.Graph,
    threshold_percentile: float = 95.0,
    use_approximate: bool = True,
    k_samples: int = 100,
) -> CentralityResult:
    """Compute betweenness centrality and identify gatekeeper nodes.

    For graphs with more than 500 nodes, approximate betweenness is used
    by default (``use_approximate=True``) to keep runtime reasonable.

    Args:
        G: Connected (or partially connected) road network graph.
        threshold_percentile: Nodes above this centrality percentile are
            flagged as gatekeepers. Default: 95 (top 5%).
        use_approximate: Use k-sample approximation for large graphs.
        k_samples: Number of pivot nodes for approximation (ignored when
            ``use_approximate=False`` or graph is small).

    Returns:
        :class:`~core.data_models.CentralityResult` with scores, gatekeeper
        list, and the threshold value used.
    """
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — returning empty centrality result.")
        return CentralityResult(
            node_centrality={},
            gatekeeper_nodes=[],
            threshold_value=0.0,
        )

    # Work on the largest connected component to avoid nx errors
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc).copy()

    n = subgraph.number_of_nodes()
    logger.info("Computing centrality on %d nodes …", n)

    if use_approximate and n > 500:
        k = min(k_samples, n)
        raw: dict[int, float] = nx.betweenness_centrality(
            subgraph, k=k, normalized=True, weight="length_pixels"
        )
    else:
        raw = nx.betweenness_centrality(
            subgraph, normalized=True, weight="length_pixels"
        )

    # Fill missing nodes (from smaller components) with 0
    node_centrality: dict[int, float] = {n: 0.0 for n in G.nodes()}
    node_centrality.update(raw)

    scores = np.array(list(node_centrality.values()), dtype=float)
    threshold_value = float(np.percentile(scores, threshold_percentile))

    gatekeeper_nodes: list[int] = [
        node_id
        for node_id, score in node_centrality.items()
        if score >= threshold_value
    ]
    # Sort descending by centrality score
    gatekeeper_nodes.sort(key=lambda n: node_centrality[n], reverse=True)

    logger.info(
        "Centrality done. Threshold=%.4f, Gatekeepers=%d",
        threshold_value,
        len(gatekeeper_nodes),
    )

    return CentralityResult(
        node_centrality=node_centrality,
        gatekeeper_nodes=gatekeeper_nodes,
        threshold_value=threshold_value,
    )


def get_heatmap_data(
    G: nx.Graph,
    node_centrality: dict[int, float],
) -> list[tuple[float, float, float]]:
    """Build heatmap data for Folium HeatMap plugin.

    Only includes nodes that have valid ``lat`` and ``lon`` attributes.

    Args:
        G: Road network graph with ``lat``/``lon`` node attributes.
        node_centrality: Mapping of node ID → centrality score.

    Returns:
        List of ``(lat, lon, intensity)`` tuples. Intensities are
        normalised to ``[0, 1]``.
    """
    points: list[tuple[float, float, float]] = []

    scores = list(node_centrality.values())
    max_score = max(scores) if scores else 1.0
    if max_score == 0:
        max_score = 1.0

    for node_id, score in node_centrality.items():
        data = G.nodes.get(node_id, {})
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is not None and lon is not None:
            intensity = score / max_score
            points.append((float(lat), float(lon), float(intensity)))

    return points


def get_top_gatekeepers(
    G: nx.Graph,
    centrality_result: CentralityResult,
    top_n: int = 10,
) -> list[dict]:
    """Return the top-N gatekeeper nodes with their attributes.

    Args:
        G: Road network graph.
        centrality_result: Output from :func:`compute_centrality`.
        top_n: How many nodes to return.

    Returns:
        List of dicts with keys: ``node_id``, ``centrality``, ``lat``,
        ``lon``, ``pixel_x``, ``pixel_y``, ``node_type``.
    """
    result = []
    scored = sorted(
        centrality_result.node_centrality.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for node_id, score in scored[:top_n]:
        attrs = G.nodes.get(node_id, {})
        result.append(
            {
                "node_id": node_id,
                "centrality": round(score, 6),
                "lat": attrs.get("lat"),
                "lon": attrs.get("lon"),
                "pixel_x": attrs.get("pixel_x"),
                "pixel_y": attrs.get("pixel_y"),
                "node_type": attrs.get("node_type", "unknown"),
            }
        )
    return result
