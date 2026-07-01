"""Disaster simulation for road network resilience analysis.

Simulates infrastructure failures by removing nodes/edges from the road
graph and quantifying the impact on connectivity, routing, and resilience.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np

from core.data_models import SimulationMetrics

logger = logging.getLogger(__name__)


def simulate_failure(
    G: nx.Graph,
    removed_nodes: Optional[list[int]] = None,
    removed_edges: Optional[list[tuple[int, int]]] = None,
) -> tuple[nx.Graph, SimulationMetrics]:
    """Simulate infrastructure failure and compute impact metrics.

    Creates a modified copy of the graph with specified nodes/edges removed,
    then calculates travel delay, connectivity, efficiency, and resilience.

    Args:
        G: Original road network graph (unmodified baseline).
        removed_nodes: Node IDs to disable (intersections, bridges).
        removed_edges: Edge tuples ``(u, v)`` to disable (road segments).

    Returns:
        Tuple of ``(modified_graph, SimulationMetrics)``.
    """
    removed_nodes = removed_nodes or []
    removed_edges = removed_edges or []

    # Work on a copy — never mutate the original
    modified = G.copy()

    # Remove edges first (before nodes, to avoid KeyErrors)
    for u, v in removed_edges:
        if modified.has_edge(u, v):
            modified.remove_edge(u, v)

    # Remove nodes (also removes their incident edges)
    for node in removed_nodes:
        if node in modified:
            modified.remove_node(node)

    metrics = _compute_metrics(G, modified, removed_nodes, removed_edges)

    logger.info(
        "Simulation: removed %d nodes, %d edges → "
        "components=%d, resilience=%.3f",
        len(removed_nodes),
        len(removed_edges),
        metrics.components,
        metrics.resilience,
    )

    return modified, metrics


def _compute_metrics(
    original: nx.Graph,
    modified: nx.Graph,
    removed_nodes: list[int],
    removed_edges: list[tuple[int, int]],
) -> SimulationMetrics:
    """Compute all simulation impact metrics."""

    # --- Connected components ---
    components = nx.number_connected_components(modified) if modified.number_of_nodes() > 0 else 0

    # --- Network efficiency (global efficiency) ---
    # Average of 1/d(u,v) over all node pairs — ranges [0, 1]
    orig_efficiency = _global_efficiency(original)
    mod_efficiency = _global_efficiency(modified)

    # --- Travel delay ---
    # Compare average shortest path length before and after.
    # Sample up to 200 random pairs to keep it fast.
    travel_delay = _estimate_travel_delay(original, modified)

    # --- Resilience index ---
    # Weighted composite: connectivity preservation + efficiency retention
    resilience = _compute_resilience(original, modified, orig_efficiency, mod_efficiency)

    return SimulationMetrics(
        travel_delay=travel_delay,
        components=components,
        efficiency=mod_efficiency,
        resilience=resilience,
    )


def _global_efficiency(G: nx.Graph) -> float:
    """Compute global network efficiency (sampled for speed on large graphs)."""
    if G.number_of_nodes() < 2:
        return 0.0
    nodes = list(G.nodes())
    if len(nodes) > 200:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(nodes), size=min(100, len(nodes)), replace=False)
        nodes = [nodes[i] for i in idx]
    total = 0.0
    n = G.number_of_nodes()
    denom = n * (n - 1) if n > 1 else 1
    for u in nodes:
        lengths = nx.single_source_shortest_path_length(G, u)
        for v, d in lengths.items():
            if v != u and d > 0:
                total += 1.0 / d
    return total / denom if denom > 0 else 0.0


def _estimate_travel_delay(
    original: nx.Graph,
    modified: nx.Graph,
    n_samples: int = 20,
) -> float:
    """Estimate average travel delay as percentage path length increase.

    Fast version: samples only 20 pairs, works on largest component only,
    uses unweighted BFS (no weight lookup overhead).

    Returns:
        Average percentage increase in path length (0.0 = no delay).
    """
    # Work only within the largest component for speed
    if original.number_of_nodes() == 0:
        return 0.0

    try:
        orig_lcc = max(nx.connected_components(original), key=len)
        orig_sub = original.subgraph(orig_lcc)
    except Exception:
        return 0.0

    orig_nodes = list(orig_lcc)
    if len(orig_nodes) < 2:
        return 0.0

    rng = np.random.default_rng(seed=42)
    sample_size = min(n_samples, len(orig_nodes))
    sampled = rng.choice(len(orig_nodes), size=(sample_size, 2), replace=False)

    delays: list[float] = []
    for idx_u, idx_v in sampled:
        u = orig_nodes[int(idx_u)]
        v = orig_nodes[int(idx_v)]
        if u == v:
            continue
        try:
            orig_len = nx.shortest_path_length(orig_sub, u, v)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        try:
            mod_len = nx.shortest_path_length(modified, u, v)
            if orig_len > 0:
                delays.append((mod_len - orig_len) / orig_len * 100.0)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            delays.append(100.0)

    return round(float(np.mean(delays)) if delays else 0.0, 2)


def _compute_resilience(
    original: nx.Graph,
    modified: nx.Graph,
    orig_efficiency: float,
    mod_efficiency: float,
) -> float:
    """Compute composite resilience index in [0, 1].

    Combines:
    - Connectivity preservation (40%)
    - Efficiency retention (40%)
    - Node survival rate (20%)

    A score of 1.0 means no degradation; 0.0 means total failure.
    """
    # Connectivity preservation
    orig_nodes = original.number_of_nodes()
    if orig_nodes == 0:
        return 0.0

    orig_comps = nx.number_connected_components(original)
    mod_comps = nx.number_connected_components(modified) if modified.number_of_nodes() > 0 else orig_nodes
    # Fewer components = better (1 is ideal)
    conn_score = orig_comps / max(mod_comps, 1)
    conn_score = min(conn_score, 1.0)

    # Efficiency retention
    eff_score = (mod_efficiency / orig_efficiency) if orig_efficiency > 0 else 0.0
    eff_score = min(eff_score, 1.0)

    # Node survival
    mod_nodes = modified.number_of_nodes()
    node_score = mod_nodes / orig_nodes

    resilience = 0.4 * conn_score + 0.4 * eff_score + 0.2 * node_score
    return round(float(np.clip(resilience, 0.0, 1.0)), 4)


def get_alternative_routes(
    G: nx.Graph,
    source: int,
    target: int,
    k: int = 3,
) -> list[list[int]]:
    """Find up to k shortest alternative routes between two nodes.

    Args:
        G: Road network graph (post-failure modified graph).
        source: Starting node ID.
        target: Destination node ID.
        k: Maximum number of paths to return.

    Returns:
        List of node-ID paths (each path is a list of node IDs).
        Empty list if no path exists.
    """
    try:
        paths = list(nx.shortest_simple_paths(G, source, target, weight="length_pixels"))
        return paths[:k]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
