"""
Graph utility functions for validation and statistics.

Provides lightweight helpers for inspecting road network graphs before
running healing, centrality, or simulation algorithms.
"""

from __future__ import annotations

import networkx as nx


def validate_graph(G: nx.Graph) -> list[str]:
    """
    Return a list of validation warnings for the given graph.

    An empty list means the graph passed all checks.

    Args:
        G: Road network graph to validate.

    Returns:
        List of human-readable warning strings.  Empty if no issues found.
    """
    warnings: list[str] = []

    if G.number_of_nodes() == 0:
        warnings.append("Graph has no nodes")
        return warnings

    # Check for self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        warnings.append(f"Graph has {len(self_loops)} self-loops")

    # Check connectivity
    components = nx.number_connected_components(G)
    if components > 1:
        warnings.append(f"Graph is disconnected: {components} components")

    return warnings


def get_graph_statistics(G: nx.Graph) -> dict:
    """
    Return a dictionary of basic statistics about the graph.

    Args:
        G: Road network graph to analyse.

    Returns:
        Dictionary with the following keys:

        - ``nodes``: total node count
        - ``edges``: total edge count
        - ``components``: number of connected components
        - ``largest_component_size``: node count of the largest component
          (only present when ``nodes > 0``)
        - ``connectivity_ratio``: fraction of nodes in the largest component
        - ``avg_degree``: mean node degree
    """
    if G.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "connectivity_ratio": 0.0,
        }

    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(components),
        "largest_component_size": largest,
        "connectivity_ratio": largest / G.number_of_nodes(),
        "avg_degree": sum(d for _, d in G.degree()) / G.number_of_nodes(),
    }
