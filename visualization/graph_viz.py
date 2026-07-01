"""Graph visualization utilities using matplotlib.

Renders NetworkX road graphs using actual pixel coordinates stored in
node attributes (pixel_x, pixel_y) — no random layouts.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def _get_colormap(name: str):
    """Return a matplotlib colormap, compatible with both old and new versions."""
    try:
        return matplotlib.colormaps[name]          # matplotlib >= 3.5
    except AttributeError:
        from matplotlib import cm
        return cm.get_cmap(name)                   # matplotlib < 3.5


def _get_pos(G: nx.Graph) -> dict[int, tuple[float, float]]:
    """Extract pixel-coordinate layout from node attributes.

    Uses (pixel_x, pixel_y) stored by graph_builder.py.
    Y-axis is flipped so that image-top = plot-top.
    """
    pos = {}
    for n, d in G.nodes(data=True):
        x = d.get("pixel_x", 0)
        y = d.get("pixel_y", 0)
        pos[n] = (float(x), -float(y))  # flip Y for image convention
    return pos


def render_graph(
    G: nx.Graph,
    title: str = "Road Graph",
    healed_edges: Optional[set[tuple[int, int]]] = None,
    figsize: tuple[int, int] = (7, 7),
) -> bytes:
    """Render a road graph to PNG bytes using actual node coordinates.

    Args:
        G: NetworkX graph with pixel_x / pixel_y node attributes.
        title: Plot title.
        healed_edges: Set of (u, v) edge tuples added by healing (drawn green).
        figsize: Matplotlib figure size in inches.

    Returns:
        PNG image as bytes.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_title(title, color="#e0e0e0", fontsize=12, pad=10)
    ax.axis("off")

    pos = _get_pos(G)
    if not pos:
        ax.text(0.5, 0.5, "No graph data", color="gray",
                ha="center", va="center", transform=ax.transAxes)
        return _fig_to_bytes(fig)

    healed_set = healed_edges or set()

    # Separate edges into original and healed
    orig_edges = [(u, v) for u, v in G.edges() if (u, v) not in healed_set and (v, u) not in healed_set]
    new_edges   = [(u, v) for u, v in G.edges() if (u, v) in healed_set or (v, u) in healed_set]

    node_size = max(2, min(8, 3000 // max(G.number_of_nodes(), 1)))
    edge_width = max(0.3, min(1.0, 500 / max(G.number_of_edges(), 1)))

    nx.draw_networkx_edges(G, pos, edgelist=orig_edges, ax=ax,
                           edge_color="#42a5f5", width=edge_width, alpha=0.7)
    if new_edges:
        nx.draw_networkx_edges(G, pos, edgelist=new_edges, ax=ax,
                               edge_color="#66bb6a", width=edge_width * 1.5, alpha=0.9)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color="white", node_size=node_size, alpha=0.9)

    # Legend
    handles = [mpatches.Patch(color="#42a5f5", label="Road edges")]
    if new_edges:
        handles.append(mpatches.Patch(color="#66bb6a", label=f"Recovered ({len(new_edges)})"))
    ax.legend(handles=handles, loc="lower right",
              facecolor="#1e2130", edgecolor="none", labelcolor="#e0e0e0", fontsize=8)

    plt.tight_layout(pad=0.5)
    return _fig_to_bytes(fig)


def render_overlay(
    image: np.ndarray,
    G: nx.Graph,
    healed_edges: Optional[set[tuple[int, int]]] = None,
    alpha: float = 0.85,
) -> bytes:
    """Overlay road graph on the satellite image using pixel coordinates.

    Args:
        image: (H, W, 3) uint8 RGB satellite image.
        G: NetworkX graph with pixel_x / pixel_y node attributes.
        healed_edges: Healed edge set (drawn in green).
        alpha: Opacity of graph overlay.

    Returns:
        PNG image as bytes.
    """
    h, w = image.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    ax.imshow(image)

    healed_set = healed_edges or set()

    for u, v in G.edges():
        u_d, v_d = G.nodes[u], G.nodes[v]
        x1, y1 = u_d.get("pixel_x", 0), u_d.get("pixel_y", 0)
        x2, y2 = v_d.get("pixel_x", 0), v_d.get("pixel_y", 0)
        is_healed = (u, v) in healed_set or (v, u) in healed_set
        color = "#66bb6a" if is_healed else "#42a5f5"
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.8, alpha=alpha)

    # Draw nodes
    for n, d in G.nodes(data=True):
        x, y = d.get("pixel_x", 0), d.get("pixel_y", 0)
        ax.plot(x, y, "o", color="white", markersize=1.5, alpha=0.9)

    return _fig_to_bytes(fig)


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    """Save a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Simulation-specific renderers ────────────────────────────────────────────

def render_simulation_comparison(
    G_before: nx.Graph,
    G_after: nx.Graph,
    removed_nodes: list[int],
    removed_edges: list[tuple[int, int]],
    path_before: Optional[list[int]] = None,
    path_after: Optional[list[int]] = None,
    show_failed: bool = True,
    show_alt_route: bool = True,
    show_disconnected: bool = True,
    show_critical: bool = True,
) -> bytes:
    """Professional Before / After comparison with clear failure visualization.

    Left panel: Original network with original route highlighted in blue.
    Right panel: Post-failure network with:
      - Large red X on failed nodes
      - Disconnected components in purple
      - Alternative route in bright green
      - Affected zone circle around failure
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0e1117")
    fig.suptitle("🗺  Network Impact Analysis — Before vs After Failure",
                 color="#e0e0e0", fontsize=14, fontweight="bold", y=0.98)

    pos = _get_pos(G_before)   # use before-graph coords for both panels
    pos_after = _get_pos(G_after)

    removed_node_set = set(removed_nodes)
    removed_edge_set = {(u, v) for u, v in removed_edges} | {(v, u) for u, v in removed_edges}

    n_nodes = G_before.number_of_nodes()
    n_edges = max(G_before.number_of_edges(), 1)
    ns  = max(4, min(14, 5000 // max(n_nodes, 1)))
    ew  = max(0.3, min(1.2, 600 / n_edges))

    # ── LEFT: Original network ────────────────────────────────────────────────
    ax_l.set_facecolor("#0d1117")
    ax_l.set_title("✅  Original Network", color="#a5d6a7", fontsize=12,
                   fontweight="bold", pad=8)
    ax_l.axis("off")

    if pos:
        # All edges — gray base
        nx.draw_networkx_edges(G_before, pos, ax=ax_l,
                               edge_color="#37474f", width=ew * 0.6, alpha=0.5)
        # All normal nodes — white
        normal_nodes = [n for n in G_before.nodes() if n not in removed_node_set]
        nx.draw_networkx_nodes(G_before, pos, nodelist=normal_nodes, ax=ax_l,
                               node_color="#90caf9", node_size=ns, alpha=0.7)

        # Original route — thick blue
        if path_before and len(path_before) > 1 and show_alt_route:
            path_edges = list(zip(path_before[:-1], path_before[1:]))
            valid = [(u, v) for u, v in path_edges if u in pos and v in pos]
            nx.draw_networkx_edges(G_before, pos, edgelist=valid, ax=ax_l,
                                   edge_color="#29b6f6", width=ew * 4, alpha=1.0,
                                   style="solid")
            # Start / end markers
            src, tgt = path_before[0], path_before[-1]
            if src in pos:
                ax_l.plot(*pos[src], "o", color="#00e5ff", markersize=ns + 4,
                          zorder=5, markeredgecolor="white", markeredgewidth=1)
                ax_l.annotate("START", pos[src], color="#00e5ff", fontsize=7,
                              fontweight="bold", ha="center", va="bottom",
                              xytext=(0, 8), textcoords="offset points")
            if tgt in pos:
                ax_l.plot(*pos[tgt], "s", color="#00e5ff", markersize=ns + 4,
                          zorder=5, markeredgecolor="white", markeredgewidth=1)
                ax_l.annotate("END", pos[tgt], color="#00e5ff", fontsize=7,
                              fontweight="bold", ha="center", va="bottom",
                              xytext=(0, 8), textcoords="offset points")

        # Mark the to-be-failed node on left too (orange warning)
        if show_failed and removed_node_set:
            present = [n for n in removed_node_set if n in pos]
            if present:
                nx.draw_networkx_nodes(G_before, pos, nodelist=present, ax=ax_l,
                                       node_color="#ffa726", node_size=ns * 10,
                                       alpha=0.9, zorder=6)
                for n in present:
                    ax_l.annotate("⚠ Target", pos[n], color="#ffa726",
                                  fontsize=8, fontweight="bold",
                                  ha="center", va="bottom",
                                  xytext=(0, 12), textcoords="offset points")

    # ── RIGHT: Post-failure network ───────────────────────────────────────────
    ax_r.set_facecolor("#0d1117")
    ax_r.set_title("❌  Network After Failure", color="#ef9a9a", fontsize=12,
                   fontweight="bold", pad=8)
    ax_r.axis("off")

    if pos_after or pos:
        p = pos_after if pos_after else pos

        # Detect connected components in modified graph
        components = list(nx.connected_components(G_after)) if G_after.number_of_nodes() > 0 else []
        largest_cc = set(max(components, key=len)) if components else set()
        small_ccs  = [c for c in components if c != largest_cc]

        # Color small disconnected components purple
        if show_disconnected:
            for i, cc in enumerate(small_ccs[:8]):   # cap at 8 for clarity
                cc_nodes = [n for n in cc if n in p]
                cc_edges = [(u, v) for u, v in G_after.edges()
                            if u in cc and v in cc and u in p and v in p]
                if cc_edges:
                    nx.draw_networkx_edges(G_after, p, edgelist=cc_edges, ax=ax_r,
                                           edge_color="#ce93d8", width=ew * 0.8,
                                           alpha=0.6)
                if cc_nodes:
                    nx.draw_networkx_nodes(G_after, p, nodelist=cc_nodes, ax=ax_r,
                                           node_color="#ab47bc", node_size=ns,
                                           alpha=0.7)

        # Draw largest component normally
        lcc_edges = [(u, v) for u, v in G_after.edges()
                     if u in largest_cc and v in largest_cc and u in p and v in p]
        nx.draw_networkx_edges(G_after, p, edgelist=lcc_edges, ax=ax_r,
                               edge_color="#37474f", width=ew * 0.6, alpha=0.5)
        lcc_nodes = [n for n in largest_cc if n in p]
        nx.draw_networkx_nodes(G_after, p, nodelist=lcc_nodes, ax=ax_r,
                               node_color="#90caf9", node_size=ns, alpha=0.6)

        # Affected zone circle around each failed node
        if show_failed and removed_node_set:
            _all_xs = [pos[n][0] for n in pos]
            _all_ys = [pos[n][1] for n in pos]
            _span = max(max(_all_xs) - min(_all_xs), 1) if _all_xs else 100
            zone_r = _span * 0.08
            for n in removed_node_set:
                if n in pos:
                    cx, cy = pos[n]
                    circle = plt.Circle((cx, cy), zone_r, color="#ef5350",
                                        fill=True, alpha=0.12, zorder=3)
                    ax_r.add_patch(circle)
                    circle_border = plt.Circle((cx, cy), zone_r, color="#ef5350",
                                               fill=False, linewidth=1.5,
                                               linestyle="--", alpha=0.6, zorder=3)
                    ax_r.add_patch(circle_border)

            # Large red X marker
            for n in removed_node_set:
                if n in pos:
                    x, y = pos[n]
                    ax_r.plot(x, y, "X", color="#f44336", markersize=ns * 2.5,
                              markeredgecolor="white", markeredgewidth=1.5,
                              zorder=8, alpha=1.0)
                    ax_r.annotate("FAILED\nJUNCTION", (x, y), color="#ff5252",
                                  fontsize=8, fontweight="bold",
                                  ha="center", va="bottom",
                                  xytext=(0, 14), textcoords="offset points",
                                  bbox=dict(boxstyle="round,pad=0.2",
                                            facecolor="#1a0000", edgecolor="#f44336",
                                            alpha=0.8))

        # Alternative route — thick bright green
        if path_after and len(path_after) > 1 and show_alt_route:
            path_edges_alt = list(zip(path_after[:-1], path_after[1:]))
            valid_alt = [(u, v) for u, v in path_edges_alt if u in p and v in p]
            if valid_alt:
                nx.draw_networkx_edges(G_after, p, edgelist=valid_alt, ax=ax_r,
                                       edge_color="#00e676", width=ew * 5, alpha=1.0,
                                       style="solid")
                src_a, tgt_a = path_after[0], path_after[-1]
                if src_a in p:
                    ax_r.plot(*p[src_a], "o", color="#00e676", markersize=ns + 6,
                              zorder=9, markeredgecolor="white", markeredgewidth=1.5)
                    ax_r.annotate("ALT START", p[src_a], color="#00e676",
                                  fontsize=7, fontweight="bold",
                                  ha="center", va="bottom",
                                  xytext=(0, 10), textcoords="offset points")
                if tgt_a in p:
                    ax_r.plot(*p[tgt_a], "s", color="#00e676", markersize=ns + 6,
                              zorder=9, markeredgecolor="white", markeredgewidth=1.5)
        elif show_alt_route and (not path_after):
            ax_r.text(0.5, 0.05, "⛔  No Alternative Route Available",
                      transform=ax_r.transAxes, color="#ff5252",
                      fontsize=11, fontweight="bold", ha="center",
                      bbox=dict(boxstyle="round", facecolor="#1a0000",
                                edgecolor="#f44336", alpha=0.85))

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(color="#37474f", label="Normal Road"),
        mpatches.Patch(color="#ab47bc", label="Disconnected Area"),
        Line2D([0], [0], color="#29b6f6", linewidth=3, label="Original Route"),
        Line2D([0], [0], color="#00e676", linewidth=3, label="Alternative Route"),
        mpatches.Patch(color="#f44336", label="Failed Node"),
        mpatches.Patch(color="#ef5350", alpha=0.2, label="Affected Zone"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               facecolor="#1e2130", edgecolor="#37474f",
               labelcolor="#e0e0e0", fontsize=8,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    return _fig_to_bytes(fig)


def render_traffic_heatmap(
    G_before: nx.Graph,
    G_after: nx.Graph,
    removed_nodes: list[int],
    path_before: Optional[list[int]] = None,
    path_after:  Optional[list[int]] = None,
    figsize: tuple[int, int] = (10, 7),
) -> bytes:
    """Traffic redistribution heatmap based on rerouted paths.

    Colors edges by traffic increase after failure:
    - Gray  = no change
    - Green → Yellow → Orange → Red = increasing congestion
    Only edges that carry additional rerouted traffic are highlighted.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_title("🌡  Traffic Redistribution Heatmap\n(Red = High Congestion · Green = Normal)",
                 color="#e0e0e0", fontsize=11, pad=10)
    ax.axis("off")

    pos = _get_pos(G_before)
    if not pos or G_after.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No data available", color="gray",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return _fig_to_bytes(fig)

    ew = max(0.4, min(1.5, 600 / max(G_before.number_of_edges(), 1)))

    # ── Compute edge traffic delta using edge betweenness ────────────────────
    # Fast approximation: edge betweenness on a sampled subgraph
    n = G_after.number_of_nodes()
    k_samples = min(50, n)

    try:
        eb_after = nx.edge_betweenness_centrality(G_after, k=k_samples, normalized=True)
    except Exception:
        eb_after = {}

    try:
        eb_before = nx.edge_betweenness_centrality(G_before, k=k_samples, normalized=True)
    except Exception:
        eb_before = {}

    # Delta = how much more traffic this edge carries post-failure
    edge_delta: dict[tuple, float] = {}
    for u, v in G_before.edges():
        key_fwd = (u, v)
        key_bwd = (v, u)
        before_val = eb_before.get(key_fwd, eb_before.get(key_bwd, 0.0))
        after_val  = eb_after.get(key_fwd,  eb_after.get(key_bwd,  0.0))
        delta = max(0.0, after_val - before_val)
        edge_delta[key_fwd] = delta

    max_delta = max(edge_delta.values()) if edge_delta else 1.0
    if max_delta == 0:
        max_delta = 1.0

    # ── Draw edges colored by traffic increase ────────────────────────────────
    cmap = _get_colormap("RdYlGn_r")

    removed_set = set(removed_nodes)
    removed_edge_set_all = {(u, v) for u, v in ([] if not removed_nodes else [])}

    for u, v in G_before.edges():
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        delta = edge_delta.get((u, v), edge_delta.get((v, u), 0.0))
        norm_delta = delta / max_delta

        if norm_delta < 0.05:
            color, lw, alpha = "#37474f", ew * 0.5, 0.4   # gray — no change
        else:
            rgba = cmap(norm_delta)
            color = rgba
            lw    = ew * (1 + norm_delta * 3)              # thicker = more congested
            alpha = 0.6 + norm_delta * 0.4

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=2)

    # ── Draw nodes ─────────────────────────────────────────────────────────────
    ns = max(3, min(8, 3000 // max(G_before.number_of_nodes(), 1)))
    node_coords = [(pos[n][0], pos[n][1]) for n in G_before.nodes() if n in pos
                   and n not in removed_set]
    if node_coords:
        xs, ys = zip(*node_coords)
        ax.scatter(xs, ys, s=ns, c="#90caf9", alpha=0.5, zorder=3)

    # Mark failed nodes
    for n in removed_set:
        if n in pos:
            ax.plot(*pos[n], "X", color="#f44336", markersize=16,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=8)
            ax.annotate("FAILED", pos[n], color="#ff5252", fontsize=8,
                        fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 10), textcoords="offset points")

    # ── Overlay alternative path ───────────────────────────────────────────────
    if path_after and len(path_after) > 1:
        for u, v in zip(path_after[:-1], path_after[1:]):
            if u in pos and v in pos:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color="#00e676", linewidth=ew * 4, alpha=1.0,
                        linestyle="--", zorder=6)

    # ── Colorbar ───────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=_get_colormap("RdYlGn_r"),
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=20)
    cbar.set_label("Traffic Increase (normalized)", color="#e0e0e0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["None", "Low", "Moderate", "High", "Critical"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e0e0e0", fontsize=7)

    plt.tight_layout()
    return _fig_to_bytes(fig)
