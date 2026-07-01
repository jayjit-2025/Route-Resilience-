"""Route Resilience — Streamlit Dashboard.

Six-page interactive demo for the ISRO Bharatiya Antariksh Hackathon.
Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import networkx as nx
import numpy as np
import streamlit as st

# ── page config must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Route Resilience",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── local imports ───────────────────────────────────────────────────────────
from preprocessing import load_and_preprocess, preprocess_for_model
from preprocessing.image_preprocessor import postprocess_mask
from graph import (
    extract_skeleton,
    build_graph_from_skeleton,
    heal_graph,
    compute_connectivity_ratio,
    get_graph_statistics,
)
from analysis.centrality import compute_centrality, get_heatmap_data, get_top_gatekeepers
from analysis.simulation import simulate_failure, get_alternative_routes, _global_efficiency


# ── CSS: dark sidebar + clean layout ───────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0e1117; }
[data-testid="stSidebar"] * { color: #fafafa; }
.metric-card {
    background: #1e2130; border-radius: 8px; padding: 1rem;
    margin: 0.25rem 0; text-align: center;
}
.metric-card h2 { color: #4fc3f7; margin: 0; }
.metric-card p  { color: #90caf9; margin: 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── session state helpers ───────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "original_image": None,
        "geo_metadata": None,
        "preprocessed": None,
        "road_mask": None,
        "skeleton": None,
        "raw_graph": None,
        "healed_graph": None,
        "centrality_result": None,
        "sim_metrics": None,
        "sim_modified_graph": None,
        "sim_removed_nodes": [],
        "threshold": 0.5,
        "reconstruction_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── sidebar navigation ──────────────────────────────────────────────────────
PAGES = [
    "📡  Upload Image",
    "🛣️  Road Detection",
    "🕸️  Road Reconstruction",
    "⚠️  Critical Bottlenecks",
    "💥  Disaster Simulation",
    "📊  Project Summary",
    "🗂️  Dataset Info",
]

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/ISRO_Logo.svg/200px-ISRO_Logo.svg.png", width=80)
    st.title("Route Resilience")
    st.caption("AI-Powered Road Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("ISRO Bharatiya Antariksh Hackathon")


# ── utility: try to load segmentation model (cached) ───────────────────────
@st.cache_resource(show_spinner="Loading segmentation model …")
def _load_model():
    """Load DeepLabV3+ with pretrained weights (cached across reruns)."""
    try:
        from segmentation import get_model
        model = get_model("deeplabv3+")
        model.load_weights("")  # uses pretrained ImageNet backbone
        return model
    except Exception as e:
        logger.warning("Could not load DL model: %s — using dummy mask.", e)
        return None


def _predict_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    """Run segmentation model and return binary mask at original resolution."""
    model = _load_model()
    h, w = image.shape[:2]
    preprocessed = preprocess_for_model(image, target_size=(512, 512))

    if model is not None:
        raw = model.predict(preprocessed)  # (512, 512) binary
        # Resize back to original image dimensions
        mask = cv2.resize(raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask >= threshold).astype(np.uint8)

    # Fallback: simple threshold on grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, int(threshold * 255), 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


# ── map helpers ─────────────────────────────────────────────────────────────
def _graph_to_folium(
    G: nx.Graph,
    gatekeeper_nodes: Optional[list[int]] = None,
    heatmap_data: Optional[list] = None,
    disabled_nodes: Optional[list[int]] = None,
    height: int = 500,
) -> Optional[str]:
    """Render a NetworkX graph to a Folium HTML map string."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        return None

    nodes_with_coords = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get("lat") is not None and d.get("lon") is not None
    ]
    if not nodes_with_coords:
        return None

    lats = [d["lat"] for _, d in nodes_with_coords]
    lons = [d["lon"] for _, d in nodes_with_coords]
    center = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB dark_matter")

    gk_set = set(gatekeeper_nodes or [])
    dis_set = set(disabled_nodes or [])

    # Draw edges
    for u, v in G.edges():
        u_d, v_d = G.nodes[u], G.nodes[v]
        if u_d.get("lat") and v_d.get("lat"):
            color = "#ef5350" if (u in dis_set or v in dis_set) else "#42a5f5"
            folium.PolyLine(
                [[u_d["lat"], u_d["lon"]], [v_d["lat"], v_d["lon"]]],
                color=color, weight=2, opacity=0.7,
            ).add_to(m)

    # Draw nodes
    for node_id, data in nodes_with_coords:
        if node_id in dis_set:
            color, radius = "#b71c1c", 8
        elif node_id in gk_set:
            color, radius = "#ff6f00", 6
        else:
            color, radius = "#26c6da", 3
        folium.CircleMarker(
            location=[data["lat"], data["lon"]],
            radius=radius, color=color, fill=True, fill_opacity=0.9,
            popup=f"Node {node_id} | {data.get('node_type','?')}",
        ).add_to(m)

    if heatmap_data:
        HeatMap(heatmap_data, radius=20, blur=15, min_opacity=0.3).add_to(m)

    return m._repr_html_()


def _metric(label: str, value: str) -> None:
    """Render a styled metric card."""
    st.markdown(
        f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload Image
# ════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("📡 Upload Satellite Image")
    st.caption("Supported formats: GeoTIFF (.tif), PNG, JPEG")

    uploaded = st.file_uploader(
        "Choose a satellite image", type=["tif", "tiff", "png", "jpg", "jpeg"]
    )

    if uploaded is not None:
        with st.spinner("Loading image …"):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            preprocessed, original, geo_meta = load_and_preprocess(tmp_path)
            st.session_state["original_image"] = original
            st.session_state["geo_metadata"] = geo_meta
            st.session_state["preprocessed"] = preprocessed

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(original, width="stretch")
        with col2:
            st.subheader("Preprocessed (model input)")
            # De-normalise for display: reverse ImageNet norm → [0,1] → uint8
            disp = np.transpose(preprocessed, (1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            disp = np.clip(disp * std + mean, 0, 1)
            st.image((disp * 255).astype(np.uint8), width="stretch")

        if geo_meta:
            st.success(f"✅ Georeferenced — CRS: {geo_meta.crs}")
        else:
            st.info("ℹ️ No geospatial metadata — pixel coordinates will be used.")

        st.success("Image loaded! Navigate to **Road Detection** →")
    else:
        st.info("Upload a satellite image to begin the demo.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Road Detection
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("🛣️ Road Detection")

    if st.session_state["original_image"] is None:
        st.warning("Please upload an image first.")
        st.stop()

    original = st.session_state["original_image"]
    h, w = original.shape[:2]

    threshold = st.slider(
        "Segmentation threshold", min_value=0.1, max_value=0.9,
        value=float(st.session_state["threshold"]), step=0.05,
    )
    st.session_state["threshold"] = threshold

    if st.button("▶  Run Road Segmentation", type="primary"):
        with st.spinner("Running DeepLabV3+ …"):
            mask = _predict_mask(original, threshold)
            st.session_state["road_mask"] = mask

    mask = st.session_state["road_mask"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(original, width="stretch")
    with col2:
        st.subheader("Road Mask")
        if mask is not None:
            st.image(mask * 255, clamp=True, width="stretch")
        else:
            st.caption("Run segmentation to see mask.")
    with col3:
        st.subheader("Overlay")
        if mask is not None:
            overlay = original.copy()
            # Ensure mask matches original size before indexing
            if mask.shape[:2] != original.shape[:2]:
                import cv2 as _cv2
                mask_resized = _cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=_cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            overlay[mask_resized == 1] = [255, 80, 80]
            st.image(overlay, width="stretch")
        else:
            st.caption("Overlay appears after segmentation.")

    if mask is not None:
        road_px = int(mask.sum())
        total_px = h * w
        st.info(f"Road pixels: **{road_px:,}** / {total_px:,} ({road_px/total_px*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Road Reconstruction
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    import time
    import pandas as pd
    from visualization.graph_viz import render_graph, render_overlay

    st.title("🕸️ Road Reconstruction")
    st.caption("Skeleton extraction → Graph construction → MST healing → Routable vector network")

    if st.session_state["road_mask"] is None:
        st.warning("Run Road Detection first.")
        st.stop()

    mask     = st.session_state["road_mask"]
    geo_meta = st.session_state["geo_metadata"]
    original = st.session_state["original_image"]

    # ── Downsample to keep processing fast ──────────────────────────────────
    MAX_DIM = 512
    h_orig, w_orig = mask.shape[:2]
    scale = min(MAX_DIM / max(h_orig, w_orig), 1.0)
    if scale < 1.0:
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        work_mask    = cv2.resize(mask,     (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        work_image   = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if original is not None else None
        st.info(f"Processing at {new_w}×{new_h} (downsampled for speed).")
    else:
        work_mask  = mask
        work_image = original

    # ── Run pipeline button ──────────────────────────────────────────────────
    if st.button("▶  Extract Skeleton & Build Graph", type="primary"):
        t0 = time.time()

        with st.spinner("Extracting skeleton …"):
            skeleton = extract_skeleton(work_mask)
            st.session_state["skeleton"] = skeleton

        with st.spinner("Building road graph …"):
            raw_graph = build_graph_from_skeleton(skeleton, geo_meta)
            st.session_state["raw_graph"] = raw_graph

        with st.spinner("Healing disconnected segments …"):
            healed = heal_graph(raw_graph, max_distance_pixels=30.0)
            st.session_state["healed_graph"] = healed

        st.session_state["reconstruction_time"] = round(time.time() - t0, 2)

    skeleton  = st.session_state["skeleton"]
    raw_graph = st.session_state["raw_graph"]
    healed    = st.session_state["healed_graph"]
    exec_time = st.session_state.get("reconstruction_time", None)

    if exec_time is not None:
        st.success(f"✅ Pipeline completed in {exec_time}s")

    # ── Row 1: Skeleton + Raw Graph + Healed Graph ───────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🦴 Skeleton")
        if skeleton is not None:
            st.image(skeleton * 255, clamp=True, width="stretch")
            st.caption(f"Road pixels: {int(skeleton.sum()):,}")
        else:
            st.caption("Run pipeline to see skeleton.")

    with col2:
        st.subheader("📍 Raw Graph")
        if raw_graph is not None and raw_graph.number_of_nodes() > 0:
            raw_png = render_graph(raw_graph, title="Raw Road Graph")
            st.image(raw_png, width="stretch")
            stats_r = get_graph_statistics(raw_graph)
            ratio_r = compute_connectivity_ratio(raw_graph)
            st.progress(ratio_r, text=f"Connectivity: {ratio_r:.1%}")
            st.caption(f"{stats_r['nodes']} nodes · {stats_r['edges']} edges · {stats_r['components']} components")
        else:
            st.caption("Run pipeline to see raw graph.")

    with col3:
        st.subheader("✅ Healed Graph")
        if healed is not None and healed.number_of_nodes() > 0:
            # Identify healed edges (edges not in raw_graph)
            raw_edge_set = set(raw_graph.edges()) | {(v, u) for u, v in raw_graph.edges()} if raw_graph else set()
            healed_edge_set = {(u, v) for u, v in healed.edges() if (u, v) not in raw_edge_set and (v, u) not in raw_edge_set}
            healed_png = render_graph(healed, title="Healed Road Graph", healed_edges=healed_edge_set)
            st.image(healed_png, width="stretch")
            stats_h = get_graph_statistics(healed)
            ratio_h = compute_connectivity_ratio(healed)
            st.progress(ratio_h, text=f"Connectivity: {ratio_h:.1%}")
            st.caption(f"{stats_h['nodes']} nodes · {stats_h['edges']} edges · {stats_h['components']} components")
        else:
            st.caption("Run pipeline to see healed graph.")

    # ── Row 2: Satellite Overlay toggle ─────────────────────────────────────
    if healed is not None and work_image is not None:
        st.markdown("---")
        show_overlay = st.checkbox("🛰️ Show Graph Overlay on Satellite Image", value=False)
        if show_overlay:
            raw_edge_set = set(raw_graph.edges()) | {(v, u) for u, v in raw_graph.edges()} if raw_graph else set()
            healed_edge_set = {(u, v) for u, v in healed.edges() if (u, v) not in raw_edge_set and (v, u) not in raw_edge_set}
            overlay_png = render_overlay(work_image, healed, healed_edges=healed_edge_set)
            st.image(overlay_png, caption="Road graph overlaid on satellite image — blue=original, green=recovered", width="stretch")

    # ── Row 3: Before / After Comparison ────────────────────────────────────
    if raw_graph is not None and healed is not None:
        st.markdown("---")
        st.subheader("📊 Before / After Comparison")

        stats_r = get_graph_statistics(raw_graph)
        stats_h = get_graph_statistics(healed)
        ratio_r = compute_connectivity_ratio(raw_graph)
        ratio_h = compute_connectivity_ratio(healed)
        new_edges_added = healed.number_of_edges() - raw_graph.number_of_edges()
        conn_improvement = ratio_h - ratio_r

        comp_data = {
            "Metric": ["Nodes", "Edges", "Connected Components",
                        "Connectivity Ratio", "New Edges Added", "Connectivity Improvement"],
            "Raw Graph": [
                str(stats_r["nodes"]), str(stats_r["edges"]), str(stats_r["components"]),
                f"{ratio_r:.1%}", "—", "—",
            ],
            "Healed Graph": [
                str(stats_h["nodes"]), str(stats_h["edges"]), str(stats_h["components"]),
                f"{ratio_h:.1%}", str(new_edges_added),
                f"+{conn_improvement:.1%}" if conn_improvement >= 0 else f"{conn_improvement:.1%}",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), hide_index=True)

        # Progress bars
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Raw connectivity")
            st.progress(ratio_r)
        with c2:
            st.caption("Healed connectivity")
            st.progress(ratio_h)

    # ── Row 4: Graph Quality Metrics ─────────────────────────────────────────
    if healed is not None:
        st.markdown("---")
        st.subheader("🔬 Graph Quality Metrics")

        stats_h = get_graph_statistics(healed)
        ratio_h = compute_connectivity_ratio(healed)
        n = healed.number_of_nodes()
        e = healed.number_of_edges()
        density = nx.density(healed)
        avg_deg = stats_h.get("avg_degree", 0.0)
        lcc     = stats_h.get("largest_component_size", 0)
        comps   = stats_h.get("components", 0)
        new_e   = healed.number_of_edges() - (raw_graph.number_of_edges() if raw_graph else 0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Nodes",    str(n))
        m2.metric("Total Edges",    str(e))
        m3.metric("Components",     str(comps))
        m4.metric("Largest CC",     str(lcc))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg Degree",      f"{avg_deg:.2f}")
        m6.metric("Connectivity",    f"{ratio_h:.1%}")
        m7.metric("Graph Density",   f"{density:.4f}")
        m8.metric("Recovered Gaps",  str(new_e))

        if exec_time is not None:
            st.caption(f"⏱️ Execution time: {exec_time}s")

    # ── Row 5: Downloads ─────────────────────────────────────────────────────
    if raw_graph is not None and healed is not None:
        st.markdown("---")
        st.subheader("⬇️ Export")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            raw_png = render_graph(raw_graph, title="Raw Road Graph")
            st.download_button("📥 Download Raw Graph",   data=raw_png,  file_name="raw_graph.png",    mime="image/png")
        with dl2:
            raw_edge_set = set(raw_graph.edges()) | {(v, u) for u, v in raw_graph.edges()}
            healed_edge_set = {(u, v) for u, v in healed.edges() if (u, v) not in raw_edge_set and (v, u) not in raw_edge_set}
            healed_png = render_graph(healed, title="Healed Road Graph", healed_edges=healed_edge_set)
            st.download_button("📥 Download Healed Graph", data=healed_png, file_name="healed_graph.png", mime="image/png")
        with dl3:
            if work_image is not None:
                overlay_png = render_overlay(work_image, healed, healed_edges=healed_edge_set)
                st.download_button("📥 Download Overlay",    data=overlay_png, file_name="overlay.png",      mime="image/png")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Critical Bottlenecks
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("⚠️ Critical Bottleneck Detection")

    if st.session_state["healed_graph"] is None:
        st.warning("Complete Road Reconstruction first.")
        st.stop()

    G = st.session_state["healed_graph"]

    percentile = st.slider("Gatekeeper threshold (percentile)", 80, 99, 95)

    if st.button("▶  Compute Betweenness Centrality", type="primary"):
        with st.spinner("Computing centrality (may take ~30s for large graphs) …"):
            result = compute_centrality(G, threshold_percentile=float(percentile))
            st.session_state["centrality_result"] = result

    result = st.session_state["centrality_result"]

    if result is not None:
        top10 = get_top_gatekeepers(G, result, top_n=10)
        heatmap = get_heatmap_data(G, result.node_centrality)

        col1, col2, col3 = st.columns(3)
        with col1:
            _metric("Total Nodes", str(G.number_of_nodes()))
        with col2:
            _metric("Gatekeeper Nodes", str(len(result.gatekeeper_nodes)))
        with col3:
            _metric("Threshold", f"{result.threshold_value:.4f}")

        st.subheader("Top 10 Critical Intersections")
        import pandas as pd
        df = pd.DataFrame(top10)[["node_id", "centrality", "node_type", "pixel_x", "pixel_y"]]
        df["centrality"] = df["centrality"].apply(lambda x: f"{x:.6f}")
        st.dataframe(df, width="stretch")

        map_html = _graph_to_folium(
            G,
            gatekeeper_nodes=result.gatekeeper_nodes,
            heatmap_data=heatmap if heatmap else None,
        )
        if map_html:
            st.subheader("Gatekeeper Map + Centrality Heatmap")
            st.components.v1.html(map_html, height=520, scrolling=False)
        else:
            st.info("Map requires georeferenced image. Showing table instead.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Disaster Simulation
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    import pandas as pd
    from visualization.graph_viz import (
        render_simulation_comparison,
        render_traffic_heatmap,
        render_graph,
    )

    st.title("💥 Disaster Simulation")
    st.caption("Decision Support Dashboard — ISRO Bharatiya Antariksh Hackathon")

    if st.session_state["healed_graph"] is None:
        st.warning("Complete Road Reconstruction first.")
        st.stop()

    G        = st.session_state["healed_graph"]
    result   = st.session_state["centrality_result"]

    # ── SIMULATION TIMELINE ─────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;
                font-size:0.8rem;color:#90caf9;">
      <span style="background:#1e3a5f;padding:4px 10px;border-radius:12px;">🟢 Normal Network</span>
      <span>→</span>
      <span style="background:#3a2a1a;padding:4px 10px;border-radius:12px;">⚡ Select Failure</span>
      <span>→</span>
      <span style="background:#3a1a1a;padding:4px 10px;border-radius:12px;">❌ Node Removed</span>
      <span>→</span>
      <span style="background:#1a3a1a;padding:4px 10px;border-radius:12px;">🔄 Routes Recalculated</span>
      <span>→</span>
      <span style="background:#1a1a3a;padding:4px 10px;border-radius:12px;">📊 Final Network</span>
    </div>
    """, unsafe_allow_html=True)

    # ── SIMULATION CONTROLS ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Simulation Controls")
    ctrl1, ctrl2 = st.columns([2, 2])

    with ctrl1:
        sim_mode = st.radio(
            "Simulation Type",
            ["🌊 Flood (Node Failure)", "🚗 Accident (Edge Failure)",
             "🌉 Bridge Collapse (Edge Failure)", "🚧 Road Construction (Multi-Edge)"],
            horizontal=False,
        )

    with ctrl2:
        is_node_failure = "Node Failure" in sim_mode
        is_edge_failure = not is_node_failure

        if result and result.gatekeeper_nodes:
            st.caption(f"💡 Top gatekeepers: {result.gatekeeper_nodes[:5]}")

        all_nodes = sorted(G.nodes())
        all_edges = list(G.edges())

        if is_node_failure:
            selected_nodes = st.multiselect(
                "Select nodes to disable",
                options=all_nodes,
                default=st.session_state["sim_removed_nodes"][:3],
                format_func=lambda n: f"Node {n} ({G.nodes[n].get('node_type','?')})",
                key="sim_node_select",
            )
            selected_edges = []

            # Quick-select buttons
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                if st.button("🎯 Critical Node", use_container_width=True):
                    if result and result.gatekeeper_nodes:
                        selected_nodes = [result.gatekeeper_nodes[0]]
            with qc2:
                if st.button("🎲 Random Node", use_container_width=True):
                    import random
                    selected_nodes = [random.choice(all_nodes)]
            with qc3:
                if st.button("🔝 Top 3 Gatekeepers", use_container_width=True):
                    if result and result.gatekeeper_nodes:
                        selected_nodes = result.gatekeeper_nodes[:3]
        else:
            selected_nodes = []
            edge_labels = [f"{u}↔{v}" for u, v in all_edges[:200]]
            selected_edge_labels = st.multiselect(
                "Select edges to disable",
                options=edge_labels,
                max_selections=5,
                key="sim_edge_select",
            )
            selected_edges = []
            edge_map = {f"{u}↔{v}": (u, v) for u, v in all_edges[:200]}
            for lbl in selected_edge_labels:
                if lbl in edge_map:
                    selected_edges.append(edge_map[lbl])

    st.session_state["sim_removed_nodes"] = selected_nodes

    col_run, col_reset = st.columns([1, 1])
    run   = col_run.button("▶  Run Simulation", type="primary", use_container_width=True)
    reset = col_reset.button("🔄  Reset", use_container_width=True)

    if reset:
        st.session_state["sim_removed_nodes"] = []
        st.session_state["sim_metrics"]        = None
        st.session_state["sim_modified_graph"] = None
        st.rerun()

    if run and (selected_nodes or selected_edges):
        with st.spinner("Simulating infrastructure failure …"):
            modified, metrics = simulate_failure(
                G, removed_nodes=selected_nodes, removed_edges=selected_edges
            )
            st.session_state["sim_metrics"]        = metrics
            st.session_state["sim_modified_graph"] = modified
        st.rerun()

    metrics  = st.session_state["sim_metrics"]
    modified = st.session_state["sim_modified_graph"]

    if metrics is None:
        st.info("👆 Select nodes/edges above and click **Run Simulation** to begin.")
        st.stop()

    # ── IMPACT METRICS BAR ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Impact Metrics")
    m1, m2, m3, m4 = st.columns(4)
    orig_eff   = _global_efficiency(G)
    orig_comps = nx.number_connected_components(G)

    m1.metric("Travel Delay",       f"+{metrics.travel_delay:.1f}%",
              delta=f"+{metrics.travel_delay:.1f}%", delta_color="inverse")
    m2.metric("Components",         str(metrics.components),
              delta=str(metrics.components - orig_comps), delta_color="inverse")
    m3.metric("Network Efficiency", f"{metrics.efficiency:.3f}",
              delta=f"{metrics.efficiency - orig_eff:.3f}", delta_color="normal")
    m4.metric("Resilience Index",   f"{metrics.resilience:.3f}",
              delta=f"{metrics.resilience - 1.0:.3f}", delta_color="normal")

    # ── BEFORE / AFTER GRAPH COMPARISON ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ Before vs After Network")

    # Visualization toggles
    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    show_failed   = vc1.checkbox("☑ Failed Node",       value=True)
    show_alt      = vc2.checkbox("☑ Alternative Route", value=True)
    show_discon   = vc3.checkbox("☑ Disconnected Areas", value=True)
    show_critical = vc4.checkbox("☑ Critical Nodes",    value=True)
    show_zone     = vc5.checkbox("☑ Affected Zone",     value=True)

    # Compute paths for overlay
    surviving = [n for n in G.nodes() if n not in selected_nodes]
    path_before, path_after = None, None
    if len(surviving) >= 2 and modified is not None:
        src, tgt = surviving[0], surviving[-1]
        try:
            path_before = nx.shortest_path(G, src, tgt)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_before = None
        try:
            path_after = nx.shortest_path(modified, src, tgt)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_after = None

    comp_png = render_simulation_comparison(
        G, modified or G,
        removed_nodes=selected_nodes,
        removed_edges=selected_edges,
        path_before=path_before,
        path_after=path_after,
        show_failed=show_failed,
        show_alt_route=show_alt,
        show_disconnected=show_discon,
        show_critical=show_critical,
    )
    st.image(comp_png, width="stretch")

    # ── ALTERNATIVE ROUTES ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔀 Route Analysis")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Original Shortest Path**")
        if path_before:
            st.code(f"{' → '.join(map(str, path_before[:12]))}{'…' if len(path_before)>12 else ''}")
            st.caption(f"Length: {len(path_before)-1} hops")
        else:
            st.caption("No path computed.")
    with r2:
        st.markdown("**Alternative Route (post-failure)**")
        if path_after:
            st.code(f"{' → '.join(map(str, path_after[:12]))}{'…' if len(path_after)>12 else ''}")
            hop_delta = len(path_after) - len(path_before) if path_before else 0
            st.caption(f"Length: {len(path_after)-1} hops  (+{hop_delta} detour)")
        else:
            st.error("⛔ No alternative route available — network is disconnected.")

    # ── TRAFFIC HEATMAP ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌡️ Traffic Impact Heatmap")
    heatmap_png = render_traffic_heatmap(G, modified or G,
                                         removed_nodes=selected_nodes,
                                         path_before=path_before,
                                         path_after=path_after)
    st.image(heatmap_png, width="stretch")
    st.caption("Edge thickness = traffic load · Red = critical congestion · Green = normal flow · Gray = no change · -- = alternative route")

    # ── BEFORE / AFTER METRICS TABLE ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Before vs After Metrics")

    orig_stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": orig_comps,
        "efficiency": orig_eff,
        "lcc": max(len(c) for c in nx.connected_components(G)),
        "avg_deg": sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1),
        "density": nx.density(G),
    }
    mod_graph = modified if modified is not None else G
    mod_comps_list = list(nx.connected_components(mod_graph)) if mod_graph.number_of_nodes() > 0 else []
    mod_stats = {
        "nodes": mod_graph.number_of_nodes(),
        "edges": mod_graph.number_of_edges(),
        "components": metrics.components,
        "efficiency": metrics.efficiency,
        "lcc": max((len(c) for c in mod_comps_list), default=0),
        "avg_deg": sum(d for _, d in mod_graph.degree()) / max(mod_graph.number_of_nodes(), 1),
        "density": nx.density(mod_graph),
    }

    table_data = {
        "Metric": [
            "Total Nodes", "Total Edges", "Connected Components",
            "Connectivity Ratio", "Average Path (travel delay)", "Network Efficiency",
            "Resilience Index", "Largest Component", "Average Degree", "Graph Density"
        ],
        "Before Failure": [
            str(orig_stats["nodes"]), str(orig_stats["edges"]),
            str(orig_stats["components"]), "100%", "0.0%",
            f"{orig_stats['efficiency']:.4f}", "1.000",
            str(orig_stats["lcc"]), f"{orig_stats['avg_deg']:.2f}",
            f"{orig_stats['density']:.4f}",
        ],
        "After Failure": [
            str(mod_stats["nodes"]), str(mod_stats["edges"]),
            str(mod_stats["components"]),
            f"{mod_stats['lcc']/max(mod_graph.number_of_nodes(),1):.1%}",
            f"+{metrics.travel_delay:.1f}%",
            f"{mod_stats['efficiency']:.4f}", f"{metrics.resilience:.3f}",
            str(mod_stats["lcc"]), f"{mod_stats['avg_deg']:.2f}",
            f"{mod_stats['density']:.4f}",
        ],
    }
    st.dataframe(pd.DataFrame(table_data), hide_index=True)

    # ── SIMULATION SUMMARY CARD ───────────────────────────────────────────────
    st.markdown("---")
    sim_type_label = sim_mode.split(" ", 1)[1] if " " in sim_mode else sim_mode
    affected_str   = ", ".join(map(str, selected_nodes[:3])) + ("…" if len(selected_nodes) > 3 else "")
    conn_before    = "100%"
    conn_after     = f"{mod_stats['lcc']/max(mod_graph.number_of_nodes(),1):.1%}"

    if metrics.resilience >= 0.9:
        recommendation = "Network is highly resilient. No immediate action required."
    elif metrics.resilience >= 0.7:
        recommendation = "Moderate impact. Deploy traffic diversions to affected corridors."
    else:
        recommendation = "Critical failure. Deploy emergency traffic diversion immediately."

    st.markdown(f"""
    <div style="background:#1e2130;border-radius:12px;padding:1.5rem;border-left:4px solid #ef5350;">
      <h4 style="color:#ef9a9a;margin-top:0;">📋 Simulation Summary</h4>
      <table style="width:100%;color:#e0e0e0;font-size:0.9rem;border-collapse:collapse;">
        <tr><td style="padding:4px 8px;color:#90caf9;"><b>Failure Type</b></td><td>{sim_type_label}</td></tr>
        <tr><td style="padding:4px 8px;color:#90caf9;"><b>Affected Node(s)</b></td><td>{affected_str or '—'}</td></tr>
        <tr><td style="padding:4px 8px;color:#90caf9;"><b>Components Created</b></td><td>{metrics.components}</td></tr>
        <tr><td style="padding:4px 8px;color:#90caf9;"><b>Travel Delay</b></td><td>+{metrics.travel_delay:.1f}%</td></tr>
        <tr><td style="padding:4px 8px;color:#90caf9;"><b>Connectivity</b></td><td>{conn_before} → {conn_after}</td></tr>
        <tr><td style="padding:4px 8px;color:#90caf9;"><b>Resilience Index</b></td><td>{metrics.resilience:.3f}</td></tr>
        <tr><td style="padding:4px 8px;color:#ffcc80;"><b>Recommended Action</b></td>
            <td style="color:#a5d6a7;">{recommendation}</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── EXPORT ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⬇️ Export Results")
    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.download_button("📥 Graph (PNG)", data=comp_png,
                           file_name="simulation_graph.png", mime="image/png")
    with e2:
        st.download_button("📥 Heatmap (PNG)", data=heatmap_png,
                           file_name="traffic_heatmap.png", mime="image/png")
    with e3:
        csv_bytes = pd.DataFrame(table_data).to_csv(index=False).encode()
        st.download_button("📥 Metrics (CSV)", data=csv_bytes,
                           file_name="simulation_metrics.csv", mime="text/csv")
    with e4:
        summary_txt = (
            f"Simulation Summary\n"
            f"Failure Type: {sim_type_label}\n"
            f"Affected Nodes: {affected_str}\n"
            f"Components: {metrics.components}\n"
            f"Travel Delay: +{metrics.travel_delay:.1f}%\n"
            f"Connectivity: {conn_before} → {conn_after}\n"
            f"Resilience Index: {metrics.resilience:.3f}\n"
            f"Recommended Action: {recommendation}\n"
        )
        st.download_button("📥 Report (TXT)", data=summary_txt.encode(),
                           file_name="simulation_report.txt", mime="text/plain")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Project Summary
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("📊 Project Summary")
    st.caption("End-to-end pipeline results for ISRO judges.")

    original  = st.session_state["original_image"]
    mask      = st.session_state["road_mask"]
    healed    = st.session_state["healed_graph"]
    result    = st.session_state["centrality_result"]
    metrics   = st.session_state["sim_metrics"]

    # Visual summary row
    cols = st.columns(4)
    labels = ["Original Image", "Road Mask", "Connectivity", "Simulation"]
    for col, label in zip(cols, labels):
        with col:
            st.markdown(f"**{label}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if original is not None:
            st.image(original, width="stretch")
    with c2:
        if mask is not None:
            st.image(mask * 255, clamp=True, width="stretch")
        else:
            st.caption("Not yet computed.")
    with c3:
        if healed is not None:
            ratio = compute_connectivity_ratio(healed)
            st.metric("Connectivity Ratio", f"{ratio:.1%}")
            st.metric("Nodes", healed.number_of_nodes())
            st.metric("Edges", healed.number_of_edges())
        else:
            st.caption("Not yet computed.")
    with c4:
        if metrics is not None:
            st.metric("Travel Delay", f"+{metrics.travel_delay:.1f}%")
            st.metric("Resilience Index", f"{metrics.resilience:.3f}")
            st.metric("Components", metrics.components)
        else:
            st.caption("Not yet simulated.")

    st.markdown("---")
    st.markdown("### Performance Metrics")
    data = {
        "Metric": [
            "Connectivity Ratio",
            "Gatekeeper Nodes",
            "Travel Delay (simulation)",
            "Network Efficiency (post-failure)",
            "Resilience Index",
        ],
        "Value": [
            f"{compute_connectivity_ratio(healed):.1%}" if healed else "—",
            str(len(result.gatekeeper_nodes)) if result else "—",
            f"+{metrics.travel_delay:.1f}%" if metrics else "—",
            f"{metrics.efficiency:.3f}" if metrics else "—",
            f"{metrics.resilience:.3f}" if metrics else "—",
        ],
    }
    import pandas as pd
    st.table(pd.DataFrame(data))

    st.markdown("---")
    st.markdown(
        "**Route Resilience** — AI-Powered Occlusion Robust Road Extraction "
        "and Graph-Theoretic Urban Road Intelligence  \n"
        "Built for the 🇮🇳 ISRO Bharatiya Antariksh Hackathon"
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Dataset Information
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    import time
    import pandas as pd
    from datasets.factory import DatasetFactory, DatasetConfig
    from datasets.evaluation import evaluate_segmentation

    st.title("🗂️ Dataset Information")
    st.caption("Configure training datasets and view evaluation metrics")

    # ── Dataset Configuration ────────────────────────────────────────────────
    st.markdown("### ⚙️ Dataset Configuration")
    col1, col2 = st.columns(2)

    with col1:
        active_ds = st.selectbox(
            "Active Dataset",
            ["spacenet", "deepglobe", "opensatmap", "osm", "combined"],
            index=0,
        )
        target_h = st.selectbox("Image Resolution", [256, 512, 1024], index=1)
        augment  = st.checkbox("Enable Augmentation", value=True)
        use_cache = st.checkbox("Use Cache", value=True)

    with col2:
        spacenet_root   = st.text_input("SpaceNet Root",   placeholder="datasets/spacenet")
        deepglobe_root  = st.text_input("DeepGlobe Root",  placeholder="datasets/deepglobe")
        opensatmap_root = st.text_input("OpenSatMap Root", placeholder="datasets/opensatmap")
        osm_root        = st.text_input("OSM Root",        placeholder="datasets/osm")

    cfg = DatasetConfig(
        active          = active_ds,
        spacenet_root   = spacenet_root   or None,
        deepglobe_root  = deepglobe_root  or None,
        opensatmap_root = opensatmap_root or None,
        osm_root        = osm_root        or None,
        target_size     = (target_h, target_h),
        augment         = augment,
        use_cache       = use_cache,
    )
    factory = DatasetFactory(cfg)
    info    = factory.get_info()

    # ── Dataset Summary Panel ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Dataset Summary")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Active Dataset",  info["active_dataset"].upper())
    d2.metric("Train Images",    str(info["train_images"]))
    d3.metric("Val Images",      str(info["val_images"]))
    d4.metric("Resolution",      f"{target_h}×{target_h}")

    # Dataset paths table
    path_data = {
        "Dataset":    ["SpaceNet", "DeepGlobe", "OpenSatMap", "OSM"],
        "Root Path":  [
            info["spacenet_root"], info["deepglobe_root"],
            info["opensatmap_root"], info["osm_root"],
        ],
        "Status": [
            "✅ Configured" if info["spacenet_root"] != "not configured" else "⚠️ Not set",
            "✅ Configured" if info["deepglobe_root"] != "not configured" else "⚠️ Not set",
            "✅ Configured" if info["opensatmap_root"] != "not configured" else "⚠️ Not set",
            "✅ Configured" if info["osm_root"] != "not configured" else "⚠️ Not set",
        ],
    }
    st.dataframe(pd.DataFrame(path_data), hide_index=True)

    # ── Augmentation Preview ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎨 Augmentation Pipeline")
    aug_items = {
        "Transform": [
            "Random Horizontal Flip", "Random Vertical Flip", "Random Rotation (±10°)",
            "Random Crop (448×448)", "Color Jitter", "Cloud Occlusion Simulation",
            "Shadow Occlusion Simulation", "ImageNet Normalization", "Channel-First (C,H,W)",
        ],
        "Split": [
            "Train only", "Train only", "Train only", "Train only",
            "Train only", "Train only", "Train only", "All", "All",
        ],
        "Enabled": ["✅" if augment else "❌"] * 7 + ["✅", "✅"],
    }
    st.dataframe(pd.DataFrame(aug_items), hide_index=True)

    # ── Model Info ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Model Information")
    m1, m2, m3 = st.columns(3)
    m1.metric("Architecture",  "DeepLabV3+ / U-Net")
    m2.metric("Backbone",      "ResNet-50 (ImageNet)")
    m3.metric("Output Classes","1 (Binary Road)")

    # ── Evaluation Metrics ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Evaluation Metrics")
    st.info("Upload an image, run segmentation, and evaluate against ground truth below.")

    mask      = st.session_state.get("road_mask")
    healed    = st.session_state.get("healed_graph")

    if mask is not None:
        st.markdown("**Run Evaluation Against Ground Truth**")
        gt_file = st.file_uploader("Upload ground truth mask (PNG)", type=["png", "jpg"])

        if gt_file is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(gt_file.read())
                gt_path = tmp.name

            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_binary = (gt_mask > 127).astype(np.uint8)

                t0      = time.time()
                results = evaluate_segmentation(mask, gt_binary, pred_graph=healed)
                elapsed = time.time() - t0

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("IoU (strict)",     f"{results['iou']:.4f}")
                e2.metric("Dice Score",       f"{results['dice']:.4f}")
                e3.metric("Relaxed IoU (±3px)", f"{results['relaxed_iou']:.4f}")
                e4.metric("Inference Time",   f"{elapsed:.2f}s")

                if results["connectivity_ratio"] is not None:
                    st.metric("Connectivity Ratio", f"{results['connectivity_ratio']:.1%}")

                # Show side-by-side comparison
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Predicted Mask")
                    st.image(mask * 255, clamp=True, width="stretch")
                with col_b:
                    st.subheader("Ground Truth")
                    st.image(gt_binary * 255, clamp=True, width="stretch")
    else:
        st.caption("Run segmentation on the Road Detection page to enable evaluation.")

    # ── Download Dataset Template ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⬇️ Dataset Setup Guide")
    readme = """# Dataset Setup Guide — Route Resilience

## SpaceNet Roads
1. Register at https://spacenet.ai/roads/
2. Download SpaceNet v2/v3/v5 road challenge data
3. Place in:  datasets/spacenet/images/  and  datasets/spacenet/masks/

## DeepGlobe Road Extraction
1. Download from https://competitions.codalab.org/competitions/18467
2. Files follow pattern: <id>_sat.jpg and <id>_mask.png
3. Place all files in:  datasets/deepglobe/

## OpenSatMap
1. Download from https://opensatmap.github.io/
2. Place images in:  datasets/opensatmap/images/
3. Place labels in:  datasets/opensatmap/labels/

## OSM Ground Truth
1. Any georeferenced GeoTIFF folder works
2. OSM road vectors are auto-fetched from the internet
3. Place images in:  datasets/osm/

## Expected Structure
datasets/
├── spacenet/
│   ├── images/   # satellite tiles
│   └── masks/    # road masks
├── deepglobe/    # flat folder with *_sat.jpg + *_mask.png
├── opensatmap/
│   ├── images/
│   └── labels/
└── osm/          # GeoTIFF satellite images (masks auto-fetched)
"""
    st.download_button("📥 Download Setup Guide",
                       data=readme.encode(),
                       file_name="dataset_setup_guide.txt",
                       mime="text/plain")
