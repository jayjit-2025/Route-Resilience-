"""Core data models for Route Resilience pipeline.

This module defines dataclass models for storing pipeline state and results.
All models use type annotations and support optional fields where appropriate.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import networkx as nx
from rasterio.transform import Affine
from rasterio.crs import CRS


@dataclass
class GeoMetadata:
    """Geospatial reference information for satellite imagery.
    
    Attributes:
        crs: Coordinate Reference System (e.g., EPSG:4326 for WGS84)
        transform: Affine transformation matrix for pixel-to-coordinate conversion
        bounds: Spatial extent as (minx, miny, maxx, maxy) in CRS units
        shape: Image dimensions as (height, width) in pixels
    """
    crs: CRS
    transform: Affine
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]


@dataclass
class PipelineState:
    """Container for all intermediate results throughout the processing pipeline.
    
    This dataclass holds the outputs of each pipeline stage, enabling
    state persistence, debugging, and partial pipeline execution.
    
    Attributes:
        raw_image: Original satellite imagery array
        geo_metadata: Geospatial reference information
        preprocessed_image: Normalized and prepared imagery
        road_mask: Binary segmentation mask from model inference
        skeleton: One-pixel-wide road centerlines
        junction_coords: List of junction pixel coordinates (x, y)
        endpoint_coords: List of endpoint pixel coordinates (x, y)
        raw_graph: Initial NetworkX graph before healing
        healed_graph: Topologically corrected graph after healing
        connectivity_ratio: Ratio of nodes in largest component to total nodes
        centrality_result: Results from centrality analysis
        simulation_metrics: Results from disaster simulation (if run)
        modified_graph: Graph after simulated infrastructure failure (if run)
    """
    raw_image: Optional[np.ndarray] = None
    geo_metadata: Optional[GeoMetadata] = None
    preprocessed_image: Optional[np.ndarray] = None
    road_mask: Optional[np.ndarray] = None
    skeleton: Optional[np.ndarray] = None
    junction_coords: list[tuple[int, int]] = field(default_factory=list)
    endpoint_coords: list[tuple[int, int]] = field(default_factory=list)
    raw_graph: Optional[nx.Graph] = None
    healed_graph: Optional[nx.Graph] = None
    connectivity_ratio: Optional[float] = None
    centrality_result: Optional['CentralityResult'] = None
    simulation_metrics: Optional['SimulationMetrics'] = None
    modified_graph: Optional[nx.Graph] = None


@dataclass
class CentralityResult:
    """Container for centrality analysis results.
    
    Attributes:
        node_centrality: Mapping from node ID to betweenness centrality score
        gatekeeper_nodes: List of node IDs exceeding the centrality threshold
        threshold_value: Actual threshold value used for gatekeeper identification
    """
    node_centrality: dict[int, float]
    gatekeeper_nodes: list[int]
    threshold_value: float


@dataclass
class SimulationMetrics:
    """Container for disaster simulation results.
    
    Metrics quantify the impact of infrastructure failures on network performance.
    
    Attributes:
        travel_delay: Average path length increase as percentage
        components: Number of disconnected regions after failure
        efficiency: Global network efficiency (average inverse shortest path length)
        resilience: Composite robustness score in [0, 1] range
    """
    travel_delay: float
    components: int
    efficiency: float
    resilience: float


@dataclass
class RenderConfig:
    """Configuration for map rendering and visualization.
    
    Attributes:
        tile_layer: Base map tile source (e.g., "OpenStreetMap", "CartoDB")
        zoom_start: Initial map zoom level
        road_color: Hex color code for road edges
        gatekeeper_color: Hex color code for high-centrality nodes
        heatmap_gradient: Optional gradient mapping for centrality heatmap
    """
    tile_layer: str = "OpenStreetMap"
    zoom_start: int = 13
    road_color: str = "#3388ff"
    gatekeeper_color: str = "#ff0000"
    heatmap_gradient: Optional[dict[float, str]] = None
