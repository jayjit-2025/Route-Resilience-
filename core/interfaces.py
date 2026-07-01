"""
Core protocol interfaces for Route Resilience system.

This module defines abstract interfaces using typing.Protocol for structural
subtyping, enabling plugin-based modularity across the system architecture.

Data model types (CentralityResult, SimulationMetrics, RenderConfig) are
imported from core.data_models to maintain a single source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Optional
import numpy as np
import networkx as nx

if TYPE_CHECKING:
    import torch

from core.data_models import CentralityResult, SimulationMetrics, RenderConfig

# Re-export for convenience — importers can use `from core.interfaces import ...`
__all__ = [
    "RoadSegmentationModel",
    "RoadHealingAlgorithm",
    "CentralityAnalyzer",
    "DisasterSimulator",
    "MapRenderer",
    "CentralityResult",
    "SimulationMetrics",
    "RenderConfig",
]


# =============================================================================
# Protocol Interface 1: Road Segmentation Model
# =============================================================================

class RoadSegmentationModel(Protocol):
    """
    Abstract interface for road segmentation models.
    
    This protocol defines the contract for deep learning models that perform
    pixel-wise road classification from satellite imagery. Implementations
    can include DeepLabV3+, U-Net, or Transformer-based architectures.
    
    The interface supports both single-image and batch prediction, with
    optional GPU acceleration.
    """
    
    def load_weights(self, weights_path: str) -> None:
        """
        Load pretrained model weights from disk.
        
        Args:
            weights_path: Absolute or relative path to model weights file
                         (e.g., .pth, .pt, .h5)
        
        Raises:
            FileNotFoundError: If weights file does not exist
            RuntimeError: If weights format is incompatible with model
        """
        ...
    
    def predict(
        self, 
        image: np.ndarray,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Predict binary road mask from preprocessed satellite image.
        
        Args:
            image: Preprocessed image array with shape (H, W, C) or (C, H, W),
                  normalized to [0, 1] or [-1, 1] depending on model
            device: PyTorch device for inference (cuda/cpu). If None, uses
                   default device (GPU if available, else CPU)
            
        Returns:
            Binary road mask with shape (H, W) containing {0, 1} values,
            where 1 indicates road pixels
        
        Raises:
            ValueError: If image dimensions are invalid
            RuntimeError: If model weights not loaded
        """
        ...
    
    def predict_batch(
        self,
        images: list[np.ndarray],
        device: Optional[torch.device] = None
    ) -> list[np.ndarray]:
        """
        Predict road masks for a batch of images.
        
        Batch processing enables throughput optimization through parallelized
        inference on GPU.
        
        Args:
            images: List of preprocessed image arrays, each with shape 
                   (H, W, C) or (C, H, W)
            device: PyTorch device for inference (cuda/cpu)
            
        Returns:
            List of binary road masks, each with shape (H, W)
        
        Raises:
            ValueError: If images have inconsistent dimensions
            RuntimeError: If model weights not loaded
        """
        ...


# =============================================================================
# Protocol Interface 2: Road Healing Algorithm
# =============================================================================

class RoadHealingAlgorithm(Protocol):
    """
    Abstract interface for road graph healing algorithms.
    
    This protocol defines the contract for algorithms that reconnect
    fragmented road segments into a unified, routable network. Implementations
    can include Minimum Spanning Tree, Union-Find, or K-Nearest Neighbor
    approaches.
    
    The primary goal is to maximize network connectivity while minimizing
    the total length of synthetic healing edges added.
    """
    
    def heal(
        self, 
        graph: nx.Graph,
        max_distance_meters: float
    ) -> nx.Graph:
        """
        Reconnect disconnected road fragments through strategic edge addition.
        
        The algorithm identifies disconnected components in the input graph
        and adds synthetic edges between nearby nodes across component
        boundaries. Healing edges are constrained by maximum distance to
        prevent unrealistic connections.
        
        Args:
            graph: Potentially disconnected road network graph. Nodes should
                  have 'x', 'y' attributes for spatial coordinates (or 'lat',
                  'lon' for geographic coordinates). Edges should have
                  'length' attribute for distance in meters.
            max_distance_meters: Maximum Euclidean distance (in meters) for
                                healing edge creation. Edges longer than this
                                threshold will not be added.
            
        Returns:
            Healed graph with improved connectivity. The returned graph
            contains all original nodes and edges plus synthetic healing edges
            marked with attribute {'healed': True}.
        
        Raises:
            ValueError: If graph has no nodes or invalid coordinate attributes
        """
        ...
    
    def compute_connectivity_ratio(self, graph: nx.Graph) -> float:
        """
        Compute connectivity ratio as quality metric for graph healing.
        
        The connectivity ratio measures what fraction of nodes belong to the
        largest connected component. A ratio of 1.0 indicates a fully
        connected graph, while lower values indicate fragmentation.
        
        Args:
            graph: Road network graph (connected or disconnected)
            
        Returns:
            Connectivity ratio in range [0, 1], calculated as:
            (nodes in largest component) / (total nodes)
        
        Raises:
            ValueError: If graph has no nodes
        """
        ...


# =============================================================================
# Protocol Interface 3: Centrality Analyzer
# =============================================================================

class CentralityAnalyzer(Protocol):
    """
    Abstract interface for network centrality computation.
    
    This protocol defines the contract for algorithms that identify critical
    infrastructure nodes through betweenness centrality analysis. High-
    centrality nodes (gatekeepers) are bottlenecks whose removal significantly
    impacts network connectivity.
    
    Implementations can provide exact or approximate centrality computation
    for scalability to large graphs.
    """
    
    def analyze(
        self,
        graph: nx.Graph,
        threshold_percentile: float = 95.0
    ) -> CentralityResult:
        """
        Compute betweenness centrality and identify gatekeeper nodes.
        
        Betweenness centrality measures how many shortest paths pass through
        each node, quantifying its importance as a connector in the network.
        Nodes exceeding the threshold percentile are classified as gatekeepers.
        
        Args:
            graph: Road network graph (should be connected for meaningful
                  centrality values)
            threshold_percentile: Percentile cutoff for gatekeeper
                                 identification (0-100). Nodes with centrality
                                 >= this percentile are marked as gatekeepers.
                                 Default: 95.0 (top 5% of nodes)
            
        Returns:
            CentralityResult containing:
            - node_centrality: dict mapping node_id to centrality score [0, 1]
            - gatekeeper_nodes: list of node IDs exceeding threshold
            - threshold_value: actual centrality value at threshold percentile
        
        Raises:
            ValueError: If threshold_percentile not in range [0, 100]
            RuntimeError: If graph is empty or has no edges
        """
        ...
    
    def compute_heatmap_data(
        self,
        graph: nx.Graph,
        centrality_scores: dict[int, float]
    ) -> list[tuple[float, float, float]]:
        """
        Generate spatial heatmap data for centrality visualization.
        
        Converts node centrality scores into geographic coordinates with
        intensity values for heatmap rendering in Folium or similar tools.
        
        Args:
            graph: Road network graph with geospatial node attributes
                  ('lat', 'lon' or 'x', 'y')
            centrality_scores: dict mapping node_id to normalized centrality
                              score [0, 1]
            
        Returns:
            List of (latitude, longitude, intensity) tuples suitable for
            folium.plugins.HeatMap. Intensity values are normalized
            centrality scores.
        
        Raises:
            ValueError: If graph nodes lack coordinate attributes
            KeyError: If centrality_scores missing node IDs present in graph
        """
        ...


# =============================================================================
# Protocol Interface 4: Disaster Simulator
# =============================================================================

class DisasterSimulator(Protocol):
    """
    Abstract interface for infrastructure failure simulation.
    
    This protocol defines the contract for disaster scenario simulation,
    allowing users to disable specific roads, bridges, or intersections and
    observe the impact on network connectivity and routing efficiency.
    
    The simulator computes metrics including travel delay, connected
    components, network efficiency, and a composite resilience index.
    """
    
    def simulate_failure(
        self,
        original_graph: nx.Graph,
        removed_nodes: list[int],
        removed_edges: list[tuple[int, int]]
    ) -> tuple[nx.Graph, SimulationMetrics]:
        """
        Simulate infrastructure failure and compute network impact metrics.
        
        Creates a modified graph with specified nodes and edges removed,
        then calculates how the failure affects routing, connectivity, and
        overall network resilience.
        
        Args:
            original_graph: Baseline road network before disaster. Should
                           include edge weights ('length') for path
                           computation.
            removed_nodes: List of node IDs to disable (simulates destroyed
                          intersections or blocked access points)
            removed_edges: List of edge tuples (u, v) to disable (simulates
                          destroyed road segments or bridges)
            
        Returns:
            Tuple of (modified_graph, metrics) where:
            - modified_graph: Copy of original with removed elements
            - metrics: SimulationMetrics dataclass with computed impact values
        
        Raises:
            ValueError: If removed_nodes or removed_edges reference non-
                       existent graph elements
        """
        ...
    
    def compute_resilience_index(
        self,
        original_graph: nx.Graph,
        modified_graph: nx.Graph,
        weights: dict[str, float]
    ) -> float:
        """
        Compute composite resilience index for network robustness assessment.
        
        The resilience index combines multiple network metrics into a single
        score representing overall network robustness after failure. Higher
        values indicate better resilience.
        
        Args:
            original_graph: Baseline road network before disaster
            modified_graph: Road network after infrastructure removal
            weights: Dict specifying importance of each metric component:
                    - 'connectivity': weight for connectivity preservation
                    - 'efficiency': weight for network efficiency retention
                    - 'gatekeeper_preservation': weight for gatekeeper survival
                    Weights should sum to 1.0
            
        Returns:
            Resilience index in range [0, 1], where:
            - 1.0 = perfect resilience (no impact)
            - 0.0 = complete network failure
            
        Raises:
            ValueError: If weights don't sum to 1.0 or contain invalid keys
        """
        ...


# =============================================================================
# Protocol Interface 5: Map Renderer
# =============================================================================

class MapRenderer(Protocol):
    """
    Abstract interface for spatial visualization of road networks.
    
    This protocol defines the contract for rendering interactive maps that
    display road graphs, satellite imagery, centrality heatmaps, and
    simulation overlays. Implementations typically use Folium/Leaflet or
    Plotly for web-based visualization.
    
    The renderer handles coordinate projection from image space to geographic
    coordinates (WGS84) for proper map alignment.
    """
    
    def render_map(
        self,
        graph: nx.Graph,
        satellite_image: Optional[np.ndarray],
        centrality_scores: Optional[dict[int, float]],
        gatekeeper_nodes: Optional[list[int]],
        config: RenderConfig
    ) -> str:
        """
        Generate interactive HTML map with road network visualization.
        
        Creates a multi-layer map displaying:
        1. Base tile layer (OpenStreetMap, satellite, etc.)
        2. Optional satellite image overlay
        3. Road graph with styled edges and nodes
        4. Optional centrality heatmap
        5. Optional gatekeeper node markers with emphasis
        
        Args:
            graph: Road network graph with geospatial node attributes
                  ('lat', 'lon'). Edges should have 'geometry' for accurate
                  rendering of curved roads.
            satellite_image: Optional satellite image array (H, W, C) to
                            overlay on map. Requires graph to have
                            geo_metadata for proper georeferencing.
            centrality_scores: Optional dict mapping node_id to centrality
                              value [0, 1] for heatmap visualization
            gatekeeper_nodes: Optional list of high-centrality node IDs to
                             highlight with special markers
            config: RenderConfig specifying colors, zoom, and style parameters
            
        Returns:
            HTML string containing complete interactive map. Can be embedded
            in Streamlit with components.html() or saved to file.
        
        Raises:
            ValueError: If graph nodes lack coordinate attributes
            RuntimeError: If satellite_image provided without geo_metadata
        """
        ...
    
    def add_simulation_overlay(
        self,
        map_html: str,
        disabled_elements: dict[str, list],
        affected_components: list[set[int]]
    ) -> str:
        """
        Add disaster simulation visualization to existing map.
        
        Modifies an existing map HTML to highlight disabled infrastructure
        and show affected network components after failure simulation.
        
        Args:
            map_html: HTML string from previous render_map() call
            disabled_elements: Dict with keys 'nodes' and 'edges', containing
                              lists of disabled node IDs and edge tuples
            affected_components: List of node sets, where each set represents
                                a connected component in the modified network.
                                Components are colored distinctly to show
                                network fragmentation.
            
        Returns:
            Updated HTML string with simulation overlay added
        
        Raises:
            ValueError: If map_html is not valid HTML or missing map div
        """
        ...
