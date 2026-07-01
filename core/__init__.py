"""Core interfaces and data models for the Route Resilience system.

Exports all Protocol interfaces and shared data model types so downstream
modules can import from a single, stable location:

    from core import RoadSegmentationModel, CentralityResult, ...
"""

# Protocol interfaces
from core.interfaces import (
    RoadSegmentationModel,
    RoadHealingAlgorithm,
    CentralityAnalyzer,
    DisasterSimulator,
    MapRenderer,
)

# Data models used as interface return / parameter types
from core.data_models import (
    GeoMetadata,
    PipelineState,
    CentralityResult,
    SimulationMetrics,
    RenderConfig,
)

__all__ = [
    # Protocols
    "RoadSegmentationModel",
    "RoadHealingAlgorithm",
    "CentralityAnalyzer",
    "DisasterSimulator",
    "MapRenderer",
    # Data models
    "GeoMetadata",
    "PipelineState",
    "CentralityResult",
    "SimulationMetrics",
    "RenderConfig",
]
