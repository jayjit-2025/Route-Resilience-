"""Configuration management using Pydantic for type-safe settings.

This module provides a simplified configuration system for the 4-hour MVP.
Most values are hardcoded with sensible defaults, only essentials are configurable.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Model-related configuration (paths and architecture)."""
    
    segmentation_weights: str = Field(
        default="models/deeplabv3_resnet50.pth",
        description="Path to pretrained segmentation model weights"
    )
    architecture: str = Field(
        default="deeplabv3+",
        description="Model architecture: 'deeplabv3+' or 'unet'"
    )
    
    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        """Validate architecture choice."""
        allowed = {"deeplabv3+", "unet"}
        if v.lower() not in allowed:
            raise ValueError(f"Architecture must be one of {allowed}, got '{v}'")
        return v.lower()


class PreprocessingConfig(BaseModel):
    """Image preprocessing configuration."""
    
    normalize_range: List[float] = Field(
        default=[0.0, 1.0],
        description="Normalization range for pixel values"
    )
    target_size: List[int] = Field(
        default=[512, 512],
        description="Target image size [height, width] for model input"
    )
    
    @field_validator("normalize_range")
    @classmethod
    def validate_normalize_range(cls, v: List[float]) -> List[float]:
        """Validate normalization range has exactly 2 values."""
        if len(v) != 2:
            raise ValueError(f"normalize_range must have 2 values [min, max], got {len(v)}")
        if v[0] >= v[1]:
            raise ValueError(f"normalize_range min ({v[0]}) must be < max ({v[1]})")
        return v
    
    @field_validator("target_size")
    @classmethod
    def validate_target_size(cls, v: List[int]) -> List[int]:
        """Validate target size has exactly 2 positive values."""
        if len(v) != 2:
            raise ValueError(f"target_size must have 2 values [height, width], got {len(v)}")
        if any(s <= 0 for s in v):
            raise ValueError(f"target_size values must be positive, got {v}")
        return v


class GraphHealingConfig(BaseModel):
    """Graph healing algorithm configuration."""
    
    max_connection_distance_meters: float = Field(
        default=50.0,
        description="Maximum distance in meters for healing edges"
    )
    min_connectivity_ratio: float = Field(
        default=0.85,
        description="Minimum target connectivity ratio after healing"
    )
    
    @field_validator("max_connection_distance_meters")
    @classmethod
    def validate_distance(cls, v: float) -> float:
        """Validate distance is positive."""
        if v <= 0:
            raise ValueError(f"max_connection_distance_meters must be positive, got {v}")
        return v
    
    @field_validator("min_connectivity_ratio")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        """Validate ratio is in valid range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_connectivity_ratio must be in [0, 1], got {v}")
        return v


class CentralityConfig(BaseModel):
    """Centrality analysis configuration."""
    
    gatekeeper_threshold_percentile: float = Field(
        default=95.0,
        description="Percentile threshold for identifying gatekeeper nodes"
    )
    
    @field_validator("gatekeeper_threshold_percentile")
    @classmethod
    def validate_percentile(cls, v: float) -> float:
        """Validate percentile is in valid range."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"gatekeeper_threshold_percentile must be in [0, 100], got {v}")
        return v


class Settings(BaseModel):
    """Main configuration container for Route Resilience system.
    
    This is a simplified configuration for the 4-hour MVP.
    Advanced features (environment overrides, simulation weights) are omitted.
    """
    
    model_config = ConfigDict(
        extra="forbid",  # Raise error for unknown fields
        validate_assignment=True  # Validate on attribute assignment
    )
    
    models: ModelConfig = Field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    graph_healing: GraphHealingConfig = Field(default_factory=GraphHealingConfig)
    centrality: CentralityConfig = Field(default_factory=CentralityConfig)


def load_config(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from YAML file with fallback to defaults.
    
    Args:
        config_path: Optional path to YAML config file. If None, uses defaults.
        
    Returns:
        Settings object with validated configuration.
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist.
        ValueError: If configuration validation fails.
    """
    if config_path is None:
        # Use default configuration
        return Settings()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML and parse with Pydantic
    import yaml
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        # Empty YAML file, use defaults
        return Settings()
    
    return Settings(**config_dict)
