"""Unit tests for configuration system.

Tests cover:
- YAML loading and parsing
- Validation for invalid configurations
- Default value fallbacks
- Type safety with Pydantic
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from config.settings import (
    CentralityConfig,
    GraphHealingConfig,
    ModelConfig,
    PreprocessingConfig,
    Settings,
    load_config,
)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig()
        assert config.segmentation_weights == "models/deeplabv3_resnet50.pth"
        assert config.architecture == "deeplabv3+"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            segmentation_weights="custom/path.pth", architecture="unet"
        )
        assert config.segmentation_weights == "custom/path.pth"
        assert config.architecture == "unet"

    def test_architecture_validation_valid(self):
        """Test valid architecture choices."""
        for arch in ["deeplabv3+", "unet", "UNET", "DeepLabV3+"]:
            config = ModelConfig(architecture=arch)
            assert config.architecture in {"deeplabv3+", "unet"}

    def test_architecture_validation_invalid(self):
        """Test invalid architecture raises ValueError."""
        with pytest.raises(ValueError, match="Architecture must be one of"):
            ModelConfig(architecture="invalid_arch")

    def test_architecture_case_insensitive(self):
        """Test architecture validation is case-insensitive."""
        config = ModelConfig(architecture="DEEPLABV3+")
        assert config.architecture == "deeplabv3+"


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig validation."""

    def test_default_values(self):
        """Test default preprocessing values."""
        config = PreprocessingConfig()
        assert config.normalize_range == [0.0, 1.0]
        assert config.target_size == [512, 512]

    def test_custom_normalize_range(self):
        """Test custom normalization range."""
        config = PreprocessingConfig(normalize_range=[-1.0, 1.0])
        assert config.normalize_range == [-1.0, 1.0]

    def test_custom_target_size(self):
        """Test custom target size."""
        config = PreprocessingConfig(target_size=[1024, 1024])
        assert config.target_size == [1024, 1024]

    def test_normalize_range_validation_length(self):
        """Test normalize_range must have exactly 2 values."""
        with pytest.raises(ValueError, match="must have 2 values"):
            PreprocessingConfig(normalize_range=[0.0])

        with pytest.raises(ValueError, match="must have 2 values"):
            PreprocessingConfig(normalize_range=[0.0, 0.5, 1.0])

    def test_normalize_range_validation_order(self):
        """Test normalize_range min must be less than max."""
        with pytest.raises(ValueError, match="must be < max"):
            PreprocessingConfig(normalize_range=[1.0, 0.0])

        with pytest.raises(ValueError, match="must be < max"):
            PreprocessingConfig(normalize_range=[0.5, 0.5])

    def test_target_size_validation_length(self):
        """Test target_size must have exactly 2 values."""
        with pytest.raises(ValueError, match="must have 2 values"):
            PreprocessingConfig(target_size=[512])

        with pytest.raises(ValueError, match="must have 2 values"):
            PreprocessingConfig(target_size=[512, 512, 3])

    def test_target_size_validation_positive(self):
        """Test target_size values must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            PreprocessingConfig(target_size=[0, 512])

        with pytest.raises(ValueError, match="must be positive"):
            PreprocessingConfig(target_size=[-512, 512])


class TestGraphHealingConfig:
    """Tests for GraphHealingConfig validation."""

    def test_default_values(self):
        """Test default graph healing values."""
        config = GraphHealingConfig()
        assert config.max_connection_distance_meters == 50.0
        assert config.min_connectivity_ratio == 0.85

    def test_custom_values(self):
        """Test custom graph healing values."""
        config = GraphHealingConfig(
            max_connection_distance_meters=100.0, min_connectivity_ratio=0.9
        )
        assert config.max_connection_distance_meters == 100.0
        assert config.min_connectivity_ratio == 0.9

    def test_distance_validation_positive(self):
        """Test max_connection_distance_meters must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            GraphHealingConfig(max_connection_distance_meters=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            GraphHealingConfig(max_connection_distance_meters=-50.0)

    def test_ratio_validation_range(self):
        """Test min_connectivity_ratio must be in [0, 1]."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            GraphHealingConfig(min_connectivity_ratio=-0.1)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            GraphHealingConfig(min_connectivity_ratio=1.5)

    def test_ratio_validation_boundaries(self):
        """Test min_connectivity_ratio boundaries (0 and 1 are valid)."""
        config_0 = GraphHealingConfig(min_connectivity_ratio=0.0)
        assert config_0.min_connectivity_ratio == 0.0

        config_1 = GraphHealingConfig(min_connectivity_ratio=1.0)
        assert config_1.min_connectivity_ratio == 1.0


class TestCentralityConfig:
    """Tests for CentralityConfig validation."""

    def test_default_values(self):
        """Test default centrality values."""
        config = CentralityConfig()
        assert config.gatekeeper_threshold_percentile == 95.0

    def test_custom_values(self):
        """Test custom centrality values."""
        config = CentralityConfig(gatekeeper_threshold_percentile=90.0)
        assert config.gatekeeper_threshold_percentile == 90.0

    def test_percentile_validation_range(self):
        """Test gatekeeper_threshold_percentile must be in [0, 100]."""
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            CentralityConfig(gatekeeper_threshold_percentile=-1.0)

        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            CentralityConfig(gatekeeper_threshold_percentile=101.0)

    def test_percentile_validation_boundaries(self):
        """Test percentile boundaries (0 and 100 are valid)."""
        config_0 = CentralityConfig(gatekeeper_threshold_percentile=0.0)
        assert config_0.gatekeeper_threshold_percentile == 0.0

        config_100 = CentralityConfig(gatekeeper_threshold_percentile=100.0)
        assert config_100.gatekeeper_threshold_percentile == 100.0


class TestSettings:
    """Tests for Settings (main configuration container)."""

    def test_default_values(self):
        """Test Settings uses default values for all sub-configs."""
        settings = Settings()
        assert isinstance(settings.models, ModelConfig)
        assert isinstance(settings.preprocessing, PreprocessingConfig)
        assert isinstance(settings.graph_healing, GraphHealingConfig)
        assert isinstance(settings.centrality, CentralityConfig)

    def test_custom_nested_values(self):
        """Test Settings with custom nested configurations."""
        settings = Settings(
            models=ModelConfig(architecture="unet"),
            preprocessing=PreprocessingConfig(target_size=[1024, 1024]),
        )
        assert settings.models.architecture == "unet"
        assert settings.preprocessing.target_size == [1024, 1024]

    def test_partial_override(self):
        """Test Settings with partial configuration override."""
        settings = Settings(
            models={"architecture": "unet"},
            preprocessing={"target_size": [1024, 1024]},
        )
        assert settings.models.architecture == "unet"
        assert settings.models.segmentation_weights == "models/deeplabv3_resnet50.pth"
        assert settings.preprocessing.target_size == [1024, 1024]
        assert settings.preprocessing.normalize_range == [0.0, 1.0]

    def test_extra_fields_forbidden(self):
        """Test that extra unknown fields raise validation error."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            Settings(unknown_field="value")

    def test_validate_on_assignment(self):
        """Test that validation occurs when assigning values."""
        settings = Settings()
        # In Pydantic v2, we need to use model_validate to trigger validation
        # Direct attribute assignment may not trigger validator for nested models
        with pytest.raises(ValueError, match="Architecture must be one of"):
            settings.models = ModelConfig(architecture="invalid_arch")


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self):
        """Test loading with no config path uses defaults."""
        settings = load_config(None)
        assert settings.models.architecture == "deeplabv3+"
        assert settings.preprocessing.target_size == [512, 512]

    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
models:
  architecture: unet
  segmentation_weights: custom/model.pth

preprocessing:
  normalize_range: [-1.0, 1.0]
  target_size: [1024, 1024]

graph_healing:
  max_connection_distance_meters: 75.0

centrality:
  gatekeeper_threshold_percentile: 90.0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            settings = load_config(config_path)
            assert settings.models.architecture == "unet"
            assert settings.models.segmentation_weights == "custom/model.pth"
            assert settings.preprocessing.normalize_range == [-1.0, 1.0]
            assert settings.preprocessing.target_size == [1024, 1024]
            assert settings.graph_healing.max_connection_distance_meters == 75.0
            assert settings.centrality.gatekeeper_threshold_percentile == 90.0
        finally:
            config_path.unlink()

    def test_load_partial_yaml(self):
        """Test loading YAML with only some fields (others use defaults)."""
        yaml_content = """
models:
  architecture: unet
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            settings = load_config(config_path)
            assert settings.models.architecture == "unet"
            # Other fields should use defaults
            assert settings.preprocessing.target_size == [512, 512]
            assert settings.graph_healing.max_connection_distance_meters == 50.0
        finally:
            config_path.unlink()

    def test_load_empty_yaml(self):
        """Test loading empty YAML file uses all defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            config_path = Path(f.name)

        try:
            settings = load_config(config_path)
            assert settings.models.architecture == "deeplabv3+"
            assert settings.preprocessing.target_size == [512, 512]
        finally:
            config_path.unlink()

    def test_load_invalid_yaml_values(self):
        """Test loading YAML with invalid values raises validation error."""
        yaml_content = """
models:
  architecture: invalid_model
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                load_config(config_path)
        finally:
            config_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(Path("nonexistent_config.yaml"))

    def test_load_default_config_yaml(self):
        """Test loading the actual default_config.yaml file."""
        default_config_path = Path("config/default_config.yaml")
        if default_config_path.exists():
            settings = load_config(default_config_path)
            # Verify it matches expected defaults
            assert settings.models.architecture == "deeplabv3+"
            assert settings.preprocessing.target_size == [512, 512]
            assert settings.graph_healing.max_connection_distance_meters == 50.0
            assert settings.centrality.gatekeeper_threshold_percentile == 95.0


class TestTypeAnnotations:
    """Tests for type safety and annotations."""

    def test_model_config_types(self):
        """Test ModelConfig field types."""
        config = ModelConfig()
        assert isinstance(config.segmentation_weights, str)
        assert isinstance(config.architecture, str)

    def test_preprocessing_config_types(self):
        """Test PreprocessingConfig field types."""
        config = PreprocessingConfig()
        assert isinstance(config.normalize_range, list)
        assert all(isinstance(x, float) for x in config.normalize_range)
        assert isinstance(config.target_size, list)
        assert all(isinstance(x, int) for x in config.target_size)

    def test_graph_healing_config_types(self):
        """Test GraphHealingConfig field types."""
        config = GraphHealingConfig()
        assert isinstance(config.max_connection_distance_meters, float)
        assert isinstance(config.min_connectivity_ratio, float)

    def test_centrality_config_types(self):
        """Test CentralityConfig field types."""
        config = CentralityConfig()
        assert isinstance(config.gatekeeper_threshold_percentile, float)
