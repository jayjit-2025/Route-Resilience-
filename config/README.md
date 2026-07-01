# Configuration System

The Route Resilience configuration system provides a type-safe, validated configuration management using Pydantic.

## Quick Start

### Using Default Configuration

```python
from config import Settings

# Use all default values
settings = Settings()
print(settings.models.architecture)  # "deeplabv3+"
print(settings.preprocessing.target_size)  # [512, 512]
```

### Loading from YAML File

```python
from pathlib import Path
from config import load_config

# Load from YAML file
settings = load_config(Path("config/default_config.yaml"))

# Load with None uses defaults
settings = load_config(None)
```

### Custom Configuration

```python
from config import Settings, ModelConfig, PreprocessingConfig

# Override specific values
settings = Settings(
    models=ModelConfig(architecture="unet"),
    preprocessing=PreprocessingConfig(target_size=[1024, 1024])
)
```

## Configuration Structure

### ModelConfig
- `segmentation_weights` (str): Path to pretrained model weights
- `architecture` (str): Model architecture ("deeplabv3+" or "unet")

### PreprocessingConfig
- `normalize_range` (List[float]): Pixel normalization range [min, max]
- `target_size` (List[int]): Target image size [height, width]

### GraphHealingConfig
- `max_connection_distance_meters` (float): Maximum distance for healing edges
- `min_connectivity_ratio` (float): Target connectivity ratio (0.0-1.0)

### CentralityConfig
- `gatekeeper_threshold_percentile` (float): Percentile for critical nodes (0.0-100.0)

## YAML Configuration Format

```yaml
models:
  architecture: "deeplabv3+"
  segmentation_weights: "models/deeplabv3_resnet50.pth"

preprocessing:
  normalize_range: [0.0, 1.0]
  target_size: [512, 512]

graph_healing:
  max_connection_distance_meters: 50.0
  min_connectivity_ratio: 0.85

centrality:
  gatekeeper_threshold_percentile: 95.0
```

## Validation

The configuration system automatically validates:
- **Architecture**: Must be "deeplabv3+" or "unet" (case-insensitive)
- **Normalize range**: Must have exactly 2 values with min < max
- **Target size**: Must have exactly 2 positive integers
- **Distance**: Must be positive
- **Ratios**: Must be in [0.0, 1.0]
- **Percentiles**: Must be in [0.0, 100.0]
- **Unknown fields**: Raises error for typos or invalid keys

### Example Validation Errors

```python
# This will raise ValueError
ModelConfig(architecture="invalid")  # Unknown architecture

PreprocessingConfig(normalize_range=[1.0, 0.0])  # min >= max

GraphHealingConfig(min_connectivity_ratio=1.5)  # Out of range
```

## Testing

Run the test suite:

```bash
python -m pytest config/test_settings.py -v
```

All 37 tests cover:
- Default value initialization
- Custom value assignment
- Validation for all config parameters
- YAML loading and parsing
- Type safety
- Error handling

## Design Principles

1. **Type Safety**: All fields have proper type hints
2. **Validation**: Automatic validation on initialization and assignment
3. **Defaults**: Sensible defaults for all parameters
4. **Simplicity**: MVP-focused, no complex features
5. **Flexibility**: Easy to extend with new configuration sections
