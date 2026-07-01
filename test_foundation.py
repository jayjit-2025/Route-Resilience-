# Quick smoke test for foundation components
from config.settings import Settings, load_config
from core.data_models import PipelineState, GeoMetadata
from utils.errors import RouteResilienceError, ImageLoadError

# Test config
cfg = Settings()
assert cfg.preprocessing.target_size == [512, 512]

# Test data models
state = PipelineState()
assert state.road_mask is None

print("✅ All foundation components verified!")
