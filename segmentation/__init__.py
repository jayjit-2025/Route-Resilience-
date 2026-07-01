# Segmentation package - Road extraction models

from segmentation.base import BaseSegmentationModel
from segmentation.model_registry import register_model, get_model, list_models

# Import model implementations so their @register_model decorators fire and
# both architectures are available via get_model() without explicit imports.
from segmentation.deeplabv3_model import DeepLabV3PlusModel
from segmentation.unet_model import UNetModel

__all__ = [
    "BaseSegmentationModel",
    "register_model",
    "get_model",
    "list_models",
    "DeepLabV3PlusModel",
    "UNetModel",
]
