"""
Model registry (factory) for road segmentation architectures.

Models are registered via the :func:`register_model` decorator and
instantiated on demand via :func:`get_model`.

Example usage::

    from segmentation.model_registry import register_model, get_model

    @register_model("my_model")
    class MyModel(BaseSegmentationModel):
        ...

    model = get_model("my_model")   # returns MyModel()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from segmentation.base import BaseSegmentationModel

# Internal registry mapping architecture name → concrete class
_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """
    Class decorator that registers a segmentation model under *name*.

    Args:
        name: Unique identifier for the architecture (e.g. ``"deeplabv3+"``).

    Returns:
        Decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If *name* is already registered by a different class.
    """

    def decorator(cls: type) -> type:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"Architecture name '{name}' is already registered by "
                f"'{_REGISTRY[name].__name__}'. "
                f"Use a different name for '{cls.__name__}'."
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_model(architecture: str) -> "BaseSegmentationModel":
    """
    Instantiate and return a registered segmentation model.

    Args:
        architecture: Registered architecture name (e.g. ``"deeplabv3+"``).

    Returns:
        A freshly constructed instance of the requested model class.

    Raises:
        ValueError: If *architecture* is not in the registry.
    """
    if architecture not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {available}"
        )
    return _REGISTRY[architecture]()


def list_models() -> list[str]:
    """Return the names of all currently registered architectures."""
    return list(_REGISTRY.keys())
