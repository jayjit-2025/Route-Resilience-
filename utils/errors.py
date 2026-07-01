"""Custom exceptions for the Route Resilience pipeline."""


class RouteResilienceError(Exception):
    """Base exception for all Route Resilience pipeline errors."""


class ImageLoadError(RouteResilienceError):
    """Raised when an image cannot be loaded (missing file, corrupt data, etc.)."""


class InvalidImageFormatError(ImageLoadError):
    """Raised when the image format is not supported by the pipeline."""


class ModelWeightsNotFoundError(RouteResilienceError):
    """Raised when model weight file cannot be found at the specified path."""
