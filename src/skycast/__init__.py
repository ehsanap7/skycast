"""SkyCast: an end-to-end MLOps weather forecasting platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("skycast")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
