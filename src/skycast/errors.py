"""Domain exceptions raised by SkyCast."""


class SkyCastError(Exception):
    """Base class for expected SkyCast failures."""


class ConfigError(SkyCastError):
    """Configuration is missing or invalid."""


class DataSourceError(SkyCastError):
    """Source data is unavailable or invalid."""


class StorageError(SkyCastError):
    """A dataset could not be read or written."""
