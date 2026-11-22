"""Domain exceptions raised by SkyCast."""


class SkyCastError(Exception):
    """Base class for expected SkyCast failures."""


class ConfigError(SkyCastError):
    """Configuration is missing or invalid."""


class DataSourceError(SkyCastError):
    """Source data is unavailable or invalid."""


class TemporarySourceError(DataSourceError):
    """A source failed in a way that may succeed on retry."""


class StorageError(SkyCastError):
    """A dataset could not be read or written."""
