"""Storage interface and local filesystem implementation."""

from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from skycast.config import StorageSettings
from skycast.errors import ConfigError, StorageError


class Layer(StrEnum):
    """Data lake layers."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class DatasetStore(Protocol):
    """Reads and writes datasets by storage key."""

    def read_bytes(self, key: str) -> bytes:
        """Return the content stored at a key."""
        ...

    def write_bytes(self, key: str, content: bytes) -> None:
        """Store content at a key."""
        ...

    def exists(self, key: str) -> bool:
        """Tell whether a key holds a dataset."""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List stored keys under a prefix."""
        ...


class LocalStore:
    """Stores datasets on the local filesystem."""

    def __init__(self, root: Path) -> None:
        """Create a store rooted at a directory."""
        self._root = root.resolve()

    def read_bytes(self, key: str) -> bytes:
        """Return the content stored at a key."""
        try:
            return self._path_for(key).read_bytes()
        except OSError as err:
            raise StorageError(f"Cannot read dataset {key!r}: {err}") from err

    def write_bytes(self, key: str, content: bytes) -> None:
        """Store content at a key using an atomic replace."""
        path = self._path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _replace_atomically(path, content)
        except OSError as err:
            raise StorageError(f"Cannot write dataset {key!r}: {err}") from err

    def exists(self, key: str) -> bool:
        """Tell whether a key holds a dataset."""
        return self._path_for(key).is_file()

    def list_keys(self, prefix: str = "") -> list[str]:
        """List stored keys under a prefix, skipping hidden files."""
        base = self._path_for(prefix)
        try:
            files = [path for path in base.rglob("*") if path.is_file()]
        except OSError as err:
            raise StorageError(f"Cannot list datasets under {prefix!r}: {err}") from err
        visible = (path for path in files if not path.name.startswith("."))
        return sorted(path.relative_to(self._root).as_posix() for path in visible)

    def _path_for(self, key: str) -> Path:
        """Resolve a key and block paths outside the root."""
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise StorageError(f"Storage key escapes the root: {key!r}")
        return path


def _replace_atomically(path: Path, content: bytes) -> None:
    """Write a temporary file, then swap it into place."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def dataset_key(layer: Layer, city_slug: str, filename: str) -> str:
    """Build the storage key for a city's dataset."""
    return f"{layer}/city={city_slug}/{filename}"


def build_store(settings: StorageSettings) -> DatasetStore:
    """Create the configured storage backend."""
    if settings.backend == "local":
        return LocalStore(settings.root)
    raise ConfigError(f"Unsupported storage backend: {settings.backend!r}")
