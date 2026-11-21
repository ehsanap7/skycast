"""Tests for dataset storage."""

from pathlib import Path

import pytest

from skycast.config import StorageSettings
from skycast.data.storage import Layer, LocalStore, build_store, dataset_key
from skycast.errors import ConfigError, StorageError


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    """Stored content can be read back."""
    store = LocalStore(tmp_path)

    store.write_bytes("silver/city=a/file.parquet", b"content")

    assert store.exists("silver/city=a/file.parquet")
    assert store.read_bytes("silver/city=a/file.parquet") == b"content"


def test_write_replaces_content_and_leaves_no_temp_file(tmp_path: Path) -> None:
    """Rewriting a key leaves only the final file."""
    store = LocalStore(tmp_path)

    store.write_bytes("gold/file.parquet", b"old")
    store.write_bytes("gold/file.parquet", b"new")

    assert store.read_bytes("gold/file.parquet") == b"new"
    assert [path.name for path in (tmp_path / "gold").iterdir()] == ["file.parquet"]


def test_failed_write_keeps_existing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed replace keeps the old file and removes the temp file."""
    store = LocalStore(tmp_path)
    store.write_bytes("gold/file.parquet", b"old")

    def fail_replace(self: Path, target: Path) -> None:
        """Simulate a filesystem failure."""
        raise OSError("disk failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(StorageError, match="Cannot write"):
        store.write_bytes("gold/file.parquet", b"new")
    assert store.read_bytes("gold/file.parquet") == b"old"
    assert [path.name for path in (tmp_path / "gold").iterdir()] == ["file.parquet"]


def test_missing_read_raises_storage_error(tmp_path: Path) -> None:
    """Reading a missing key is a storage error."""
    with pytest.raises(StorageError, match="Cannot read"):
        LocalStore(tmp_path).read_bytes("missing.csv")


@pytest.mark.parametrize("key", ["../outside.csv", "/etc/hosts", "a/../../outside.csv"])
def test_key_cannot_escape_root(tmp_path: Path, key: str) -> None:
    """Keys outside the root are rejected."""
    with pytest.raises(StorageError, match="escapes"):
        LocalStore(tmp_path / "root").exists(key)


def test_list_keys_is_sorted_and_skips_hidden_files(tmp_path: Path) -> None:
    """Listing returns visible keys under a prefix."""
    store = LocalStore(tmp_path)
    store.write_bytes("bronze/b.csv", b"b")
    store.write_bytes("bronze/a.csv", b"a")
    store.write_bytes("bronze/.DS_Store", b"hidden")
    store.write_bytes("silver/c.parquet", b"c")

    assert store.list_keys("bronze") == ["bronze/a.csv", "bronze/b.csv"]


def test_list_keys_for_missing_prefix_is_empty(tmp_path: Path) -> None:
    """A missing prefix has no keys."""
    assert LocalStore(tmp_path).list_keys("missing") == []


def test_dataset_key_uses_partition_layout() -> None:
    """Keys follow the layer and city layout."""
    assert dataset_key(Layer.GOLD, "sample", "x.parquet") == "gold/city=sample/x.parquet"


def test_build_store_creates_local_backend(tmp_path: Path) -> None:
    """The factory builds the local backend."""
    store = build_store(StorageSettings(backend="local", root=tmp_path))

    assert isinstance(store, LocalStore)


def test_build_store_rejects_unknown_backend(tmp_path: Path) -> None:
    """Unknown backends are config errors."""
    with pytest.raises(ConfigError, match="Unsupported"):
        build_store(StorageSettings(backend="unknown", root=tmp_path))
