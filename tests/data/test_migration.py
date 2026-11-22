"""Tests for create-only storage and data migration."""

from pathlib import Path

import pytest

from skycast.cli import main
from skycast.data.migration import migrate_store
from skycast.data.storage import LocalStore
from skycast.errors import StorageError

DATA = {
    "bronze/city=a/source.csv": b"raw",
    "bronze/city=a/batches/source=recent/b.json": b"{}",
    "silver/city=a/hourly.parquet": b"silver",
    "gold/city=a/training.parquet": b"gold",
    "state/city=a/ingestion.json": b"state",
}
CONFIG = """cities:
  - slug: a
    name: A
    latitude: 1.0
    longitude: 2.0
    timezone: UTC
    source_file: source.csv
    archive_source: archive-source
    recent_source: recent-source
    archive_cutoff: "2024-01-01T00:00:00+00:00"
"""
SETTINGS = """active_city: a
storage:
  backend: local
  root: data
  bucket: sample-bucket
forecast:
  horizon_days: 2
gold:
  max_fill_hours: 3
  lag_days: [1]
  rolling_days: [2]
ingestion:
  archive_url: https://archive.example.test/v1/archive
  archive_model: sample-model
  recent_url: https://recent.example.test/v1/forecast
  recent_forecast_days: 1
  max_past_days: 92
  timeout_seconds: 5
schedule:
  recent_cron: "10 */4 * * *"
  archive_cron: "40 6 * * *"
  retries: 3
  retry_base_seconds: 0
"""


class BucketStore(LocalStore):
    """A local target that behaves like a bucketed store."""

    def ensure_bucket(self) -> None:
        """Accept the bucket check."""


class CorruptingStore(LocalStore):
    """A target that damages files as they are created."""

    def create_bytes(self, key: str, content: bytes) -> None:
        """Store changed content."""
        super().create_bytes(key, content + b"damaged")


def filled_store(root: Path) -> LocalStore:
    """Create a store holding the sample datasets."""
    store = LocalStore(root)
    for key, content in DATA.items():
        store.write_bytes(key, content)
    return store


def test_local_create_stores_new_key(tmp_path: Path) -> None:
    """A new key is stored with no temporary file left."""
    store = LocalStore(tmp_path)

    store.create_bytes("bronze/a.json", b"content")

    assert store.read_bytes("bronze/a.json") == b"content"
    assert [path.name for path in (tmp_path / "bronze").iterdir()] == ["a.json"]


def test_local_create_refuses_existing_key(tmp_path: Path) -> None:
    """A create-only write keeps the original."""
    store = LocalStore(tmp_path)
    store.create_bytes("bronze/a.json", b"original")

    with pytest.raises(StorageError, match="already exists"):
        store.create_bytes("bronze/a.json", b"changed")

    assert store.read_bytes("bronze/a.json") == b"original"
    assert [path.name for path in (tmp_path / "bronze").iterdir()] == ["a.json"]


def test_migration_copies_all_layers(tmp_path: Path) -> None:
    """Every dataset reaches the target unchanged."""
    source = filled_store(tmp_path / "source")
    target = LocalStore(tmp_path / "target")

    summary = migrate_store(source, target)

    assert summary.files == len(DATA)
    assert summary.copied == len(DATA)
    assert summary.bytes_verified == sum(len(content) for content in DATA.values())
    assert all(target.read_bytes(key) == content for key, content in DATA.items())


def test_migration_ignores_files_outside_layers(tmp_path: Path) -> None:
    """Project files in the data folder are not copied."""
    source = filled_store(tmp_path / "source")
    source.write_bytes("README.md", b"docs")
    target = LocalStore(tmp_path / "target")

    migrate_store(source, target)

    assert not target.exists("README.md")


def test_migration_never_changes_source(tmp_path: Path) -> None:
    """The source keeps every file."""
    source = filled_store(tmp_path / "source")
    keys = source.list_keys()

    migrate_store(source, LocalStore(tmp_path / "target"))

    assert source.list_keys() == keys
    assert all(source.read_bytes(key) == content for key, content in DATA.items())


def test_rerun_copies_nothing(tmp_path: Path) -> None:
    """A repeated migration verifies existing files."""
    source = filled_store(tmp_path / "source")
    target = LocalStore(tmp_path / "target")
    migrate_store(source, target)

    summary = migrate_store(source, target)

    assert summary.copied == 0
    assert summary.already_present == len(DATA)


def test_partial_copy_can_resume(tmp_path: Path) -> None:
    """A rerun copies only the missing files."""
    source = filled_store(tmp_path / "source")
    target = LocalStore(tmp_path / "target")
    target.write_bytes("gold/city=a/training.parquet", b"gold")

    summary = migrate_store(source, target)

    assert summary.copied == len(DATA) - 1
    assert summary.already_present == 1


def test_conflicting_file_is_not_overwritten(tmp_path: Path) -> None:
    """A different target file stops the migration."""
    source = filled_store(tmp_path / "source")
    target = LocalStore(tmp_path / "target")
    target.write_bytes("gold/city=a/training.parquet", b"different")

    with pytest.raises(StorageError, match="differs from source"):
        migrate_store(source, target)

    assert target.read_bytes("gold/city=a/training.parquet") == b"different"


def test_damaged_copy_is_detected(tmp_path: Path) -> None:
    """Each copied file is checked by hash."""
    source = filled_store(tmp_path / "source")

    with pytest.raises(StorageError, match="differs from source"):
        migrate_store(source, CorruptingStore(tmp_path / "target"))


def test_empty_source_raises(tmp_path: Path) -> None:
    """An empty migration is not reported as success."""
    with pytest.raises(StorageError, match="No datasets"):
        migrate_store(LocalStore(tmp_path / "source"), LocalStore(tmp_path / "target"))


def test_migrate_command_reports_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command copies local data to the bucket."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cities.yaml").write_text(CONFIG, encoding="utf-8")
    (config_dir / "data.yaml").write_text(SETTINGS, encoding="utf-8")
    filled_store(tmp_path / "data")
    target = BucketStore(tmp_path / "bucket")
    monkeypatch.setattr("skycast.data.migration.create_s3_store", lambda _bucket: target)

    exit_code = main(["data", "migrate", "--config-dir", str(config_dir)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"Files verified: {len(DATA)}" in output
    assert f"Copied:         {len(DATA)}" in output
    assert target.read_bytes("gold/city=a/training.parquet") == b"gold"


def test_migrate_command_reports_missing_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing storage settings give a clean error."""
    for name in ("SKYCAST_S3_ENDPOINT", "SKYCAST_S3_ACCESS_KEY", "SKYCAST_S3_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cities.yaml").write_text(CONFIG, encoding="utf-8")
    (config_dir / "data.yaml").write_text(SETTINGS, encoding="utf-8")

    exit_code = main(["data", "migrate", "--config-dir", str(config_dir)])

    assert exit_code == 1
    assert "Missing environment variable" in capsys.readouterr().err
