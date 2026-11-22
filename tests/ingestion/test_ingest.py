"""Tests for saving batches and recovering from crashes."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from skycast.config import City
from skycast.data.storage import LocalStore
from skycast.errors import StorageError
from skycast.ingestion.batches import batch_key, manifest_key, save_batch
from skycast.ingestion.checkpoint import SourceKind, checkpoint_key, read_checkpoint
from skycast.ingestion.client import WeatherResponse
from skycast.ingestion.ingest import ingest_response

NOW = pd.Timestamp("2024-01-01 06:10", tz="UTC")
LATER = NOW + pd.Timedelta(minutes=5)
RAW = b'{"exact": "api response"}'
HourlyFactory = Callable[[str, int], pd.DataFrame]


class FailingStore(LocalStore):
    """A store that fails when writing matching keys."""

    def __init__(self, root: Path, suffix: str) -> None:
        """Create a store that fails for a key suffix."""
        super().__init__(root)
        self._suffix = suffix

    def write_bytes(self, key: str, content: bytes) -> None:
        """Fail for matching keys and write the others."""
        if key.endswith(self._suffix):
            raise StorageError(f"Simulated failure for {key}")
        super().write_bytes(key, content)

    def create_bytes(self, key: str, content: bytes) -> None:
        """Fail for matching keys and create the others."""
        if key.endswith(self._suffix):
            raise StorageError(f"Simulated failure for {key}")
        super().create_bytes(key, content)


@pytest.fixture
def response(make_hourly: HourlyFactory) -> WeatherResponse:
    """Build a response with old, new, and future hours."""
    return WeatherResponse(raw=RAW, hourly=make_hourly("2024-01-01 00:00", 12))


def test_records_new_hours_and_moves_checkpoint(
    seeded_root: Path, city: City, response: WeatherResponse
) -> None:
    """A normal run saves a batch and advances its source."""
    store = LocalStore(seeded_root)

    result = ingest_response(store, city, SourceKind.RECENT, response, NOW)

    checkpoint = read_checkpoint(store, city)
    assert result.rows == 4
    assert result.last_hour == pd.Timestamp("2024-01-01 06:00", tz="UTC")
    assert checkpoint.recent == result.last_hour
    assert checkpoint.archive == pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_batch_keeps_exact_response(
    seeded_root: Path, city: City, response: WeatherResponse
) -> None:
    """The raw batch has the API's exact bytes."""
    store = LocalStore(seeded_root)

    result = ingest_response(store, city, SourceKind.RECENT, response, NOW)

    assert result.batch_key == "bronze/city=sample/batches/source=recent/2024-01-01T061000Z.json"
    assert store.read_bytes(result.batch_key) == RAW


def test_manifest_excludes_future_hours(
    seeded_root: Path, city: City, response: WeatherResponse
) -> None:
    """The receipt records only accepted hours."""
    store = LocalStore(seeded_root)

    result = ingest_response(store, city, SourceKind.RECENT, response, NOW)

    manifest = json.loads(store.read_bytes(manifest_key(result.batch_key)))
    assert manifest["source"] == "recent"
    assert manifest["first_hour"] == "2024-01-01T03:00:00+00:00"
    assert manifest["last_hour"] == "2024-01-01T06:00:00+00:00"
    assert manifest["rows"] == 4
    assert manifest["sha256"] == hashlib.sha256(RAW).hexdigest()


def test_rerun_adds_nothing(seeded_root: Path, city: City, response: WeatherResponse) -> None:
    """A second run with the same data writes no batch."""
    store = LocalStore(seeded_root)
    ingest_response(store, city, SourceKind.RECENT, response, NOW)
    keys = store.list_keys()

    result = ingest_response(store, city, SourceKind.RECENT, response, LATER)

    assert result.rows == 0
    assert result.batch_key is None
    assert store.list_keys() == keys


def test_no_new_hours_writes_nothing(
    seeded_root: Path, city: City, make_hourly: HourlyFactory
) -> None:
    """An empty result leaves storage unchanged."""
    store = LocalStore(seeded_root)
    keys = store.list_keys()
    stale = WeatherResponse(raw=RAW, hourly=make_hourly("2024-01-01 00:00", 3))

    result = ingest_response(store, city, SourceKind.RECENT, stale, NOW)

    assert result.rows == 0
    assert store.list_keys() == keys


def test_sources_advance_independently(
    seeded_root: Path, city: City, response: WeatherResponse
) -> None:
    """Archive and recent batches keep separate positions."""
    store = LocalStore(seeded_root)

    archive = ingest_response(store, city, SourceKind.ARCHIVE, response, NOW)
    recent = ingest_response(store, city, SourceKind.RECENT, response, LATER)

    assert archive.rows == 6
    assert recent.rows == 4
    assert "source=archive" in archive.batch_key
    assert "source=recent" in recent.batch_key


def test_existing_batch_is_never_overwritten(
    tmp_path: Path, city: City, make_hourly: HourlyFactory
) -> None:
    """A batch key can only be written once."""
    store = LocalStore(tmp_path)
    selected = make_hourly("2024-01-01 03:00", 2)
    key = save_batch(store, city, SourceKind.RECENT, RAW, selected, NOW)

    with pytest.raises(StorageError, match="already exists"):
        save_batch(store, city, SourceKind.RECENT, b"changed", selected, NOW)

    assert store.read_bytes(key) == RAW


def test_orphan_response_blocks_same_key(
    tmp_path: Path, city: City, make_hourly: HourlyFactory
) -> None:
    """A response without a manifest is not replaced."""
    store = LocalStore(tmp_path)
    key = batch_key(city.slug, SourceKind.RECENT, NOW)
    store.write_bytes(key, RAW)

    with pytest.raises(StorageError, match="already exists"):
        save_batch(store, city, SourceKind.RECENT, b"new", make_hourly("2024-01-01 03:00", 1), NOW)

    assert store.read_bytes(key) == RAW


@pytest.mark.parametrize(
    ("failing_suffix", "response_saved", "manifest_saved"),
    [
        ("Z.json", False, False),
        (".manifest.json", True, False),
        ("ingestion.json", True, True),
    ],
)
def test_crash_keeps_checkpoint_and_recovers(
    seeded_root: Path,
    city: City,
    response: WeatherResponse,
    failing_suffix: str,
    response_saved: bool,
    manifest_saved: bool,
) -> None:
    """A crash at any write leaves a state the next run can repair."""
    failing = FailingStore(seeded_root, failing_suffix)
    store = LocalStore(seeded_root)
    key = batch_key(city.slug, SourceKind.RECENT, NOW)

    with pytest.raises(StorageError, match="Simulated"):
        ingest_response(failing, city, SourceKind.RECENT, response, NOW)

    assert store.exists(key) is response_saved
    assert store.exists(manifest_key(key)) is manifest_saved
    assert not store.exists(checkpoint_key(city.slug))

    result = ingest_response(store, city, SourceKind.RECENT, response, LATER)

    assert result.rows == 4
    assert read_checkpoint(store, city).recent == pd.Timestamp("2024-01-01 06:00", tz="UTC")
