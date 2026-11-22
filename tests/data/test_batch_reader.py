"""Tests for loading stored API batches."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from skycast.config import City
from skycast.data.api_response import parse_weather_response
from skycast.data.batch_reader import load_batches
from skycast.data.contracts import MEASUREMENT_RANGES, SourceKind
from skycast.data.storage import LocalStore
from skycast.errors import DataSourceError
from skycast.ingestion.batches import batch_key, manifest_key, save_batch

CITY = City(
    slug="sample",
    name="Sample",
    latitude=1.0,
    longitude=2.0,
    timezone="UTC",
    source_file="sample.csv",
    archive_source="archive-source",
    recent_source="recent-source",
    archive_cutoff=datetime(2024, 1, 1, tzinfo=UTC),
)
FETCHED = pd.Timestamp("2024-01-01 03:10", tz="UTC")
KEY = batch_key("sample", SourceKind.RECENT, FETCHED)


def api_raw(periods: int = 6) -> bytes:
    """Build a raw API response body."""
    times = pd.date_range("2024-01-01 00:00", periods=periods, freq="h")
    hourly: dict[str, list[object]] = {"time": [f"{time:%Y-%m-%dT%H:%M}" for time in times]}
    hourly |= {column: [5.0] * periods for column in MEASUREMENT_RANGES}
    payload = {
        "utc_offset_seconds": 0,
        "hourly_units": {"wind_speed_10m": "m/s"},
        "hourly": hourly,
    }
    return json.dumps(payload).encode()


def store_batch(root: Path) -> LocalStore:
    """Store one batch that accepts hours 01 to 03."""
    store = LocalStore(root)
    raw = api_raw()
    selected = parse_weather_response(raw).iloc[1:4]
    save_batch(store, CITY, SourceKind.RECENT, raw, selected, FETCHED)
    return store


def edit_manifest(store: LocalStore, **changes: object) -> None:
    """Change fields in the stored manifest."""
    manifest = json.loads(store.read_bytes(manifest_key(KEY)))
    store.write_bytes(manifest_key(KEY), json.dumps({**manifest, **changes}).encode())


def test_no_batches_is_empty(tmp_path: Path) -> None:
    """A city without batches loads nothing."""
    assert load_batches(LocalStore(tmp_path), CITY) == ()


def test_loads_only_accepted_hours(tmp_path: Path) -> None:
    """Hours outside the manifest never leave the raw file."""
    batches = load_batches(store_batch(tmp_path), CITY)

    assert len(batches) == 1
    assert batches[0].kind is SourceKind.RECENT
    assert batches[0].fetched_at == FETCHED
    assert batches[0].hourly["time"].dt.hour.tolist() == [1, 2, 3]


def test_response_without_manifest_is_ignored(tmp_path: Path) -> None:
    """A half-finished save does not count as a batch."""
    store = LocalStore(tmp_path)
    store.write_bytes(KEY, api_raw())

    assert load_batches(store, CITY) == ()


def test_damaged_response_raises_with_file_name(tmp_path: Path) -> None:
    """A changed response stops the build."""
    store = store_batch(tmp_path)
    store.write_bytes(KEY, api_raw(periods=5))

    with pytest.raises(DataSourceError, match=r"damaged.*2024-01-01T031000Z\.json"):
        load_batches(store, CITY)


def test_missing_response_raises(tmp_path: Path) -> None:
    """A manifest needs its response."""
    store = store_batch(tmp_path)
    (tmp_path / KEY).unlink()

    with pytest.raises(DataSourceError, match="response is missing"):
        load_batches(store, CITY)


def test_unparseable_manifest_raises(tmp_path: Path) -> None:
    """A broken manifest stops the build."""
    store = store_batch(tmp_path)
    store.write_bytes(manifest_key(KEY), b"not json")

    with pytest.raises(DataSourceError, match="Cannot parse batch manifest"):
        load_batches(store, CITY)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source": "other"}, "unknown source"),
        ({"rows": 99}, "do not match"),
        ({"first_hour": "bad"}, "invalid first_hour"),
        ({"last_hour": None}, "no valid last_hour"),
        ({"fetched_at": "2024-01-01T03:10:00-05:00"}, "must be UTC"),
    ],
)
def test_invalid_manifest_fields_raise(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    """Manifest fields must match the batch contract."""
    store = store_batch(tmp_path)
    edit_manifest(store, **changes)

    with pytest.raises(DataSourceError, match=message):
        load_batches(store, CITY)
