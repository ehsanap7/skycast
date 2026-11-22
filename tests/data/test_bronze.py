"""Tests for bronze loading and validation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from skycast.config import City
from skycast.data.bronze import load_bronze
from skycast.data.contracts import BRONZE_COLUMNS
from skycast.data.storage import LocalStore
from skycast.errors import DataSourceError

CITY = City(
    slug="sample",
    name="Sample",
    latitude=1.0,
    longitude=2.0,
    timezone="UTC",
    source_file="sample.csv",
    archive_source="archive-source",
    recent_source="recent-source",
    archive_cutoff=datetime(2010, 1, 1, 12, tzinfo=UTC),
)
KEY = "bronze/city=sample/sample.csv"
HEADER = ",".join(BRONZE_COLUMNS)
ROW = "2010-01-01 00:00:00+00:00,1.6,93,0.6,0.0,1013.7,94,2.8,215"
NEXT_ROW = "2010-01-01 01:00:00+00:00,1.3,93,0.3,0.0,1013.9,96,2.91,229"
VALID_CSV = f"{HEADER}\n{ROW}\n{NEXT_ROW}\n".encode()


def store_with(tmp_path: Path, content: bytes) -> LocalStore:
    """Create a store holding one bronze file."""
    store = LocalStore(tmp_path)
    store.write_bytes(KEY, content)
    return store


def test_loads_valid_file_as_raw_text(tmp_path: Path) -> None:
    """Valid input keeps its rows and original text."""
    frame = load_bronze(store_with(tmp_path, VALID_CSV), CITY)

    assert list(frame.columns) == list(BRONZE_COLUMNS)
    assert len(frame) == 2
    assert frame.loc[1, "wind_speed_10m"] == "2.91"
    assert frame.loc[0, "time"] == "2010-01-01 00:00:00+00:00"


def test_source_file_is_unchanged(tmp_path: Path) -> None:
    """Loading leaves the source bytes untouched."""
    store = store_with(tmp_path, VALID_CSV)

    load_bronze(store, CITY)

    assert store.read_bytes(KEY) == VALID_CSV


def test_blank_measurement_is_left_for_silver(tmp_path: Path) -> None:
    """Bronze does not judge measurement values."""
    content = f"{HEADER}\n{ROW.replace('1013.7', '')}\n".encode()

    frame = load_bronze(store_with(tmp_path, content), CITY)

    assert frame.loc[0, "pressure_msl"] == ""


def test_missing_file_raises(tmp_path: Path) -> None:
    """A missing source is a data source error."""
    with pytest.raises(DataSourceError, match="not found"):
        load_bronze(LocalStore(tmp_path), CITY)


def test_empty_file_raises(tmp_path: Path) -> None:
    """An empty file cannot be parsed."""
    with pytest.raises(DataSourceError, match="Cannot parse"):
        load_bronze(store_with(tmp_path, b""), CITY)


def test_header_only_file_raises(tmp_path: Path) -> None:
    """A file without rows is rejected."""
    with pytest.raises(DataSourceError, match="no rows"):
        load_bronze(store_with(tmp_path, f"{HEADER}\n".encode()), CITY)


def test_missing_column_raises(tmp_path: Path) -> None:
    """Missing contract columns are named in the error."""
    content = VALID_CSV.replace(b"pressure_msl", b"pressure")

    with pytest.raises(DataSourceError, match="pressure_msl"):
        load_bronze(store_with(tmp_path, content), CITY)


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-a-date",
        "",
        "2010-13-01 00:00:00+00:00",
        "2010-01-01 00:00:00",
        "2010-01-01 00:00:00-05:00",
    ],
)
def test_invalid_timestamp_raises(tmp_path: Path, timestamp: str) -> None:
    """Timestamps must be valid and explicitly UTC."""
    row = ROW.replace("2010-01-01 00:00:00+00:00", timestamp)
    content = f"{HEADER}\n{row}\n{NEXT_ROW}\n".encode()

    with pytest.raises(DataSourceError, match="1 invalid timestamps"):
        load_bronze(store_with(tmp_path, content), CITY)
