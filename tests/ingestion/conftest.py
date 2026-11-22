"""Shared fixtures for ingestion tests."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from skycast.config import City
from skycast.data.contracts import BRONZE_COLUMNS, MEASUREMENT_RANGES
from skycast.data.storage import LocalStore

SEED_ROWS = (
    "2024-01-01 00:00:00+00:00,1.6,93,0.6,0.0,1013.7,94,2.8,215",
    "2024-01-01 01:00:00+00:00,1.3,93,0.3,0.0,1013.9,96,2.91,229",
    "2024-01-01 02:00:00+00:00,1.1,92,0.1,0.0,1014.0,97,2.7,230",
)


def _hourly_frame(start: str, periods: int) -> pd.DataFrame:
    """Build complete hourly API rows."""
    times = pd.date_range(start, periods=periods, freq="h", tz="UTC", name="time")
    values = {column: 1.0 for column in MEASUREMENT_RANGES}
    return pd.DataFrame({"time": times, **values})[list(BRONZE_COLUMNS)]


@pytest.fixture
def make_hourly() -> Callable[[str, int], pd.DataFrame]:
    """Provide an hourly frame factory."""
    return _hourly_frame


@pytest.fixture
def city() -> City:
    """Provide a sample city whose archive ends at the first seed hour."""
    return City(
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


@pytest.fixture
def seeded_root(tmp_path: Path) -> Path:
    """Create a storage root containing a seed CSV."""
    content = "\n".join((",".join(BRONZE_COLUMNS), *SEED_ROWS)) + "\n"
    LocalStore(tmp_path).write_bytes("bronze/city=sample/sample.csv", content.encode())
    return tmp_path
