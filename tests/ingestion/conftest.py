"""Shared fixtures for ingestion tests."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fake_weather_api import FakeApi

from skycast.config import City
from skycast.data.contracts import BRONZE_COLUMNS, MEASUREMENT_RANGES
from skycast.data.storage import LocalStore

SEED_ROWS = (
    "2024-01-01 00:00:00+00:00,1.6,93,0.6,0.0,1013.7,94,2.8,215",
    "2024-01-01 01:00:00+00:00,1.3,93,0.3,0.0,1013.9,96,2.91,229",
    "2024-01-01 02:00:00+00:00,1.1,92,0.1,0.0,1014.0,97,2.7,230",
)


PROJECT_NOW = pd.Timestamp("2024-01-11 03:10", tz="UTC")
PROJECT_CITIES = """cities:
  - slug: sample
    name: Sample
    latitude: 1.0
    longitude: 2.0
    timezone: UTC
    source_file: sample.csv
    archive_source: archive-source
    recent_source: recent-source
    archive_cutoff: "2024-01-10T23:00:00+00:00"
"""
PROJECT_DATA = """active_city: sample
storage:
  backend: local
  root: data
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


def _project_row(hour: int) -> str:
    """Build one raw seed row with a known daily pattern."""
    time = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour)
    temperature = 5 + hour // 24 + time.hour / 100
    return f"{time:%Y-%m-%d %H:%M:%S}+00:00,{temperature},90,1.0,0.0,1010.0,50,3.0,180"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a sample project and return its config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cities.yaml").write_text(PROJECT_CITIES, encoding="utf-8")
    (config_dir / "data.yaml").write_text(PROJECT_DATA, encoding="utf-8")
    lines = [",".join(BRONZE_COLUMNS), *(_project_row(hour) for hour in range(240))]
    content = ("\n".join(lines) + "\n").encode()
    LocalStore(tmp_path / "data").write_bytes("bronze/city=sample/sample.csv", content)
    return config_dir


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    """Replace the network and clock used by the runner."""
    api = FakeApi()
    transport = httpx.MockTransport(api.handle)
    monkeypatch.setattr(
        "skycast.ingestion.runner._open_http", lambda _settings: httpx.Client(transport=transport)
    )
    monkeypatch.setattr("skycast.ingestion.runner._utc_now", lambda: PROJECT_NOW)
    return api


@pytest.fixture
def seeded_root(tmp_path: Path) -> Path:
    """Create a storage root containing a seed CSV."""
    content = "\n".join((",".join(BRONZE_COLUMNS), *SEED_ROWS)) + "\n"
    LocalStore(tmp_path).write_bytes("bronze/city=sample/sample.csv", content.encode())
    return tmp_path
