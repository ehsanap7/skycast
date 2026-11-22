"""Tests for settings loading."""

from pathlib import Path

import pytest

from skycast.config import load_settings
from skycast.errors import ConfigError

VALID_CITY = """cities:
  - slug: sample-city
    name: Sample City
    latitude: 10.5
    longitude: -20.5
    timezone: UTC
    source_file: sample.csv
    archive_source: archive-source
    recent_source: recent-source
    archive_cutoff: "2010-01-01T12:00:00+00:00"
"""
VALID_DATA = """active_city: sample-city
storage:
  backend: local
  root: datasets
forecast:
  horizon_days: 7
gold:
  max_fill_hours: 3
  lag_days: [1, 2]
  rolling_days: [3, 7]
ingestion:
  archive_url: https://archive.example.test/v1/archive
  archive_model: sample-model
  recent_url: https://recent.example.test/v1/forecast
  recent_forecast_days: 1
  max_past_days: 92
  timeout_seconds: 60
"""


def write_config(root: Path, cities: str = VALID_CITY, data: str = VALID_DATA) -> Path:
    """Write config files and return their directory."""
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "cities.yaml").write_text(cities, encoding="utf-8")
    (config_dir / "data.yaml").write_text(data, encoding="utf-8")
    return config_dir


def test_loads_valid_settings(tmp_path: Path) -> None:
    """Valid files become typed settings."""
    settings = load_settings(write_config(tmp_path))

    assert settings.active_city.slug == "sample-city"
    assert settings.active_city.latitude == 10.5
    assert settings.storage.root == tmp_path.resolve() / "datasets"
    assert settings.horizon_days == 7


def test_repository_config_is_valid() -> None:
    """The committed config loads successfully."""
    settings = load_settings(Path(__file__).parents[1] / "config")

    assert settings.active_city in settings.cities
    assert settings.horizon_days == 7


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    """A missing config directory is a config error."""
    with pytest.raises(ConfigError, match="Cannot load"):
        load_settings(tmp_path / "missing")


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    """Broken YAML is a config error."""
    with pytest.raises(ConfigError, match="Cannot load"):
        load_settings(write_config(tmp_path, cities="cities: ["))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("sample-city\n    name", "../escape\n    name", "Invalid city slug"),
        ("latitude: 10.5", "latitude: 91", "latitude"),
        ("longitude: -20.5", "longitude: -181", "longitude"),
        ("timezone: UTC", "timezone: Mars/Base", "Invalid timezone"),
        ("source_file: sample.csv", "source_file: ../sample.csv", "Invalid source file"),
        ("source_file: sample.csv", "source_file: sample.txt", "Invalid source file"),
        ("    archive_source: archive-source\n", "", "archive_source"),
        ("    recent_source: recent-source\n", "", "recent_source"),
        ("2010-01-01T12:00:00+00:00", "not-a-date", "Invalid archive cutoff"),
        ("2010-01-01T12:00:00+00:00", "2010-01-01T12:00:00", "must be UTC"),
        ("2010-01-01T12:00:00+00:00", "2010-01-01T12:00:00-05:00", "must be UTC"),
        ("latitude: 10.5", "latitude: true", "invalid type"),
    ],
)
def test_invalid_city_raises_config_error(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Invalid city values are rejected."""
    cities = VALID_CITY.replace(old, new)

    with pytest.raises(ConfigError, match=message):
        load_settings(write_config(tmp_path, cities=cities))


def test_duplicate_city_raises_config_error(tmp_path: Path) -> None:
    """City slugs must be unique."""
    entry = VALID_CITY.removeprefix("cities:\n")

    with pytest.raises(ConfigError, match="unique"):
        load_settings(write_config(tmp_path, cities=VALID_CITY + entry))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("active_city: sample-city", "active_city: other", "not configured"),
        ("horizon_days: 7", "horizon_days: 0", "at least 1"),
        ("  root: datasets\n", "", "root"),
        ("max_fill_hours: 3", "max_fill_hours: -1", "cannot be negative"),
        ("lag_days: [1, 2]", "lag_days: []", "lag_days"),
        ("lag_days: [1, 2]", "lag_days: [1, 1]", "lag_days"),
        ("lag_days: [1, 2]", "lag_days: [0, 1]", "lag_days"),
        ("rolling_days: [3, 7]", "rolling_days: [1, 7]", "rolling_days"),
        ("rolling_days: [3, 7]", "rolling_days: [3, true]", "rolling_days"),
        ("https://archive.example", "http://archive.example", "archive_url"),
        ("https://recent.example", "ftp://recent.example", "recent_url"),
        ("recent_forecast_days: 1", "recent_forecast_days: 17", "recent_forecast_days"),
        ("max_past_days: 92", "max_past_days: 93", "max_past_days"),
        ("timeout_seconds: 60", "timeout_seconds: 0", "timeout_seconds"),
        ("  archive_model: sample-model\n", "", "archive_model"),
    ],
)
def test_invalid_data_config_raises_config_error(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Invalid data settings are rejected."""
    data = VALID_DATA.replace(old, new)

    with pytest.raises(ConfigError, match=message):
        load_settings(write_config(tmp_path, data=data))
