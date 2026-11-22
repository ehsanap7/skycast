"""Typed application settings loaded from YAML files."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from skycast.errors import ConfigError

DEFAULT_CONFIG_DIR = Path("config")
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
SOURCE_FILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.csv")


@dataclass(frozen=True, slots=True)
class City:
    """A forecast location."""

    slug: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    source_file: str
    archive_source: str
    recent_source: str
    archive_cutoff: datetime


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Where datasets are stored."""

    backend: str
    root: Path


@dataclass(frozen=True, slots=True)
class GoldSettings:
    """How the gold dataset is built."""

    max_fill_hours: int
    lag_days: tuple[int, ...]
    rolling_days: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    """How weather data is requested from the APIs."""

    archive_url: str
    archive_model: str
    recent_url: str
    recent_forecast_days: int
    max_past_days: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class Settings:
    """All settings needed by the data section."""

    cities: tuple[City, ...]
    active_city: City
    storage: StorageSettings
    horizon_days: int
    gold: GoldSettings
    ingestion: IngestionSettings


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """Read a YAML file that must contain a mapping."""
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        raise ConfigError(f"Cannot load config file {path}: {err}") from err
    return _as_mapping(content, str(path))


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return a value only if it is a mapping."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _field(raw: Mapping[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    """Read a required field and check its type."""
    if key not in raw:
        raise ConfigError(f"Missing config field: {key}")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ConfigError(f"Config field {key} has an invalid type")
    return value


def _parse_slug(raw: Mapping[str, Any]) -> str:
    """Read a city slug that is safe to use in storage paths."""
    slug = _field(raw, "slug", str)
    if not SLUG_PATTERN.fullmatch(slug):
        raise ConfigError(f"Invalid city slug: {slug!r}")
    return slug


def _parse_coordinate(raw: Mapping[str, Any], key: str, limit: float) -> float:
    """Read a coordinate within its allowed range."""
    value = float(_field(raw, key, (int, float)))
    if not -limit <= value <= limit:
        raise ConfigError(f"{key} must be between {-limit} and {limit}")
    return value


def _parse_timezone(raw: Mapping[str, Any]) -> str:
    """Read a valid IANA timezone name."""
    timezone = _field(raw, "timezone", str)
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, OSError) as err:
        raise ConfigError(f"Invalid timezone: {timezone!r}") from err
    return timezone


def _parse_source_file(raw: Mapping[str, Any]) -> str:
    """Read a plain CSV filename without any path."""
    source_file = _field(raw, "source_file", str)
    if not SOURCE_FILE_PATTERN.fullmatch(source_file):
        raise ConfigError(f"Invalid source file: {source_file!r}")
    return source_file


def _parse_archive_cutoff(raw: Mapping[str, Any]) -> datetime:
    """Read the last archive timestamp as an explicit UTC time."""
    text = _field(raw, "archive_cutoff", str)
    try:
        cutoff = datetime.fromisoformat(text)
    except ValueError as err:
        raise ConfigError(f"Invalid archive cutoff: {text!r}") from err
    if cutoff.utcoffset() != timedelta(0):
        raise ConfigError(f"Archive cutoff must be UTC: {text!r}")
    return cutoff


def _parse_city(value: Any) -> City:
    """Build a city from its raw config."""
    raw = _as_mapping(value, "City entry")
    return City(
        slug=_parse_slug(raw),
        name=_field(raw, "name", str),
        latitude=_parse_coordinate(raw, "latitude", 90),
        longitude=_parse_coordinate(raw, "longitude", 180),
        timezone=_parse_timezone(raw),
        source_file=_parse_source_file(raw),
        archive_source=_field(raw, "archive_source", str),
        recent_source=_field(raw, "recent_source", str),
        archive_cutoff=_parse_archive_cutoff(raw),
    )


def _parse_cities(raw: Mapping[str, Any]) -> tuple[City, ...]:
    """Build the configured cities and reject duplicates."""
    cities = tuple(_parse_city(entry) for entry in _field(raw, "cities", list))
    slugs = [city.slug for city in cities]
    if not cities or len(slugs) != len(set(slugs)):
        raise ConfigError("Cities must be non-empty with unique slugs")
    return cities


def _find_city(cities: tuple[City, ...], slug: str) -> City:
    """Find a configured city by slug."""
    for city in cities:
        if city.slug == slug:
            return city
    raise ConfigError(f"Active city is not configured: {slug!r}")


def _parse_storage(raw: Mapping[str, Any], project_root: Path) -> StorageSettings:
    """Build storage settings with an absolute root."""
    storage = _as_mapping(_field(raw, "storage", Mapping), "storage")
    root = project_root / _field(storage, "root", str)
    return StorageSettings(backend=_field(storage, "backend", str), root=root)


def _parse_horizon(raw: Mapping[str, Any]) -> int:
    """Read a positive forecast horizon in days."""
    forecast = _as_mapping(_field(raw, "forecast", Mapping), "forecast")
    horizon = _field(forecast, "horizon_days", int)
    if horizon < 1:
        raise ConfigError("horizon_days must be at least 1")
    return horizon


def _parse_day_list(raw: Mapping[str, Any], key: str, minimum: int) -> tuple[int, ...]:
    """Read a non-empty list of unique day counts."""
    values = _field(raw, key, list)
    valid = all(type(value) is int and value >= minimum for value in values)
    if not values or not valid or len(values) != len(set(values)):
        raise ConfigError(f"{key} must be unique integers of at least {minimum}")
    return tuple(values)


def _parse_gold(raw: Mapping[str, Any]) -> GoldSettings:
    """Build the gold dataset settings."""
    gold = _as_mapping(_field(raw, "gold", Mapping), "gold")
    max_fill_hours = _field(gold, "max_fill_hours", int)
    if max_fill_hours < 0:
        raise ConfigError("max_fill_hours cannot be negative")
    return GoldSettings(
        max_fill_hours=max_fill_hours,
        lag_days=_parse_day_list(gold, "lag_days", 1),
        rolling_days=_parse_day_list(gold, "rolling_days", 2),
    )


def _parse_https_url(raw: Mapping[str, Any], key: str) -> str:
    """Read a URL that must use HTTPS."""
    url = _field(raw, key, str)
    if not url.startswith("https://"):
        raise ConfigError(f"{key} must be an HTTPS URL")
    return url


def _parse_int_range(raw: Mapping[str, Any], key: str, low: int, high: int) -> int:
    """Read an integer within an allowed range."""
    value = _field(raw, key, int)
    if not low <= value <= high:
        raise ConfigError(f"{key} must be between {low} and {high}")
    return value


def _parse_timeout(raw: Mapping[str, Any]) -> float:
    """Read a positive timeout in seconds."""
    timeout = float(_field(raw, "timeout_seconds", (int, float)))
    if timeout <= 0:
        raise ConfigError("timeout_seconds must be positive")
    return timeout


def _parse_ingestion(raw: Mapping[str, Any]) -> IngestionSettings:
    """Build the API ingestion settings."""
    ingestion = _as_mapping(_field(raw, "ingestion", Mapping), "ingestion")
    return IngestionSettings(
        archive_url=_parse_https_url(ingestion, "archive_url"),
        archive_model=_field(ingestion, "archive_model", str),
        recent_url=_parse_https_url(ingestion, "recent_url"),
        recent_forecast_days=_parse_int_range(ingestion, "recent_forecast_days", 1, 16),
        max_past_days=_parse_int_range(ingestion, "max_past_days", 1, 92),
        timeout_seconds=_parse_timeout(ingestion),
    )


def load_settings(config_dir: Path = DEFAULT_CONFIG_DIR) -> Settings:
    """Load and validate settings from a config directory."""
    resolved_dir = config_dir.resolve()
    cities = _parse_cities(_read_yaml(resolved_dir / "cities.yaml"))
    data = _read_yaml(resolved_dir / "data.yaml")
    return Settings(
        cities=cities,
        active_city=_find_city(cities, _field(data, "active_city", str)),
        storage=_parse_storage(data, resolved_dir.parent),
        horizon_days=_parse_horizon(data),
        gold=_parse_gold(data),
        ingestion=_parse_ingestion(data),
    )
