"""Typed application settings loaded from YAML files."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from skycast.errors import ConfigError

DEFAULT_CONFIG_DIR = Path("config")
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class City:
    """A forecast location."""

    slug: str
    name: str
    latitude: float
    longitude: float
    timezone: str


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Where datasets are stored."""

    backend: str
    root: Path


@dataclass(frozen=True, slots=True)
class Settings:
    """All settings needed by the data section."""

    cities: tuple[City, ...]
    active_city: City
    storage: StorageSettings
    horizon_days: int


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


def _parse_city(value: Any) -> City:
    """Build a city from its raw config."""
    raw = _as_mapping(value, "City entry")
    return City(
        slug=_parse_slug(raw),
        name=_field(raw, "name", str),
        latitude=_parse_coordinate(raw, "latitude", 90),
        longitude=_parse_coordinate(raw, "longitude", 180),
        timezone=_parse_timezone(raw),
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
    )
