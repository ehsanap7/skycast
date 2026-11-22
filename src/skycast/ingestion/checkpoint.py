"""Remember the last ingested hour for each source."""

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import pandas as pd

from skycast.config import City
from skycast.data.bronze import load_bronze
from skycast.data.contracts import TIME_COLUMN, TIMESTAMP_FORMAT
from skycast.data.storage import DatasetStore
from skycast.errors import DataSourceError


class SourceKind(StrEnum):
    """The API sources that feed bronze."""

    ARCHIVE = "archive"
    RECENT = "recent"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The last ingested hour per source."""

    archive: pd.Timestamp
    recent: pd.Timestamp

    def last_hour(self, kind: SourceKind) -> pd.Timestamp:
        """Return the last ingested hour for a source."""
        return getattr(self, kind.value)

    def advanced(self, kind: SourceKind, hour: pd.Timestamp) -> Checkpoint:
        """Return a checkpoint moved forward for a source."""
        if hour <= self.last_hour(kind):
            raise DataSourceError(f"Checkpoint for {kind} cannot move back to {hour}")
        return replace(self, **{kind.value: hour})


def checkpoint_key(city_slug: str) -> str:
    """Build the storage key for a city's checkpoint."""
    return f"state/city={city_slug}/ingestion.json"


def _parse_hour(value: Any, name: str) -> pd.Timestamp:
    """Parse a stored hour that must be explicit UTC."""
    if not isinstance(value, str):
        raise DataSourceError(f"Checkpoint is missing a valid {name} hour")
    try:
        hour = pd.Timestamp(value)
    except ValueError as err:
        raise DataSourceError(f"Invalid checkpoint {name} hour: {value!r}") from err
    if hour.utcoffset() != pd.Timedelta(0):
        raise DataSourceError(f"Checkpoint {name} hour must be UTC: {value!r}")
    return hour.tz_convert("UTC")


def _parse_checkpoint(content: bytes, key: str) -> Checkpoint:
    """Decode a stored checkpoint."""
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeDecodeError) as err:
        raise DataSourceError(f"Cannot parse checkpoint {key}: {err}") from err
    if not isinstance(payload, dict):
        raise DataSourceError(f"Checkpoint must be a JSON object: {key}")
    return Checkpoint(
        archive=_parse_hour(payload.get(SourceKind.ARCHIVE), SourceKind.ARCHIVE),
        recent=_parse_hour(payload.get(SourceKind.RECENT), SourceKind.RECENT),
    )


def _initial_checkpoint(store: DatasetStore, city: City) -> Checkpoint:
    """Start from the seed file's archive cutoff and last hour."""
    times = load_bronze(store, city)[TIME_COLUMN]
    last_hour = pd.to_datetime(times, format=TIMESTAMP_FORMAT, utc=True).max()
    archive = pd.Timestamp(city.archive_cutoff).tz_convert("UTC")
    return Checkpoint(archive=archive, recent=last_hour)


def read_checkpoint(store: DatasetStore, city: City) -> Checkpoint:
    """Load a city's checkpoint, or start one from the seed file."""
    key = checkpoint_key(city.slug)
    if not store.exists(key):
        return _initial_checkpoint(store, city)
    return _parse_checkpoint(store.read_bytes(key), key)


def write_checkpoint(store: DatasetStore, city: City, checkpoint: Checkpoint) -> None:
    """Store a city's checkpoint."""
    payload = {
        SourceKind.ARCHIVE.value: checkpoint.archive.isoformat(),
        SourceKind.RECENT.value: checkpoint.recent.isoformat(),
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    store.write_bytes(checkpoint_key(city.slug), content)
