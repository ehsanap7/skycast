"""Load the accepted hours from stored raw API batches."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from skycast.config import City
from skycast.data.api_response import parse_weather_response
from skycast.data.contracts import BATCHES_FOLDER, MANIFEST_SUFFIX, TIME_COLUMN, SourceKind
from skycast.data.storage import DatasetStore, Layer, dataset_key
from skycast.errors import DataSourceError


@dataclass(frozen=True, slots=True)
class BronzeBatch:
    """Accepted hourly rows from one API fetch."""

    kind: SourceKind
    fetched_at: pd.Timestamp
    hourly: pd.DataFrame


def batch_manifest_keys(store: DatasetStore, city: City) -> list[str]:
    """List the committed batch manifests for a city."""
    prefix = dataset_key(Layer.BRONZE, city.slug, BATCHES_FOLDER)
    return [key for key in store.list_keys(prefix) if key.endswith(MANIFEST_SUFFIX)]


def _read_manifest(content: bytes, key: str) -> Mapping[str, Any]:
    """Decode a manifest that must be a JSON object."""
    try:
        manifest = json.loads(content)
    except (ValueError, UnicodeDecodeError) as err:
        raise DataSourceError(f"Cannot parse batch manifest {key}: {err}") from err
    if not isinstance(manifest, dict):
        raise DataSourceError(f"Batch manifest must be a JSON object: {key}")
    return manifest


def _manifest_time(manifest: Mapping[str, Any], name: str, key: str) -> pd.Timestamp:
    """Read a manifest time that must be explicit UTC."""
    value = manifest.get(name)
    if not isinstance(value, str):
        raise DataSourceError(f"Batch manifest {key} has no valid {name}")
    try:
        time = pd.Timestamp(value)
    except ValueError as err:
        raise DataSourceError(f"Batch manifest {key} has an invalid {name}") from err
    if time.utcoffset() != pd.Timedelta(0):
        raise DataSourceError(f"Batch manifest {key} {name} must be UTC")
    return time.tz_convert("UTC")


def _manifest_kind(manifest: Mapping[str, Any], key: str) -> SourceKind:
    """Read the source named by a manifest."""
    try:
        return SourceKind(manifest.get("source"))
    except ValueError as err:
        raise DataSourceError(f"Batch manifest {key} has an unknown source") from err


def _read_verified_response(store: DatasetStore, manifest: Mapping[str, Any], key: str) -> bytes:
    """Read a raw response and check it against its manifest hash."""
    response_key = key.removesuffix(MANIFEST_SUFFIX) + ".json"
    if not store.exists(response_key):
        raise DataSourceError(f"Batch response is missing: {response_key}")
    raw = store.read_bytes(response_key)
    if hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
        raise DataSourceError(f"Batch file is damaged, hash mismatch: {response_key}")
    return raw


def _accepted_hours(raw: bytes, manifest: Mapping[str, Any], key: str) -> pd.DataFrame:
    """Keep only the hours accepted by a manifest."""
    try:
        hourly = parse_weather_response(raw)
    except DataSourceError as err:
        raise DataSourceError(f"Invalid batch for {key}: {err}") from err
    first = _manifest_time(manifest, "first_hour", key)
    last = _manifest_time(manifest, "last_hour", key)
    accepted = hourly.loc[hourly[TIME_COLUMN].between(first, last)].reset_index(drop=True)
    if len(accepted) != manifest.get("rows"):
        raise DataSourceError(f"Batch rows do not match manifest: {key}")
    return accepted


def _load_batch(store: DatasetStore, key: str) -> BronzeBatch:
    """Load and verify one committed batch."""
    manifest = _read_manifest(store.read_bytes(key), key)
    raw = _read_verified_response(store, manifest, key)
    return BronzeBatch(
        kind=_manifest_kind(manifest, key),
        fetched_at=_manifest_time(manifest, "fetched_at", key),
        hourly=_accepted_hours(raw, manifest, key),
    )


def load_batches(store: DatasetStore, city: City) -> tuple[BronzeBatch, ...]:
    """Load every committed batch for a city."""
    return tuple(_load_batch(store, key) for key in batch_manifest_keys(store, city))
