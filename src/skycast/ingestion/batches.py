"""Store exact API responses as immutable raw batches."""

import hashlib
import json
from dataclasses import asdict, dataclass

import pandas as pd

from skycast.config import City
from skycast.data.contracts import TIME_COLUMN
from skycast.data.storage import DatasetStore, Layer, dataset_key
from skycast.errors import StorageError
from skycast.ingestion.checkpoint import SourceKind


@dataclass(frozen=True, slots=True)
class BatchManifest:
    """A receipt for the hours accepted from a raw batch."""

    source: str
    fetched_at: str
    first_hour: str
    last_hour: str
    rows: int
    sha256: str


def batch_key(city_slug: str, kind: SourceKind, fetched_at: pd.Timestamp) -> str:
    """Build the storage key for a raw response."""
    stamp = fetched_at.tz_convert("UTC").strftime("%Y-%m-%dT%H%M%SZ")
    return dataset_key(Layer.BRONZE, city_slug, f"batches/source={kind}/{stamp}.json")


def manifest_key(key: str) -> str:
    """Build the manifest key for a raw response key."""
    return key.removesuffix(".json") + ".manifest.json"


def _build_manifest(
    kind: SourceKind, raw: bytes, selected: pd.DataFrame, fetched_at: pd.Timestamp
) -> BatchManifest:
    """Describe the accepted hours in a response."""
    times = selected[TIME_COLUMN]
    return BatchManifest(
        source=kind.value,
        fetched_at=fetched_at.tz_convert("UTC").isoformat(),
        first_hour=times.iloc[0].isoformat(),
        last_hour=times.iloc[-1].isoformat(),
        rows=len(selected),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def save_batch(
    store: DatasetStore,
    city: City,
    kind: SourceKind,
    raw: bytes,
    selected: pd.DataFrame,
    fetched_at: pd.Timestamp,
) -> str:
    """Save a response, then its manifest, and return the batch key."""
    key = batch_key(city.slug, kind, fetched_at)
    receipt = manifest_key(key)
    if store.exists(key) or store.exists(receipt):
        raise StorageError(f"Raw batch already exists: {key}")
    manifest = _build_manifest(kind, raw, selected, fetched_at)
    store.write_bytes(key, raw)
    store.write_bytes(receipt, (json.dumps(asdict(manifest), indent=2) + "\n").encode())
    return key
