"""Build the model-ready gold dataset."""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from skycast.config import City, GoldSettings
from skycast.data.batch_reader import batch_manifest_keys
from skycast.data.contracts import (
    CITY_COLUMN,
    DATE_COLUMN,
    GOLD_CONTRACT_VERSION,
    GOLD_DATA_FILE,
    GOLD_METADATA_FILE,
    SILVER_DATA_FILE,
    SOURCE_COLUMN,
)
from skycast.data.daily import build_daily
from skycast.data.features import build_features, build_targets
from skycast.data.parquet import to_parquet_bytes
from skycast.data.storage import DatasetStore, Layer, dataset_key
from skycast.errors import DataSourceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GoldDataset:
    """Model-ready rows and their column roles."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


def _archive_rows(silver: pd.DataFrame, city: City) -> pd.DataFrame:
    """Keep only rows from the archive source."""
    archive = silver.loc[silver[SOURCE_COLUMN].eq(city.archive_source)]
    if archive.empty:
        raise DataSourceError(f"No archive rows for city: {city.slug}")
    return archive.reset_index(drop=True)


def _complete_rows(features: pd.DataFrame, targets: pd.DataFrame, city: City) -> pd.DataFrame:
    """Join features and targets, keeping only complete rows."""
    rows = pd.concat([features, targets], axis=1).dropna()
    if rows.empty:
        raise DataSourceError(f"No complete gold rows for city: {city.slug}")
    frame = rows.reset_index()
    frame.insert(0, CITY_COLUMN, city.slug)
    return frame


def build_gold(
    silver: pd.DataFrame, city: City, settings: GoldSettings, horizon_days: int
) -> GoldDataset:
    """Turn silver data into model-ready daily rows."""
    archive = _archive_rows(silver, city)
    daily = build_daily(archive, city.timezone, settings.max_fill_hours)
    features = build_features(daily, settings.lag_days, settings.rolling_days)
    targets = build_targets(daily, horizon_days)
    return GoldDataset(
        frame=_complete_rows(features, targets, city),
        feature_columns=tuple(features.columns),
        target_columns=tuple(targets.columns),
    )


def _sha256(store: DatasetStore, key: str) -> str:
    """Hash a stored dataset."""
    return hashlib.sha256(store.read_bytes(key)).hexdigest()


def _batch_lineage(store: DatasetStore, city: City) -> dict[str, Any]:
    """Fingerprint the raw batches behind a dataset."""
    keys = batch_manifest_keys(store, city)
    manifests = b"".join(store.read_bytes(key) for key in keys)
    return {"batch_count": len(keys), "batches_sha256": hashlib.sha256(manifests).hexdigest()}


def _build_metadata(
    store: DatasetStore,
    city: City,
    gold: GoldDataset,
    settings: GoldSettings,
    horizon_days: int,
) -> dict[str, Any]:
    """Describe how a gold dataset was produced."""
    dates = gold.frame[DATE_COLUMN]
    return {
        "contract_version": GOLD_CONTRACT_VERSION,
        "city": city.slug,
        "source_file": city.source_file,
        "source_sha256": _sha256(store, dataset_key(Layer.BRONZE, city.slug, city.source_file)),
        **_batch_lineage(store, city),
        "silver_sha256": _sha256(store, dataset_key(Layer.SILVER, city.slug, SILVER_DATA_FILE)),
        "archive_source": city.archive_source,
        "archive_cutoff": city.archive_cutoff.isoformat(),
        "rows": len(gold.frame),
        "start_date": dates.iloc[0].date().isoformat(),
        "end_date": dates.iloc[-1].date().isoformat(),
        "horizon_days": horizon_days,
        "max_fill_hours": settings.max_fill_hours,
        "lag_days": list(settings.lag_days),
        "rolling_days": list(settings.rolling_days),
        "feature_columns": list(gold.feature_columns),
        "target_columns": list(gold.target_columns),
    }


def write_gold(
    store: DatasetStore,
    city: City,
    gold: GoldDataset,
    settings: GoldSettings,
    horizon_days: int,
) -> None:
    """Store gold data and its metadata."""
    metadata = _build_metadata(store, city, gold, settings, horizon_days)
    data_key = dataset_key(Layer.GOLD, city.slug, GOLD_DATA_FILE)
    metadata_key = dataset_key(Layer.GOLD, city.slug, GOLD_METADATA_FILE)
    store.write_bytes(data_key, to_parquet_bytes(gold.frame))
    store.write_bytes(metadata_key, (json.dumps(metadata, indent=2) + "\n").encode())
    logger.info("Wrote %d gold rows to %s", len(gold.frame), data_key)
