"""Load and validate raw source files without changing them."""

import logging
from io import BytesIO

import pandas as pd

from skycast.config import City
from skycast.data.contracts import BRONZE_COLUMNS, TIME_COLUMN, TIMESTAMP_FORMAT, UTC_SUFFIX
from skycast.data.storage import DatasetStore, Layer, dataset_key
from skycast.errors import DataSourceError

logger = logging.getLogger(__name__)


def _read_content(store: DatasetStore, key: str) -> bytes:
    """Read a bronze file that must exist."""
    if not store.exists(key):
        raise DataSourceError(f"Bronze file not found: {key}")
    return store.read_bytes(key)


def _parse_csv(content: bytes, key: str) -> pd.DataFrame:
    """Parse CSV bytes while keeping every value as raw text."""
    try:
        return pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as err:
        raise DataSourceError(f"Cannot parse bronze file {key}: {err}") from err


def _require_columns(frame: pd.DataFrame, key: str) -> None:
    """Check that every contract column is present."""
    missing = [column for column in BRONZE_COLUMNS if column not in frame.columns]
    if missing:
        raise DataSourceError(f"Missing columns in {key}: {', '.join(missing)}")


def _require_rows(frame: pd.DataFrame, key: str) -> None:
    """Check that the file contains data."""
    if frame.empty:
        raise DataSourceError(f"Bronze file has no rows: {key}")


def _require_utc_timestamps(frame: pd.DataFrame, key: str) -> None:
    """Check that every timestamp is valid and explicitly UTC."""
    times = frame[TIME_COLUMN]
    parsed = pd.to_datetime(times, format=TIMESTAMP_FORMAT, utc=True, errors="coerce")
    invalid = parsed.isna() | ~times.str.endswith(UTC_SUFFIX, na=False)
    if invalid.any():
        first = times[invalid].iloc[0]
        raise DataSourceError(f"{invalid.sum()} invalid timestamps in {key}, first: {first!r}")


def load_bronze(store: DatasetStore, city: City) -> pd.DataFrame:
    """Load a city's raw file after checking its structure."""
    key = dataset_key(Layer.BRONZE, city.slug, city.source_file)
    frame = _parse_csv(_read_content(store, key), key)
    _require_columns(frame, key)
    _require_rows(frame, key)
    _require_utc_timestamps(frame, key)
    logger.info("Loaded %d bronze rows from %s", len(frame), key)
    return frame
