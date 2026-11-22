"""Build the clean hourly silver dataset."""

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pandas as pd

from skycast.config import City
from skycast.data.batch_reader import BronzeBatch
from skycast.data.contracts import (
    CITY_COLUMN,
    MEASUREMENT_RANGES,
    QUALITY_REPORT_FILE,
    SILVER_COLUMNS,
    SILVER_DATA_FILE,
    SOURCE_COLUMN,
    TIME_COLUMN,
    TIMESTAMP_FORMAT,
    SourceKind,
)
from skycast.data.parquet import read_parquet_bytes, to_parquet_bytes
from skycast.data.storage import DatasetStore, Layer, dataset_key
from skycast.errors import DataSourceError

logger = logging.getLogger(__name__)

SOURCE_RANK = "_source_rank"
FETCH_RANK = "_fetch_rank"
RECENT_RANK = 0
ARCHIVE_RANK = 1
SEED_FETCH_RANK = 0


@dataclass(frozen=True, slots=True)
class QualityReport:
    """What silver found and changed."""

    source_rows: int
    rows: int
    duplicates_removed: int
    missing_hours: int
    start: str
    end: str
    null_values: dict[str, int]
    out_of_range: dict[str, int]
    rows_by_source: dict[str, int]
    batches_used: int
    recent_replaced_by_archive: int


@dataclass(frozen=True, slots=True)
class SilverDataset:
    """Clean hourly data and its quality report."""

    frame: pd.DataFrame
    report: QualityReport


def _parse_measurement(values: pd.Series, column: str) -> pd.Series:
    """Convert text to numbers, keeping blanks as null."""
    text = values.fillna("").str.strip()
    numbers = pd.to_numeric(text, errors="coerce").astype("float64")
    invalid = numbers.isna() & text.ne("")
    if invalid.any():
        first = text[invalid].iloc[0]
        raise DataSourceError(f"{invalid.sum()} non-numeric values in {column}, first: {first!r}")
    return numbers


def _typed_frame(bronze: pd.DataFrame) -> pd.DataFrame:
    """Build a typed frame from raw bronze text."""
    times = pd.to_datetime(bronze[TIME_COLUMN], format=TIMESTAMP_FORMAT, utc=True)
    measurements = {
        column: _parse_measurement(bronze[column], column) for column in MEASUREMENT_RANGES
    }
    return pd.DataFrame({TIME_COLUMN: times, **measurements})


def _ranked_seed(bronze: pd.DataFrame, city: City) -> pd.DataFrame:
    """Rank seed rows by the archive cutoff."""
    typed = _typed_frame(bronze)
    source_ranks = typed[TIME_COLUMN].le(city.archive_cutoff).astype(int)
    return typed.assign(**{SOURCE_RANK: source_ranks, FETCH_RANK: SEED_FETCH_RANK})


def _ranked_batches(batches: Sequence[BronzeBatch]) -> list[pd.DataFrame]:
    """Rank batch rows by source and fetch order."""
    ordered = sorted(batches, key=lambda batch: batch.fetched_at)
    return [
        batch.hourly.assign(
            **{
                SOURCE_RANK: ARCHIVE_RANK if batch.kind is SourceKind.ARCHIVE else RECENT_RANK,
                FETCH_RANK: position,
            }
        )
        for position, batch in enumerate(ordered, start=SEED_FETCH_RANK + 1)
    ]


def _combine_sources(
    bronze: pd.DataFrame, batches: Sequence[BronzeBatch], city: City
) -> pd.DataFrame:
    """Stack seed and batch rows with their priorities."""
    return pd.concat([_ranked_seed(bronze, city), *_ranked_batches(batches)], ignore_index=True)


def _require_whole_hours(frame: pd.DataFrame) -> None:
    """Reject timestamps that do not fall exactly on an hour."""
    times = frame[TIME_COLUMN]
    invalid = times.ne(times.dt.floor("h"))
    if invalid.any():
        raise DataSourceError(f"{invalid.sum()} timestamps are not on a whole hour")


def _remove_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the highest-priority row for each timestamp."""
    ordered = frame.sort_values([TIME_COLUMN, SOURCE_RANK, FETCH_RANK], kind="stable")
    unique = ordered.drop_duplicates(subset=TIME_COLUMN, keep="last")
    return unique, len(frame) - len(unique)


def _recent_replaced(combined: pd.DataFrame, unique: pd.DataFrame) -> int:
    """Count hours where an archive value replaced a recent one."""
    recent = combined.loc[combined[SOURCE_RANK].eq(RECENT_RANK), TIME_COLUMN]
    archive = unique.loc[unique[SOURCE_RANK].eq(ARCHIVE_RANK), TIME_COLUMN]
    return int(recent[recent.isin(archive)].nunique())


def _complete_timeline(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add null rows for hours missing from the timeline."""
    times = frame[TIME_COLUMN]
    hours = pd.date_range(times.min(), times.max(), freq="h", unit=times.dt.unit, name=TIME_COLUMN)
    complete = frame.set_index(TIME_COLUMN).reindex(hours).reset_index()
    return complete, len(hours) - len(frame)


def _apply_ranges(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace physically impossible values with null and count them."""
    clean = frame.copy()
    counts: dict[str, int] = {}
    for column, (low, high) in MEASUREMENT_RANGES.items():
        invalid = frame[column].notna() & ~frame[column].between(low, high)
        counts[column] = int(invalid.sum())
        clean[column] = frame[column].mask(invalid)
    return clean, counts


def _source_labels(frame: pd.DataFrame, city: City) -> pd.Series:
    """Label each hour with the source that supplied it."""
    missing_archive = frame[SOURCE_RANK].isna() & frame[TIME_COLUMN].le(city.archive_cutoff)
    is_archive = frame[SOURCE_RANK].eq(ARCHIVE_RANK) | missing_archive
    return is_archive.map({True: city.archive_source, False: city.recent_source})


def _add_identity(frame: pd.DataFrame, city: City) -> pd.DataFrame:
    """Add city and source columns in contract order."""
    identity = {CITY_COLUMN: city.slug, SOURCE_COLUMN: _source_labels(frame, city)}
    return frame.assign(**identity)[list(SILVER_COLUMNS)]


def _source_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count rows per source."""
    return {str(name): int(count) for name, count in frame[SOURCE_COLUMN].value_counts().items()}


def _null_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count null values per measurement."""
    return {column: int(frame[column].isna().sum()) for column in MEASUREMENT_RANGES}


def build_silver(
    bronze: pd.DataFrame, city: City, batches: Sequence[BronzeBatch] = ()
) -> SilverDataset:
    """Merge bronze sources into clean hourly data."""
    combined = _combine_sources(bronze, batches, city)
    _require_whole_hours(combined)
    unique, duplicates = _remove_duplicates(combined)
    complete, missing = _complete_timeline(unique)
    clean, out_of_range = _apply_ranges(complete)
    frame = _add_identity(clean, city)
    report = QualityReport(
        source_rows=len(combined),
        rows=len(frame),
        duplicates_removed=duplicates,
        missing_hours=missing,
        start=frame[TIME_COLUMN].iloc[0].isoformat(),
        end=frame[TIME_COLUMN].iloc[-1].isoformat(),
        null_values=_null_counts(frame),
        out_of_range=out_of_range,
        rows_by_source=_source_counts(frame),
        batches_used=len(batches),
        recent_replaced_by_archive=_recent_replaced(combined, unique),
    )
    return SilverDataset(frame=frame, report=report)


def _to_json_bytes(report: QualityReport) -> bytes:
    """Serialize a quality report to stable JSON bytes."""
    return (json.dumps(asdict(report), indent=2, sort_keys=True) + "\n").encode()


def write_silver(store: DatasetStore, city: City, silver: SilverDataset) -> None:
    """Store silver data and its quality report."""
    data_key = dataset_key(Layer.SILVER, city.slug, SILVER_DATA_FILE)
    report_key = dataset_key(Layer.SILVER, city.slug, QUALITY_REPORT_FILE)
    store.write_bytes(data_key, to_parquet_bytes(silver.frame))
    store.write_bytes(report_key, _to_json_bytes(silver.report))
    logger.info("Wrote %d silver rows to %s", silver.report.rows, data_key)


def read_silver(store: DatasetStore, city: City) -> pd.DataFrame:
    """Load a city's stored silver data and check its columns."""
    key = dataset_key(Layer.SILVER, city.slug, SILVER_DATA_FILE)
    if not store.exists(key):
        raise DataSourceError(f"Silver dataset not found: {key}")
    frame = read_parquet_bytes(store.read_bytes(key), key)
    missing = [column for column in SILVER_COLUMNS if column not in frame.columns]
    if missing:
        raise DataSourceError(f"Missing columns in {key}: {', '.join(missing)}")
    return frame
