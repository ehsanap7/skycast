"""Record an API response safely in bronze."""

import logging
from dataclasses import dataclass

import pandas as pd

from skycast.config import City
from skycast.data.contracts import TIME_COLUMN
from skycast.data.storage import DatasetStore
from skycast.ingestion.batches import save_batch
from skycast.ingestion.checkpoint import SourceKind, read_checkpoint, write_checkpoint
from skycast.ingestion.client import WeatherResponse
from skycast.ingestion.selection import select_new_hours

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What one ingestion run recorded."""

    rows: int
    last_hour: pd.Timestamp
    batch_key: str | None


def ingest_response(
    store: DatasetStore,
    city: City,
    kind: SourceKind,
    response: WeatherResponse,
    now: pd.Timestamp,
) -> IngestResult:
    """Save new hours, then move the checkpoint."""
    checkpoint = read_checkpoint(store, city)
    after = checkpoint.last_hour(kind)
    selected = select_new_hours(response.hourly, after, now)
    if selected.empty:
        logger.info("No new %s hours after %s", kind, after)
        return IngestResult(rows=0, last_hour=after, batch_key=None)
    key = save_batch(store, city, kind, response.raw, selected, now)
    last_hour = selected[TIME_COLUMN].iloc[-1]
    write_checkpoint(store, city, checkpoint.advanced(kind, last_hour))
    logger.info("Recorded %d %s hours in %s", len(selected), kind, key)
    return IngestResult(rows=len(selected), last_hour=last_hour, batch_key=key)
