"""Fetch, record, and rebuild data for one source."""

from dataclasses import dataclass

import httpx
import pandas as pd

from skycast.config import City, IngestionSettings, Settings
from skycast.data.contracts import SourceKind
from skycast.data.pipeline import PipelineSummary, run_data_pipeline
from skycast.data.storage import build_store
from skycast.ingestion.checkpoint import read_checkpoint
from skycast.ingestion.client import WeatherClient, WeatherResponse
from skycast.ingestion.ingest import IngestResult, ingest_response
from skycast.ingestion.selection import archive_dates, recent_past_days


@dataclass(frozen=True, slots=True)
class UpdateSummary:
    """What one source update produced."""

    source: str
    new_hours: int
    last_hour: str
    gold_rows: int | None


def _utc_now() -> pd.Timestamp:
    """Return the current UTC time."""
    return pd.Timestamp.now(tz="UTC")


def _open_http(settings: IngestionSettings) -> httpx.Client:
    """Open an HTTP client with the configured timeout."""
    return httpx.Client(timeout=settings.timeout_seconds)


def _request(
    client: WeatherClient,
    city: City,
    kind: SourceKind,
    last_hour: pd.Timestamp,
    max_past_days: int,
) -> WeatherResponse:
    """Request the hours a source still needs."""
    now = _utc_now()
    if kind is SourceKind.ARCHIVE:
        start, end = archive_dates(last_hour, now)
        return client.fetch_archive(city, start, end)
    return client.fetch_recent(city, recent_past_days(last_hour, now, max_past_days))


def fetch_source(settings: Settings, kind: SourceKind) -> WeatherResponse:
    """Fetch new data for one source."""
    city = settings.active_city
    last_hour = read_checkpoint(build_store(settings.storage), city).last_hour(kind)
    with _open_http(settings.ingestion) as http:
        client = WeatherClient(settings.ingestion, http)
        return _request(client, city, kind, last_hour, settings.ingestion.max_past_days)


def record_source(settings: Settings, kind: SourceKind, response: WeatherResponse) -> IngestResult:
    """Save a fetched response and move the checkpoint."""
    store = build_store(settings.storage)
    return ingest_response(store, settings.active_city, kind, response, _utc_now())


def summarize_update(
    kind: SourceKind, result: IngestResult, pipeline: PipelineSummary | None
) -> UpdateSummary:
    """Describe a finished update."""
    return UpdateSummary(
        source=kind.value,
        new_hours=result.rows,
        last_hour=result.last_hour.isoformat(),
        gold_rows=pipeline.gold_rows if pipeline else None,
    )


def update_source(settings: Settings, kind: SourceKind) -> UpdateSummary:
    """Fetch a source, record it, and rebuild when it has new hours."""
    response = fetch_source(settings, kind)
    result = record_source(settings, kind, response)
    pipeline = run_data_pipeline(settings) if result.rows else None
    return summarize_update(kind, result, pipeline)
