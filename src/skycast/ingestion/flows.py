"""Prefect flows and schedules for weather ingestion."""

from pathlib import Path
from typing import Any

from prefect import flow, serve, task
from prefect.cache_policies import NO_CACHE
from prefect.deployments.runner import RunnerDeployment
from prefect.tasks import exponential_backoff

from skycast.config import DEFAULT_CONFIG_DIR, Settings, load_settings
from skycast.data.contracts import SourceKind
from skycast.data.pipeline import PipelineSummary, run_data_pipeline
from skycast.errors import TemporarySourceError
from skycast.ingestion.client import WeatherResponse
from skycast.ingestion.ingest import IngestResult
from skycast.ingestion.runner import UpdateSummary, fetch_source, record_source, summarize_update

MAX_CONCURRENT_RUNS = 1


def _is_temporary(_task: Any, _task_run: Any, state: Any) -> bool:
    """Retry only failures that may succeed later."""
    try:
        state.result()
    except TemporarySourceError:
        return True
    except Exception:
        return False
    return False


@task(name="fetch-weather", cache_policy=NO_CACHE, retry_condition_fn=_is_temporary)
def fetch_task(settings: Settings, kind: SourceKind) -> WeatherResponse:
    """Fetch new hours from the API."""
    return fetch_source(settings, kind)


@task(name="record-batch", cache_policy=NO_CACHE)
def record_task(settings: Settings, kind: SourceKind, response: WeatherResponse) -> IngestResult:
    """Save the raw batch and move the checkpoint."""
    return record_source(settings, kind, response)


@task(name="build-datasets", cache_policy=NO_CACHE)
def build_task(settings: Settings) -> PipelineSummary:
    """Rebuild silver and gold."""
    return run_data_pipeline(settings)


def _update(kind: SourceKind, config_dir: str) -> UpdateSummary:
    """Run the update steps for one source."""
    settings = load_settings(Path(config_dir))
    schedule = settings.schedule
    delays = exponential_backoff(schedule.retry_base_seconds)(schedule.retries)
    fetch = fetch_task.with_options(retries=schedule.retries, retry_delay_seconds=delays)
    result = record_task(settings, kind, fetch(settings, kind))
    pipeline = build_task(settings) if result.rows else None
    return summarize_update(kind, result, pipeline)


@flow(name="update-recent-weather")
def update_recent(config_dir: str = str(DEFAULT_CONFIG_DIR)) -> UpdateSummary:
    """Bring in the latest recent hours."""
    return _update(SourceKind.RECENT, config_dir)


@flow(name="update-archive-weather")
def update_archive(config_dir: str = str(DEFAULT_CONFIG_DIR)) -> UpdateSummary:
    """Bring in newly published archive hours."""
    return _update(SourceKind.ARCHIVE, config_dir)


def build_deployments(settings: Settings, config_dir: Path) -> list[RunnerDeployment]:
    """Create the scheduled flow deployments."""
    parameters = {"config_dir": str(config_dir.resolve())}
    return [
        update_recent.to_deployment(
            name="recent", cron=settings.schedule.recent_cron, parameters=parameters
        ),
        update_archive.to_deployment(
            name="archive", cron=settings.schedule.archive_cron, parameters=parameters
        ),
    ]


def serve_schedules(settings: Settings, config_dir: Path) -> None:
    """Run scheduled flows, one at a time, until stopped."""
    serve(*build_deployments(settings, config_dir), limit=MAX_CONCURRENT_RUNS)
