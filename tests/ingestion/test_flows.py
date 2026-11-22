"""Tests for Prefect flows, retries, and schedules."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_weather_api import FakeApi
from prefect.testing.utilities import prefect_test_harness

from skycast.cli import main
from skycast.config import load_settings
from skycast.data.storage import build_store
from skycast.errors import DataSourceError, TemporarySourceError
from skycast.ingestion import flows
from skycast.ingestion.checkpoint import checkpoint_key
from skycast.ingestion.flows import build_deployments, update_archive, update_recent


@pytest.fixture(scope="module", autouse=True)
def prefect_server() -> Iterator[None]:
    """Run flows against a temporary Prefect server."""
    with prefect_test_harness():
        yield


def test_recent_flow_updates_data(project: Path, fake_api: FakeApi) -> None:
    """The recent flow saves hours and rebuilds."""
    summary = update_recent(config_dir=str(project))

    assert summary.source == "recent"
    assert summary.new_hours == 4
    assert summary.gold_rows == 7


def test_archive_flow_updates_data(project: Path, fake_api: FakeApi) -> None:
    """The archive flow saves hours and rebuilds."""
    summary = update_archive(config_dir=str(project))

    assert summary.source == "archive"
    assert summary.new_hours == 4


def test_second_run_skips_build(project: Path, fake_api: FakeApi) -> None:
    """A run without new hours finishes without rebuilding."""
    update_recent(config_dir=str(project))

    summary = update_recent(config_dir=str(project))

    assert summary.new_hours == 0
    assert summary.gold_rows is None


def test_temporary_failure_is_retried(project: Path, fake_api: FakeApi) -> None:
    """A temporary API failure succeeds on retry."""
    fake_api.failures = [503, 429]

    summary = update_recent(config_dir=str(project))

    assert fake_api.calls == 3
    assert summary.new_hours == 4


def test_retries_stop_at_limit(project: Path, fake_api: FakeApi) -> None:
    """A lasting outage fails after the configured retries."""
    fake_api.failures = [503, 503, 503, 503]
    settings = load_settings(project)

    with pytest.raises(TemporarySourceError):
        update_recent(config_dir=str(project))

    assert fake_api.calls == 4
    assert not build_store(settings.storage).exists(checkpoint_key(settings.active_city.slug))


def test_bad_request_is_not_retried(project: Path, fake_api: FakeApi) -> None:
    """A permanent failure stops immediately."""
    fake_api.failures = [400]

    with pytest.raises(DataSourceError, match="400"):
        update_recent(config_dir=str(project))

    assert fake_api.calls == 1


def test_deployments_use_configured_schedules(project: Path) -> None:
    """Each flow is scheduled from configuration."""
    settings = load_settings(project)

    deployments = build_deployments(settings, project)

    crons = {item.name: item.schedules[0].schedule.cron for item in deployments}
    assert crons == {"recent": "10 */4 * * *", "archive": "40 6 * * *"}
    assert all(item.parameters["config_dir"] == str(project.resolve()) for item in deployments)


def test_schedule_command_serves_one_run_at_a_time(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The schedule command starts both flows with a limit of one."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(flows, "serve", lambda *args, **kwargs: calls.append((args, kwargs)))

    exit_code = main(["data", "schedule", "--config-dir", str(project)])

    assert exit_code == 0
    assert len(calls[0][0]) == 2
    assert calls[0][1]["limit"] == 1
