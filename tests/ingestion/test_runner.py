"""Tests for updating one source and its commands."""

import subprocess
import sys
from pathlib import Path

import pytest
from fake_weather_api import FakeApi

from skycast.cli import main
from skycast.config import load_settings
from skycast.data.contracts import SourceKind
from skycast.data.storage import build_store
from skycast.errors import DataSourceError
from skycast.ingestion.checkpoint import checkpoint_key, read_checkpoint
from skycast.ingestion.runner import update_source


@pytest.mark.parametrize("kind", list(SourceKind))
def test_update_saves_hours_and_rebuilds(
    project: Path, fake_api: FakeApi, kind: SourceKind
) -> None:
    """New hours are saved and the datasets are rebuilt."""
    settings = load_settings(project)

    summary = update_source(settings, kind)

    checkpoint = read_checkpoint(build_store(settings.storage), settings.active_city)
    assert summary.source == kind.value
    assert summary.new_hours == 4
    assert summary.last_hour == "2024-01-11T03:00:00+00:00"
    assert summary.gold_rows == 7
    assert checkpoint.last_hour(kind).isoformat() == summary.last_hour


def test_second_update_skips_rebuild(project: Path, fake_api: FakeApi) -> None:
    """No new hours means no rebuild."""
    settings = load_settings(project)
    update_source(settings, SourceKind.RECENT)

    summary = update_source(settings, SourceKind.RECENT)

    assert summary.new_hours == 0
    assert summary.gold_rows is None


def test_failed_fetch_changes_nothing(project: Path, fake_api: FakeApi) -> None:
    """A failed request leaves storage unchanged."""
    settings = load_settings(project)
    store = build_store(settings.storage)
    keys = store.list_keys()
    fake_api.failures = [400]

    with pytest.raises(DataSourceError, match="400"):
        update_source(settings, SourceKind.RECENT)

    assert store.list_keys() == keys
    assert not store.exists(checkpoint_key(settings.active_city.slug))


def test_update_command_reports_result(
    project: Path, fake_api: FakeApi, capsys: pytest.CaptureFixture[str]
) -> None:
    """The update command prints what changed."""
    exit_code = main(["data", "update", "--source", "recent", "--config-dir", str(project)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Source:    recent" in output
    assert "New hours: 4" in output
    assert "Gold:      7 rows" in output


def test_update_command_reports_failure(
    project: Path, fake_api: FakeApi, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed update exits with a clean error."""
    fake_api.failures = [400]

    exit_code = main(["data", "update", "--source", "archive", "--config-dir", str(project)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: Weather API returned 400" in captured.err
    assert "New hours" not in captured.out


def test_core_cli_does_not_import_prefect() -> None:
    """Core commands work without the optional extra."""
    code = "import sys, skycast.cli; raise SystemExit('prefect' in sys.modules)"

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_schedule_without_prefect_reports_install_step(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing extra gives a clear instruction."""
    monkeypatch.setitem(sys.modules, "prefect", None)
    monkeypatch.delitem(sys.modules, "skycast.ingestion.flows", raising=False)

    exit_code = main(["data", "schedule", "--config-dir", str(project)])

    assert exit_code == 1
    assert "uv sync --extra orchestration" in capsys.readouterr().err
