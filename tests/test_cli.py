"""Tests for the SkyCast command-line entry point."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from skycast import __version__
from skycast.cli import main
from skycast.config import load_settings
from skycast.data.contracts import BRONZE_COLUMNS, SourceKind
from skycast.data.storage import LocalStore
from skycast.ingestion.batches import save_batch

CITIES = """cities:
  - slug: sample
    name: Sample
    latitude: 1.0
    longitude: 2.0
    timezone: UTC
    source_file: sample.csv
    archive_source: archive-source
    recent_source: recent-source
    archive_cutoff: "2030-01-01T00:00:00+00:00"
"""
DATA = """active_city: sample
storage:
  backend: local
  root: data
forecast:
  horizon_days: 2
gold:
  max_fill_hours: 3
  lag_days: [1]
  rolling_days: [2]
ingestion:
  archive_url: https://archive.example.test/v1/archive
  archive_model: sample-model
  recent_url: https://recent.example.test/v1/forecast
  recent_forecast_days: 1
  max_past_days: 92
  timeout_seconds: 60
schedule:
  recent_cron: "10 */4 * * *"
  archive_cron: "40 6 * * *"
  retries: 3
  retry_base_seconds: 0
"""


def csv_row(hour: int) -> str:
    """Build one raw row with a known daily pattern."""
    time = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour)
    temperature = 5 + hour // 24 + time.hour / 100
    return f"{time:%Y-%m-%d %H:%M:%S}+00:00,{temperature},90,1.0,0.0,1010.0,50,3.0,180"


def write_project(root: Path, hours: int = 240) -> Path:
    """Create config and a sample CSV, then return the config directory."""
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "cities.yaml").write_text(CITIES, encoding="utf-8")
    (config_dir / "data.yaml").write_text(DATA, encoding="utf-8")
    bronze_dir = root / "data" / "bronze" / "city=sample"
    bronze_dir.mkdir(parents=True)
    lines = [",".join(BRONZE_COLUMNS), *(csv_row(hour) for hour in range(hours))]
    (bronze_dir / "sample.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_dir


def build_args(config_dir: Path) -> list[str]:
    """Build the data command arguments."""
    return ["data", "build", "--config-dir", str(config_dir)]


def test_main_prints_hello_world(capsys: pytest.CaptureFixture[str]) -> None:
    """``main()`` greets the user and includes the installed version."""
    exit_code = main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == f"Hello from SkyCast v{__version__}!\n"
    assert captured.err == ""


def test_version_is_set() -> None:
    """The package exposes a real version from installed metadata."""
    assert __version__
    assert __version__ != "0.0.0+unknown"


def test_data_build_creates_silver_and_gold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command builds every output from a bronze CSV."""
    exit_code = main(build_args(write_project(tmp_path)))

    output = capsys.readouterr().out
    data = tmp_path / "data"
    gold = pd.read_parquet(data / "gold" / "city=sample" / "training.parquet")
    assert exit_code == 0
    assert "Bronze: 240 rows" in output
    assert "Silver: 240 rows" in output
    assert "Gold:   7 rows" in output
    assert "Saved:  gold/city=sample/training.parquet" in output
    assert (data / "silver" / "city=sample" / "hourly.parquet").is_file()
    assert (data / "silver" / "city=sample" / "quality_report.json").is_file()
    assert (data / "gold" / "city=sample" / "metadata.json").is_file()
    assert len(gold) == 7
    assert not gold.isna().any().any()


def test_data_build_is_idempotent(tmp_path: Path) -> None:
    """Running twice gives the same datasets."""
    config_dir = write_project(tmp_path)
    gold_dir = tmp_path / "data" / "gold" / "city=sample"

    assert main(build_args(config_dir)) == 0
    first = pd.read_parquet(gold_dir / "training.parquet")
    first_metadata = json.loads((gold_dir / "metadata.json").read_text(encoding="utf-8"))
    assert main(build_args(config_dir)) == 0
    second = pd.read_parquet(gold_dir / "training.parquet")
    second_metadata = json.loads((gold_dir / "metadata.json").read_text(encoding="utf-8"))

    pd.testing.assert_frame_equal(first, second)
    assert first_metadata["source_sha256"] == second_metadata["source_sha256"]
    assert first_metadata["rows"] == second_metadata["rows"]


def test_data_build_leaves_bronze_unchanged(tmp_path: Path) -> None:
    """The source file keeps its exact bytes."""
    config_dir = write_project(tmp_path)
    source = tmp_path / "data" / "bronze" / "city=sample" / "sample.csv"
    original = source.read_bytes()

    main(build_args(config_dir))

    assert source.read_bytes() == original


def test_missing_source_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing CSV exits with an error and no success output."""
    config_dir = write_project(tmp_path)
    (tmp_path / "data" / "bronze" / "city=sample" / "sample.csv").unlink()

    exit_code = main(build_args(config_dir))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: Bronze file not found" in captured.err
    assert "Saved" not in captured.out


def test_failed_gold_stage_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A later stage failure is not reported as success."""
    exit_code = main(build_args(write_project(tmp_path, hours=24)))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No complete gold rows" in captured.err
    assert "Saved" not in captured.out
    assert not (tmp_path / "data" / "gold").exists()


def test_invalid_config_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing config directory exits cleanly."""
    exit_code = main(build_args(tmp_path / "missing"))

    assert exit_code == 1
    assert "Error: Cannot load config file" in capsys.readouterr().err


def test_unexpected_error_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unexpected failure is logged and exits non-zero."""

    def fail(_settings: object) -> None:
        """Simulate a programming error."""
        raise RuntimeError("boom")

    monkeypatch.setattr("skycast.cli.run_data_pipeline", fail)

    assert main(build_args(write_project(tmp_path))) == 1
    assert "Data build failed unexpectedly" in caplog.text


def test_damaged_batch_stops_build_and_keeps_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A damaged batch fails the build before silver or gold change."""
    config_dir = write_project(tmp_path)
    assert main(build_args(config_dir)) == 0
    data = tmp_path / "data"
    outputs = [
        data / "silver" / "city=sample" / "hourly.parquet",
        data / "gold" / "city=sample" / "training.parquet",
    ]
    before = [path.read_bytes() for path in outputs]
    store = LocalStore(data)
    city = load_settings(config_dir).active_city
    fetched_at = pd.Timestamp("2024-01-11 00:00", tz="UTC")
    selected = pd.DataFrame({"time": [fetched_at]})
    key = save_batch(store, city, SourceKind.RECENT, b"{}", selected, fetched_at)
    store.write_bytes(key, b"changed")
    capsys.readouterr()

    exit_code = main(build_args(config_dir))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "damaged" in captured.err
    assert "Saved" not in captured.out
    assert [path.read_bytes() for path in outputs] == before
