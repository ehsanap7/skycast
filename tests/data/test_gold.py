"""Tests for the gold dataset."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from skycast.config import City, GoldSettings
from skycast.data.gold import build_gold, write_gold
from skycast.data.storage import LocalStore
from skycast.errors import DataSourceError

CITY = City(
    slug="sample",
    name="Sample",
    latitude=1.0,
    longitude=2.0,
    timezone="UTC",
    source_file="sample.csv",
    archive_source="archive-source",
    recent_source="recent-source",
    archive_cutoff=datetime(2024, 1, 10, 23, tzinfo=UTC),
)
SETTINGS = GoldSettings(max_fill_hours=3, lag_days=(1, 2), rolling_days=(2, 3))
HORIZON = 2
SilverFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def silver(make_silver: SilverFactory) -> pd.DataFrame:
    """Build ten days where each day has a known high and low."""
    frame = make_silver("2024-01-01 00:00", 240)
    day_number = (frame["time"] - frame["time"].iloc[0]).dt.days
    frame["temperature_2m"] = day_number + frame["time"].dt.hour / 100
    return frame


def row(frame: pd.DataFrame, date: str) -> pd.Series:
    """Select the gold row for a date."""
    return frame.loc[frame["date"].eq(pd.Timestamp(date))].iloc[0]


def test_keeps_only_complete_rows(silver: pd.DataFrame) -> None:
    """Warm-up rows and rows without future targets are dropped."""
    frame = build_gold(silver, CITY, SETTINGS, HORIZON).frame

    assert frame["date"].iloc[0] == pd.Timestamp("2024-01-03")
    assert frame["date"].iloc[-1] == pd.Timestamp("2024-01-08")
    assert len(frame) == 6
    assert not frame.isna().any().any()


def test_targets_match_future_days(silver: pd.DataFrame) -> None:
    """Targets are the real future highs and lows."""
    gold_row = row(build_gold(silver, CITY, SETTINGS, HORIZON).frame, "2024-01-04")

    assert gold_row["temp_max_d1"] == pytest.approx(4.23)
    assert gold_row["temp_min_d1"] == pytest.approx(4.0)
    assert gold_row["temp_max_d2"] == pytest.approx(5.23)
    assert gold_row["temp_min_d2"] == pytest.approx(5.0)


def test_lags_and_rolling_use_past_days(silver: pd.DataFrame) -> None:
    """Lags look back and rolling windows end on the row's day."""
    gold_row = row(build_gold(silver, CITY, SETTINGS, HORIZON).frame, "2024-01-04")

    assert gold_row["temp_max"] == pytest.approx(3.23)
    assert gold_row["temp_max_lag_1"] == pytest.approx(2.23)
    assert gold_row["temp_max_lag_2"] == pytest.approx(1.23)
    assert gold_row["temp_mean_roll_3_mean"] == pytest.approx(2.115)


def test_future_changes_do_not_change_features(silver: pd.DataFrame) -> None:
    """Data after a forecast origin cannot affect its features."""
    original = build_gold(silver, CITY, SETTINGS, HORIZON)
    changed_silver = silver.copy()
    future = changed_silver["time"].gt(pd.Timestamp("2024-01-06 00:00", tz="UTC"))
    changed_silver.loc[future, ["temperature_2m", "pressure_msl", "precipitation"]] = 40.0
    changed = build_gold(changed_silver, CITY, SETTINGS, HORIZON)
    columns = list(original.feature_columns)

    before = original.frame.loc[original.frame["date"].le("2024-01-05"), columns]
    after = changed.frame.loc[changed.frame["date"].le("2024-01-05"), columns]

    pd.testing.assert_frame_equal(before, after)
    assert row(changed.frame, "2024-01-05")["temp_max_d1"] == 40.0


def test_targets_are_not_features(silver: pd.DataFrame) -> None:
    """Feature and target columns never overlap."""
    gold = build_gold(silver, CITY, SETTINGS, HORIZON)

    assert not set(gold.feature_columns) & set(gold.target_columns)
    assert len(gold.target_columns) == 2 * HORIZON


def test_recent_rows_are_excluded(silver: pd.DataFrame) -> None:
    """Forecast-source days cannot become features or targets."""
    recent = silver["time"].ge(pd.Timestamp("2024-01-10 00:00", tz="UTC"))
    silver.loc[recent, "source"] = "recent-source"

    frame = build_gold(silver, CITY, SETTINGS, HORIZON).frame

    assert frame["date"].iloc[-1] == pd.Timestamp("2024-01-07")


def test_no_archive_rows_raises(make_silver: SilverFactory) -> None:
    """Gold needs archive data."""
    silver = make_silver("2024-01-01 00:00", 240, source="recent-source")

    with pytest.raises(DataSourceError, match="No archive rows"):
        build_gold(silver, CITY, SETTINGS, HORIZON)


def test_too_little_history_raises(make_silver: SilverFactory) -> None:
    """Gold needs enough days for features and targets."""
    with pytest.raises(DataSourceError, match="No complete gold rows"):
        build_gold(make_silver("2024-01-01 00:00", 48), CITY, SETTINGS, HORIZON)


def test_rebuild_is_identical(silver: pd.DataFrame) -> None:
    """The same input gives the same rows."""
    first = build_gold(silver, CITY, SETTINGS, HORIZON).frame
    second = build_gold(silver, CITY, SETTINGS, HORIZON).frame

    pd.testing.assert_frame_equal(first, second)


def test_write_round_trip_and_metadata(tmp_path: Path, silver: pd.DataFrame) -> None:
    """Stored gold matches and metadata records its lineage."""
    store = LocalStore(tmp_path)
    store.write_bytes("bronze/city=sample/sample.csv", b"raw")
    store.write_bytes("silver/city=sample/hourly.parquet", b"silver")
    city = replace(CITY)
    gold = build_gold(silver, city, SETTINGS, HORIZON)

    write_gold(store, city, gold, SETTINGS, HORIZON)

    stored = pd.read_parquet(BytesIO(store.read_bytes("gold/city=sample/training.parquet")))
    metadata = json.loads(store.read_bytes("gold/city=sample/metadata.json"))
    pd.testing.assert_frame_equal(stored, gold.frame, check_dtype=False)
    assert metadata["contract_version"] == "1.0.0"
    assert metadata["source_sha256"] == hashlib.sha256(b"raw").hexdigest()
    assert metadata["silver_sha256"] == hashlib.sha256(b"silver").hexdigest()
    assert metadata["rows"] == 6
    assert metadata["start_date"] == "2024-01-03"
    assert metadata["end_date"] == "2024-01-08"
    assert metadata["target_columns"] == list(gold.target_columns)
