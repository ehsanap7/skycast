"""Tests for the silver dataset."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from skycast.config import City
from skycast.data.contracts import BRONZE_COLUMNS, SILVER_COLUMNS
from skycast.data.silver import build_silver, write_silver
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
    archive_cutoff=datetime(2010, 1, 1, 12, tzinfo=UTC),
)
VALUES = ("1.5", "90", "0.5", "0.0", "1013.5", "80", "2.5", "215")


def bronze_row(hour: int, **overrides: str) -> dict[str, str]:
    """Build one raw row for an hour."""
    row = dict(zip(BRONZE_COLUMNS[1:], VALUES, strict=True))
    return {"time": f"2010-01-01 {hour:02d}:00:00+00:00", **row, **overrides}


def bronze_frame(*rows: dict[str, str]) -> pd.DataFrame:
    """Build a raw text frame."""
    return pd.DataFrame(list(rows), columns=list(BRONZE_COLUMNS), dtype="str")


def test_builds_typed_contract_columns() -> None:
    """Silver has contract columns and real types."""
    frame = build_silver(bronze_frame(bronze_row(0)), CITY).frame

    assert tuple(frame.columns) == SILVER_COLUMNS
    assert str(frame["time"].dt.tz) == "UTC"
    assert frame.loc[0, "temperature_2m"] == 1.5
    assert frame.loc[0, "city"] == "sample"
    assert frame.loc[0, "source"] == "archive-source"


def test_source_changes_after_archive_cutoff() -> None:
    """The cutoff hour is archive and the next hour is recent."""
    city = replace(CITY, archive_cutoff=datetime(2010, 1, 1, 1, tzinfo=UTC))

    silver = build_silver(bronze_frame(bronze_row(0), bronze_row(1), bronze_row(2)), city)

    assert silver.frame["source"].tolist() == [
        "archive-source",
        "archive-source",
        "recent-source",
    ]
    assert silver.report.rows_by_source == {"archive-source": 2, "recent-source": 1}


def test_blank_value_becomes_null() -> None:
    """Blank measurements stay missing."""
    silver = build_silver(bronze_frame(bronze_row(0, pressure_msl="")), CITY)

    assert pd.isna(silver.frame.loc[0, "pressure_msl"])
    assert silver.report.null_values["pressure_msl"] == 1
    assert silver.report.out_of_range["pressure_msl"] == 0


def test_non_numeric_value_raises() -> None:
    """Broken text stops the build."""
    with pytest.raises(DataSourceError, match="cloud_cover"):
        build_silver(bronze_frame(bronze_row(0, cloud_cover="abc")), CITY)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("relative_humidity_2m", "101"),
        ("precipitation", "-0.1"),
        ("wind_speed_10m", "-1"),
        ("wind_direction_10m", "361"),
        ("temperature_2m", "inf"),
    ],
)
def test_out_of_range_value_becomes_null(column: str, value: str) -> None:
    """Impossible values are removed and counted."""
    silver = build_silver(bronze_frame(bronze_row(0, **{column: value})), CITY)

    assert pd.isna(silver.frame.loc[0, column])
    assert silver.report.out_of_range[column] == 1


@pytest.mark.parametrize(
    ("column", "value"),
    [("relative_humidity_2m", "100"), ("cloud_cover", "0"), ("wind_direction_10m", "360")],
)
def test_range_limits_are_valid(column: str, value: str) -> None:
    """Values on a range limit are kept."""
    silver = build_silver(bronze_frame(bronze_row(0, **{column: value})), CITY)

    assert silver.frame.loc[0, column] == float(value)


def test_duplicate_keeps_last_row() -> None:
    """The newest row wins for a repeated timestamp."""
    bronze = bronze_frame(bronze_row(0, temperature_2m="1"), bronze_row(0, temperature_2m="2"))

    silver = build_silver(bronze, CITY)

    assert len(silver.frame) == 1
    assert silver.frame.loc[0, "temperature_2m"] == 2.0
    assert silver.report.duplicates_removed == 1


def test_missing_hour_becomes_null_row() -> None:
    """A gap appears as an explicit null row."""
    silver = build_silver(bronze_frame(bronze_row(0), bronze_row(2)), CITY)

    assert len(silver.frame) == 3
    assert silver.frame.loc[1, "time"] == pd.Timestamp("2010-01-01 01:00", tz="UTC")
    assert silver.frame.loc[1, "city"] == "sample"
    assert pd.isna(silver.frame.loc[1, "temperature_2m"])
    assert silver.report.missing_hours == 1


def test_unsorted_rows_are_sorted() -> None:
    """Silver is ordered by time."""
    frame = build_silver(bronze_frame(bronze_row(2), bronze_row(1)), CITY).frame

    assert frame["time"].is_monotonic_increasing


def test_partial_hour_raises() -> None:
    """Timestamps must fall on whole hours."""
    row = bronze_row(0, time="2010-01-01 00:30:00+00:00")

    with pytest.raises(DataSourceError, match="whole hour"):
        build_silver(bronze_frame(row), CITY)


def test_input_frame_is_not_changed() -> None:
    """Building silver leaves bronze data untouched."""
    bronze = bronze_frame(bronze_row(0, cloud_cover="101"))
    original = bronze.copy(deep=True)

    build_silver(bronze, CITY)

    pd.testing.assert_frame_equal(bronze, original)


def test_rebuild_is_identical() -> None:
    """The same input gives the same output."""
    bronze = bronze_frame(bronze_row(0), bronze_row(2))

    first = build_silver(bronze, CITY)
    second = build_silver(bronze, CITY)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.report == second.report


def test_write_round_trip(tmp_path: Path) -> None:
    """Stored Parquet and report match the built dataset."""
    store = LocalStore(tmp_path)
    silver = build_silver(bronze_frame(bronze_row(0), bronze_row(2)), CITY)

    write_silver(store, CITY, silver)

    parquet = store.read_bytes("silver/city=sample/hourly.parquet")
    report = json.loads(store.read_bytes("silver/city=sample/quality_report.json"))
    stored = pd.read_parquet(BytesIO(parquet))
    pd.testing.assert_frame_equal(stored, silver.frame, check_dtype=False)
    assert str(stored["time"].dt.tz) == "UTC"
    assert report["rows"] == 3
    assert report["missing_hours"] == 1
