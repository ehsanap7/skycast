"""Tests for local daily aggregation."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from skycast.data.daily import build_daily, expected_hours, fill_short_gaps

TIMEZONE = "America/New_York"
SilverFactory = Callable[..., pd.DataFrame]


def at(frame: pd.DataFrame, time: str) -> pd.Series:
    """Select the row for a UTC time."""
    return frame["time"].eq(pd.Timestamp(time, tz="UTC"))


def test_fill_short_gap_only() -> None:
    """Short inside gaps fill, long and edge gaps stay null."""
    values = pd.Series([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan])

    filled = fill_short_gaps(values, limit=2)

    assert filled[2] == 2.0
    assert filled[[0, 4, 5, 6, 8]].isna().all()


def test_zero_limit_fills_nothing() -> None:
    """A zero limit disables filling."""
    values = pd.Series([1.0, np.nan, 3.0])

    assert pd.isna(fill_short_gaps(values, limit=0)[1])


@pytest.mark.parametrize(
    ("day", "hours"),
    [("2024-03-10", 23), ("2024-11-03", 25), ("2024-06-01", 24)],
)
def test_expected_hours_follow_clock_changes(day: str, hours: int) -> None:
    """Local days can have 23, 24, or 25 hours."""
    days = pd.DatetimeIndex([day])

    assert expected_hours(days, TIMEZONE).iloc[0] == hours


@pytest.mark.parametrize(
    ("start", "periods", "day"),
    [("2024-03-10 05:00", 23, "2024-03-10"), ("2024-11-03 04:00", 25, "2024-11-03")],
)
def test_clock_change_day_is_complete(
    make_silver: SilverFactory, start: str, periods: int, day: str
) -> None:
    """Short and long days count as complete."""
    daily = build_daily(make_silver(start, periods), TIMEZONE, 3)

    assert daily.loc[day, "temp_max"] == 10.0


def test_days_use_local_time(make_silver: SilverFactory) -> None:
    """A late local hour belongs to the local day."""
    frame = make_silver("2024-01-01 05:00", 48)
    frame.loc[at(frame, "2024-01-02 03:00"), "temperature_2m"] = 30.0

    daily = build_daily(frame, TIMEZONE, 3)

    assert daily.loc["2024-01-01", "temp_max"] == 30.0
    assert daily.loc["2024-01-02", "temp_max"] == 10.0


def test_precipitation_belongs_to_previous_hour(make_silver: SilverFactory) -> None:
    """Rain stamped at local midnight belongs to the previous day."""
    frame = make_silver("2024-01-01 05:00", 49)
    frame.loc[at(frame, "2024-01-02 05:00"), "precipitation"] = 2.0

    daily = build_daily(frame, TIMEZONE, 3)

    assert daily.loc["2024-01-01", "precip_sum"] == 2.0
    assert daily.loc["2024-01-02", "precip_sum"] == 0.0


def test_partial_day_is_null(make_silver: SilverFactory) -> None:
    """A day without every hour has no daily value."""
    daily = build_daily(make_silver("2024-01-01 05:00", 30), TIMEZONE, 3)

    assert daily.loc["2024-01-01", "temp_max"] == 10.0
    assert pd.isna(daily.loc["2024-01-02", "temp_max"])


def test_long_gap_makes_day_null(make_silver: SilverFactory) -> None:
    """A gap over the limit is not filled."""
    frame = make_silver("2024-01-01 05:00", 24)
    frame.loc[5:9, "temperature_2m"] = np.nan

    daily = build_daily(frame, TIMEZONE, 3)

    assert pd.isna(daily.loc["2024-01-01", "temp_max"])
    assert daily.loc["2024-01-01", "pressure_mean"] == 1010.0


def test_short_gap_keeps_day(make_silver: SilverFactory) -> None:
    """A short temperature gap is filled."""
    frame = make_silver("2024-01-01 05:00", 24)
    frame.loc[5:7, "temperature_2m"] = np.nan

    daily = build_daily(frame, TIMEZONE, 3)

    assert daily.loc["2024-01-01", "temp_mean"] == 10.0


def test_precipitation_is_never_filled(make_silver: SilverFactory) -> None:
    """A missing rain hour makes the daily total null."""
    frame = make_silver("2024-01-01 05:00", 49)
    frame.loc[at(frame, "2024-01-01 12:00"), "precipitation"] = np.nan

    daily = build_daily(frame, TIMEZONE, 3)

    assert pd.isna(daily.loc["2024-01-01", "precip_sum"])


def test_wind_direction_is_circular(make_silver: SilverFactory) -> None:
    """Directions around north average near north."""
    frame = make_silver("2024-01-01 05:00", 24)
    frame["wind_direction_10m"] = [359.0, 1.0] * 12

    daily = build_daily(frame, TIMEZONE, 3)

    assert daily.loc["2024-01-01", "wind_dir_sin"] == pytest.approx(0.0, abs=1e-9)
    assert daily.loc["2024-01-01", "wind_dir_cos"] == pytest.approx(1.0, abs=1e-3)
