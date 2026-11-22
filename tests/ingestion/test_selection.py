"""Tests for choosing which hours to request and accept."""

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from skycast.errors import DataSourceError
from skycast.ingestion.selection import archive_dates, recent_past_days, select_new_hours

AFTER = pd.Timestamp("2024-01-01 02:00", tz="UTC")
NOW = pd.Timestamp("2024-01-01 06:10", tz="UTC")
HourlyFactory = Callable[[str, int], pd.DataFrame]


def hours(frame: pd.DataFrame) -> list[int]:
    """Return the selected UTC hours."""
    return frame["time"].dt.hour.tolist()


def test_selects_only_new_finished_hours(make_hourly: HourlyFactory) -> None:
    """Old and future hours are excluded."""
    selected = select_new_hours(make_hourly("2024-01-01 00:00", 12), AFTER, NOW)

    assert hours(selected) == [3, 4, 5, 6]


def test_no_new_hours_is_empty(make_hourly: HourlyFactory) -> None:
    """A response with nothing new selects no rows."""
    assert select_new_hours(make_hourly("2024-01-01 00:00", 3), AFTER, NOW).empty


def test_only_future_hours_is_empty(make_hourly: HourlyFactory) -> None:
    """Future forecast hours are never accepted."""
    assert select_new_hours(make_hourly("2024-01-01 07:00", 3), AFTER, NOW).empty


def test_stops_before_null_hour(make_hourly: HourlyFactory) -> None:
    """A null value ends the accepted run."""
    frame = make_hourly("2024-01-01 00:00", 12)
    frame.loc[frame["time"].dt.hour.eq(5), "pressure_msl"] = np.nan

    assert hours(select_new_hours(frame, AFTER, NOW)) == [3, 4]


def test_first_hour_null_selects_nothing(make_hourly: HourlyFactory) -> None:
    """The checkpoint cannot jump over a null first hour."""
    frame = make_hourly("2024-01-01 00:00", 12)
    frame.loc[frame["time"].dt.hour.eq(3), "cloud_cover"] = np.nan

    assert select_new_hours(frame, AFTER, NOW).empty


def test_stops_before_missing_hour(make_hourly: HourlyFactory) -> None:
    """A missing row ends the accepted run."""
    frame = make_hourly("2024-01-01 00:00", 12)
    frame = frame.loc[frame["time"].dt.hour.ne(5)]

    assert hours(select_new_hours(frame, AFTER, NOW)) == [3, 4]


def test_gap_right_after_checkpoint_selects_nothing(make_hourly: HourlyFactory) -> None:
    """Rows that do not touch the checkpoint are not accepted."""
    assert select_new_hours(make_hourly("2024-01-01 04:00", 3), AFTER, NOW).empty


def test_stops_at_duplicate_hour(make_hourly: HourlyFactory) -> None:
    """A repeated hour is not treated as continuous data."""
    frame = make_hourly("2024-01-01 03:00", 3)
    duplicated = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)

    assert hours(select_new_hours(duplicated, AFTER, NOW)) == [3, 4]


def test_unsorted_response_is_sorted(make_hourly: HourlyFactory) -> None:
    """Accepted rows are ordered by time."""
    frame = make_hourly("2024-01-01 03:00", 3).iloc[::-1]

    assert hours(select_new_hours(frame, AFTER, NOW)) == [3, 4, 5]


def test_recent_past_days_adds_safety_day() -> None:
    """The lookback covers the gap plus one day."""
    last = pd.Timestamp("2024-01-01 23:00", tz="UTC")
    now = pd.Timestamp("2024-01-04 02:00", tz="UTC")

    assert recent_past_days(last, now, 92) == 4


def test_recent_past_days_same_day() -> None:
    """A same-day update still has one safety day."""
    assert recent_past_days(AFTER, NOW, 92) == 1


def test_recent_past_days_at_limit() -> None:
    """A lookback exactly at the limit is allowed."""
    now = AFTER + pd.Timedelta(days=91)

    assert recent_past_days(AFTER, now, 92) == 92


def test_recent_gap_over_limit_raises() -> None:
    """A gap beyond the API limit stops the run."""
    now = AFTER + pd.Timedelta(days=92)

    with pytest.raises(DataSourceError, match="API limit"):
        recent_past_days(AFTER, now, 92)


def test_archive_dates_cover_checkpoint_to_today() -> None:
    """The archive request starts on the checkpoint date."""
    now = pd.Timestamp("2024-01-05 03:00", tz="UTC")

    assert archive_dates(AFTER, now) == (date(2024, 1, 1), date(2024, 1, 5))
