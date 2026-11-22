"""Decide what to request and which returned hours to accept."""

from datetime import date

import pandas as pd

from skycast.data.contracts import MEASUREMENT_RANGES, TIME_COLUMN
from skycast.errors import DataSourceError


def _finished_new_hours(
    hourly: pd.DataFrame, after: pd.Timestamp, now: pd.Timestamp
) -> pd.DataFrame:
    """Keep hours after the checkpoint and not in the future."""
    times = hourly[TIME_COLUMN]
    finished = times.gt(after) & times.le(now.floor("h"))
    return hourly.loc[finished].sort_values(TIME_COLUMN).reset_index(drop=True)


def _unbroken_complete(candidates: pd.DataFrame, after: pd.Timestamp) -> pd.Series:
    """Mark the run of complete hours that directly follows the checkpoint."""
    times = candidates[TIME_COLUMN]
    first = after + pd.Timedelta(hours=1)
    expected = pd.date_range(first, periods=len(times), freq="h", unit=times.dt.unit)
    on_time = times.eq(pd.Series(expected))
    complete = candidates[list(MEASUREMENT_RANGES)].notna().all(axis=1)
    return (on_time & complete).cummin().astype(bool)


def select_new_hours(hourly: pd.DataFrame, after: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """Select the unbroken complete hours that can be ingested."""
    candidates = _finished_new_hours(hourly, after, now)
    if candidates.empty:
        return candidates
    return candidates.loc[_unbroken_complete(candidates, after)]


def recent_past_days(last_hour: pd.Timestamp, now: pd.Timestamp, max_past_days: int) -> int:
    """Calculate the recent API lookback, including one safety day."""
    gap_days = max((now.date() - last_hour.date()).days, 0)
    past_days = gap_days + 1
    if past_days > max_past_days:
        raise DataSourceError(
            f"Recent data gap needs {past_days} days, but the API limit is {max_past_days}"
        )
    return past_days


def archive_dates(last_hour: pd.Timestamp, now: pd.Timestamp) -> tuple[date, date]:
    """Calculate the archive date range to request."""
    return last_hour.date(), now.date()
