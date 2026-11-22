"""Summarize hourly weather into complete city-local days."""

import numpy as np
import pandas as pd

from skycast.data.contracts import DATE_COLUMN, INTERPOLATED_COLUMNS, TIME_COLUMN

WIND_SIN = "wind_direction_sin"
WIND_COS = "wind_direction_cos"

DAILY_STATS: dict[str, tuple[str, str]] = {
    "temp_max": ("temperature_2m", "max"),
    "temp_min": ("temperature_2m", "min"),
    "temp_mean": ("temperature_2m", "mean"),
    "humidity_mean": ("relative_humidity_2m", "mean"),
    "dew_point_mean": ("dew_point_2m", "mean"),
    "pressure_mean": ("pressure_msl", "mean"),
    "cloud_cover_mean": ("cloud_cover", "mean"),
    "wind_speed_mean": ("wind_speed_10m", "mean"),
    "wind_dir_sin": (WIND_SIN, "mean"),
    "wind_dir_cos": (WIND_COS, "mean"),
}


def fill_short_gaps(values: pd.Series, limit: int) -> pd.Series:
    """Interpolate only gaps no longer than the limit."""
    missing = values.isna()
    runs = missing.ne(missing.shift()).cumsum()
    run_lengths = missing.groupby(runs).transform("sum")
    fillable = missing & run_lengths.le(limit)
    return values.where(~fillable, values.interpolate(limit_area="inside"))


def local_days(times: pd.Series, timezone: str) -> pd.Series:
    """Convert UTC times to their city-local calendar day."""
    return times.dt.tz_convert(timezone).dt.tz_localize(None).dt.normalize()


def _localize_midnights(days: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    """Place calendar days at local midnight, even around clock changes."""
    earliest = np.ones(len(days), dtype=bool)
    return days.tz_localize(timezone, ambiguous=earliest, nonexistent="shift_forward")


def expected_hours(days: pd.DatetimeIndex, timezone: str) -> pd.Series:
    """Count the real hours in each local day."""
    starts = _localize_midnights(days, timezone)
    ends = _localize_midnights(days + pd.Timedelta(days=1), timezone)
    return pd.Series((ends - starts) / pd.Timedelta(hours=1), index=days)


def _add_wind_components(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add circular wind direction components."""
    radians = np.deg2rad(hourly["wind_direction_10m"])
    return hourly.assign(**{WIND_SIN: np.sin(radians), WIND_COS: np.cos(radians)})


def _fill_hourly(hourly: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Fill short gaps in smooth measurements."""
    filled = {column: fill_short_gaps(hourly[column], limit) for column in INTERPOLATED_COLUMNS}
    return hourly.assign(**filled)


def _daily_stat(values: pd.Series, days: pd.Series, expected: pd.Series, stat: str) -> pd.Series:
    """Aggregate one measurement, keeping only complete days."""
    grouped = values.groupby(days)
    result = grouped.agg(stat).reindex(expected.index)
    counts = grouped.count().reindex(expected.index, fill_value=0)
    return result.where(counts.eq(expected))


def build_daily(hourly: pd.DataFrame, timezone: str, max_fill_hours: int) -> pd.DataFrame:
    """Build one row per local calendar day."""
    filled = _add_wind_components(_fill_hourly(hourly, max_fill_hours))
    days = local_days(filled[TIME_COLUMN], timezone)
    rain_days = local_days(filled[TIME_COLUMN] - pd.Timedelta(hours=1), timezone)
    calendar = pd.date_range(days.min(), days.max(), freq="D", name=DATE_COLUMN)
    expected = expected_hours(calendar, timezone)
    columns = {
        name: _daily_stat(filled[column], days, expected, stat)
        for name, (column, stat) in DAILY_STATS.items()
    }
    columns["precip_sum"] = _daily_stat(filled["precipitation"], rain_days, expected, "sum")
    return pd.DataFrame(columns, index=calendar)
