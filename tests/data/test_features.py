"""Tests for feature and target building."""

import numpy as np
import pandas as pd
import pytest

from skycast.data.features import build_features, build_targets


def daily_frame(start: str, periods: int) -> pd.DataFrame:
    """Build daily data with increasing values."""
    index = pd.date_range(start, periods=periods, freq="D", name="date")
    values = np.arange(periods, dtype=float)
    columns = ("temp_max", "temp_min", "temp_mean", "precip_sum", "pressure_mean")
    return pd.DataFrame({column: values for column in columns}, index=index)


def test_season_is_continuous_across_new_year() -> None:
    """December 31 sits close to January 1."""
    features = build_features(daily_frame("2023-12-31", 2), (1,), (2,))

    gap = features["day_of_year_sin"].diff().abs().iloc[1]

    assert gap < 0.02
    assert features.loc["2024-01-01", "day_of_year_cos"] == pytest.approx(1.0)


def test_season_handles_leap_year() -> None:
    """The middle of a leap year is opposite January 1."""
    features = build_features(daily_frame("2024-07-02", 1), (1,), (2,))

    assert features["day_of_year_cos"].iloc[0] == pytest.approx(-1.0)


def test_rolling_window_includes_only_past_and_today() -> None:
    """A rolling mean ends on the current day."""
    features = build_features(daily_frame("2024-01-01", 5), (1,), (3,))

    assert pd.isna(features["temp_mean_roll_3_mean"].iloc[1])
    assert features["temp_mean_roll_3_mean"].iloc[2] == 1.0
    assert features["temp_mean_roll_3_mean"].iloc[4] == 3.0


def test_targets_look_forward() -> None:
    """Targets come from future days and end with nulls."""
    targets = build_targets(daily_frame("2024-01-01", 4), 2)

    assert targets["temp_max_d1"].iloc[0] == 1.0
    assert targets["temp_min_d2"].iloc[0] == 2.0
    assert pd.isna(targets["temp_max_d2"].iloc[2])
