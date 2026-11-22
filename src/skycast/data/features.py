"""Build past-only features and future targets from daily weather."""

import numpy as np
import pandas as pd

LAGGED_COLUMNS: tuple[str, ...] = ("temp_max", "temp_min")
ROLLING_MEAN_COLUMNS: tuple[str, ...] = ("temp_mean", "precip_sum", "pressure_mean")
ROLLING_STD_COLUMN = "temp_mean"
TARGET_COLUMNS: tuple[str, ...] = ("temp_max", "temp_min")


def _lag_features(daily: pd.DataFrame, lag_days: tuple[int, ...]) -> dict[str, pd.Series]:
    """Build values from earlier days."""
    return {
        f"{column}_lag_{lag}": daily[column].shift(lag)
        for column in LAGGED_COLUMNS
        for lag in lag_days
    }


def _rolling_features(daily: pd.DataFrame, windows: tuple[int, ...]) -> dict[str, pd.Series]:
    """Build trailing window statistics ending on each day."""
    features: dict[str, pd.Series] = {}
    for window in windows:
        for column in ROLLING_MEAN_COLUMNS:
            features[f"{column}_roll_{window}_mean"] = daily[column].rolling(window).mean()
        rolling = daily[ROLLING_STD_COLUMN].rolling(window)
        features[f"{ROLLING_STD_COLUMN}_roll_{window}_std"] = rolling.std()
    return features


def _calendar_features(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Encode the time of year as a repeating cycle."""
    year_length = 365 + index.is_leap_year.astype(int)
    angle = 2 * np.pi * (index.day_of_year - 1) / year_length
    return {
        "day_of_year_sin": pd.Series(np.sin(angle), index=index),
        "day_of_year_cos": pd.Series(np.cos(angle), index=index),
    }


def build_features(
    daily: pd.DataFrame, lag_days: tuple[int, ...], rolling_days: tuple[int, ...]
) -> pd.DataFrame:
    """Combine everything known by the end of each day."""
    columns = {column: daily[column] for column in daily.columns}
    columns |= _lag_features(daily, lag_days)
    columns |= _rolling_features(daily, rolling_days)
    columns |= _calendar_features(daily.index)
    return pd.DataFrame(columns, index=daily.index)


def build_targets(daily: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Build the future daily values to predict."""
    columns = {
        f"{column}_d{day}": daily[column].shift(-day)
        for column in TARGET_COLUMNS
        for day in range(1, horizon_days + 1)
    }
    return pd.DataFrame(columns, index=daily.index)
