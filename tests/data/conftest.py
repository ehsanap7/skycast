"""Shared fixtures for data tests."""

from collections.abc import Callable

import pandas as pd
import pytest

from skycast.data.contracts import SILVER_COLUMNS


def _silver_frame(start: str, periods: int, source: str = "archive-source") -> pd.DataFrame:
    """Build a constant hourly silver frame."""
    times = pd.date_range(start, periods=periods, freq="h", tz="UTC", name="time")
    frame = pd.DataFrame(
        {
            "city": "sample",
            "source": source,
            "time": times,
            "temperature_2m": 10.0,
            "relative_humidity_2m": 50.0,
            "dew_point_2m": 5.0,
            "precipitation": 0.0,
            "pressure_msl": 1010.0,
            "cloud_cover": 50.0,
            "wind_speed_10m": 3.0,
            "wind_direction_10m": 90.0,
        }
    )
    return frame[list(SILVER_COLUMNS)]


@pytest.fixture
def make_silver() -> Callable[..., pd.DataFrame]:
    """Provide a silver frame factory."""
    return _silver_frame
