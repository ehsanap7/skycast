"""Check raw weather API responses and turn them into hourly rows."""

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from skycast.data.contracts import (
    API_TIME_FORMAT,
    API_WIND_SPEED_LABEL,
    BRONZE_COLUMNS,
    MEASUREMENT_RANGES,
    TIME_COLUMN,
)
from skycast.errors import DataSourceError

WIND_SPEED_COLUMN = "wind_speed_10m"


def _parse_payload(raw: bytes) -> Mapping[str, Any]:
    """Decode a response body that must be a JSON object."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as err:
        raise DataSourceError(f"Weather API returned invalid JSON: {err}") from err
    if not isinstance(payload, dict):
        raise DataSourceError("Weather API response must be a JSON object")
    return payload


def _require_expected_units(payload: Mapping[str, Any]) -> None:
    """Check that the response uses UTC and the contract wind unit."""
    if payload.get("utc_offset_seconds") != 0:
        raise DataSourceError("Weather API response is not in UTC")
    units = payload.get("hourly_units")
    if not isinstance(units, dict) or units.get(WIND_SPEED_COLUMN) != API_WIND_SPEED_LABEL:
        raise DataSourceError(f"Weather API wind speed must be in {API_WIND_SPEED_LABEL}")


def _hourly_block(payload: Mapping[str, Any]) -> Mapping[str, list[Any]]:
    """Return hourly data with every column at the same length."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise DataSourceError("Weather API response has no hourly data")
    missing = [column for column in BRONZE_COLUMNS if not isinstance(hourly.get(column), list)]
    if missing:
        raise DataSourceError(f"Weather API response is missing: {', '.join(missing)}")
    lengths = {len(hourly[column]) for column in BRONZE_COLUMNS}
    if len(lengths) != 1 or 0 in lengths:
        raise DataSourceError("Weather API hourly columns are empty or uneven")
    return hourly


def _parse_times(values: list[Any]) -> pd.Series:
    """Convert API timestamps to UTC."""
    times = pd.to_datetime(
        pd.Series(values, dtype="object"), format=API_TIME_FORMAT, utc=True, errors="coerce"
    )
    if times.isna().any():
        raise DataSourceError(f"Weather API returned {times.isna().sum()} invalid timestamps")
    return times


def _parse_values(values: list[Any], column: str) -> pd.Series:
    """Convert API values to numbers, keeping nulls."""
    original = pd.Series(values, dtype="object")
    numbers = pd.to_numeric(original, errors="coerce").astype("float64")
    if (numbers.isna() & original.notna()).any():
        raise DataSourceError(f"Weather API returned non-numeric values in {column}")
    return numbers


def parse_weather_response(raw: bytes) -> pd.DataFrame:
    """Check a response body and build its hourly rows."""
    payload = _parse_payload(raw)
    _require_expected_units(payload)
    hourly = _hourly_block(payload)
    columns = {column: _parse_values(hourly[column], column) for column in MEASUREMENT_RANGES}
    frame = pd.DataFrame({TIME_COLUMN: _parse_times(hourly[TIME_COLUMN]), **columns})
    return frame[list(BRONZE_COLUMNS)]
