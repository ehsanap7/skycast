"""Fetch and check hourly weather from the Open-Meteo APIs."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx
import pandas as pd

from skycast.config import City, IngestionSettings
from skycast.data.contracts import (
    API_TIME_FORMAT,
    API_TIMEZONE,
    API_WIND_SPEED_LABEL,
    API_WIND_SPEED_UNIT,
    BRONZE_COLUMNS,
    MEASUREMENT_RANGES,
    TIME_COLUMN,
)
from skycast.errors import DataSourceError

WIND_SPEED_COLUMN = "wind_speed_10m"


@dataclass(frozen=True, slots=True)
class WeatherResponse:
    """The exact API response and its checked hourly rows."""

    raw: bytes
    hourly: pd.DataFrame


def _error_reason(response: httpx.Response) -> str:
    """Extract a short reason from a failed response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    return str(payload.get("reason", "unknown")) if isinstance(payload, dict) else "unknown"


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


def _parse_response(raw: bytes) -> WeatherResponse:
    """Check a response body and build its hourly rows."""
    payload = _parse_payload(raw)
    _require_expected_units(payload)
    hourly = _hourly_block(payload)
    columns = {column: _parse_values(hourly[column], column) for column in MEASUREMENT_RANGES}
    frame = pd.DataFrame({TIME_COLUMN: _parse_times(hourly[TIME_COLUMN]), **columns})
    return WeatherResponse(raw=raw, hourly=frame[list(BRONZE_COLUMNS)])


def _base_params(city: City) -> dict[str, str | float]:
    """Build the parameters shared by both APIs."""
    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "hourly": ",".join(MEASUREMENT_RANGES),
        "wind_speed_unit": API_WIND_SPEED_UNIT,
        "timezone": API_TIMEZONE,
    }


class WeatherClient:
    """Requests hourly weather for a city."""

    def __init__(self, settings: IngestionSettings, http_client: httpx.Client) -> None:
        """Create a client using a shared HTTP connection."""
        self._settings = settings
        self._http = http_client

    def fetch_archive(self, city: City, start: date, end: date) -> WeatherResponse:
        """Fetch archive hours for an inclusive date range."""
        if start > end:
            raise DataSourceError(f"Invalid archive range: {start} to {end}")
        params = {
            **_base_params(city),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "models": self._settings.archive_model,
        }
        return _parse_response(self._get(self._settings.archive_url, params))

    def fetch_recent(self, city: City, past_days: int) -> WeatherResponse:
        """Fetch recent hours looking back a number of days."""
        if not 0 <= past_days <= self._settings.max_past_days:
            raise DataSourceError(
                f"past_days must be between 0 and {self._settings.max_past_days}: {past_days}"
            )
        params = {
            **_base_params(city),
            "past_days": past_days,
            "forecast_days": self._settings.recent_forecast_days,
        }
        return _parse_response(self._get(self._settings.recent_url, params))

    def _get(self, url: str, params: Mapping[str, str | float]) -> bytes:
        """Send one request and return its successful body."""
        try:
            response = self._http.get(url, params=dict(params))
        except httpx.HTTPError as err:
            raise DataSourceError(f"Weather API request failed: {err}") from err
        if response.is_error:
            reason = _error_reason(response)
            raise DataSourceError(f"Weather API returned {response.status_code}: {reason}")
        return response.content
