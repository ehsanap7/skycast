"""Fetch hourly weather from the Open-Meteo APIs."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import httpx
import pandas as pd

from skycast.config import City, IngestionSettings
from skycast.data.api_response import parse_weather_response
from skycast.data.contracts import API_TIMEZONE, API_WIND_SPEED_UNIT, MEASUREMENT_RANGES
from skycast.errors import DataSourceError


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


def _parse_response(raw: bytes) -> WeatherResponse:
    """Pair a response body with its checked hourly rows."""
    return WeatherResponse(raw=raw, hourly=parse_weather_response(raw))


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
