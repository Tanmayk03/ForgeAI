"""
weather.py - current weather and a 3-day forecast from open-meteo.com
(free, no API key). Input: a place name, e.g. "Mumbai".

Two calls: the geocoding API turns the name into coordinates, then the
forecast API returns the weather. Network problems become ToolError.
"""

import requests

from forgeai.tools.base import ToolContext, ToolError

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 15


def _get_json(context: ToolContext, url: str, params: dict) -> dict:
    try:
        response = context.http_get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise ToolError(f"weather service unavailable: {exc}") from exc
    except ValueError as exc:
        raise ToolError("weather service returned invalid data") from exc


def weather_api(tool_input: str, context: ToolContext) -> str:
    place = tool_input.strip()
    if not place:
        raise ToolError("give a place name, e.g. 'Mumbai'")

    found = _get_json(context, GEOCODE_URL, {"name": place, "count": 1}).get("results") or []
    if not found:
        raise ToolError(f"place not found: {place}")
    loc = found[0]

    data = _get_json(context, FORECAST_URL, {
        "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
        "current": "temperature_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 3,
    })
    try:
        now, daily = data["current"], data["daily"]
        days = [
            f"{d}: max {hi} C, min {lo} C, rain {rain} mm"
            for d, hi, lo, rain in zip(daily["time"], daily["temperature_2m_max"],
                                       daily["temperature_2m_min"], daily["precipitation_sum"])
        ]
    except (KeyError, TypeError) as exc:
        raise ToolError("weather service returned unexpected data") from exc
    name = ", ".join(p for p in (loc.get("name"), loc.get("country")) if p)
    return (f"Weather for {name}: now {now['temperature_2m']} C, wind {now['wind_speed_10m']} km/h. "
            f"Forecast: " + "; ".join(days) + ".")
