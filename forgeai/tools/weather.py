"""
weather.py - current weather and a 3-day forecast from open-meteo.com
(free, no API key). Input: a place name, e.g. "Mumbai".

Two calls: the geocoding API turns the name into coordinates, then the
forecast API returns the weather. Network problems become ToolError.
"""

import re

import requests

from forgeai.tools.base import ToolContext, ToolError

# "Port of Kochi" / "Kochi port" / "the city of Pune" -> also try "Kochi" / "Pune"
PLACE_NOISE = re.compile(r"\b(the|port|harbou?r|city|town|of|terminal|airport|area)\b", re.IGNORECASE)


def place_candidates(place: str) -> list[str]:
    """The place as given, then without words like 'port of', then its first remaining word
    (usually the city: "Kochi port, India" -> "Kochi", not the country)."""
    candidates = [place]
    cleaned = " ".join(PLACE_NOISE.sub(" ", place).replace(",", " ").split())
    if cleaned and cleaned.lower() != place.lower():
        candidates.append(cleaned)
    if cleaned and len(cleaned.split()) > 1:
        candidates.append(cleaned.split()[0])
    return list(dict.fromkeys(candidates))

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


def choose_place(results: list[dict], name: str) -> dict:
    """
    Several places share a name ("Kochi" is in India and in Japan). Prefer exact
    name matches, then the largest population - a better guess than the
    geocoder's first result, though still a guess; the tool output names the
    country so the answer shows which place was used.
    """
    exact = [r for r in results if str(r.get("name", "")).lower() == name.lower()] or results
    return max(exact, key=lambda r: r.get("population") or 0)


def weather_api(tool_input: str, context: ToolContext) -> str:
    place = tool_input.strip()
    if not place:
        raise ToolError("give a place name, e.g. 'Mumbai'")

    # "Kochi, India" -> name "Kochi", country "India"
    country = place.split(",", 1)[1].strip().lower() if "," in place else ""
    loc = None
    for candidate in place_candidates(place.split(",", 1)[0].strip()):
        found = _get_json(context, GEOCODE_URL, {"name": candidate, "count": 10}).get("results") or []
        if country:
            found = [r for r in found if country in str(r.get("country", "")).lower()] or found
        if found:
            loc = choose_place(found, candidate)
            break
    if loc is None:
        raise ToolError(f"place not found: {place}")

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
