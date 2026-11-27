"""Tests for the four tools and run_tool (no network: the weather API is faked)."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import requests

from forgeai import config
from forgeai.tools.base import ToolContext, ToolError
from forgeai.tools.calculator import calculator
from forgeai.tools.csv_lookup import csv_lookup
from forgeai.tools.current_date import current_date
from forgeai.tools.registry import TOOL_NAMES, run_tool
from forgeai.tools.weather import place_candidates, weather_api

CSV = config.DATA_DIR / "sample_docs" / "spare_parts.csv"


class TestCalculator(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculator("24 * 2 + 1", ToolContext()), "24 * 2 + 1 = 49")
        self.assertEqual(calculator("6,000 * 3", ToolContext()), "6000 * 3 = 18000")
        self.assertEqual(calculator("10 / 4", ToolContext()), "10 / 4 = 2.5")
        self.assertEqual(calculator("-(2 ** 3) % 5", ToolContext()), "-(2 ** 3) % 5 = 2")

    def test_rejects_code_and_bad_input(self):
        for bad in ("__import__('os').system('dir')", "open('x')", "a + 1", "2 ** 1000", "", "1 / 0", "2 +"):
            with self.subTest(bad=bad), self.assertRaises(ToolError):
                calculator(bad, ToolContext())


class TestCsvLookup(unittest.TestCase):
    def test_finds_row_with_file_and_row_number(self):
        out = csv_lookup("fuel injector", ToolContext(csv_paths=[CSV]))
        self.assertIn("spare_parts.csv row 2", out)
        self.assertIn("unit_price_usd=145.00", out)
        self.assertNotIn("SP-1002", out)

    def test_part_number_and_no_match(self):
        self.assertIn("Cylinder head gasket", csv_lookup("SP-1002", ToolContext(csv_paths=[CSV])))
        self.assertIn("No rows match", csv_lookup("propeller", ToolContext(csv_paths=[CSV])))

    def test_no_csv_or_empty_query(self):
        with self.assertRaises(ToolError):
            csv_lookup("x", ToolContext())
        with self.assertRaises(ToolError):
            csv_lookup("the", ToolContext(csv_paths=[CSV]))

    def test_windows_encoded_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "c.csv")
            path.write_bytes("name,city\nCaf\xe9 Uno,Pune\n".encode("cp1252"))
            self.assertIn("Café Uno", csv_lookup("pune", ToolContext(csv_paths=[path])))


class TestCurrentDate(unittest.TestCase):
    def test_uses_context_clock(self):
        ctx = ToolContext(now=lambda: datetime(2026, 11, 27, 9, 5))
        self.assertEqual(current_date("", ctx), "Today is Friday, 2026-11-27. Local time is 09:05.")


def fake_get(responses):
    """Returns a fake requests.get that answers with the given JSON bodies in order."""
    def get(url, params=None, timeout=None):
        body = responses.pop(0)
        if isinstance(body, Exception):
            raise body
        r = mock.Mock()
        r.json.return_value = body
        r.raise_for_status.return_value = None
        return r
    return get


GEO = {"results": [{"name": "Mumbai", "country": "India", "latitude": 19.07, "longitude": 72.88}]}
FORECAST = {"current": {"temperature_2m": 29.4, "wind_speed_10m": 11.2},
            "daily": {"time": ["2026-11-27"], "temperature_2m_max": [32.1],
                      "temperature_2m_min": [24.0], "precipitation_sum": [0.0]}}


class TestWeather(unittest.TestCase):
    def test_weather_report(self):
        out = weather_api("Mumbai", ToolContext(http_get=fake_get([GEO, FORECAST])))
        self.assertIn("Weather for Mumbai, India: now 29.4 C", out)
        self.assertIn("2026-11-27: max 32.1 C", out)

    def test_place_name_fallbacks(self):
        self.assertEqual(place_candidates("Port of Kochi"), ["Port of Kochi", "Kochi"])
        self.assertEqual(place_candidates("Mumbai"), ["Mumbai"])
        self.assertEqual(place_candidates("Kochi port, India"), ["Kochi port, India", "Kochi India", "Kochi"])

    def test_port_of_kochi_found_on_second_try(self):
        # Real chat: "Port of Kochi" is not a geocoder name; "Kochi" is.
        out = weather_api("Port of Kochi", ToolContext(http_get=fake_get([{"results": []}, GEO, FORECAST])))
        self.assertIn("Weather for Mumbai, India", out)   # the fake geocoder answers GEO

    def test_errors(self):
        cases = [
            ("Nowhereville", [{"results": []}]),
            ("Port of Nowhere", [{"results": []}, {"results": []}]),
            ("Mumbai", [requests.exceptions.ConnectionError("offline")]),
            ("Mumbai", [GEO, {"unexpected": 1}]),
            ("", []),
        ]
        for place, responses in cases:
            with self.subTest(place=place, responses=str(responses)[:30]), self.assertRaises(ToolError):
                weather_api(place, ToolContext(http_get=fake_get(responses)))


class TestRunTool(unittest.TestCase):
    def test_registry_has_four_tools(self):
        self.assertEqual(TOOL_NAMES, {"csv_lookup", "calculator", "weather_api", "current_date"})

    def test_allowed_tool_runs(self):
        r = run_tool("calculator", "2+2", ToolContext(), allowed=["calculator"])
        self.assertTrue(r.ok)
        self.assertEqual(r.output, "2+2 = 4")

    def test_not_allowed_unknown_and_failing_tools_never_raise(self):
        for name, inp, allowed, text in [
            ("weather_api", "Mumbai", ["calculator"], "not available to this agent"),
            ("search_web", "x", ["calculator"], "unknown tool"),
            ("calculator", "1/0", ["calculator"], "division by zero"),
        ]:
            with self.subTest(name=name):
                r = run_tool(name, inp, ToolContext(), allowed=allowed)
                self.assertFalse(r.ok)
                self.assertIn(text, r.output)


if __name__ == "__main__":
    unittest.main()
