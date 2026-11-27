"""Smoke test: the results viewer renders every committed run without errors."""

import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:   # streamlit not installed: skip, the factory itself does not need it
    AppTest = None

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
RUNS = Path(__file__).resolve().parent.parent / "runs"


@unittest.skipIf(AppTest is None, "streamlit not installed")
class TestViewer(unittest.TestCase):
    def test_every_run_renders(self):
        runs = sorted(p.name for p in RUNS.iterdir() if (p / "report.md").is_file()) if RUNS.is_dir() else []
        if not runs:
            self.skipTest("no committed runs")
        for name in runs:
            with self.subTest(run=name):
                at = AppTest.from_file(APP, default_timeout=60).run()
                at.sidebar.selectbox[0].select(name).run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertEqual(len(at.tabs), 6)
                self.assertIn("/", at.metric[0].value)      # e.g. "18/22 (82%)"


if __name__ == "__main__":
    unittest.main()
