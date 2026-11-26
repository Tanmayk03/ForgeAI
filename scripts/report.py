"""report.py - (re)write runs/<run>/report.md from the saved artifacts.

    python -m scripts.report --run demo_test
"""

import argparse
import sys

from forgeai import config
from forgeai.roles.reporter import write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI Reporter")
    parser.add_argument("--run", required=True)
    path = write_report(config.RUNS_DIR / parser.parse_args().run)
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
