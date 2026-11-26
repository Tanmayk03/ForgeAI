"""
guard.py - run the groundability guard on a run's requirements.json.

    python -m scripts.guard --run demo_test
"""

import argparse
import sys

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.llm_json import LLMJSONError
from forgeai.roles.guard import run_guard, save_guard_report
from forgeai.roles.manager import load_requirements


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI groundability guard")
    parser.add_argument("--run", required=True, help="run name (folder in runs/)")
    args = parser.parse_args()
    run_dir = config.RUNS_DIR / args.run
    try:
        req = load_requirements(run_dir / "requirements.json")
    except FileNotFoundError:
        print(f"No requirements.json in {run_dir}; run scripts.intake first.")
        return 1

    log: list = []
    try:
        req = run_guard(req, log=log)
    except (LLMError, LLMJSONError) as exc:
        print(f"Guard failed: {exc}")
        return 1
    path = save_guard_report(req, run_dir, log=log)

    for s in req.topic_support:
        mark = "OK " if s.status == "supported" else "GAP"
        print(f"[{mark}] {s.topic:35} {s.status:11} ({s.method}, coverage {s.coverage:.2f}) {s.reason}")
    for w in req.guard_warnings:
        print(f"  warning: {w}")
    print(f"\nSaved {path}  ({len(log)} model call(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
