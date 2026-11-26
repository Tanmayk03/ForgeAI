"""
repair.py - classify failures, apply targeted fixes, re-run (max 3 iterations).

    python -m scripts.repair --run demo_test

Uses runs/<run>/scorecard_iteration_0.json as the starting score (run
scripts.run_evals first). Writes agent_iteration_N.json, scorecard_iteration_N.json,
agent_final.json and repair_log.json.
"""

import argparse
import sys

from forgeai import config
from forgeai.roles.eval_generator import load_evals
from forgeai.roles.judge import Scorecard
from forgeai.roles.repair import classify_all, run_repair
from forgeai.runtime.agent_schema import load_agent_definition


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI Repair")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = config.RUNS_DIR / args.run
    baseline_path = run_dir / "scorecard_iteration_0.json"
    if not baseline_path.is_file():
        print(f"No {baseline_path.name}; run scripts.run_evals first.")
        return 1
    evals = load_evals(run_dir / "evals.json")
    baseline = Scorecard.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    agent = load_agent_definition(run_dir / "agent.json")

    print("Failures in iteration 0:")
    for f in classify_all(baseline, evals):
        print(f"  {f.id} {f.category:15} ({f.subtype}) {f.evidence}")
    best, card, logs = run_repair(agent, evals, baseline, run_dir, progress=print)
    print(f"\nFinal: {card.passed}/{card.passed + card.failed} ({card.pass_rate:.0%})  "
          f"after {sum(1 for l in logs if l.change)} change(s); saved agent_final.json and repair_log.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
