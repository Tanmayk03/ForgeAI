"""
run_evals.py - run the frozen eval set against an agent and score it.

    python -m scripts.run_evals --run demo_test
    python -m scripts.run_evals --run demo_test --agent runs\\demo_test\\agent.json --label iteration_0
"""

import argparse
import sys

from forgeai import config
from forgeai.roles.eval_generator import load_evals
from forgeai.roles.judge import JudgeError, run_evals, save_scorecard
from forgeai.runtime.runner import AgentRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI Auditor: run and judge the evals")
    parser.add_argument("--run", required=True)
    parser.add_argument("--agent", help="agent.json (default: runs/<run>/agent.json)")
    parser.add_argument("--label", default="iteration_0")
    args = parser.parse_args()
    run_dir = config.RUNS_DIR / args.run
    runner = AgentRunner.from_file(args.agent or run_dir / "agent.json")
    evals = load_evals(run_dir / "evals.json")
    log: list = []
    try:
        card = run_evals(evals, runner, label=args.label, log=log, progress=print)
    except JudgeError as exc:
        print(f"Judge refused: {exc}")
        return 1
    path = save_scorecard(card, run_dir)
    print(f"\n{card.agent_name}: {card.passed}/{card.passed + card.failed} passed "
          f"({card.pass_rate:.0%}), {card.errors} error(s), {card.judge_llm_calls} judge call(s), {card.total_seconds:.0f} s")
    for t, row in card.by_type.items():
        print(f"  {t:13} pass {row['pass']}  fail {row['fail']}  error {row['error']}")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
