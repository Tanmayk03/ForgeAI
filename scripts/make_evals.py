"""
make_evals.py - the Auditor generates, checks and freezes the test set.
Reads only requirements.json and the documents (never agent.json).

    python -m scripts.make_evals --run demo_test
"""

import argparse
import sys

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.roles.eval_generator import generate_evals, save_evals
from forgeai.roles.manager import load_requirements


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI Auditor: generate evals.json")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = config.RUNS_DIR / args.run
    try:
        req = load_requirements(run_dir / "requirements.json")
    except FileNotFoundError:
        print(f"No requirements.json in {run_dir}")
        return 1
    log: list = []
    try:
        evals, rejected = generate_evals(req, log=log)
    except LLMError as exc:
        print(f"Eval generation failed: {exc}")
        return 1
    path = save_evals(evals, rejected, run_dir, log=log)

    for c in evals.cases:
        facts = f"  facts={c.key_facts} docs={c.expected_docs}" if c.key_facts else ""
        print(f"{c.id} [{c.type:12}] {c.expected_behavior:6} {c.question}{facts}")
    print(f"\nCounts: {evals.counts}")
    for n in evals.notes:
        print(f"  note: {n}")
    reasons = {}
    for r in rejected:
        key = r.reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"Rejected by checks: {reasons}")
    print(f"Frozen sha256: {evals.sha256[:16]}...  Saved {path}  ({len(log)} model call(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
