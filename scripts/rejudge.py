"""
rejudge.py - re-score a run's SAVED answers with the current judge (code checks
only: no agent, no model, instant). Shows which cases change and why.

    python -m scripts.rejudge --run demo2_spares
    python -m scripts.rejudge --run demo2_spares --label iteration_1
"""

import argparse
import sys

from forgeai import config
from forgeai.roles.judge import Scorecard, rejudge, save_scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-judge saved answers with the current judge")
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", default="iteration_0")
    args = parser.parse_args()
    path = config.RUNS_DIR / args.run / f"scorecard_{args.label}.json"
    if not path.is_file():
        print(f"Not found: {path}")
        return 1
    old = Scorecard.model_validate_json(path.read_text(encoding="utf-8"))
    new = rejudge(old)
    before = {r.id: r for r in old.results}
    for r in new.results:
        if before[r.id].status != r.status:
            how = ", ".join(f"{f.fact!r} -> {f.method}" for f in r.facts)
            print(f"  {r.id}: {before[r.id].status} -> {r.status}   answer {r.answer[:60]!r}   ({how})")
    print(f"\n{old.label}: {old.passed}/{old.passed + old.failed} ({old.pass_rate:.0%})   "
          f"re-judged: {new.passed}/{new.passed + new.failed} ({new.pass_rate:.0%})")
    print(f"Saved {save_scorecard(new, config.RUNS_DIR / args.run)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
