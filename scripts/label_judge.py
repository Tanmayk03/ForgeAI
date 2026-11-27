"""
label_judge.py - you label saved answers; it reports how often the judge agrees.

    python -m scripts.label_judge --run demo2_spares
    python -m scripts.label_judge --run demo2_spares --label iteration_0_rejudged --n 20

For each case you see the question, what the agent should do, the answer key
and the agent's actual response. Type y (acceptable), n (wrong) or s (skip).
Labels are saved in runs/<run>/human_labels.json; re-running only asks about
cases you have not labelled yet.
"""

import argparse
import sys

from forgeai import config
from forgeai.roles.eval_generator import load_evals
from forgeai.roles.judge import Scorecard
from forgeai.roles.judge_agreement import HumanLabel, agreement, choose_sample, load_labels, save_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the judge against your own labels")
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", default="iteration_0", help="scorecard label, e.g. iteration_0_rejudged")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    run_dir = config.RUNS_DIR / args.run
    card = Scorecard.model_validate_json((run_dir / f"scorecard_{args.label}.json").read_text(encoding="utf-8"))
    facts = {c.id: c.key_facts for c in load_evals(run_dir / "evals.json").cases}
    labels = [l for l in load_labels(run_dir) if l.scorecard == card.label]
    done = {l.id for l in labels}

    todo = [r for r in choose_sample(card.results, args.n) if r.id not in done]
    print(f"{len(done)} already labelled; {len(todo)} to go. y = acceptable, n = wrong, s = skip, q = quit\n")
    for r in todo:
        print("-" * 70)
        print(f"{r.id} [{r.type}]  {r.question}")
        if r.expected_behavior == "refuse":
            print("  SHOULD: refuse (out of scope)")
        else:
            print(f"  SHOULD: answer, containing {facts.get(r.id, [])}")
        print(f"  AGENT : {'REFUSED - ' if r.refused else ''}{r.answer}")
        while True:
            key = input("  Is the agent's response acceptable? [y/n/s/q] ").strip().lower()
            if key in ("y", "n", "s", "q"):
                break
        if key == "q":
            break
        if key == "s":
            continue
        labels.append(HumanLabel(scorecard=card.label, id=r.id, question=r.question, answer=r.answer,
                                 judge=r.status, human_ok=key == "y"))
        save_labels(labels + [l for l in load_labels(run_dir) if l.scorecard != card.label], run_dir)

    a = agreement(labels)
    print(f"\nJudge agrees with you on {a.agree}/{a.labelled} ({a.rate:.0%}) of labelled cases.")
    print(f"  judge too lenient (passed, you said wrong): {a.false_pass or '-'}")
    print(f"  judge too strict  (failed, you said fine) : {a.false_fail or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
