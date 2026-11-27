"""
compare_scorecards.py - case-by-case comparison of two scorecards run on the
SAME frozen eval set (e.g. before and after a code change).

    python -m scripts.compare_scorecards runs\\demo2_check\\scorecard_iteration_1.json runs\\demo2_check\\scorecard_plural_check.json
"""

import sys

from forgeai.roles.judge import Scorecard


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    a, b = (Scorecard.model_validate_json(open(p, encoding="utf-8").read()) for p in sys.argv[1:])
    if a.eval_sha256 != b.eval_sha256:
        print("WARNING: different eval sets (hash mismatch) - the comparison is not like for like.")
    old = {r.id: r for r in a.results}
    fixed, broken = [], []
    for r in b.results:
        before = old.get(r.id)
        if before and before.status != r.status:
            (fixed if r.status == "pass" else broken).append(r.id)
            print(f"  {r.id}: {before.status} -> {r.status}   {r.failure or before.failure}")
    print(f"\n{a.label}: {a.passed}/{a.passed + a.failed}   {b.label}: {b.passed}/{b.passed + b.failed}")
    print(f"fixed {fixed or '-'}   broken {broken or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
