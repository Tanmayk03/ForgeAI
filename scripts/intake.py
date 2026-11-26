"""
intake.py - run the Manager: brief + documents -> runs/<run>/requirements.json

    python -m scripts.intake --brief "I need an assistant that ..." --docs data\\sample_docs --run demo1
    python -m scripts.intake --brief data\\my_brief.txt --docs data\\sample_docs --run demo1 --auto

--brief is either the text itself or a path to a .txt file.
--auto answers every clarifying question with its default (for unattended runs).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.llm_json import LLMJSONError
from forgeai.roles.manager import ManagerError, run_manager, save_requirements


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI Manager: write requirements.json")
    parser.add_argument("--brief", required=True, help="brief text, or a path to a .txt file")
    parser.add_argument("--docs", required=True, help="document folder")
    parser.add_argument("--run", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"), help="run name")
    parser.add_argument("--auto", action="store_true", help="use default answers, ask nothing")
    args = parser.parse_args()

    brief_path = Path(args.brief)
    brief = brief_path.read_text(encoding="utf-8-sig") if brief_path.suffix == ".txt" and brief_path.is_file() else args.brief

    log: list = []
    try:
        req = run_manager(brief, args.docs, auto=args.auto, log=log)
    except (ManagerError, LLMJSONError, LLMError) as exc:
        print(f"Manager failed: {exc}")
        return 1

    path = save_requirements(req, config.RUNS_DIR / args.run, log=log)
    print(f"\nAgent name : {req.agent_name}\nPurpose    : {req.purpose}\nAudience   : {req.audience}")
    print(f"Tone       : {req.tone}\nMust answer: {req.must_answer_topics}\nMust refuse: {req.must_refuse_topics}")
    for c in req.clarifications:
        print(f"  [{c.source}] {c.question} -> {c.answer}")
    for n in req.notes:
        print(f"  note: {n}")
    print(f"\nSaved {path}  ({len(log)} model call(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
