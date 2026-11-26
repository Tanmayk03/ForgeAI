"""
forge.py - the whole factory in one command:
  Manager -> Guard -> Builder -> Auditor (evals) -> Judge -> Repair -> Reporter

    python -m scripts.forge --brief "I need an assistant that ..." --docs data\\my_docs --run demo1 --auto
    python -m scripts.forge --brief data\\brief.txt --docs data\\my_docs --run demo1

--auto: no questions (unattended); the interview defaults are recorded as assumptions.
Afterwards:  python -m scripts.chat --agent runs\\<run>\\agent_final.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.llm_json import LLMJSONError
from forgeai.roles.builder import BuilderError, run_builder, save_build
from forgeai.roles.eval_generator import generate_evals, save_evals
from forgeai.roles.guard import run_guard, save_guard_report
from forgeai.roles.judge import run_evals, save_scorecard
from forgeai.roles.manager import ManagerError, run_manager, save_requirements
from forgeai.roles.repair import run_repair
from forgeai.roles.reporter import write_report
from forgeai.runtime.runner import AgentRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI: build, test and repair an agent from a brief")
    parser.add_argument("--brief", required=True, help="brief text, or a path to a .txt file")
    parser.add_argument("--docs", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--auto", action="store_true", help="unattended: use interview defaults")
    args = parser.parse_args()
    brief_path = Path(args.brief)
    brief = brief_path.read_text(encoding="utf-8-sig") if brief_path.suffix == ".txt" and brief_path.is_file() else args.brief
    run_dir = config.RUNS_DIR / args.run
    timings: dict[str, float] = {}

    def stage(name, fn):
        print(f"\n=== {name} ===", flush=True)
        start = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - start, 1)
        print(f"({timings[name]:.0f} s)")
        return out

    try:
        log: list = []
        req = stage("manager", lambda: run_manager(brief, args.docs, auto=args.auto, log=log))
        save_requirements(req, run_dir, log=log)
        log = []
        req = stage("guard", lambda: run_guard(req, log=log))
        save_guard_report(req, run_dir, log=log)
        for s in req.topic_support:
            print(f"  {s.status:11} {s.topic}")
        log = []
        agent, build_log = stage("builder", lambda: run_builder(req, log=log))
        save_build(agent, build_log, run_dir, log=log)
        print(f"  tools: {agent.tools or 'none'}")
        log = []
        evals, rejected = stage("eval generation", lambda: generate_evals(req, log=log))
        save_evals(evals, rejected, run_dir, log=log)
        print(f"  {evals.counts}")
        baseline = stage("evaluation", lambda: run_evals(evals, AgentRunner(agent), progress=lambda m: print("  " + m)))
        save_scorecard(baseline, run_dir)
        print(f"  baseline {baseline.passed}/{baseline.passed + baseline.failed}")
        stage("repair", lambda: run_repair(agent, evals, baseline, run_dir, progress=lambda m: print("  " + m)))
    except (ManagerError, BuilderError, LLMJSONError, LLMError, FileNotFoundError) as exc:
        print(f"\nFactory stopped: {exc}")
        (run_dir / "timings.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        return 1

    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    report = write_report(run_dir)
    print(f"\nTotal {sum(timings.values()) / 60:.1f} min. Report: {report}")
    print(f"Chat: python -m scripts.chat --agent {(run_dir / 'agent_final.json').relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
