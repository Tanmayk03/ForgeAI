"""
build.py - run the Builder on a run's requirements.json (after the guard).

    python -m scripts.build --run demo_test
    python -m scripts.chat --agent runs\\demo_test\\agent.json
"""

import argparse
import sys

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.roles.builder import BuilderError, run_builder, save_build
from forgeai.roles.manager import load_requirements


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAI Builder: requirements.json -> agent.json")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = config.RUNS_DIR / args.run
    try:
        req = load_requirements(run_dir / "requirements.json")
    except FileNotFoundError:
        print(f"No requirements.json in {run_dir}; run scripts.intake and scripts.guard first.")
        return 1
    log: list = []
    try:
        agent, build_log = run_builder(req, log=log)
    except (BuilderError, LLMError) as exc:
        print(f"Builder failed: {exc}")
        return 1
    path = save_build(agent, build_log, run_dir, log=log)

    print(f"Agent      : {agent.name}")
    print(f"Persona    : {agent.system_instructions}  [{build_log['system_instructions_source']}]")
    print(f"In scope   : {agent.scope.in_scope_topics}")
    print(f"Refuse     : {agent.scope.refusal_topics}")
    print(f"Retrieval  : {build_log['retrieval']}")
    for t in build_log["tools"]:
        print(f"  tool {t['name']:13} {'ATTACHED' if t['attached'] else 'no      '}  {t['reason']}")
    print(f"\nSaved {path}\nChat with it: python -m scripts.chat --agent {path.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
