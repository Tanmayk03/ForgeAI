"""
check_tools.py - runs each tool for real, then asks the parts agent two
questions that need tools (Ollama must be running for part 2).

    python -m scripts.check_tools
"""

import sys

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.runtime.runner import AgentRunner
from forgeai.tools.base import ToolContext
from forgeai.tools.registry import TOOL_NAMES, run_tool

CSV = config.DATA_DIR / "sample_docs" / "spare_parts.csv"


def main() -> int:
    print("Part 1: each tool directly")
    ctx = ToolContext(csv_paths=[CSV])
    for name, tool_input in [("calculator", "145 * 4"), ("csv_lookup", "fuel injector"),
                             ("current_date", ""), ("weather_api", "Mumbai")]:
        r = run_tool(name, tool_input, ctx, allowed=sorted(TOOL_NAMES))
        print(f"  [{'ok' if r.ok else 'FAILED'}] {name}({tool_input!r}) -> {r.output}")

    print("\nPart 2: parts agent with real model")
    agent = AgentRunner.from_file(config.DATA_DIR / "sample_agents" / "parts_assistant.json")
    for q in ["What is the unit price of the fuel injector nozzle?",
              "How much do 4 fuel injector nozzles cost in total?"]:
        try:
            resp = agent.ask(q)
        except LLMError as exc:
            print(f"  MODEL ERROR: {exc}")
            return 1
        t = resp.trace
        print(f"\n  Q: {q}\n  {'REFUSED' if resp.refused else 'ANSWER'}: {resp.answer}")
        print(f"  decision: {t.decision}, tool calls: {[(c.name, c.tool_input, c.ok) for c in t.tool_calls]}, "
              f"{t.latency_seconds:.1f} s")
        if resp.sources:
            print(f"  sources: {resp.sources}")
        for w in t.warnings:
            print(f"  warning: {w}")
    print("\nExpected: 145.00 USD; 4 x 145 = 580 USD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
