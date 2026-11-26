"""
chat.py - chat with any agent definition from the terminal.

Run from the project root:
    python -m scripts.chat --agent data\\sample_agents\\policy_assistant.json
    python -m scripts.chat --agent <path to agent.json> --trace

Type a question and press Enter. Type "exit" (or press Ctrl+C) to quit.
--trace also prints retrieved chunks, scores, coverage and which stage decided.
"""

import argparse
import sys

from forgeai.llm_client import LLMError
from forgeai.runtime.agent_schema import AgentDefinitionError
from forgeai.runtime.response import AgentResponse
from forgeai.runtime.runner import AgentLoadError, AgentRunner


def print_response(response: AgentResponse, show_trace: bool) -> None:
    label = "REFUSED" if response.refused else "ANSWER"
    print(f"\n{label}: {response.answer}")
    if response.refused:
        print(f"  reason : {response.refusal_reason}")
    else:
        print(f"  sources: {', '.join(response.sources)}")

    t = response.trace
    print(f"  ({t.decision}, {t.latency_seconds:.1f} s)")
    if show_trace:
        print(f"  query coverage: {t.query_coverage:.2f}  covered={t.covered_terms}  all={t.query_terms}")
        for r in t.retrieved:
            print(f"  [{r.source_number}] score {r.score:5.2f}  {r.chunk_id}")
        print(f"  model called: {t.model_called}, attempts: {t.model_attempts}")
        if t.model_answer_before_checks and response.refused:
            print(f"  model's own answer (overridden): {t.model_answer_before_checks}")
    for warning in t.warnings:
        print(f"  warning: {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a ForgeAI agent.")
    parser.add_argument("--agent", required=True, help="path to agent.json")
    parser.add_argument("--trace", action="store_true", help="show retrieval and decision details")
    args = parser.parse_args()

    try:
        runner = AgentRunner.from_file(args.agent)
    except (AgentDefinitionError, AgentLoadError) as exc:
        print(f"Could not load agent:\n{exc}")
        return 1

    agent = runner.agent
    print(f"Agent: {agent.name} - {agent.description}")
    print(f"Documents: {len(runner.load_result.documents)} loaded, "
          f"{len(runner.load_result.skipped)} skipped, {len(runner.chunks)} chunks")
    print('Type a question, or "exit" to quit.')

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            print_response(runner.ask(question), args.trace)
        except LLMError as exc:
            # The model could not be reached: say so, never pretend it was a refusal.
            print(f"\nMODEL ERROR (not an answer):\n{exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
