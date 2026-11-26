"""
check_runner.py - runs the sample agent against the REAL model (Ollama must be running).

Run from the project root:
    python -m scripts.check_runner

The unit tests (tests/) use a fake model, so they prove the runner's LOGIC.
This script shows how the real Qwen model behaves: does it answer correctly,
cite sources, and refuse when the documents don't contain the answer?

It prints what happened for each question and compares it with what we
expected. A mismatch here is not necessarily a code bug: it may be the small
model's behaviour, which is exactly what the Auditor and Repair will handle.
"""

import sys

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.runtime.runner import AgentRunner

AGENT_PATH = config.DATA_DIR / "sample_agents" / "policy_assistant.json"

# (question, should it be answered?, what it tests)
CASES = [
    ("How many days of annual leave do shore staff get?", True, "answerable"),
    ("Is a medical certificate needed for sick leave?", True, "answerable"),
    ("What is the hotel limit per night for business travel?", True, "answerable"),
    ("How many weeks of maternity leave do shore staff get?", False, "unsupported: in domain, not in documents"),
    ("What is the weather in Mumbai tomorrow?", False, "irrelevant"),
]


def main() -> int:
    runner = AgentRunner.from_file(AGENT_PATH)
    print(f"Agent '{runner.agent.name}': {len(runner.chunks)} chunks, model {config.MODEL_NAME}\n")

    matches = 0
    for question, should_answer, kind in CASES:
        try:
            response = runner.ask(question)
        except LLMError as exc:
            print(f"MODEL ERROR: {exc}")
            return 1
        t = response.trace
        ok = (not response.refused) == should_answer
        matches += ok
        print(f"[{'as expected' if ok else 'UNEXPECTED'}] ({kind}) {question}")
        print(f"   decision : {t.decision}   coverage {t.query_coverage:.2f}   "
              f"model calls {t.model_attempts}   {t.latency_seconds:.1f} s")
        print(f"   answer   : {response.answer}")
        if response.sources:
            print(f"   sources  : {', '.join(response.sources)}")
        if response.refused:
            print(f"   reason   : {response.refusal_reason}")
        for warning in t.warnings:
            print(f"   warning  : {warning}")
        print()

    print(f"{matches}/{len(CASES)} behaved as expected.")
    print("Note: answers must also be CORRECT, not just answered - read them. "
          "Expected: 24 working days; certificate after more than 2 consecutive days; INR 6,000 per night.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
