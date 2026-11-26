"""
check_connection.py - checks that ForgeAI can talk to Qwen through Ollama.

Run from the project root (the ForgeAI folder):
    python -m scripts.check_connection

It runs three checks in order and stops at the first failure:
    1. Is the Ollama server running?
    2. Is the configured model downloaded?
    3. Does the model answer a simple prompt?

Exit code 0 = all checks passed, 1 = a check failed (useful for automation later).
"""

import sys
import time

from forgeai import config
from forgeai.llm_client import LLMError, check_ollama_running, generate, is_model_available


def main() -> int:
    print("ForgeAI - LLM connection check")
    print(f"  Ollama URL : {config.OLLAMA_BASE_URL}")
    print(f"  Model      : {config.MODEL_NAME}")
    print()

    # Check 1: server
    print("[1/3] Is Ollama running? ...", end=" ", flush=True)
    try:
        version = check_ollama_running()
    except LLMError as exc:
        print("FAILED")
        print(exc)
        return 1
    print(f"OK (Ollama version {version})")

    # Check 2: model
    print(f"[2/3] Is '{config.MODEL_NAME}' downloaded? ...", end=" ", flush=True)
    try:
        available = is_model_available()
    except LLMError as exc:
        print("FAILED")
        print(exc)
        return 1
    if not available:
        print("FAILED")
        print(f"Model '{config.MODEL_NAME}' is not downloaded.")
        print(f"  Fix: run `ollama pull {config.MODEL_NAME}`")
        return 1
    print("OK")

    # Check 3: a real prompt
    print("[3/3] Sending a test prompt (the first call can take a while) ...", end=" ", flush=True)
    start = time.perf_counter()
    try:
        reply = generate(
            prompt="Reply with exactly one word: READY",
            system="You are a test assistant. Follow the instruction exactly.",
            max_tokens=10,
        )
    except LLMError as exc:
        print("FAILED")
        print(exc)
        return 1
    seconds = time.perf_counter() - start
    print(f"OK ({seconds:.1f} s)")
    print(f"      Model replied: {reply!r}")

    if "READY" not in reply.upper():
        # The connection works; the small model just didn't follow the instruction exactly.
        print("      Note: the reply was not exactly 'READY'. The connection works, but this")
        print("      shows small models don't always follow instructions perfectly.")

    print()
    print("All checks passed. ForgeAI can talk to the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
