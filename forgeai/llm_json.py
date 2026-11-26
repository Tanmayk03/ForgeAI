"""
llm_json.py - ask the model for JSON matching a Pydantic model, with one retry.

Used by the factory roles (Manager, Builder, ...). The agent runner keeps its
own version because it also records a detailed trace.
"""

import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from forgeai import config, llm_client

T = TypeVar("T", bound=BaseModel)


class LLMJSONError(Exception):
    """The model did not return usable JSON, even after retrying."""


def generate_json(
    prompt: str,
    model_cls: type[T],
    system: str = "Reply with JSON only.",
    llm_fn: Callable[..., str] | None = None,
    log: list | None = None,
    max_tokens: int | None = None,
) -> T:
    """
    Call the model with model_cls's JSON schema, parse and validate the reply.
    Every prompt and raw reply is appended to `log` (evidence for the run).
    LLM errors (Ollama down...) propagate unchanged.
    """
    llm_fn = llm_fn or llm_client.generate
    schema = model_cls.model_json_schema()
    current, reason = prompt, ""
    for attempt in range(1 + config.MAX_OUTPUT_RETRIES):
        raw = llm_fn(prompt=current, system=system, json_schema=schema, max_tokens=max_tokens)
        if log is not None:
            log.append({"attempt": attempt + 1, "prompt": current, "raw_output": raw})
        try:
            return model_cls.model_validate(json.loads(raw))
        except (ValueError, ValidationError) as exc:
            reason = str(exc).splitlines()[0][:200]
            current = prompt + f"\n\nYour previous reply could not be used ({reason}). Reply with valid JSON only."
    raise LLMJSONError(f"model output unusable after {1 + config.MAX_OUTPUT_RETRIES} attempts: {reason}")
