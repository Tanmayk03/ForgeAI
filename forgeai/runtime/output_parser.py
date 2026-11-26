"""
output_parser.py - turns the model's raw text into a validated ModelOutput.

Even with Ollama's JSON mode, small models sometimes:
  * wrap the JSON in ```json ... ``` fences,
  * add a sentence before or after it,
  * leave out a field, or use the wrong type ("sources": "1"),
  * return nothing at all.
This module handles each case and reports a clear reason when it can't.
"""

import json
import re

from pydantic import ValidationError

from forgeai.runtime.response import ModelOutput

FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json_text(raw: str) -> str:
    """Strip code fences and anything outside the outermost { ... }."""
    text = FENCE_PATTERN.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return text  # no object found; json.loads will report the problem
    return text[start:end + 1]


def parse_model_output(raw: str) -> tuple[ModelOutput | None, str]:
    """
    Try to read the model's reply.

    Returns (output, "") on success, or (None, reason) on failure, so the
    caller can decide whether to retry.
    """
    if raw is None or not raw.strip():
        return None, "empty reply"

    try:
        data = json.loads(_extract_json_text(raw))
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc.msg})"

    if not isinstance(data, dict):
        return None, "JSON is not an object"

    try:
        output = ModelOutput.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'object'}: {e['msg']}" for e in exc.errors()
        )
        return None, f"wrong fields ({problems})"

    output.answer = output.answer.strip()
    output.refusal_reason = output.refusal_reason.strip()
    output.tool_name = output.tool_name.strip()
    if not output.refused and not output.answer and not output.tool_name:
        return None, "answer is empty but refused is false"
    return output, ""
