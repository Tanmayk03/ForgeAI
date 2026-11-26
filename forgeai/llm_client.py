"""
llm_client.py - the only place in ForgeAI that talks to Ollama.

Every role (Manager, Builder, Auditor, Repair, Reporter) will call `generate()`
from this file. Keeping all LLM access here means:
  * error handling is written once,
  * the generation settings (temperature, seed, context size) are always applied,
  * switching models later is a config change, not a code change.

We call Ollama's HTTP API directly with `requests` (no LLM framework), so every
step is visible and easy to explain.

Ollama endpoints used:
    GET  /api/version   -> is the server running?
    GET  /api/tags      -> which models are downloaded?
    POST /api/generate  -> send a prompt, get a reply
"""

from typing import Optional

import requests

from forgeai import config


# ---------------------------------------------------------------------------
# Errors
#
# Each problem gets its own error type, so callers can react differently
# (e.g. "start Ollama" vs "download the model") and the message tells the
# user exactly how to fix it.
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for every error raised by this module."""


class OllamaNotRunningError(LLMError):
    """The Ollama server cannot be reached."""


class ModelNotFoundError(LLMError):
    """Ollama is running, but the requested model has not been downloaded."""


class LLMTimeoutError(LLMError):
    """Ollama did not answer within the configured timeout."""


class LLMResponseError(LLMError):
    """Ollama answered, but with an error or an unusable reply."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _normalise_model_name(name: str) -> str:
    """
    Ollama treats "qwen2.5" and "qwen2.5:latest" as the same model.
    Adding ":latest" when no tag is given lets us compare names reliably.
    """
    return name if ":" in name else f"{name}:latest"


def _not_running_message(base_url: str) -> str:
    return (
        f"Cannot connect to Ollama at {base_url}.\n"
        "  Fix: start Ollama (open the Ollama app, or run `ollama serve` in a terminal),\n"
        "  then try again. If Ollama uses a different address, set FORGEAI_OLLAMA_URL."
    )


def _model_missing_message(model: str) -> str:
    return (
        f"Model '{model}' is not available in Ollama.\n"
        f"  Fix: run `ollama pull {model}`, or set FORGEAI_MODEL to a model you have\n"
        "  (see the list with `ollama list`)."
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_ollama_running(base_url: Optional[str] = None) -> str:
    """
    Check that the Ollama server is reachable.

    Returns the Ollama version string (e.g. "0.5.7") if it is running.
    Raises OllamaNotRunningError if it is not.
    """
    base_url = base_url or config.OLLAMA_BASE_URL
    try:
        response = requests.get(f"{base_url}/api/version", timeout=5)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaNotRunningError(_not_running_message(base_url)) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaNotRunningError(
            f"Ollama at {base_url} did not respond within 5 seconds."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaNotRunningError(
            f"Something answered at {base_url}, but it does not look like Ollama: {exc}"
        ) from exc

    return response.json().get("version", "unknown")


def list_local_models(base_url: Optional[str] = None) -> list[str]:
    """
    Return the names of all models downloaded in Ollama, e.g. ["qwen2.5:1.5b"].
    Raises OllamaNotRunningError if the server cannot be reached.
    """
    base_url = base_url or config.OLLAMA_BASE_URL
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaNotRunningError(_not_running_message(base_url)) from exc
    except requests.exceptions.RequestException as exc:
        raise LLMResponseError(f"Could not list Ollama models: {exc}") from exc

    return [m.get("name", "") for m in response.json().get("models", [])]


def is_model_available(model: Optional[str] = None, base_url: Optional[str] = None) -> bool:
    """Return True if `model` (default: config.MODEL_NAME) is downloaded in Ollama."""
    model = model or config.MODEL_NAME
    wanted = _normalise_model_name(model)
    installed = {_normalise_model_name(name) for name in list_local_models(base_url)}
    return wanted in installed


# ---------------------------------------------------------------------------
# The main function: send a prompt, get text back
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_schema: Optional[dict] = None,
) -> str:
    """
    Send one prompt to the local model and return its reply as plain text.

    Args:
        prompt:      what you want the model to respond to.
        system:      optional system prompt (the role's standing instructions).
        model:       Ollama model name; defaults to config.MODEL_NAME.
        temperature: randomness; defaults to config.TEMPERATURE (0.0).
        max_tokens:  reply length limit; defaults to config.MAX_OUTPUT_TOKENS.
        json_schema: optional JSON schema. If given, Ollama forces the reply to be
                     JSON in this shape (the reply is still returned as text;
                     the caller parses and validates it).

    Returns:
        The model's reply, with leading/trailing whitespace removed.

    Raises:
        ValueError             if the prompt is empty.
        OllamaNotRunningError  if Ollama cannot be reached.
        ModelNotFoundError     if the model is not downloaded.
        LLMTimeoutError        if the reply takes longer than the timeout.
        LLMResponseError       for any other Ollama error or an empty reply.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    model = model or config.MODEL_NAME
    base_url = config.OLLAMA_BASE_URL

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # wait for the full reply instead of receiving it word by word
        "keep_alive": config.KEEP_ALIVE,  # keep the model loaded between calls
        "options": {
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "seed": config.SEED,
            "num_ctx": config.NUM_CTX,
            "num_predict": config.MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens,
        },
    }
    if system:
        payload["system"] = system
    if json_schema:
        payload["format"] = json_schema

    # 1. Send the request, turning network problems into clear errors.
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaNotRunningError(_not_running_message(base_url)) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMTimeoutError(
            f"Model '{model}' did not reply within {config.REQUEST_TIMEOUT_SECONDS} seconds.\n"
            "  The first call is slower because the model is loading into memory.\n"
            "  Fix: try again, or raise FORGEAI_TIMEOUT."
        ) from exc

    # 2. Ollama replied, but maybe with an error. Its error body looks like {"error": "..."}.
    if response.status_code != 200:
        try:
            error_text = response.json().get("error", response.text)
        except ValueError:
            error_text = response.text

        # Ollama answers 404 with "model ... not found" when the model is not downloaded.
        if response.status_code == 404 and "not found" in error_text.lower():
            raise ModelNotFoundError(_model_missing_message(model))
        raise LLMResponseError(f"Ollama returned HTTP {response.status_code}: {error_text}")

    # 3. Read the reply text.
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMResponseError("Ollama returned a reply that is not valid JSON.") from exc

    text = (data.get("response") or "").strip()
    if not text:
        raise LLMResponseError(f"Model '{model}' returned an empty reply.")
    return text
