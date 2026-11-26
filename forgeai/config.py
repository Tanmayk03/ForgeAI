"""
config.py - every ForgeAI setting lives in one place.

Why a config file?
    The model name, the Ollama address and similar values are used by many
    modules. Keeping them here means you change a setting once, not in ten files.

Overriding a setting without editing code:
    Each value can be replaced by an environment variable. Example (Windows PowerShell):
        $env:FORGEAI_MODEL = "qwen2.5:3b"
        python -m scripts.check_connection
"""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    """Return the environment variable `name` if it is set, otherwise `default`."""
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Ollama connection
# ---------------------------------------------------------------------------

# Address of the local Ollama server. 11434 is Ollama's default port.
OLLAMA_BASE_URL: str = _env("FORGEAI_OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Model used by every role (Manager, Builder, Auditor, Repair, Reporter) for now.
MODEL_NAME: str = _env("FORGEAI_MODEL", "qwen2.5:1.5b")

# How long (seconds) to wait for one LLM reply before giving up.
# CPU inference is slow, and the first call also loads the model into memory.
REQUEST_TIMEOUT_SECONDS: int = int(_env("FORGEAI_TIMEOUT", "180"))


# ---------------------------------------------------------------------------
# Generation settings (sent to Ollama as "options")
# ---------------------------------------------------------------------------

# 0.0 = least random. We want repeatable runs so score changes mean something.
TEMPERATURE: float = float(_env("FORGEAI_TEMPERATURE", "0.0"))

# A fixed seed also helps runs repeat. Changing it gives a different, but still repeatable, run.
SEED: int = int(_env("FORGEAI_SEED", "42"))

# Context window in tokens (prompt + reply). Ollama's default may be smaller, and if a
# prompt is longer than this, Ollama silently cuts it off. Setting it explicitly avoids that.
NUM_CTX: int = int(_env("FORGEAI_NUM_CTX", "4096"))

# Maximum tokens in one reply. Short replies keep CPU runs fast.
MAX_OUTPUT_TOKENS: int = int(_env("FORGEAI_MAX_TOKENS", "512"))

# How long Ollama keeps the model in memory after a call. Ollama's own default is
# 5 minutes; after that the next call pays a 5-10 s reload. "30m" keeps it loaded
# through a factory run.
KEEP_ALIVE: str = _env("FORGEAI_KEEP_ALIVE", "30m")

# If the model's reply is not valid JSON, ask again this many times before giving up.
MAX_OUTPUT_RETRIES: int = int(_env("FORGEAI_OUTPUT_RETRIES", "1"))

# Maximum tool calls the model may make while answering one question.
MAX_TOOL_CALLS: int = int(_env("FORGEAI_MAX_TOOL_CALLS", "2"))


# ---------------------------------------------------------------------------
# Document chunking and retrieval
# ---------------------------------------------------------------------------

# Target chunk length in words. A starting point; Repair may change it per agent later.
CHUNK_SIZE_WORDS: int = int(_env("FORGEAI_CHUNK_SIZE", "300"))

# Words repeated between neighbouring chunks, so a sentence cut at a chunk
# boundary still appears whole in one of the two chunks.
CHUNK_OVERLAP_WORDS: int = int(_env("FORGEAI_CHUNK_OVERLAP", "50"))

# How many chunks a search returns by default.
DEFAULT_TOP_K: int = int(_env("FORGEAI_TOP_K", "3"))


# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------

# Project root = the folder that contains the `forgeai` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"        # input briefs and document folders
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"  # prompt templates (committed, as the assignment requires)
RUNS_DIR: Path = PROJECT_ROOT / "runs"        # one sub-folder of JSON artifacts per factory run
