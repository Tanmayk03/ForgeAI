"""
helpers.py - shared test tools.

FakeLLM stands in for llm_client.generate so tests run without Ollama,
give the same result every time, and can simulate any model behaviour
(good answers, refusals, broken JSON, errors).
"""

import copy
import json

from forgeai import config

SAMPLE_DOCS = str(config.DATA_DIR / "sample_docs")


class FakeLLM:
    """
    Returns scripted replies in order. A reply can be:
      * a dict      -> sent back as JSON text
      * a str       -> sent back as-is (use this to simulate broken output)
      * an Exception instance -> raised (simulates Ollama errors)
    Every call is recorded in .calls so tests can check what was sent.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, system=None, json_schema=None, **kwargs):
        self.calls.append({"prompt": prompt, "system": system, "json_schema": json_schema})
        if not self.replies:
            raise AssertionError("FakeLLM was called more times than expected")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply) if isinstance(reply, dict) else reply


def answer(text, sources):
    return {"answer": text, "sources": sources, "refused": False, "refusal_reason": ""}


def refusal(reason="not in sources"):
    return {"answer": "", "sources": [], "refused": True, "refusal_reason": reason}


VALID_AGENT = {
    "schema_version": "1.0",
    "name": "test_agent",
    "description": "Agent used in tests.",
    "system_instructions": "You answer questions about the provided documents.",
    "scope": {
        "in_scope_topics": ["leave", "travel"],
        "refusal_topics": ["legal advice"],
        "refusal_message": "Sorry, I can't answer that.",
    },
    "documents": {"folder": SAMPLE_DOCS},
    "retrieval": {
        "chunk_size_words": 300, "chunk_overlap_words": 50, "top_k": 3,
        "min_bm25_score": 0.0, "min_query_coverage": 0.5,
    },
    "tools": [],
    "output_contract": {"max_answer_words": 120, "require_citations": True},
}


def agent_dict(**changes):
    """A fresh copy of VALID_AGENT with top-level keys replaced."""
    data = copy.deepcopy(VALID_AGENT)
    data.update(changes)
    return data
