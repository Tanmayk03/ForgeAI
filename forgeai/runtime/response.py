"""
response.py - the shapes of what the model returns and what the runner returns.

Two layers:
  ModelOutput   - what we ask Qwen to produce (sources are numbers like 1, 2, 3)
  AgentResponse - what the runner returns to the caller (sources are real chunk IDs),
                  plus a RunTrace with everything an auditor needs to check the answer.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ModelOutput(BaseModel):
    """
    The JSON shape Qwen must reply with. Its JSON schema is sent to Ollama, which
    forces the reply into this shape. We still validate it here, because a forced
    shape can still hold bad content (e.g. citing source 7 when only 3 were shown).
    """

    answer: str
    sources: list[int] = Field(description="Numbers of the sources used, e.g. [1, 3]")
    refused: bool
    refusal_reason: str = ""


# Which stage decided the outcome. Lets an auditor see WHY an answer looks the way it does.
Decision = Literal[
    "answered",               # model answered with valid citations
    "refused_no_evidence",    # gate: no chunk matched the question
    "refused_low_coverage",   # gate: chunks cover too little of the question
    "refused_by_model",       # model chose to refuse
    "refused_no_citation",    # model answered but cited no valid source -> refused (fail closed)
    "refused_unsupported_number",  # answer has a number not in the cited sources -> refused (fail closed)
    "refused_invalid_output", # model output unusable even after retry -> refused (fail closed)
]


class RetrievedChunkInfo(BaseModel):
    """One retrieved chunk, as recorded in the trace (text included, for auditing)."""

    source_number: int = Field(description="Number shown to the model, e.g. [1]")
    chunk_id: str
    doc_name: str
    page: int
    score: float
    rank: int
    text: str


class RunTrace(BaseModel):
    """Everything that happened while answering one question."""

    agent_name: str
    question: str
    decision: Decision
    retrieved: list[RetrievedChunkInfo] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    covered_terms: list[str] = Field(default_factory=list)
    query_coverage: float = 0.0
    model_called: bool = False
    model_attempts: int = 0
    raw_model_outputs: list[str] = Field(default_factory=list)
    model_answer_before_checks: str = ""   # kept when the runner overrides the model
    unsupported_numbers: list[str] = Field(default_factory=list)
    estimated_prompt_tokens: int = 0
    warnings: list[str] = Field(default_factory=list)
    latency_seconds: float = 0.0


class AgentResponse(BaseModel):
    """What every caller (chat CLI now, Auditor later) receives."""

    answer: str
    sources: list[str] = Field(description="Chunk IDs the answer is based on")
    refused: bool
    refusal_reason: str = ""
    trace: RunTrace
