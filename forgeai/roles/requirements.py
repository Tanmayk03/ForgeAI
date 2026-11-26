"""
requirements.py - the shape of requirements.json, written by the Manager.

requirements.json is the contract between the roles: the Builder builds the
agent from it, and the Auditor writes tests from it (plus the documents).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REQUIREMENTS_VERSION = "1.0"


class Clarification(BaseModel):
    question: str
    answer: str
    source: Literal["requester", "assumed"] = Field(
        description="requester = typed by the user; assumed = default used (--auto or empty answer)"
    )


class DocumentSummary(BaseModel):
    name: str
    file_type: str
    pages: int
    words: int
    chunked: bool
    preview: str
    keywords: list[str]


class TopicSupport(BaseModel):
    """Groundability verdict for one must-answer topic (written by the guard)."""

    topic: str
    status: Literal["supported", "unsupported"]
    method: Literal["code", "llm"] = Field(description="code = decided by word matching; llm = model verified")
    coverage: float = Field(description="Share of the topic's key words found in the best matching chunk")
    evidence: list[str] = Field(default_factory=list, description="Chunk IDs or csv:<file> supporting it")
    quote: str = ""
    reason: str = ""


class RequirementFields(BaseModel):
    """The fields the model writes (draft and final)."""

    agent_name: str
    purpose: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[str]
    must_refuse_topics: list[str]


class Requirements(RequirementFields):
    """The saved requirements.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = REQUIREMENTS_VERSION
    brief: str
    documents_folder: str
    documents: list[DocumentSummary]
    skipped_files: list[str] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="Fixes the Manager's code applied, for transparency")
    # Filled by the groundability guard (empty until it runs).
    topic_support: list[TopicSupport] = Field(default_factory=list)
    guard_warnings: list[str] = Field(default_factory=list)
    must_answer_topics: list[str] = Field(min_length=1)
