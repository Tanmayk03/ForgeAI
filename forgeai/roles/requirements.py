"""
requirements.py - the shape of requirements.json, written by the Manager.

requirements.json is the contract between the roles: the Builder builds the
agent from it, and the Auditor writes tests from it (plus the documents).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REQUIREMENTS_VERSION = "1.0"


class ClarifyingQuestion(BaseModel):
    question: str
    default_answer: str = Field(description="Assumption used if the requester gives no answer")


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


class RequirementFields(BaseModel):
    """The fields the model writes (draft and final)."""

    agent_name: str
    purpose: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[str]
    must_refuse_topics: list[str]


class DraftRequirements(RequirementFields):
    questions: list[ClarifyingQuestion] = Field(default_factory=list)


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
    must_answer_topics: list[str] = Field(min_length=1)
