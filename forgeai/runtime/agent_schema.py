"""
agent_schema.py - what a valid agent.json looks like.

An agent definition is DATA, not code. One generic runner reads any
agent.json and behaves accordingly. Later, the Builder will write these
files and Repair will patch individual fields (e.g. retrieval.top_k).

Every model uses extra="forbid": an unknown key (often a typo such as
"top_kk") is an error instead of being silently ignored.

Example agent.json:
{
  "schema_version": "1.0",
  "name": "policy_assistant",
  "description": "Answers questions about the company's leave and travel policies.",
  "system_instructions": "You help employees understand company policies...",
  "scope": {
    "in_scope_topics": ["annual leave", "sick leave", "travel claims"],
    "refusal_topics": ["legal advice", "salary negotiation"],
    "refusal_message": "I can't answer that from the available policy documents."
  },
  "documents": {"folder": "data/sample_docs"},
  "retrieval": {"chunk_size_words": 300, "chunk_overlap_words": 50, "top_k": 3,
                "min_bm25_score": 0.0, "min_query_coverage": 0.5},
  "tools": [],
  "output_contract": {"max_answer_words": 120, "require_citations": true}
}
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from forgeai import config

SCHEMA_VERSION = "1.0"

# Tool names an agent may list. The tool registry module (built later) will own
# this list and actually run the tools; for now we only validate names, so a typo
# like "calculater" is caught. The runner does not execute tools yet.
KNOWN_TOOL_NAMES = {"csv_lookup", "calculator", "weather_api", "current_date"}

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


class AgentDefinitionError(Exception):
    """agent.json is missing, is not valid JSON, or breaks the schema."""


class StrictModel(BaseModel):
    """Base class: reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


class ScopeConfig(StrictModel):
    in_scope_topics: list[str] = Field(
        min_length=1, description="Topics the agent is meant to answer about"
    )
    refusal_topics: list[str] = Field(
        default_factory=list, description="Topics the agent must refuse, even if documents mention them"
    )
    refusal_message: str = Field(
        default="I can't answer that from the available documents.",
        min_length=1,
        description="Exact text shown to the user whenever the agent refuses",
    )


class DocumentsConfig(StrictModel):
    folder: str = Field(
        min_length=1,
        description="Document folder. A relative path is resolved from the project root.",
    )

    def resolved_path(self) -> Path:
        path = Path(self.folder)
        return path if path.is_absolute() else config.PROJECT_ROOT / path


class RetrievalConfig(StrictModel):
    chunk_size_words: int = Field(default=config.CHUNK_SIZE_WORDS, ge=20, le=2000)
    chunk_overlap_words: int = Field(default=config.CHUNK_OVERLAP_WORDS, ge=0)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20)
    # Evidence gate (see runtime/evidence.py). Chunks scoring at or below
    # min_bm25_score are ignored. If retrieved chunks contain less than
    # min_query_coverage of the question's key words, the runner refuses without
    # calling the model.
    min_bm25_score: float = Field(default=0.0, ge=0.0)
    min_query_coverage: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self):
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        return self


class OutputContract(StrictModel):
    """
    Rules for the answer. The response SHAPE is fixed for every agent
    (answer, sources, refused, refusal_reason) so the Auditor can check any agent
    the same way; these settings tune how the agent fills it.
    """

    max_answer_words: int = Field(default=150, ge=10, le=1000)
    require_citations: bool = Field(
        default=True, description="If true, an answer without a valid source is turned into a refusal"
    )


class AgentDefinition(StrictModel):
    schema_version: str = SCHEMA_VERSION
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1)
    system_instructions: str = Field(min_length=10)
    scope: ScopeConfig
    documents: DocumentsConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: list[str] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)

    @field_validator("schema_version")
    @classmethod
    def supported_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version '{value}' (expected '{SCHEMA_VERSION}')")
        return value

    @field_validator("name")
    @classmethod
    def simple_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError("name must be lowercase letters, digits, '_' or '-', e.g. 'policy_assistant'")
        return value

    @field_validator("tools")
    @classmethod
    def known_unique_tools(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - KNOWN_TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown tool(s) {unknown}; allowed: {sorted(KNOWN_TOOL_NAMES)}")
        if len(value) != len(set(value)):
            raise ValueError("tools must not contain duplicates")
        return value


def _readable_errors(exc: ValidationError) -> str:
    """Turn Pydantic's error list into one line per problem, e.g. 'retrieval.top_k: ...'."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(top level)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def parse_agent_definition(data: dict) -> AgentDefinition:
    """Validate a dict (already loaded from JSON) and return an AgentDefinition."""
    try:
        return AgentDefinition.model_validate(data)
    except ValidationError as exc:
        raise AgentDefinitionError("Invalid agent definition:\n" + _readable_errors(exc)) from exc


def load_agent_definition(path: str | Path) -> AgentDefinition:
    """Read and validate an agent.json file."""
    path = Path(path)
    if not path.is_file():
        raise AgentDefinitionError(f"Agent definition file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise AgentDefinitionError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentDefinitionError(f"{path} must contain a JSON object at the top level")
    return parse_agent_definition(data)


def save_agent_definition(agent: AgentDefinition, path: str | Path) -> None:
    """Write an agent definition as readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(agent.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
