"""
builder.py - the Builder role: requirements.json -> agent.json

Most decisions are made by CODE with a logged reason, because they must be
predictable and explainable; the model writes only the persona paragraph.

  TOOLS       rule-based. A tool is attached only if the requirements show a need:
                csv_lookup   - the folder has a CSV whose words overlap the brief/topics
                calculator   - the requirements mention totals, prices, sums, ...
                weather_api  - the requirements mention weather, forecast, ...
                current_date - the requirements mention today, deadlines, ...
              Every tool gets an attached/not-attached reason in build_log.json.
  SCOPE       in-scope = must-answer topics the guard marked SUPPORTED.
  REFUSALS    must-refuse topics + topics the guard marked UNSUPPORTED, so the
              agent refuses documentation gaps instead of inventing answers.
  RETRIEVAL   chunk size / top_k from document sizes, kept within the context window.
  CONTRACT    citations required, unsupported numbers blocked, answer length from tone.
  PROMPT      the model writes 2-4 persona sentences; code checks them and falls back
              to a template if they are unusable. Grounding and refusal rules are NOT
              written by the model: they come from prompts/runner_system.txt.
"""

import json
from collections.abc import Callable
from pathlib import Path
from string import Template

from pydantic import BaseModel

from forgeai import config
from forgeai.llm_json import LLMJSONError, generate_json
from forgeai.retrieval.bm25_index import tokenize
from forgeai.retrieval.loader import load_documents, read_text_file
from forgeai.roles.requirements import Requirements
from forgeai.runtime.agent_schema import AgentDefinition, parse_agent_definition

# Words that signal a need for each tool (matched against the requirements' words).
TOOL_TRIGGERS = {
    "calculator": {"total", "totals", "cost", "costs", "price", "prices", "pricing", "sum", "calculate",
                   "calculation", "quantity", "quantities", "amount", "amounts", "budget", "average",
                   "percent", "percentage", "discount"},
    "weather_api": {"weather", "forecast", "forecasts", "temperature", "rain", "wind", "climate"},
    "current_date": {"today", "date", "dates", "deadline", "deadlines", "schedule", "calendar",
                     "expiry", "expire", "expires", "remaining", "overdue"},
}
# Words that weaken a "persona" paragraph's grounding; such text is replaced by the template.
FORBIDDEN_IN_PERSONA = ("general knowledge", "your own knowledge", "make up", "guess", "speculate")
CONTEXT_SHARE = 0.6   # retrieved chunks may use at most this share of the context window


class BuilderError(Exception):
    """The Builder cannot build an agent from these requirements."""


class PersonaOutput(BaseModel):
    system_instructions: str


def _requirement_words(req: Requirements) -> set[str]:
    text = " ".join([req.brief, req.purpose, req.scope, *req.must_answer_topics])
    return set(tokenize(text))


def select_tools(req: Requirements, csv_paths: list[Path]) -> list[dict]:
    """Decide every tool, with a reason. Returns [{"name", "attached", "reason"}]."""
    words = _requirement_words(req)
    decisions = []

    related = []
    for path in csv_paths:
        overlap = sorted(words & set(tokenize(read_text_file(path))))
        if overlap:
            related.append(f"{path.name} (shared words: {', '.join(overlap[:5])})")
    if related:
        decisions.append({"name": "csv_lookup", "attached": True, "reason": "CSV data related to the brief: " + "; ".join(related)})
    elif csv_paths:
        decisions.append({"name": "csv_lookup", "attached": False,
                          "reason": f"CSV file(s) present ({', '.join(p.name for p in csv_paths)}) but unrelated to the brief"})
    else:
        decisions.append({"name": "csv_lookup", "attached": False, "reason": "no CSV files in the document folder"})

    for name, triggers in TOOL_TRIGGERS.items():
        hits = sorted(words & triggers)
        decisions.append({"name": name, "attached": bool(hits),
                          "reason": f"requirements mention: {', '.join(hits)}" if hits else "no need found in the requirements"})
    return decisions


def choose_retrieval(req: Requirements) -> tuple[dict, str]:
    """Chunk size and top_k from document sizes, kept inside the context window."""
    text_docs = [d for d in req.documents if d.chunked]
    words_per_page = (sum(d.words for d in text_docs) / max(sum(d.pages for d in text_docs), 1)) if text_docs else 0
    chunk = 200 if words_per_page and words_per_page < 200 else 300
    top_k = 4 if len(text_docs) > 4 else 3
    while top_k > 1 and top_k * chunk * 1.3 > CONTEXT_SHARE * config.NUM_CTX:
        top_k -= 1
    reason = (f"{len(text_docs)} text document(s), about {words_per_page:.0f} words per page -> "
              f"chunks of {chunk} words, top_k {top_k} (fits {CONTEXT_SHARE:.0%} of a {config.NUM_CTX}-token context)")
    settings = {"chunk_size_words": chunk, "chunk_overlap_words": chunk // 6, "top_k": top_k,
                "min_bm25_score": 0.0, "min_query_coverage": 0.5}
    return settings, reason


def template_persona(req: Requirements) -> str:
    return (f"You are an assistant whose purpose is: {req.purpose.rstrip('.')}. "
            f"You help {req.audience.rstrip('.')}. Use a {req.tone.rstrip('.').lower()} tone.")


def write_persona(req: Requirements, llm_fn, log: list) -> tuple[str, str]:
    """Model-written persona, checked in code; template fallback. Returns (text, how)."""
    prompt = Template((config.PROMPTS_DIR / "builder_instructions.txt").read_text(encoding="utf-8")).substitute(
        purpose=req.purpose, audience=req.audience, tone=req.tone, scope=req.scope)
    try:
        text = " ".join(generate_json(prompt, PersonaOutput, llm_fn=llm_fn, log=log, max_tokens=250)
                        .system_instructions.split())
    except LLMJSONError as exc:
        return template_persona(req), f"template (model output unusable: {exc})"
    words = len(text.split())
    if not 10 <= words <= 120:
        return template_persona(req), f"template (model text had {words} words; allowed 10-120)"
    bad = [p for p in FORBIDDEN_IN_PERSONA if p in text.lower()]
    if bad:
        return template_persona(req), f"template (model text weakened grounding: {bad})"
    return text, "model"


def _portable_folder(folder: str) -> str:
    """Store the folder relative to the project when possible, so runs can be committed."""
    path = Path(folder)
    path = path if path.is_absolute() else Path.cwd() / path
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run_builder(req: Requirements, llm_fn: Callable[..., str] | None = None,
                log: list | None = None) -> tuple[AgentDefinition, dict]:
    """Return (agent definition, build log)."""
    log = [] if log is None else log
    if not req.topic_support:
        raise BuilderError("requirements have no groundability results; run the guard first")

    supported = [s.topic for s in req.topic_support if s.status == "supported"]
    unsupported = [s.topic for s in req.topic_support if s.status == "unsupported"]
    if not supported:
        raise BuilderError("the documents support none of the must-answer topics; "
                           "building an agent would only produce refusals or hallucinations")

    loaded = load_documents(req.documents_folder)
    csv_paths = [Path(d.path) for d in loaded.documents if d.file_type == "csv"]
    tool_decisions = select_tools(req, csv_paths)
    retrieval, retrieval_reason = choose_retrieval(req)
    persona, persona_how = write_persona(req, llm_fn, log)
    refusals = list(dict.fromkeys([*req.must_refuse_topics, *unsupported]))
    concise = any(w in req.tone.lower() for w in ("concise", "brief", "short"))

    agent = parse_agent_definition({
        "schema_version": "1.0",
        "name": req.agent_name,
        "description": req.purpose,
        "system_instructions": persona,
        "scope": {
            "in_scope_topics": supported,
            "refusal_topics": refusals,
            "refusal_message": "I can't answer that from the available documents.",
        },
        "documents": {"folder": _portable_folder(req.documents_folder)},
        "retrieval": retrieval,
        "tools": [t["name"] for t in tool_decisions if t["attached"]],
        "output_contract": {"max_answer_words": 80 if concise else 150,
                            "require_citations": True, "block_unsupported_numbers": True},
    })
    build_log = {
        "tools": tool_decisions,
        "retrieval": retrieval_reason,
        "system_instructions_source": persona_how,
        "in_scope_topics": supported,
        "refusal_topics_added_from_guard": unsupported,
        "known_risks": req.guard_warnings,
    }
    return agent, build_log


def save_build(agent: AgentDefinition, build_log: dict, run_dir: str | Path, log: list | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "agent.json"
    path.write_text(json.dumps(agent.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "build_log.json").write_text(json.dumps(build_log, indent=2, ensure_ascii=False), encoding="utf-8")
    if log is not None:
        (run_dir / "builder_llm_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
