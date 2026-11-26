"""
manager.py - the Manager role: brief + document folder -> requirements.json.

Steps:
  1. INVENTORY  (code)  list the documents with a short preview and keywords
  2. DRAFT      (LLM)   draft requirements + at most 5 clarifying questions,
                        each with a default answer
  3. ASK        (CLI)   ask the requester; an empty answer, or --auto mode,
                        uses the default and records it as "assumed"
  4. FINALIZE   (LLM)   update the draft with the requester's answers
                        (skipped when every answer was assumed)
  5. CLEAN UP   (code)  enforce limits, remove duplicates and contradictions,
                        and record every fix in `notes`

The Manager never sees an agent definition or test; it only writes the spec.
"""

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from string import Template

from forgeai import config
from forgeai.llm_json import generate_json
from forgeai.retrieval.bm25_index import tokenize
from forgeai.retrieval.loader import load_documents, read_text_file
from forgeai.retrieval.models import LoadResult
from forgeai.roles.requirements import (
    Clarification, DocumentSummary, DraftRequirements, RequirementFields, Requirements,
)

MAX_QUESTIONS = 5          # assignment limit
MAX_TOPICS = 12
MAX_TOPIC_WORDS = 8
PREVIEW_WORDS = 50
MAX_DOCS_IN_PROMPT = 15


class ManagerError(Exception):
    """The Manager cannot produce requirements (no documents, unusable brief...)."""


# ---------------------------------------------------------------- 1. inventory

def summarise_documents(loaded: LoadResult) -> list[DocumentSummary]:
    """Code-only overview of each document: size, opening words and top keywords."""
    summaries = []
    for doc in loaded.documents:
        if doc.file_type == "csv":
            lines = read_text_file(Path(doc.path)).splitlines()
            text = " ".join(lines[:4])
            preview = f"CSV columns: {lines[0] if lines else '(empty)'}; {max(len(lines) - 1, 0)} rows"
            pages = 1
        else:
            text = " ".join(p.text for p in doc.pages)
            preview = " ".join(text.split()[:PREVIEW_WORDS])
            pages = len(doc.pages)
        words = [t for t in tokenize(text) if len(t) > 2 and not t.isdigit()]
        summaries.append(DocumentSummary(
            name=doc.name, file_type=doc.file_type, pages=pages, words=len(text.split()),
            chunked=doc.chunked, preview=preview,
            keywords=[w for w, _ in Counter(words).most_common(8)],
        ))
    return summaries


def _documents_text(summaries: list[DocumentSummary]) -> str:
    lines = []
    for s in summaries[:MAX_DOCS_IN_PROMPT]:
        lines.append(f"- {s.name} ({s.file_type}, {s.pages} page(s), {s.words} words)")
        lines.append(f"  keywords: {', '.join(s.keywords)}")
        lines.append(f"  starts: {s.preview}")
    if len(summaries) > MAX_DOCS_IN_PROMPT:
        lines.append(f"- ... and {len(summaries) - MAX_DOCS_IN_PROMPT} more documents")
    return "\n".join(lines)


# ---------------------------------------------------------------- 5. clean up

def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48].strip("_")
    return slug or "agent"


def _clean_topics(topics: list[str], label: str, notes: list[str]) -> list[str]:
    cleaned, seen = [], set()
    for topic in topics:
        topic = " ".join(str(topic).split()[:MAX_TOPIC_WORDS]).strip(" .;,")
        if topic and topic.lower() not in seen:
            seen.add(topic.lower())
            cleaned.append(topic)
    if len(cleaned) > MAX_TOPICS:
        notes.append(f"{label}: kept the first {MAX_TOPICS} of {len(cleaned)} topics")
        cleaned = cleaned[:MAX_TOPICS]
    return cleaned


def clean_fields(fields: RequirementFields, notes: list[str]) -> RequirementFields:
    """Enforce limits and remove contradictions, recording each fix."""
    data = fields.model_dump()
    data["agent_name"] = _slug(data["agent_name"])
    data["must_answer_topics"] = _clean_topics(data["must_answer_topics"], "must_answer_topics", notes)
    data["must_refuse_topics"] = _clean_topics(data["must_refuse_topics"], "must_refuse_topics", notes)

    refuse = {t.lower() for t in data["must_refuse_topics"]}
    both = [t for t in data["must_answer_topics"] if t.lower() in refuse]
    if both:
        notes.append(f"topics listed as both answer and refuse were kept as refuse only: {both}")
        data["must_answer_topics"] = [t for t in data["must_answer_topics"] if t.lower() not in refuse]
    for key in ("purpose", "scope", "audience", "tone"):
        data[key] = " ".join(str(data[key]).split())
    if not data["must_answer_topics"]:
        raise ManagerError("the model produced no must-answer topics; the brief may be too vague")
    return RequirementFields(**{k: data[k] for k in RequirementFields.model_fields})


# ---------------------------------------------------------------- main entry

InputFunction = Callable[[str], str]


def run_manager(
    brief: str,
    docs_folder: str | Path,
    auto: bool = False,
    input_fn: InputFunction = input,
    llm_fn: Callable[..., str] | None = None,
    log: list | None = None,
) -> Requirements:
    """Produce validated Requirements. `log` collects every prompt and raw reply."""
    brief = " ".join(brief.split())
    if len(brief.split()) < 5:
        raise ManagerError("the brief is too short; describe the assistant in at least one sentence")
    log = [] if log is None else log

    # 1. INVENTORY
    try:
        loaded = load_documents(docs_folder)
    except FileNotFoundError as exc:
        raise ManagerError(str(exc)) from exc
    summaries = summarise_documents(loaded)
    if not summaries:
        raise ManagerError(f"no usable documents in {docs_folder}")

    # 2. DRAFT
    draft_prompt = Template((config.PROMPTS_DIR / "manager_draft.txt").read_text(encoding="utf-8")).substitute(
        brief=brief, documents=_documents_text(summaries),
    )
    draft = generate_json(draft_prompt, DraftRequirements, llm_fn=llm_fn, log=log, max_tokens=900)

    notes: list[str] = []
    questions = draft.questions
    if len(questions) > MAX_QUESTIONS:
        notes.append(f"model proposed {len(questions)} questions; only the first {MAX_QUESTIONS} were asked")
        questions = questions[:MAX_QUESTIONS]

    # 3. ASK
    clarifications = []
    for number, q in enumerate(questions, start=1):
        answer = ""
        if not auto:
            answer = input_fn(f"\nQ{number}/{len(questions)}: {q.question}\n  (Enter = {q.default_answer})\n  > ").strip()
        if answer:
            clarifications.append(Clarification(question=q.question, answer=answer, source="requester"))
        else:
            clarifications.append(Clarification(question=q.question, answer=q.default_answer, source="assumed"))

    # 4. FINALIZE (only if the requester actually answered something)
    fields: RequirementFields = draft
    if any(c.source == "requester" for c in clarifications):
        answers = "\n".join(f"- Q: {c.question}\n  A: {c.answer}" for c in clarifications)
        draft_json = json.dumps(draft.model_dump(exclude={"questions"}), indent=1)
        final_prompt = Template((config.PROMPTS_DIR / "manager_finalize.txt").read_text(encoding="utf-8")).substitute(
            brief=brief, draft=draft_json, answers=answers,
        )
        fields = generate_json(final_prompt, RequirementFields, llm_fn=llm_fn, log=log, max_tokens=900)

    # 5. CLEAN UP
    fields = clean_fields(fields, notes)
    return Requirements(
        **fields.model_dump(),
        brief=brief,
        documents_folder=str(docs_folder),
        documents=summaries,
        skipped_files=[f"{s.name}: {s.reason}" for s in loaded.skipped],
        clarifications=clarifications,
        notes=notes,
    )


def save_requirements(requirements: Requirements, run_dir: str | Path, log: list | None = None) -> Path:
    """Write requirements.json (and manager_log.json with prompts and raw replies)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "requirements.json"
    path.write_text(json.dumps(requirements.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    if log is not None:
        (run_dir / "manager_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_requirements(path: str | Path) -> Requirements:
    return Requirements.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
