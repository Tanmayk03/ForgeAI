"""
guard.py - the groundability guard: can the documents support what the brief asks for?

For each must-answer topic:
  1. CODE   search the chunks (BM25) and the CSV files for the topic's key words.
            None found anywhere           -> unsupported (no model call)
            all found in one chunk / CSV  -> supported   (no model call)
  2. LLM    only for in-between cases: the model reads the best chunks and says
            supported / unsupported, quoting one sentence as proof.
  3. CODE   the quote must really be in the chunks; an unverifiable quote means
            "unsupported" (fail closed: better to refuse than hallucinate a gap away).

It also warns (it does not decide) when:
  * a must-REFUSE topic appears in the documents (risk of wrong refusals), and
  * the documents clearly cover keywords that no must-answer topic mentions.

Unsupported topics are recorded in requirements.topic_support; the Builder
turns them into refusal topics, so the agent never "answers" a documentation gap.
"""

import json
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Literal

from pydantic import BaseModel

from forgeai import config
from forgeai.llm_json import generate_json
from forgeai.retrieval.bm25_index import BM25Index, tokenize
from forgeai.retrieval.chunker import chunk_documents
from forgeai.retrieval.loader import load_documents, read_text_file
from forgeai.roles.requirements import Requirements, TopicSupport

TOP_K = 3
QUOTE_MATCH = 0.8      # share of the quote's words that must appear in the chunks


class GuardVerdict(BaseModel):
    verdict: Literal["supported", "unsupported"]
    evidence_quote: str = ""


def _coverage(terms: list[str], text_tokens: set[str]) -> float:
    return sum(t in text_tokens for t in terms) / len(terms) if terms else 0.0


def _quote_found(quote: str, texts: list[str]) -> bool:
    words = tokenize(quote)
    if len(words) < 3:
        return False
    tokens = set()
    for t in texts:
        tokens.update(tokenize(t))
    return _coverage(words, tokens) >= QUOTE_MATCH


def run_guard(
    requirements: Requirements,
    llm_fn: Callable[..., str] | None = None,
    log: list | None = None,
) -> Requirements:
    """Return a copy of requirements with topic_support and guard_warnings filled in."""
    log = [] if log is None else log
    loaded = load_documents(requirements.documents_folder)
    chunks = chunk_documents(loaded.documents)
    index = BM25Index(chunks) if chunks else None
    chunk_tokens = {c.chunk_id: set(tokenize(c.text)) for c in chunks}
    csv_tokens = {
        d.name: set(tokenize(read_text_file(Path(d.path)))) for d in loaded.documents if d.file_type == "csv"
    }
    anywhere = set().union(*chunk_tokens.values(), *csv_tokens.values())  # every word in every document
    template = Template((config.PROMPTS_DIR / "guard_verify.txt").read_text(encoding="utf-8"))

    support: list[TopicSupport] = []
    for topic in requirements.must_answer_topics:
        terms = list(dict.fromkeys(tokenize(topic)))
        results = index.search(topic, top_k=TOP_K) if index else []
        best = max((_coverage(terms, chunk_tokens[r.chunk.chunk_id]) for r in results), default=0.0)
        csv_hits = [name for name, toks in csv_tokens.items() if terms and _coverage(terms, toks) == 1.0]

        if not terms or _coverage(terms, anywhere) == 0.0:
            support.append(TopicSupport(topic=topic, status="unsupported", method="code", coverage=0.0,
                                        reason="none of the topic's key words appear in any document"))
            continue
        full_chunks = [r.chunk.chunk_id for r in results if _coverage(terms, chunk_tokens[r.chunk.chunk_id]) == 1.0]
        if full_chunks or csv_hits:
            support.append(TopicSupport(topic=topic, status="supported", method="code", coverage=1.0,
                                        evidence=full_chunks + [f"csv:{n}" for n in csv_hits],
                                        reason="all key words found together in one passage"))
            continue

        # In-between: some key words found, never all together -> ask the model.
        texts = [r.chunk.text for r in results[:2]]
        verdict = generate_json(
            template.substitute(topic=topic, text="\n\n".join(texts)), GuardVerdict, llm_fn=llm_fn, log=log,
            max_tokens=200,
        )
        if verdict.verdict == "supported" and _quote_found(verdict.evidence_quote, texts):
            support.append(TopicSupport(topic=topic, status="supported", method="llm", coverage=round(best, 3),
                                        evidence=[r.chunk.chunk_id for r in results[:2]],
                                        quote=verdict.evidence_quote, reason="model verified, quote found in text"))
        else:
            reason = ("model said supported but its quote is not in the documents"
                      if verdict.verdict == "supported" else "model found no information on this topic")
            support.append(TopicSupport(topic=topic, status="unsupported", method="llm",
                                        coverage=round(best, 3), quote=verdict.evidence_quote, reason=reason))

    # Warnings (information only)
    warnings = []
    for topic in requirements.must_refuse_topics:
        terms = tokenize(topic)
        hits = [cid for cid, toks in chunk_tokens.items() if terms and _coverage(terms, toks) == 1.0]
        if hits:
            warnings.append(f"refuse topic '{topic}' appears in the documents ({', '.join(hits[:3])}); "
                            "questions the documents answer may be wrongly refused")
    answered_words = {t for topic in requirements.must_answer_topics for t in tokenize(topic)}
    uncovered = [k for d in requirements.documents for k in d.keywords[:5] if k not in answered_words]
    if uncovered:
        warnings.append("documents also cover keywords not in any must-answer topic: "
                        + ", ".join(dict.fromkeys(uncovered)))
    unsupported = [s.topic for s in support if s.status == "unsupported"]
    if unsupported:
        warnings.append(f"UNSUPPORTED by the documents (the agent must refuse these): {unsupported}")

    return requirements.model_copy(update={"topic_support": support, "guard_warnings": warnings})


def save_guard_report(requirements: Requirements, run_dir: str | Path, log: list | None = None) -> Path:
    """Update requirements.json and write groundability.json (+ guard_log.json)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "requirements.json").write_text(
        json.dumps(requirements.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "topic_support": [s.model_dump() for s in requirements.topic_support],
        "warnings": requirements.guard_warnings,
    }
    path = run_dir / "groundability.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if log is not None:
        (run_dir / "guard_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
