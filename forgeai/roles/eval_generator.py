"""
eval_generator.py - the Auditor's test-set generator.

INDEPENDENCE: this module reads only requirements.json and the documents.
It never reads agent.json, and it chunks the documents with its own default
settings, so expected sources are DOCUMENT NAMES (chunk IDs would differ
between the agent and the Auditor).

Case types (targets for a 24-case set):
  answerable    12  model writes questions from one passage; key facts must be
                    copied from that passage (checked in code)
  multi_doc      3  code combines two accepted questions from DIFFERENT documents into one;
                    the answer must contain both facts. (Qwen 1.5B, asked to write such
                    questions, wrote single-document questions with unrelated answer keys.
                    Limitation: this tests answers spanning two documents, not reasoning
                    across them.)
  ambiguous      3  informal / vague rewrites of accepted answerable questions
  out_of_scope   6+ refuse-topic questions, guard-unsupported topics, and
                    unrelated questions (checked: the documents barely mention them)
  (+ csv rows)      if a CSV relates to the brief, row questions built by code

Every generated case is CHECKED in code before it is kept:
  * every key fact appears in its source passage (no invented answer keys)
  * the question does not contain its own answer
  * no near-duplicate questions
Rejected cases are saved with the reason. The final set is FROZEN with a
SHA-256 hash, so later changes to the tests would be detectable.
Fewer than 20 valid cases is reported honestly, never padded.
"""

import csv
import hashlib
import io
import json
import math
import random
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Literal

from pydantic import BaseModel, Field

from forgeai import config
from forgeai.llm_json import LLMJSONError, generate_json
from forgeai.retrieval.bm25_index import tokenize
from forgeai.retrieval.chunker import chunk_documents
from forgeai.retrieval.loader import load_documents, read_text_file
from forgeai.retrieval.models import Chunk
from forgeai.roles.requirements import Requirements

TARGETS = {"answerable": 12, "multi_doc": 3, "ambiguous": 3, "unrelated": 3}
MIN_CASES = 20
MAX_PER_CHUNK = 4
MAX_FACT_WORDS = 8
DUPLICATE_JACCARD = 0.75
UNRELATED_MAX_COVERAGE = 0.34   # an "unrelated" question may share at most this share of words with the docs


# ------------------------------------------------------------------ data shapes

class EvalCase(BaseModel):
    id: str
    type: Literal["answerable", "multi_doc", "ambiguous", "out_of_scope"]
    question: str
    expected_behavior: Literal["answer", "refuse"]
    key_facts: list[str] = Field(default_factory=list, description="Phrases a correct answer must contain")
    expected_docs: list[str] = Field(default_factory=list, description="Documents that contain the facts")
    source_text: str = Field(default="", description="Passage(s) the case was generated from (for the judge)")
    origin: str = Field(description="How the case was made, e.g. 'passage leave_policy.pdf p1'")


class RejectedCase(BaseModel):
    question: str
    reason: str
    origin: str


class EvalSet(BaseModel):
    schema_version: str = "1.0"
    created_at: str
    model: str
    sha256: str = ""
    counts: dict[str, int]
    notes: list[str] = Field(default_factory=list)
    cases: list[EvalCase]


class _QA(BaseModel):
    question: str
    key_facts: list[str]


class _QAList(BaseModel):
    cases: list[_QA]


class _Rewrites(BaseModel):
    rewrites: list[str]


class _TopicQuestion(BaseModel):
    topic: str
    question: str


class _RefuseSet(BaseModel):
    topic_questions: list[_TopicQuestion]
    unrelated_questions: list[str]


# ------------------------------------------------------------------ text checks

def normalise(text: str) -> str:
    """Lowercase, '6,000' -> '6000', punctuation -> spaces, single spaces."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", text).replace(". ", " ").split()).strip(" .")


def fact_in_text(fact: str, text: str) -> bool:
    f = normalise(fact)
    return bool(f) and f" {f} " in f" {normalise(text)} "


FACT_STOPWORDS = {"of", "the", "a", "an", "and", "or", "per", "to", "in", "for", "on", "at", "is", "are"}
# A number followed by 1-2 words, ignoring list markers such as "3." at a sentence start.
NUMBER_PHRASE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?!\.(?:\s|$))\s+([A-Za-z]+)(?:\s+([A-Za-z]+))?")


def strengthen_fact(fact: str, passage: str) -> str:
    """
    Make a key fact precise, in code:
      bare number  '24'  -> '24 working days'   (next words from the passage)
      long sentence      -> its first number phrase, e.g. '5 unused days'
    Anything else is returned unchanged (the normal checks still apply).
    """
    f = normalise(fact)
    if re.fullmatch(r"[\d.]+", f):
        tokens = normalise(passage).split()
        if f in tokens:
            i = tokens.index(f)
            words = [t for t in tokens[i + 1:i + 3]]
            while words and words[-1] in FACT_STOPWORDS:
                words.pop()
            return " ".join([f, *words])
        return fact
    if len(fact.split()) > MAX_FACT_WORDS:
        for m in NUMBER_PHRASE.finditer(fact):
            words = [w for w in m.groups()[1:] if w]
            while words and words[-1].lower() in FACT_STOPWORDS:
                words.pop()
            phrase = " ".join([m.group(1), *words])
            if fact_in_text(phrase, passage):
                return phrase
    return fact


def _jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x | y else 1.0


def check_answerable(question: str, facts: list[str], passages: list[str], kept: list[EvalCase]) -> str:
    """Return '' if the case is valid, otherwise the rejection reason."""
    if len(tokenize(question)) < 2 or len(question.split()) < 4:
        return "question too short"
    facts = [f.strip() for f in facts if f.strip()]
    if not facts:
        return "no key facts"
    for fact in facts:
        if len(fact.split()) > MAX_FACT_WORDS:
            return f"key fact longer than {MAX_FACT_WORDS} words: {fact!r}"
        if not any(fact_in_text(fact, p) for p in passages):
            return f"key fact not found in the source passage: {fact!r}"
    if all(fact_in_text(f, question) for f in facts):
        return "question contains its own answer"
    if any(_jaccard(question, k.question) >= DUPLICATE_JACCARD for k in kept):
        return "near-duplicate of an existing question"
    return ""


# ------------------------------------------------------------------ generator

class _Builder:
    """Holds the state of one generation run."""

    def __init__(self, req: Requirements, llm_fn, log):
        self.req, self.llm_fn, self.log = req, llm_fn, log
        self.cases: list[EvalCase] = []
        self.rejected: list[RejectedCase] = []
        self.notes: list[str] = []
        self.rng = random.Random(config.SEED)

    def prompt(self, name: str, **values) -> str:
        return Template((config.PROMPTS_DIR / name).read_text(encoding="utf-8")).substitute(**values)

    def ask(self, prompt: str, model_cls, origin: str):
        try:
            return generate_json(prompt, model_cls, llm_fn=self.llm_fn, log=self.log, max_tokens=700)
        except LLMJSONError as exc:
            self.rejected.append(RejectedCase(question="(none)", reason=f"model output unusable: {exc}", origin=origin))
            return None

    def add(self, type_, question, behavior, facts, docs, source, origin):
        self.cases.append(EvalCase(
            id=f"q{len(self.cases) + 1:02d}", type=type_, question=" ".join(question.split()),
            expected_behavior=behavior, key_facts=facts, expected_docs=docs, source_text=source, origin=origin,
        ))

    def reject(self, question, reason, origin):
        self.rejected.append(RejectedCase(question=question, reason=reason, origin=origin))

    def docs_with_facts(self, facts: list[str], chunks: list[Chunk]) -> list[str]:
        """Every document containing all the facts: an answer citing any of them is grounded."""
        return sorted({c.doc_name for c in chunks if all(fact_in_text(f, c.text) for f in facts)})

    # -- answerable
    def answerable(self, chunks: list[Chunk]):
        target = TARGETS["answerable"]
        per_chunk = min(MAX_PER_CHUNK, max(1, math.ceil(target / len(chunks))))
        order = chunks[:]
        self.rng.shuffle(order)
        for round_ in range(2):  # second pass only if the first did not reach the target
            for chunk in order:
                done = sum(c.type == "answerable" for c in self.cases)
                if done >= target:
                    return
                existing = [c.question for c in self.cases if c.origin.startswith(f"passage {chunk.chunk_id}")]
                avoid = ("Do not repeat these questions:\n" + "\n".join(f"- {q}" for q in existing)) if existing else ""
                origin = f"passage {chunk.chunk_id}"
                out = self.ask(self.prompt("auditor_answerable.txt", audience=self.req.audience, doc=chunk.doc_name,
                                           passage=chunk.text, count=min(per_chunk, target - done), avoid=avoid),
                               _QAList, origin)
                for qa in (out.cases if out else []):
                    facts = [strengthen_fact(f.strip(), chunk.text) for f in qa.key_facts if f.strip()]
                    reason = check_answerable(qa.question, facts, [chunk.text], self.cases)
                    if reason:
                        self.reject(qa.question, reason, origin)
                        continue
                    self.add("answerable", qa.question, "answer", facts,
                             self.docs_with_facts(facts, chunks) or [chunk.doc_name], chunk.text, origin)

    # -- multi-document (code-built)
    def combined_multi_doc(self):
        """Join two accepted questions from different documents into one question needing both."""
        need = TARGETS["multi_doc"] - sum(c.type == "multi_doc" for c in self.cases)
        base = [c for c in self.cases if c.type == "answerable" and len(c.expected_docs) == 1]
        pairs = [(x, y) for i, x in enumerate(base) for y in base[i + 1:] if x.expected_docs != y.expected_docs]
        self.rng.shuffle(pairs)
        used: set[str] = set()
        for x, y in pairs:
            if need <= 0:
                break
            if x.id in used or y.id in used:
                continue
            second = y.question[0].lower() + y.question[1:]
            question = f"{x.question.rstrip(' ?')}, and {second.rstrip(' ?')}?"
            self.add("multi_doc", question, "answer", x.key_facts + y.key_facts,
                     sorted({*x.expected_docs, *y.expected_docs}), x.source_text + "\n\n" + y.source_text,
                     f"combined {x.id} + {y.id}")
            used.update({x.id, y.id})
            need -= 1
        if need > 0:
            self.notes.append(f"{need} multi-document case(s) short: not enough questions from different documents")

    # -- ambiguous rewrites
    def ambiguous(self):
        base = [c for c in self.cases if c.type == "answerable"]
        picked = self.rng.sample(base, min(TARGETS["ambiguous"], len(base)))
        if not picked:
            return
        out = self.ask(self.prompt("auditor_rephrase.txt", questions="\n".join(f"- {c.question}" for c in picked)),
                       _Rewrites, "rephrase")
        for original, rewrite in zip(picked, out.rewrites if out else []):
            origin = f"rephrase of {original.id}"
            if len(rewrite.split()) < 3:
                self.reject(rewrite, "rewrite too short", origin)
            elif _jaccard(rewrite, original.question) >= 0.8:
                self.reject(rewrite, "rewrite too similar to the original", origin)
            elif all(fact_in_text(f, rewrite) for f in original.key_facts):
                self.reject(rewrite, "question contains its own answer", origin)
            else:
                self.add("ambiguous", rewrite, "answer", original.key_facts, original.expected_docs,
                         original.source_text, origin)

    # -- must refuse
    def out_of_scope(self, all_tokens: set[str]):
        unsupported = [s.topic for s in self.req.topic_support if s.status == "unsupported"]
        topics = self.req.must_refuse_topics
        if topics or TARGETS["unrelated"]:
            out = self.ask(self.prompt("auditor_refuse.txt", scope=self.req.scope, topics="; ".join(topics) or "(none)",
                                       unrelated=TARGETS["unrelated"] + 1), _RefuseSet, "refuse topics")
            if out:
                for tq in out.topic_questions:
                    origin = f"refuse topic '{tq.topic}'"
                    topic_words = set(tokenize(tq.topic))
                    if tq.topic not in topics:
                        self.reject(tq.question, "topic is not one of the must-refuse topics", origin)
                    elif len(tq.question.split()) < 4:
                        self.reject(tq.question, "question too short", origin)
                    else:
                        if not topic_words & set(tokenize(tq.question)):
                            self.notes.append(f"{origin}: question does not name the topic word; kept")
                        self.add("out_of_scope", tq.question, "refuse", [], [], "", origin)
                kept = 0
                for q in out.unrelated_questions:
                    words = tokenize(q)
                    coverage = sum(w in all_tokens for w in words) / len(words) if words else 1.0
                    if len(q.split()) < 4:
                        self.reject(q, "question too short", "unrelated")
                    elif coverage > UNRELATED_MAX_COVERAGE:
                        self.reject(q, f"not clearly unrelated: {coverage:.0%} of its words appear in the documents",
                                    "unrelated")
                    elif kept < TARGETS["unrelated"]:
                        self.add("out_of_scope", q, "refuse", [], [], "", "unrelated question")
                        kept += 1
        for topic in unsupported:   # documentation gaps found by the guard: built by code
            self.add("out_of_scope", f"What information is available about {topic}?", "refuse", [], [], "",
                     f"guard: unsupported topic '{topic}'")

    # -- CSV rows (code-built)
    def csv_rows(self, loaded, chunks):
        words = set(tokenize(" ".join([self.req.brief, self.req.scope, *self.req.must_answer_topics])))
        for doc in loaded.documents:
            if doc.file_type != "csv":
                continue
            text = read_text_file(Path(doc.path))
            if not words & set(tokenize(text)):
                self.notes.append(f"{doc.name}: unrelated to the brief; no CSV cases")
                continue
            rows = list(csv.DictReader(io.StringIO(text)))
            if not rows or len(rows[0]) < 2:
                continue
            key_col, *value_cols = list(rows[0].keys())
            for row in self.rng.sample(rows, min(3, len(rows))):
                col = self.rng.choice(value_cols)
                if not row.get(key_col) or not row.get(col):
                    continue
                self.add("answerable", f"What is the {col.replace('_', ' ')} for {row[key_col]}?", "answer",
                         [row[col]], [doc.name], f"{doc.name}: " + "; ".join(f"{k}={v}" for k, v in row.items()),
                         f"csv row {doc.name} {key_col}={row[key_col]}")


def generate_evals(req: Requirements, llm_fn: Callable[..., str] | None = None,
                   log: list | None = None) -> tuple[EvalSet, list[RejectedCase]]:
    """Build, check and freeze the test set. Reads requirements + documents only."""
    log = [] if log is None else log
    loaded = load_documents(req.documents_folder)
    chunks = chunk_documents(loaded.documents)   # Auditor's own chunking (default settings)
    b = _Builder(req, llm_fn, log)
    all_tokens = set().union(*(set(tokenize(c.text)) for c in chunks)) if chunks else set()

    if chunks:
        b.answerable(chunks)
        b.combined_multi_doc()
    b.csv_rows(loaded, chunks)
    b.ambiguous()
    b.out_of_scope(all_tokens)

    counts = {t: sum(c.type == t for c in b.cases) for t in ("answerable", "multi_doc", "ambiguous", "out_of_scope")}
    counts["total"] = len(b.cases)
    counts["rejected"] = len(b.rejected)
    if len(b.cases) < MIN_CASES:
        b.notes.append(f"only {len(b.cases)} valid cases (target at least {MIN_CASES}); "
                       "the documents may be too small. The set was NOT padded.")
    evals = EvalSet(created_at=datetime.now().isoformat(timespec="seconds"), model=config.MODEL_NAME,
                    counts=counts, notes=b.notes, cases=b.cases)
    return freeze(evals), b.rejected


def _cases_hash(evals: EvalSet) -> str:
    payload = json.dumps([c.model_dump() for c in evals.cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze(evals: EvalSet) -> EvalSet:
    return evals.model_copy(update={"sha256": _cases_hash(evals)})


def verify_frozen(evals: EvalSet) -> bool:
    """True if the cases are unchanged since freezing."""
    return bool(evals.sha256) and evals.sha256 == _cases_hash(evals)


def save_evals(evals: EvalSet, rejected: list[RejectedCase], run_dir: str | Path, log: list | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "evals.json"
    path.write_text(json.dumps(evals.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "evals_rejected.json").write_text(
        json.dumps([r.model_dump() for r in rejected], indent=2, ensure_ascii=False), encoding="utf-8")
    if log is not None:
        (run_dir / "eval_generator_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_evals(path: str | Path) -> EvalSet:
    return EvalSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
