"""
judge.py - the Auditor's judge: runs every frozen eval case through the SAME
runner users chat with, and scores it.

Checks per case, code first:
  should ANSWER  passed = not refused AND every key fact present AND grounded
    facts     1. code: exact phrase match (after normalising "6,000" -> "6000")
              2. code: >= 80% of the fact's words present and every number exact
              3. LLM, only if 1-2 fail: one yes/no question per fact, and the
                 model must quote the part of the answer that states it; code
                 checks the quote is really in the answer and has the fact's numbers
    grounded  the cited sources come from the expected documents
              (multi-document cases: every expected document must be cited)
  should REFUSE  passed = refused

Known LLM-judge biases and the defences used here:
  position bias     never compares two answers; one answer, one fact at a time
  verbosity bias    yes/no on a single fact; length never asked about
  self-preference   code decides most facts; the model's "yes" counts only
                    with a quote that code verifies inside the answer
Runner errors (Ollama down...) are recorded as "error", never as pass or fail.
"""

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field

from forgeai import config
from forgeai.llm_client import LLMError
from forgeai.llm_json import LLMJSONError, generate_json
from forgeai.retrieval.bm25_index import tokenize
from forgeai.roles.eval_generator import EvalCase, EvalSet, fact_in_text, verify_frozen
from forgeai.runtime.runner import AgentRunner

FUZZY_SHARE = 0.8
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


class JudgeError(Exception):
    """The eval set cannot be judged (e.g. it was changed after freezing)."""


class _FactVerdict(BaseModel):
    states_fact: bool
    quote: str = ""


class FactCheck(BaseModel):
    fact: str
    found: bool
    method: str = Field(description="exact | fuzzy | llm | llm-rejected | missing")


class CaseResult(BaseModel):
    id: str
    type: str
    question: str
    expected_behavior: str
    status: str = Field(description="pass | fail | error")
    refused: bool = False
    answer: str = ""
    sources: list[str] = Field(default_factory=list)
    cited_docs: list[str] = Field(default_factory=list)
    retrieved_docs: list[str] = Field(default_factory=list)
    expected_docs: list[str] = Field(default_factory=list)
    facts: list[FactCheck] = Field(default_factory=list)
    grounded: bool = False
    runner_decision: str = ""
    refusal_reason: str = ""
    model_answer_before_checks: str = ""
    failure: str = Field(default="", description="Plain description of why it failed")
    latency_seconds: float = 0.0


class Scorecard(BaseModel):
    agent_name: str
    eval_sha256: str
    label: str
    passed: int
    failed: int
    errors: int
    total: int
    pass_rate: float
    by_type: dict[str, dict[str, int]]
    judge_llm_calls: int
    total_seconds: float
    results: list[CaseResult]


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def _doc_of(source_id: str, csv_docs: list[str]) -> list[str]:
    if source_id.startswith("tool:csv_lookup"):
        return csv_docs
    if source_id.startswith("tool:"):
        return []
    return [source_id.split("::")[0]]


def check_fact_by_code(fact: str, answer: str) -> str:
    """'exact', 'fuzzy' or '' (not found)."""
    if fact_in_text(fact, answer):
        return "exact"
    words = tokenize(fact)
    answer_words = set(tokenize(answer))
    if words and sum(w in answer_words for w in words) / len(words) >= FUZZY_SHARE \
            and _numbers(fact) <= _numbers(answer):
        return "fuzzy"
    return ""


class Judge:
    def __init__(self, llm_fn: Callable[..., str] | None = None, log: list | None = None):
        self.llm_fn = llm_fn
        self.log = [] if log is None else log
        self.llm_calls = 0
        self.template = Template((config.PROMPTS_DIR / "judge_fact.txt").read_text(encoding="utf-8"))

    def check_fact(self, fact: str, answer: str) -> FactCheck:
        method = check_fact_by_code(fact, answer)
        if method:
            return FactCheck(fact=fact, found=True, method=method)
        self.llm_calls += 1
        try:
            v = generate_json(self.template.substitute(answer=answer, fact=fact), _FactVerdict,
                              llm_fn=self.llm_fn, log=self.log, max_tokens=150)
        except LLMJSONError:
            return FactCheck(fact=fact, found=False, method="missing")
        if not v.states_fact:
            return FactCheck(fact=fact, found=False, method="missing")
        quote_ok = (bool(tokenize(v.quote)) and check_fact_by_code(v.quote, answer) != ""
                    and _numbers(fact) <= _numbers(v.quote))
        return FactCheck(fact=fact, found=quote_ok, method="llm" if quote_ok else "llm-rejected")

    def judge_case(self, case: EvalCase, runner: AgentRunner) -> CaseResult:
        csv_docs = [d.name for d in runner.load_result.documents if d.file_type == "csv"]
        started = time.perf_counter()
        base = dict(id=case.id, type=case.type, question=case.question,
                    expected_behavior=case.expected_behavior, expected_docs=case.expected_docs)
        try:
            r = runner.ask(case.question)
        except LLMError as exc:
            return CaseResult(**base, status="error", failure=f"runner error: {exc}".splitlines()[0])
        cited = sorted({d for s in r.sources for d in _doc_of(s, csv_docs)})
        result = CaseResult(
            **base, status="fail", refused=r.refused, answer=r.answer, sources=r.sources, cited_docs=cited,
            retrieved_docs=sorted({c.doc_name for c in r.trace.retrieved}),
            runner_decision=r.trace.decision, refusal_reason=r.refusal_reason,
            model_answer_before_checks=r.trace.model_answer_before_checks,
            latency_seconds=round(time.perf_counter() - started, 2),
        )
        if case.expected_behavior == "refuse":
            result.status = "pass" if r.refused else "fail"
            result.failure = "" if r.refused else "answered a question it should refuse"
            return result
        if r.refused:
            result.failure = f"refused an answerable question ({r.trace.decision})"
            return result
        result.facts = [self.check_fact(f, r.answer) for f in case.key_facts]
        need = set(case.expected_docs)
        result.grounded = need <= set(cited) if case.type == "multi_doc" else bool(need & set(cited))
        missing = [f.fact for f in result.facts if not f.found]
        if not missing and result.grounded:
            result.status = "pass"
        else:
            problems = []
            if missing:
                problems.append(f"missing key fact(s): {missing}")
            if not result.grounded:
                problems.append(f"cited {cited or 'nothing'}, expected {sorted(need)}")
            result.failure = "; ".join(problems)
        return result


def run_evals(evals: EvalSet, runner: AgentRunner, label: str = "iteration_0",
              llm_fn: Callable[..., str] | None = None, log: list | None = None,
              progress: Callable[[str], None] | None = None) -> Scorecard:
    if not verify_frozen(evals):
        raise JudgeError("evals.json was changed after it was frozen (hash mismatch); refusing to judge")
    judge = Judge(llm_fn=llm_fn, log=log)
    started = time.perf_counter()
    results = []
    for case in evals.cases:
        res = judge.judge_case(case, runner)
        results.append(res)
        if progress:
            progress(f"{res.id} {res.status:5} [{res.type}] {res.failure}")
    by_type: dict[str, dict[str, int]] = {}
    for res in results:
        row = by_type.setdefault(res.type, {"pass": 0, "fail": 0, "error": 0})
        row[res.status] += 1
    passed = sum(r.status == "pass" for r in results)
    failed = sum(r.status == "fail" for r in results)
    scored = passed + failed
    return Scorecard(
        agent_name=runner.agent.name, eval_sha256=evals.sha256, label=label, passed=passed, failed=failed,
        errors=sum(r.status == "error" for r in results), total=len(results),
        pass_rate=round(passed / scored, 3) if scored else 0.0, by_type=by_type,
        judge_llm_calls=judge.llm_calls, total_seconds=round(time.perf_counter() - started, 1), results=results,
    )


def save_scorecard(card: Scorecard, run_dir: str | Path) -> Path:
    path = Path(run_dir) / f"scorecard_{card.label}.json"
    path.write_text(json.dumps(card.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
