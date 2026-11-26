"""
repair.py - the Repair role: classify every failure, apply ONE targeted fix,
re-run ALL evals, keep the fix only if MORE cases pass. Max 3 iterations.

Repair is CODE ONLY (no model):
  * classification uses the judge's evidence with fixed rules (below), so it is
    repeatable and cannot be talked into a convenient label;
  * fixes come from a fixed library of small patches to agent.json, so a fix can
    never paste test questions into the prompt (no overfitting by copying) and is
    never a full regeneration.

Classification (should-answer cases):
  refused by the evidence gate, expected document retrieved   -> wrong_refusal  (gate)
  refused, expected document NOT retrieved                    -> retrieval_miss
  model refused, expected document retrieved                  -> wrong_refusal  (model)
  blocked number / no citation, model's own answer HAD facts  -> prompt_defect  (citation)
  blocked number / no citation, model's answer lacked facts   -> hallucination
  answered, expected document not retrieved                   -> retrieval_miss
  answered, facts missing, multi-document case                -> prompt_defect  (multi_part)
  answered, facts missing                                     -> prompt_defect  (incomplete)
  answered, facts present, wrong documents cited              -> prompt_defect  (citation)
Should-refuse cases that were answered:                         -> prompt_defect  (refusal_rule)
"""

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from forgeai.roles.eval_generator import EvalSet
from forgeai.roles.judge import CaseResult, Scorecard, check_fact_by_code, run_evals, save_scorecard
from forgeai.runtime.agent_schema import AgentDefinition, parse_agent_definition, save_agent_definition
from forgeai.runtime.runner import AgentRunner

MAX_ITERATIONS = 3
MAX_TOP_K = 6
MIN_COVERAGE_FLOOR = 0.2
COVERAGE_STEP = 0.15

INSTRUCTIONS = {
    "incomplete": "Always include the exact figures, units and conditions from the SOURCES in your answer.",
    "citation": "Cite every source you use. If you combine facts from two sources, cite both source numbers.",
    "multi_part": "If the question asks more than one thing, answer every part, each from its own source.",
    "hallucination": "Never state a number that is not written in the SOURCES; refuse instead.",
    "model": "If the SOURCES contain the answer, give it, even when the question uses different words.",
    "refusal_rule": "Refuse questions about the refusal topics even when the SOURCES mention related words.",
}
GATE_DECISIONS = {"refused_no_evidence", "refused_low_coverage"}
BLOCKED_DECISIONS = {"refused_unsupported_number", "refused_no_citation", "refused_invalid_output"}


class Failure(BaseModel):
    id: str
    category: str = Field(description="retrieval_miss | prompt_defect | hallucination | wrong_refusal")
    subtype: str
    evidence: str


class IterationLog(BaseModel):
    iteration: int
    failures_by_category: dict[str, int]
    failures: list[Failure]
    target_category: str = ""
    target_subtype: str = ""
    targeted_cases: list[str] = Field(default_factory=list)
    change: str = ""
    field: str = ""
    old_value: str = ""
    new_value: str = ""
    score_before: str = ""
    score_after: str = ""
    fixed_cases: list[str] = Field(default_factory=list)
    broken_cases: list[str] = Field(default_factory=list)
    kept: bool = False
    note: str = ""


def classify(result: CaseResult, key_facts: list[str]) -> Failure:
    retrieved_ok = bool(set(result.expected_docs) & set(result.retrieved_docs))
    if result.expected_behavior == "refuse":
        return Failure(id=result.id, category="prompt_defect", subtype="refusal_rule",
                       evidence="answered a question that should be refused")
    if result.refused:
        d = result.runner_decision
        if d in BLOCKED_DECISIONS:
            own = result.model_answer_before_checks
            had_facts = bool(own) and all(check_fact_by_code(f, own) for f in key_facts)
            if had_facts:
                return Failure(id=result.id, category="prompt_defect", subtype="citation",
                               evidence=f"model's answer had the facts but was blocked ({d})")
            return Failure(id=result.id, category="hallucination", subtype="hallucination",
                           evidence=f"model's answer lacked the facts and was blocked ({d}): {own[:80]!r}")
        if not retrieved_ok:
            return Failure(id=result.id, category="retrieval_miss", subtype="retrieval",
                           evidence=f"expected {result.expected_docs}, retrieved {result.retrieved_docs}")
        if d in GATE_DECISIONS:
            return Failure(id=result.id, category="wrong_refusal", subtype="gate",
                           evidence=f"evidence gate refused ({d}) although the right document was retrieved")
        return Failure(id=result.id, category="wrong_refusal", subtype="model",
                       evidence="model refused although the right document was retrieved")
    if not set(result.expected_docs) <= set(result.retrieved_docs):
        return Failure(id=result.id, category="retrieval_miss", subtype="retrieval",
                       evidence=f"expected {result.expected_docs}, retrieved {result.retrieved_docs}")
    missing = [f.fact for f in result.facts if not f.found]
    if missing:
        sub = "multi_part" if result.type == "multi_doc" else "incomplete"
        return Failure(id=result.id, category="prompt_defect", subtype=sub, evidence=f"missing {missing}")
    return Failure(id=result.id, category="prompt_defect", subtype="citation",
                   evidence=f"cited {result.cited_docs}, expected {result.expected_docs}")


def classify_all(card: Scorecard, evals: EvalSet) -> list[Failure]:
    facts = {c.id: c.key_facts for c in evals.cases}
    return [classify(r, facts[r.id]) for r in card.results if r.status == "fail"]


def propose_patch(agent: AgentDefinition, category: str, subtype: str) -> tuple[dict, str, str, str, str] | None:
    """Return (new agent data, change, field, old, new), or None if this fix is used up."""
    data = agent.model_dump()
    if category == "retrieval_miss":
        k = data["retrieval"]["top_k"]
        if k < MAX_TOP_K:
            data["retrieval"]["top_k"] = k + 1
            return data, "retrieve one more chunk per question", "retrieval.top_k", str(k), str(k + 1)
        size = data["retrieval"]["chunk_size_words"]
        if size > 120:
            new = max(120, size - 100)
            data["retrieval"]["chunk_size_words"] = new
            data["retrieval"]["chunk_overlap_words"] = new // 6
            return data, "smaller chunks (more precise matches)", "retrieval.chunk_size_words", str(size), str(new)
        return None
    if subtype == "gate":
        cov = data["retrieval"]["min_query_coverage"]
        if cov > MIN_COVERAGE_FLOOR:
            new = round(max(MIN_COVERAGE_FLOOR, cov - COVERAGE_STEP), 2)
            data["retrieval"]["min_query_coverage"] = new
            return data, "relax the evidence gate", "retrieval.min_query_coverage", str(cov), str(new)
        return None
    sentence = INSTRUCTIONS.get(subtype)
    if not sentence or sentence in data["system_instructions"]:
        return None
    old = data["system_instructions"]
    data["system_instructions"] = old.rstrip() + " " + sentence
    return data, f"add instruction: {sentence}", "system_instructions", old, data["system_instructions"]


def _score(card: Scorecard) -> str:
    return f"{card.passed}/{card.passed + card.failed} ({card.pass_rate:.0%})"


def run_repair(agent: AgentDefinition, evals: EvalSet, baseline: Scorecard, run_dir: str | Path,
               agent_llm_fn: Callable[..., str] | None = None, judge_llm_fn: Callable[..., str] | None = None,
               progress: Callable[[str], None] = lambda _: None,
               max_iterations: int = MAX_ITERATIONS) -> tuple[AgentDefinition, Scorecard, list[IterationLog]]:
    """Returns (best agent, its scorecard, iteration logs). Saves every artifact in run_dir."""
    run_dir = Path(run_dir)
    best_agent, best_card = agent, baseline
    logs: list[IterationLog] = []
    tried: set[str] = set()   # descriptions of changes already tested (kept or rolled back)
    for iteration in range(1, min(max_iterations, MAX_ITERATIONS) + 1):
        failures = classify_all(best_card, evals)
        log = IterationLog(iteration=iteration, failures=failures, score_before=_score(best_card),
                           failures_by_category=dict(Counter(f.category for f in failures)))
        logs.append(log)
        if not failures:
            log.note = "no failures left; stopping"
            break
        # Largest category first; within it the largest subtype; skip fixes already used up.
        patch = None
        groups = Counter((f.category, f.subtype) for f in failures)
        for (category, subtype), _ in sorted(groups.items(),
                                             key=lambda kv: (-log.failures_by_category[kv[0][0]], -kv[1])):
            patch = propose_patch(best_agent, category, subtype)
            if patch and patch[1] in tried:
                patch = None              # already tested and rolled back
            if patch:
                log.target_category, log.target_subtype = category, subtype
                break
        if not patch:
            log.note = "no untried fix left for the remaining failures; stopping"
            break
        data, log.change, log.field, log.old_value, log.new_value = patch
        tried.add(log.change)
        log.targeted_cases = [f.id for f in failures if (f.category, f.subtype) == (log.target_category, log.target_subtype)]
        progress(f"iteration {iteration}: {log.change}  (targets {log.targeted_cases})")

        candidate = parse_agent_definition(data)
        save_agent_definition(candidate, run_dir / f"agent_iteration_{iteration}.json")
        card = run_evals(evals, AgentRunner(candidate, llm_fn=agent_llm_fn), label=f"iteration_{iteration}",
                         llm_fn=judge_llm_fn)
        save_scorecard(card, run_dir)
        before = {r.id: r.status for r in best_card.results}
        after = {r.id: r.status for r in card.results}
        log.fixed_cases = [i for i in after if before.get(i) == "fail" and after[i] == "pass"]
        log.broken_cases = [i for i in after if before.get(i) == "pass" and after[i] == "fail"]
        log.score_after = _score(card)
        # Keep a change only if it pays for itself: strictly more cases pass.
        log.kept = card.passed > best_card.passed
        if log.kept:
            best_agent, best_card = candidate, card
        else:
            log.note = ("score dropped: change rolled back" if card.passed < best_card.passed
                        else "no gain: change rolled back")
        progress(f"  {log.score_before} -> {log.score_after}  fixed {log.fixed_cases} broken {log.broken_cases}"
                 f"  {'KEPT' if log.kept else 'ROLLED BACK'}")

    save_agent_definition(best_agent, run_dir / "agent_final.json")
    (run_dir / "repair_log.json").write_text(
        json.dumps([l.model_dump() for l in logs], indent=2, ensure_ascii=False), encoding="utf-8")
    return best_agent, best_card, logs
