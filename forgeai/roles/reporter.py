"""
reporter.py - writes report.md from a run's saved artifacts.

Code only (no model): every sentence comes from the artifacts, so the report
cannot overstate the result. It states the final score, the score of every
iteration, what changed, and - for every case that still fails - its category,
the evidence and a plain explanation. Known limitations are listed every time.
"""

import json
from pathlib import Path

from forgeai.roles.eval_generator import load_evals
from forgeai.roles.judge import Scorecard
from forgeai.roles.manager import load_requirements
from forgeai.roles.repair import classify_all
from forgeai.runtime.agent_schema import load_agent_definition

EXPLAIN = {
    "retrieval_miss": "the search did not bring the passage with the answer to the model",
    "wrong_refusal": "the agent refused although the documents contain the answer",
    "prompt_defect": "the right passage was found, but the answer was incomplete, badly cited or should have been a refusal",
    "hallucination": "the model's own answer did not match the documents (the runner blocked it or it was wrong)",
}
LIMITATIONS = [
    "One small local model (Qwen 2.5 1.5B) writes the persona, generates tests and judges unclear facts; "
    "code checks limit, but do not remove, its influence.",
    "Words are matched exactly (no stemming): 'policies' does not match 'policy', which can cause retrieval "
    "misses and wrong gate refusals.",
    "Multi-document cases join two questions from different documents; they test answers that span two "
    "documents, not reasoning across them.",
    "Key facts are short phrases; a correct answer that states a fact in different words (e.g. spelled-out "
    "numbers) can be scored as missing.",
    "Repair tunes the agent on the same eval set it is scored on, so the final score may be optimistic.",
    "The guard counts a topic as supported when at least half of its key words appear in a CSV, so a "
    "two-word topic can pass on one word.",
]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _card(path: Path) -> Scorecard | None:
    return Scorecard.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_report(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    req = load_requirements(run / "requirements.json")
    evals = load_evals(run / "evals.json")
    agent = load_agent_definition(run / ("agent_final.json" if (run / "agent_final.json").is_file() else "agent.json"))
    build = _read(run / "build_log.json") or {}
    repair = _read(run / "repair_log.json") or []
    rejected = _read(run / "evals_rejected.json") or []
    timings = _read(run / "timings.json") or {}
    baseline = _card(run / "scorecard_iteration_0.json")
    if baseline is None:
        raise FileNotFoundError("scorecard_iteration_0.json missing: run the evals first")

    final, final_label = baseline, "iteration_0"
    for log in repair:
        if log.get("kept"):
            final_label = f"iteration_{log['iteration']}"
            final = _card(run / f"scorecard_{final_label}.json") or final
    scored = final.passed + final.failed

    L = [f"# ForgeAI report: {agent.name}", "",
         f"**Final score: {final.passed}/{scored} cases passed ({final.pass_rate:.0%})**, "
         f"{final.errors} error(s), from {final_label}. Baseline: {baseline.passed}/{baseline.passed + baseline.failed} "
         f"({baseline.pass_rate:.0%}).", "",
         f"Brief: _{req.brief}_", ""]
    if timings:
        L += ["Time per stage: " + ", ".join(f"{k} {v:.0f} s" for k, v in timings.items()), ""]

    L += ["## 1. Requirements", "", f"- Audience: {req.audience}", f"- Tone: {req.tone}",
          f"- Must answer: {', '.join(req.must_answer_topics)}", f"- Must refuse: {', '.join(req.must_refuse_topics)}",
          f"- Clarifications: {sum(c.source == 'requester' for c in req.clarifications)} answered by the requester, "
          f"{sum(c.source == 'assumed' for c in req.clarifications)} assumed (defaults)", ""]

    L += ["## 2. Groundability (can the documents support the brief?)", "", "| Topic | Status | Decided by | Reason |",
          "|---|---|---|---|"]
    L += [f"| {s.topic} | {s.status} | {s.method} | {s.reason} |" for s in req.topic_support]
    L += [""] + [f"- Warning: {w}" for w in req.guard_warnings] + [""]

    L += ["## 3. Agent definition", "", "| Tool | Attached | Reason |", "|---|---|---|"]
    L += [f"| {t['name']} | {'yes' if t['attached'] else 'no'} | {t['reason']} |" for t in build.get("tools", [])]
    r = agent.retrieval
    L += ["", f"- Retrieval: chunks of {r.chunk_size_words} words, top_k {r.top_k}, "
          f"evidence-gate coverage {r.min_query_coverage}",
          f"- Refusal topics: {', '.join(agent.scope.refusal_topics)}",
          f"- Persona written by: {build.get('system_instructions_source', 'unknown')}", ""]

    L += ["## 4. Eval set (generated, checked, frozen)", "",
          f"- {evals.counts.get('total', len(evals.cases))} cases: " +
          ", ".join(f"{k} {v}" for k, v in evals.counts.items() if k not in ("total", "rejected")),
          f"- {len(rejected)} generated cases rejected by code checks (see evals_rejected.json)",
          f"- SHA-256 `{evals.sha256[:16]}...` (the judge refuses a changed set)"]
    L += [f"- Note: {n}" for n in evals.notes] + [""]

    L += ["## 5. Scorecard per iteration", "", "| Iteration | Change | Score | Fixed | Broken | Result |",
          "|---|---|---|---|---|---|",
          f"| 0 | (baseline agent) | {baseline.passed}/{baseline.passed + baseline.failed} | | | |"]
    for log in repair:
        if log.get("change"):
            L.append(f"| {log['iteration']} | {log['change']} | {log['score_after']} | "
                     f"{', '.join(log['fixed_cases']) or '-'} | {', '.join(log['broken_cases']) or '-'} | "
                     f"{'kept' if log['kept'] else log.get('note', 'rolled back')} |")
        elif log.get("note"):
            L.append(f"| {log['iteration']} | - | - | | | {log['note']} |")
    for path in sorted(run.glob("scorecard_*_rejudged.json")):
        rc = _card(path)
        source = rc.label.replace("_rejudged", "")
        L.append(f"| {source} re-judged | same answers, current judge (code checks only) | "
                 f"{rc.passed}/{rc.passed + rc.failed} ({rc.pass_rate:.0%}) | | | see judge note |")
    L += ["", "By case type (final): " + "; ".join(
        f"{t} {row['pass']}/{row['pass'] + row['fail']}" for t, row in final.by_type.items()), ""]

    from forgeai.roles.judge_agreement import agreement, load_labels
    labels = load_labels(run)
    if labels:
        a = agreement(labels)
        L += ["### Is the judge trustworthy?", "",
              f"A human labelled {a.labelled} saved answers. The judge agreed on **{a.agree}/{a.labelled} "
              f"({a.rate:.0%})**. Judge too lenient (passed, human said wrong): {', '.join(a.false_pass) or 'none'}. "
              f"Judge too strict (failed, human said fine): {', '.join(a.false_fail) or 'none'}.", ""]

    failures = classify_all(final, evals)
    L += ["## 6. What still fails, and why", ""]
    if not failures:
        L += ["Nothing in this eval set fails. Treat this with suspicion: the set is small and generated by the "
              "same model family as the agent.", ""]
    questions = {c.id: c for c in evals.cases}
    for f in failures:
        c = questions[f.id]
        L += [f"- **{f.id}** ({c.type}) _{c.question}_  ",
              f"  {f.category} ({f.subtype}): {EXPLAIN[f.category]}. Evidence: {f.evidence}."]
    L += ["", "## 7. Known limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [
        "", "## 8. Files", "",
        "requirements.json, groundability.json, agent.json (built), agent_final.json (best after repair), "
        "build_log.json, evals.json, evals_rejected.json, scorecard_iteration_N.json, repair_log.json, "
        "and *_log.json files with every model prompt and reply.", ""]

    path = run / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
