"""
streamlit_app.py - ForgeAI results viewer (read-only).

Shows the artifacts of every committed factory run in runs/: the brief and
requirements, the groundability check, the agent that was built, the frozen
eval set, per-case results for every iteration, the repair log and the report.

It reads JSON files only - no model, no Ollama - so it can be hosted for free
(Streamlit Community Cloud). Building new agents runs locally:
    python -m scripts.forge --brief ... --docs ... --run ... --auto

Run locally:  streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import streamlit as st

RUNS_DIR = Path(__file__).resolve().parent / "runs"


# ------------------------------------------------------------------ loading

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data
def list_runs() -> list[str]:
    if not RUNS_DIR.is_dir():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if (p / "report.md").is_file())


@st.cache_data
def load_run(name: str) -> dict:
    run = RUNS_DIR / name
    scorecards = {}
    for path in sorted(run.glob("scorecard_*.json")):
        card = read_json(path)
        if card:
            scorecards[card.get("label", path.stem)] = card
    report = run / "report.md"
    return {
        "requirements": read_json(run / "requirements.json") or {},
        "build_log": read_json(run / "build_log.json") or {},
        "agent": read_json(run / "agent_final.json") or read_json(run / "agent.json") or {},
        "evals": read_json(run / "evals.json") or {},
        "rejected": read_json(run / "evals_rejected.json") or [],
        "repair": read_json(run / "repair_log.json") or [],
        "timings": read_json(run / "timings.json") or {},
        "scorecards": scorecards,
        "report": report.read_text(encoding="utf-8") if report.is_file() else "",
    }


def final_scorecard(data: dict) -> tuple[str, dict | None]:
    """The scorecard of the last kept repair iteration, else the baseline."""
    label = "iteration_0"
    for log in data["repair"]:
        if log.get("kept"):
            label = f"iteration_{log['iteration']}"
    cards = data["scorecards"]
    return label, cards.get(label) or cards.get("iteration_0")


def score_text(card: dict | None) -> str:
    if not card:
        return "-"
    scored = card["passed"] + card["failed"]
    return f"{card['passed']}/{scored} ({card['pass_rate']:.0%})"


# ------------------------------------------------------------------ page

st.set_page_config(page_title="ForgeAI results", layout="wide")
st.title("ForgeAI - agent factory results")
st.caption(
    "Read-only viewer of committed factory runs. Each run: brief + documents -> requirements -> "
    "groundability check -> agent -> generated, frozen evals -> judge -> repair -> report. "
    "Agents are built locally with a CPU model (Qwen 2.5 1.5B via Ollama); this page needs no model."
)

runs = list_runs()
if not runs:
    st.error(f"No runs with a report.md found in {RUNS_DIR}.")
    st.stop()

default = runs.index("demo1_hr") if "demo1_hr" in runs else 0
name = st.sidebar.selectbox("Run", runs, index=default)
data = load_run(name)
req, agent, evals = data["requirements"], data["agent"], data["evals"]
baseline = data["scorecards"].get("iteration_0")
final_label, final = final_scorecard(data)

st.sidebar.markdown(f"**Agent:** `{agent.get('name', '?')}`")
st.sidebar.markdown(f"**Model:** `{evals.get('model', '?')}`")
if data["timings"]:
    st.sidebar.markdown(f"**Total time:** {sum(data['timings'].values()) / 60:.1f} min")
st.sidebar.markdown("---")
st.sidebar.markdown("[Source code and README](https://github.com/Tanmayk03/ForgeAI)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final score", score_text(final), help=f"from {final_label}")
c2.metric("Baseline", score_text(baseline))
c3.metric("Eval cases", evals.get("counts", {}).get("total", len(evals.get("cases", []))))
c4.metric("Tools attached", len(agent.get("tools", [])))

st.markdown(f"**Brief:** _{req.get('brief', '')}_")

tabs = st.tabs(["Requirements & guard", "Agent", "Eval set", "Results per case", "Repair log", "Report"])

# --- 1. requirements and groundability
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.subheader("Requirements")
        st.markdown(f"- **Audience:** {req.get('audience', '')}\n- **Tone:** {req.get('tone', '')}\n"
                    f"- **Scope:** {req.get('scope', '')}")
        st.markdown("**Must answer:** " + ", ".join(req.get("must_answer_topics", [])))
        st.markdown("**Must refuse:** " + ", ".join(req.get("must_refuse_topics", [])))
        st.markdown("**Interview** (requester = typed answer, assumed = default used in --auto)")
        st.dataframe([{"question": c["question"], "answer": c["answer"], "source": c["source"]}
                      for c in req.get("clarifications", [])], width="stretch", hide_index=True)
    with right:
        st.subheader("Groundability guard")
        st.dataframe([{"topic": s["topic"], "status": s["status"], "decided by": s["method"],
                       "reason": s["reason"]} for s in req.get("topic_support", [])],
                     width="stretch", hide_index=True)
        for w in req.get("guard_warnings", []):
            st.warning(w)
        st.markdown("**Documents**")
        st.dataframe([{"name": d["name"], "type": d["file_type"], "pages": d["pages"], "words": d["words"]}
                      for d in req.get("documents", [])], width="stretch", hide_index=True)
        if req.get("skipped_files"):
            st.caption("Skipped: " + "; ".join(req["skipped_files"]))

# --- 2. agent
with tabs[1]:
    st.subheader("Tool decisions (Builder)")
    st.dataframe([{"tool": t["name"], "attached": "yes" if t["attached"] else "no", "reason": t["reason"]}
                  for t in data["build_log"].get("tools", [])], width="stretch", hide_index=True)
    st.markdown(f"**Retrieval:** {data['build_log'].get('retrieval', '')}")
    st.markdown(f"**Persona written by:** {data['build_log'].get('system_instructions_source', '')}")
    st.subheader("Final agent definition (agent_final.json)")
    st.json(agent, expanded=False)

# --- 3. eval set
with tabs[2]:
    counts = evals.get("counts", {})
    st.markdown(", ".join(f"**{k}** {v}" for k, v in counts.items()))
    st.caption(f"Frozen with SHA-256 {evals.get('sha256', '')[:16]}... - the judge refuses a changed set.")
    for note in evals.get("notes", []):
        st.info(note)
    st.dataframe([{"id": c["id"], "type": c["type"], "expected": c["expected_behavior"],
                   "question": c["question"], "key facts": ", ".join(c["key_facts"]),
                   "expected docs": ", ".join(c["expected_docs"])} for c in evals.get("cases", [])],
                 width="stretch", hide_index=True)
    with st.expander(f"{len(data['rejected'])} generated cases rejected by code checks"):
        st.dataframe([{"question": r["question"], "reason": r["reason"], "origin": r["origin"]}
                      for r in data["rejected"]], width="stretch", hide_index=True)

# --- 4. results per case
with tabs[3]:
    labels = list(data["scorecards"])
    if labels:
        label = st.selectbox("Scorecard", labels, index=labels.index(final_label) if final_label in labels else 0)
        card = data["scorecards"][label]
        st.markdown(f"**{score_text(card)}** passed, {card['errors']} error(s), "
                    f"{card['judge_llm_calls']} LLM judge call(s), {card['total_seconds']:.0f} s")
        st.dataframe([{"type": t, "pass": r["pass"], "fail": r["fail"]} for t, r in card["by_type"].items()],
                     hide_index=True)
        only_failed = st.checkbox("Show only failed cases")
        rows = [r for r in card["results"] if not only_failed or r["status"] == "fail"]
        st.dataframe([{"id": r["id"], "status": r["status"], "type": r["type"], "question": r["question"],
                       "answer": r["answer"], "why it failed": r["failure"], "runner decision": r["runner_decision"],
                       "cited docs": ", ".join(r["cited_docs"])} for r in rows],
                     width="stretch", hide_index=True)

# --- 5. repair log
with tabs[4]:
    if not data["repair"]:
        st.info("No repair log for this run.")
    for log in data["repair"]:
        title = f"Iteration {log['iteration']}: {log.get('change') or log.get('note', '')}"
        with st.expander(title, expanded=bool(log.get("change"))):
            if log.get("change"):
                result = "KEPT" if log["kept"] else f"ROLLED BACK ({log.get('note', '')})"
                st.markdown(f"- **Targeted:** {log['target_category']} ({log['target_subtype']}) - "
                            f"{', '.join(log['targeted_cases'])}\n"
                            f"- **Field:** `{log['field']}`\n"
                            f"- **Score:** {log['score_before']} -> {log['score_after']}\n"
                            f"- **Fixed:** {', '.join(log['fixed_cases']) or '-'}; "
                            f"**broken:** {', '.join(log['broken_cases']) or '-'}\n- **Result:** {result}")
            st.markdown("**Failures classified before this iteration**")
            st.dataframe([{"id": f["id"], "category": f["category"], "subtype": f["subtype"],
                           "evidence": f["evidence"]} for f in log.get("failures", [])],
                         width="stretch", hide_index=True)

# --- 6. report
with tabs[5]:
    st.markdown(data["report"])
