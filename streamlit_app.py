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
from collections import Counter
from pathlib import Path

import pandas as pd
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


def pipeline_dot(data: dict, final: dict | None) -> str:
    """Graphviz diagram of the factory, labelled with this run's real numbers."""
    req, agent, evals = data["requirements"], data["agent"], data["evals"]
    support = req.get("topic_support", [])
    ok = sum(s["status"] == "supported" for s in support)
    gaps = len(support) - ok
    tools = "\\n".join(agent.get("tools", [])) or "no tools"
    counts = evals.get("counts", {})
    baseline = data["scorecards"].get("iteration_0")
    changes = [l for l in data["repair"] if l.get("change")]
    kept = sum(1 for l in changes if l.get("kept"))
    t = data["timings"]

    def secs(key):
        return f"\\n{t[key]:.0f} s" if key in t else ""

    nodes = [
        ("in", "Brief + documents", f"{len(req.get('documents', []))} document(s)", "#F1EFE8", "#444441"),
        ("m", "Manager", f"{len(req.get('must_answer_topics', []))} topics to answer" + secs("manager"), "#EEEDFE", "#3C3489"),
        ("g", "Guard", f"{ok} supported, {gaps} gap(s)" + secs("guard"), "#EEEDFE", "#3C3489"),
        ("b", "Builder", tools + secs("builder"), "#EEEDFE", "#3C3489"),
        ("e", "Exam (Auditor)", f"{counts.get('total', 0)} questions, {counts.get('rejected', 0)} rejected" + secs("eval generation"), "#E1F5EE", "#085041"),
        ("j", "Judge", f"baseline {score_text(baseline)}" + secs("evaluation"), "#E1F5EE", "#085041"),
        ("r", "Repair", f"{len(changes)} tried, {kept} kept" + secs("repair"), "#E1F5EE", "#085041"),
        ("out", "Chatbot + report", f"final {score_text(final)}", "#F1EFE8", "#444441"),
    ]
    lines = ['digraph G { rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, penwidth=0.6, margin="0.18,0.08"];',
             'edge [color="#888780", arrowsize=0.6];']
    for key, title, sub, fill, ink in nodes:
        lines.append(f'{key} [label="{title}\\n{sub}", fillcolor="{fill}", color="{ink}", fontcolor="{ink}"];')
    # Two rows: building the chatbot, then testing and fixing it.
    lines.append("{ rank=same; in -> m -> g -> b; }")
    lines.append("{ rank=same; e -> j -> r -> out; }")
    lines.append("b -> e; in -> e [style=invis]; }")
    return "\n".join(lines)


def colour_status(value):
    if value == "pass":
        return "background-color: #EAF3DE; color: #27500A"
    if value in ("fail", "error"):
        return "background-color: #FCEBEB; color: #791F1F"
    return ""


def colour_topic(value):
    return colour_status("pass" if value == "supported" else "fail" if value == "unsupported" else "")


def all_runs_page(runs: list[str]) -> None:
    st.subheader("All runs")
    rows = []
    for run in runs:
        d = load_run(run)
        label, fin = final_scorecard(d)
        base = d["scorecards"].get("iteration_0")
        rows.append({
            "run": run, "agent": d["agent"].get("name", "?"),
            "tools": ", ".join(d["agent"].get("tools", [])) or "-",
            "questions": d["evals"].get("counts", {}).get("total", 0),
            "baseline %": round(100 * base["pass_rate"]) if base else None,
            "final %": round(100 * fin["pass_rate"]) if fin else None,
            "minutes": round(sum(d["timings"].values()) / 60, 1) if d["timings"] else None,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown("**Score before and after repair (% of questions passed)**")
    st.bar_chart(df.set_index("run")[["baseline %", "final %"]], stack=False, color=["#B4B2A9", "#1D9E75"],
                 y_label="% passed")
    st.caption("Each run generates its own exam, so scores of different runs measure different questions.")


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

ALL = "All runs (overview)"
options = [ALL, *runs]
# A link like ...streamlit.app/?run=demo1_hr opens that run directly.
wanted = st.query_params.get("run", ALL)
name = st.sidebar.selectbox("Run", options, index=options.index(wanted) if wanted in options else 0)
st.query_params["run"] = name
if name == ALL:
    all_runs_page(runs)
    st.sidebar.markdown("[Source code and README](https://github.com/Tanmayk03/ForgeAI)")
    st.stop()
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
st.markdown("**How this run went** (each box shows what that stage produced)")
st.graphviz_chart(pipeline_dot(data, final), width="stretch")

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
        guard_df = pd.DataFrame([{"topic": s["topic"], "status": s["status"], "decided by": s["method"],
                                  "reason": s["reason"]} for s in req.get("topic_support", [])])
        if not guard_df.empty:
            st.dataframe(guard_df.style.map(colour_topic, subset=["status"]), width="stretch", hide_index=True)
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
    type_counts = {k: v for k, v in counts.items() if k not in ("total", "rejected")}
    if type_counts:
        st.bar_chart(pd.Series(type_counts, name="questions"), color="#7F77DD", horizontal=True)
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
        by_type = pd.DataFrame(card["by_type"]).T[["pass", "fail"]]
        st.markdown("**Passed and failed questions by type**")
        st.bar_chart(by_type, color=["#639922", "#E24B4A"], horizontal=True)
        only_failed = st.checkbox("Show only failed cases")
        rows = [r for r in card["results"] if not only_failed or r["status"] == "fail"]
        results_df = pd.DataFrame([{"id": r["id"], "status": r["status"], "type": r["type"], "question": r["question"],
                                    "answer": r["answer"], "why it failed": r["failure"],
                                    "runner decision": r["runner_decision"], "cited docs": ", ".join(r["cited_docs"])}
                                   for r in rows])
        if not results_df.empty:
            st.dataframe(results_df.style.map(colour_status, subset=["status"]), width="stretch", hide_index=True)

# --- 5. repair log
with tabs[4]:
    if not data["repair"]:
        st.info("No repair log for this run.")
    attempts = [l for l in data["repair"] if l.get("change")]
    if baseline and attempts:
        def passed(score: str) -> int:
            return int(score.split("/")[0])
        points = [{"attempt": "0 baseline", "passed": baseline["passed"], "result": "baseline"}]
        points += [{"attempt": f"{l['iteration']} {'kept' if l['kept'] else 'rolled back'}",
                    "passed": passed(l["score_after"]), "result": "kept" if l["kept"] else "rolled back"}
                   for l in attempts]
        st.markdown("**Questions passed after each repair attempt** (only changes that improve the score are kept)")
        st.bar_chart(pd.DataFrame(points), x="attempt", y="passed", color="result", y_label="questions passed")
    first = data["repair"][0] if data["repair"] else None
    if first and first.get("failures"):
        st.markdown("**Why the baseline failures failed** (failure category)")
        st.bar_chart(pd.Series(Counter(f["category"] for f in first["failures"]), name="cases"),
                     color="#D85A30", horizontal=True)
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
