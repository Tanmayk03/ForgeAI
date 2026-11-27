"""
runner.py - the ONE generic runtime that runs any agent.json.

The chat CLI uses it now, and the Auditor will use exactly the same class
later, so what gets tested is exactly what the user chats with.

Answering a question has four stages:

  1. RETRIEVE   BM25 top-k chunks (agent's own chunking + top_k settings)
  2. GATE       code-only evidence check. No matching chunk, or chunks that
                cover too little of the question  -> refuse WITHOUT calling the model.
                (Skipped for agents with tools: a tool may supply the evidence.)
  2b. TOOLS     small models rarely decide to call tools by themselves, so:
                csv_lookup and current_date are run automatically by the runner;
                calculator and weather_api are chosen in a short planning call
                whose answer must be one name from a fixed list (or "none").
  3. GENERATE   ask the model for JSON {answer, sources, refused, refusal_reason,
                tool_name, tool_input}, retrying once if the reply is unusable.
                If it asks for a tool, the runner runs it (only if agent.json
                allows it), adds the result as a numbered source, and asks again.
  4. VERIFY     code-only checks on the reply. An answer citing no valid source, or
                containing a number found in none of the cited chunks, is turned
                into a refusal (fail closed; the number check can be switched off
                with output_contract.block_unsupported_numbers).

Errors from Ollama (not running, model missing, timeout) are NOT turned into
refusals: they are raised to the caller. An outage is not an answer, and the
Auditor must never score it as a correct refusal.
"""

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

from forgeai import config, llm_client
from forgeai.retrieval.bm25_index import BM25Index
from forgeai.retrieval.chunker import chunk_documents
from forgeai.retrieval.loader import load_documents
from forgeai.retrieval.models import LoadResult, SearchResult
from forgeai.runtime import evidence
from forgeai.runtime.agent_schema import AgentDefinition, load_agent_definition
from forgeai.runtime.output_parser import parse_model_output
from forgeai.runtime.prompt_builder import (
    AUTO_TOOLS, PLANNED_TOOLS, PromptSource, build_planner_prompt, build_retry_note,
    build_system_prompt, build_user_prompt, planner_schema,
)
from forgeai.runtime.response import AgentResponse, ModelOutput, RetrievedChunkInfo, RunTrace, ToolCallRecord
from forgeai.tools.base import ToolContext
from forgeai.tools.registry import run_tool

# Type of the function used to call the model. Normally llm_client.generate;
# tests pass a fake with the same signature so they run without Ollama.
LLMFunction = Callable[..., str]

# The planning call costs ~5-10 s on CPU, so it runs only when the question hints at a tool.
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "twice", "double", "half", "dozen"}
PLANNER_HINTS = {
    "calculator": {"total", "sum", "cost", "costs", "discount", "discounts", "percent", "percentage",
                   "average", "difference", "calculate", "altogether", "combined", "times"} | NUMBER_WORDS,
    "weather_api": {"weather", "forecast", "temperature", "rain", "wind"},
}


def tool_hinted(tool: str, question: str) -> bool:
    words = set(question.lower().replace("?", " ").split())
    has_digit = any(ch.isdigit() for ch in question)
    return bool(words & PLANNER_HINTS.get(tool, set())) or (tool == "calculator" and has_digit)


# An arithmetic operator between numbers: "145 * 4" yes, "145.00" no.
LIMIT_NOTE = "\n\nYou cannot use more tools. Answer now using only the SOURCES, or refuse."

ARITHMETIC = re.compile(r"[\d)]\s*(\*\*|[-+*/%])\s*[\d(.]")


class AgentLoadError(Exception):
    """The agent's documents could not be loaded (missing folder, nothing searchable)."""


class AgentRunner:
    """Loads one agent definition and its documents, then answers questions."""

    def __init__(self, agent: AgentDefinition, llm_fn: LLMFunction | None = None):
        self.agent = agent
        self.llm_fn = llm_fn or llm_client.generate
        self.system_prompt = build_system_prompt(agent)

        folder = agent.documents.resolved_path()
        try:
            self.load_result: LoadResult = load_documents(folder)
        except FileNotFoundError as exc:
            raise AgentLoadError(f"Agent '{agent.name}': {exc}") from exc

        self.chunks = chunk_documents(
            self.load_result.documents,
            chunk_size=agent.retrieval.chunk_size_words,
            overlap=agent.retrieval.chunk_overlap_words,
        )
        # An agent needs searchable text, unless it has tools (e.g. a CSV-only folder + csv_lookup).
        if not self.chunks and not agent.tools:
            raise AgentLoadError(
                f"Agent '{agent.name}': no searchable text found in {folder} "
                f"({len(self.load_result.skipped)} file(s) skipped)."
            )
        self.index = BM25Index(self.chunks) if self.chunks else None
        self.tool_context = ToolContext(
            csv_paths=[Path(d.path) for d in self.load_result.documents if d.file_type == "csv"]
        )

    @classmethod
    def from_file(cls, path: str | Path, llm_fn: LLMFunction | None = None) -> "AgentRunner":
        """Load agent.json (raises AgentDefinitionError if invalid) and build a runner."""
        return cls(load_agent_definition(path), llm_fn=llm_fn)

    # ------------------------------------------------------------------ helpers

    def _refusal(self, trace: RunTrace, reason: str) -> AgentResponse:
        """Every refusal looks the same to the user: the agent's refusal message."""
        return AgentResponse(
            answer=self.agent.scope.refusal_message,
            sources=[],
            refused=True,
            refusal_reason=reason,
            trace=trace,
        )

    def _call_model(self, user_prompt: str, trace: RunTrace) -> tuple[ModelOutput | None, str]:
        """Ask the model, retrying if the reply is unusable. LLM errors propagate."""
        schema = ModelOutput.model_json_schema()
        prompt = user_prompt
        reason = ""
        for _ in range(1 + config.MAX_OUTPUT_RETRIES):
            trace.model_called = True
            trace.model_attempts += 1
            raw = self.llm_fn(prompt=prompt, system=self.system_prompt, json_schema=schema)
            trace.raw_model_outputs.append(raw)
            output, reason = parse_model_output(raw)
            if output is not None:
                return output, ""
            trace.warnings.append(f"attempt {trace.model_attempts}: unusable model output: {reason}")
            prompt = user_prompt + build_retry_note(reason)
        return None, reason

    @staticmethod
    def _normalise_tool_name(name: str) -> str:
        """'tool weather_api', 'Weather-API ' -> 'weather_api' (small models garble names)."""
        name = re.sub(r"^\s*tool[\s:_-]+", "", name.strip(), flags=re.IGNORECASE)
        return re.sub(r"[\s-]+", "_", name.strip()).lower()

    def _add_tool_result(self, name, tool_input, requested_by, sources, trace) -> None:
        """Run a tool and add its result as the next numbered source."""
        name = self._normalise_tool_name(name)
        result = run_tool(name, tool_input, self.tool_context, self.agent.tools)
        trace.tool_calls.append(ToolCallRecord(
            source_number=len(sources) + 1, requested_by=requested_by, name=result.name,
            tool_input=result.tool_input, ok=result.ok, output=result.output,
        ))
        sources.append(PromptSource(
            f"tool:{result.name}:{len(trace.tool_calls)}",
            f"tool {result.name}, input: {result.tool_input or '(none)'}",
            result.output,
        ))

    def _run_auto_tools(self, question, sources, trace) -> None:
        """Run the cheap tools that need no model decision."""
        if "csv_lookup" in self.agent.tools:
            self._add_tool_result("csv_lookup", question, "runner", sources, trace)
            if not trace.tool_calls[-1].ok or trace.tool_calls[-1].output.startswith("No rows match"):
                sources.pop()  # nothing useful found; don't show the model an empty result
        if "current_date" in self.agent.tools:
            self._add_tool_result("current_date", "", "runner", sources, trace)

    def _run_planner(self, question, sources, trace) -> None:
        """Short model call: which of calculator / weather_api (or none) is needed?"""
        planned = [t for t in self.agent.tools if t in PLANNED_TOOLS and tool_hinted(t, question)]
        if not planned:
            return  # no hint of a calculation or weather question: skip the extra model call
        trace.model_called = True
        trace.model_attempts += 1
        raw = self.llm_fn(
            prompt=build_planner_prompt(question, sources, planned),
            system="You choose tools. Reply with JSON only.",
            json_schema=planner_schema(planned),
        )
        trace.raw_model_outputs.append(raw)
        try:
            plan = json.loads(raw)
            name, tool_input = str(plan["tool_name"]).strip(), str(plan.get("tool_input", "")).strip()
        except (ValueError, KeyError, TypeError):
            trace.warnings.append("tool planner reply unusable; continuing without a planned tool")
            return
        if name == "calculator" and not ARITHMETIC.search(tool_input):
            trace.warnings.append(f"planner chose calculator with no calculation ({tool_input!r}); skipped")
            return
        if name and name != "none":
            self._add_tool_result(name, tool_input, "planner", sources, trace)

    # --------------------------------------------------------------------- main

    def ask(self, question: str) -> AgentResponse:
        """Answer one question. Always returns an AgentResponse unless the LLM itself fails."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        started = time.perf_counter()
        retrieval = self.agent.retrieval
        trace = RunTrace(agent_name=self.agent.name, question=question, decision="refused_no_evidence")

        try:
            # 1. RETRIEVE
            results: list[SearchResult] = [
                r for r in (self.index.search(question, top_k=retrieval.top_k) if self.index else [])
                if r.score > retrieval.min_bm25_score
            ]
            trace.retrieved = [
                RetrievedChunkInfo(
                    source_number=n, chunk_id=r.chunk.chunk_id, doc_name=r.chunk.doc_name,
                    page=r.chunk.page, score=r.score, rank=r.rank, text=r.chunk.text,
                )
                for n, r in enumerate(results, start=1)
            ]

            # 2. GATE (coverage is computed first so the trace always shows the question's key terms)
            terms, covered, coverage = evidence.query_coverage(question, [r.chunk.text for r in results])
            trace.query_terms, trace.covered_terms, trace.query_coverage = terms, covered, round(coverage, 3)
            has_tools = bool(self.agent.tools)
            if not results and not has_tools:
                trace.decision = "refused_no_evidence"
                return self._refusal(trace, "No document passage matches this question.")

            if coverage < retrieval.min_query_coverage and not has_tools:
                missing = [t for t in terms if t not in covered]
                trace.decision = "refused_low_coverage"
                return self._refusal(
                    trace, f"The documents do not mention key terms of the question: {', '.join(missing)}."
                )
            if has_tools and (not results or coverage < retrieval.min_query_coverage):
                trace.warnings.append("evidence gate skipped: agent has tools that may supply evidence")

            # 3. GENERATE (with up to MAX_TOOL_CALLS tool rounds)
            sources = [
                PromptSource(r.chunk.chunk_id, f"{r.chunk.doc_name}, page {r.chunk.page}", r.chunk.text)
                for r in results
            ]
            self._run_auto_tools(question, sources, trace)
            self._run_planner(question, sources, trace)
            note = ""
            while True:
                user_prompt = build_user_prompt(question, sources) + note
                trace.estimated_prompt_tokens = evidence.estimate_tokens(self.system_prompt + user_prompt)
                output, reason = self._call_model(user_prompt, trace)
                if output is None:
                    trace.decision = "refused_invalid_output"
                    return self._refusal(trace, f"The model's reply could not be used ({reason}).")
                if not output.tool_name:
                    break
                model_calls = sum(1 for c in trace.tool_calls if c.requested_by == "model")
                if model_calls >= config.MAX_TOOL_CALLS:
                    if note.startswith(LIMIT_NOTE):
                        trace.decision = "refused_tool_limit"
                        return self._refusal(trace, f"Stopped after {config.MAX_TOOL_CALLS} tool calls.")
                    # One last chance: answer from what is already in the SOURCES.
                    trace.warnings.append("tool limit reached; asked the model to answer from the sources")
                    note = LIMIT_NOTE
                    continue
                wanted = self._normalise_tool_name(output.tool_name)
                duplicate = next((c for c in trace.tool_calls
                                  if c.name == wanted and c.tool_input == output.tool_input), None)
                if duplicate:
                    # Don't run the same call twice: point the model at the existing result.
                    trace.tool_calls.append(duplicate.model_copy(update={"requested_by": "model"}))
                    trace.warnings.append(f"repeated tool call {output.tool_name}({output.tool_input!r}) not re-run")
                    note = (f"\n\nThe result of {output.tool_name} is already source [{duplicate.source_number}]. "
                            "Answer now using the SOURCES.")
                    continue
                self._add_tool_result(output.tool_name, output.tool_input, "model", sources, trace)

            if trace.estimated_prompt_tokens > 0.8 * config.NUM_CTX:
                trace.warnings.append(
                    f"prompt is about {trace.estimated_prompt_tokens} tokens, close to the "
                    f"{config.NUM_CTX}-token context window; text may be cut off"
                )

            trace.model_answer_before_checks = output.answer
            if output.refused:
                trace.decision = "refused_by_model"
                return self._refusal(trace, output.refusal_reason or "The model declined to answer.")

            # 4. VERIFY
            # A failed tool call ("ERROR: ...") is not evidence: citing it does not count.
            failed = {c.source_number for c in trace.tool_calls if not c.ok}
            if failed & set(output.sources):
                trace.warnings.append(f"model cited failed tool result(s) {sorted(failed & set(output.sources))}; ignored")
            valid_numbers = [n for n in dict.fromkeys(output.sources) if 1 <= n <= len(sources) and n not in failed]
            invalid_numbers = [n for n in output.sources if n not in valid_numbers]
            if invalid_numbers:
                trace.warnings.append(f"model cited source number(s) that were not provided: {invalid_numbers}")

            if not valid_numbers and self.agent.output_contract.require_citations:
                trace.decision = "refused_no_citation"
                return self._refusal(trace, "The answer did not cite any of the provided sources.")

            cited = [sources[n - 1] for n in valid_numbers]
            # Without citations (allowed only if require_citations is false), check against all sources.
            checked_against = cited or sources
            trace.unsupported_numbers = evidence.unsupported_numbers(
                output.answer, [src.text for src in checked_against]
            )
            if trace.unsupported_numbers:
                trace.warnings.append(
                    f"numbers not found in cited sources: {trace.unsupported_numbers}"
                )
                if self.agent.output_contract.block_unsupported_numbers:
                    trace.decision = "refused_unsupported_number"
                    return self._refusal(
                        trace,
                        "The answer contained figures not found in the cited sources: "
                        + ", ".join(trace.unsupported_numbers) + ".",
                    )

            word_count = len(output.answer.split())
            if word_count > self.agent.output_contract.max_answer_words:
                trace.warnings.append(
                    f"answer has {word_count} words; contract limit is "
                    f"{self.agent.output_contract.max_answer_words}"
                )

            trace.decision = "answered"
            return AgentResponse(
                answer=output.answer,
                sources=[src.source_id for src in cited],
                refused=False,
                refusal_reason="",
                trace=trace,
            )
        finally:
            trace.latency_seconds = round(time.perf_counter() - started, 3)
