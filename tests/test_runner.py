"""
Tests for the generic agent runner, using a fake model.

Covers: answerable, unsupported and irrelevant questions, evidence-aware
refusal, malformed model output, missing documents, invalid agent
definitions and Ollama errors.
"""

import json
import tempfile
import unittest
from pathlib import Path

from forgeai.llm_client import LLMTimeoutError, ModelNotFoundError, OllamaNotRunningError
from forgeai.runtime.agent_schema import AgentDefinitionError, parse_agent_definition
from forgeai.runtime.runner import AgentLoadError, AgentRunner
from tests.helpers import FakeLLM, agent_dict, answer, refusal

LEAVE_CHUNK = "leave_policy.pdf::p1::c0"  # chunk with "24 working days of paid annual leave"


def make_runner(fake, **agent_changes):
    return AgentRunner(parse_agent_definition(agent_dict(**agent_changes)), llm_fn=fake)


class TestAnswerable(unittest.TestCase):
    def test_answer_with_valid_citation(self):
        fake = FakeLLM(answer("Shore staff get 24 working days of annual leave.", [1]))
        response = make_runner(fake).ask("How many days of annual leave do shore staff get?")

        self.assertFalse(response.refused)
        self.assertEqual(response.sources, [LEAVE_CHUNK])
        self.assertEqual(response.trace.decision, "answered")
        self.assertEqual(response.trace.unsupported_numbers, [])
        self.assertEqual(len(fake.calls), 1)

    def test_prompts_contain_agent_settings_and_numbered_sources(self):
        fake = FakeLLM(answer("24 working days.", [1]))
        make_runner(fake).ask("How many days of annual leave do shore staff get?")
        call = fake.calls[0]
        self.assertIn("legal advice", call["system"])          # refusal topic from agent.json
        self.assertIn("You answer questions about the provided documents.", call["system"])
        self.assertIn("[1] (leave_policy.pdf, page 1)", call["prompt"])
        self.assertTrue(call["prompt"].rstrip().endswith("shore staff get?"))
        self.assertIn("properties", call["json_schema"])        # JSON schema sent to Ollama

    def test_trace_keeps_retrieval_metadata_for_auditing(self):
        fake = FakeLLM(answer("24 working days.", [1]))
        trace = make_runner(fake).ask("How many days of annual leave do shore staff get?").trace
        self.assertEqual(trace.retrieved[0].chunk_id, LEAVE_CHUNK)
        self.assertEqual(trace.retrieved[0].source_number, 1)
        self.assertGreater(trace.retrieved[0].score, 0)
        self.assertIn("24 working days", trace.retrieved[0].text)
        self.assertGreater(trace.query_coverage, 0.5)
        self.assertEqual(len(trace.raw_model_outputs), 1)
        self.assertGreaterEqual(trace.latency_seconds, 0)
        json.dumps(trace.model_dump())  # the whole trace must be saveable as JSON

    def test_invented_number_is_flagged(self):
        fake = FakeLLM(answer("Shore staff get 30 days.", [1]))
        response = make_runner(fake).ask("How many days of annual leave do shore staff get?")
        self.assertFalse(response.refused)
        self.assertEqual(response.trace.unsupported_numbers, ["30"])
        self.assertTrue(any("numbers not found" in w for w in response.trace.warnings))

    def test_mixed_valid_and_invalid_citations(self):
        fake = FakeLLM(answer("24 working days.", [1, 9]))
        response = make_runner(fake).ask("How many days of annual leave do shore staff get?")
        self.assertEqual(response.sources, [LEAVE_CHUNK])
        self.assertTrue(any("[9]" in w for w in response.trace.warnings))

    def test_agent_chunk_settings_are_used(self):
        small = agent_dict()
        small["retrieval"]["chunk_size_words"] = 40
        small["retrieval"]["chunk_overlap_words"] = 10
        runner_small = AgentRunner(parse_agent_definition(small), llm_fn=FakeLLM())
        runner_default = make_runner(FakeLLM())
        self.assertGreater(len(runner_small.chunks), len(runner_default.chunks))


class TestUnsupported(unittest.TestCase):
    """In the domain, but the documents don't contain the answer."""

    QUESTION = "How many weeks of maternity leave do shore staff get?"

    def test_model_refusal_is_respected(self):
        fake = FakeLLM(refusal("maternity leave is not covered"))
        response = make_runner(fake).ask(self.QUESTION)
        self.assertTrue(response.refused)
        self.assertEqual(response.answer, "Sorry, I can't answer that.")  # agent's refusal message
        self.assertEqual(response.trace.decision, "refused_by_model")
        self.assertEqual(response.sources, [])

    def test_answer_without_citation_becomes_refusal(self):
        fake = FakeLLM(answer("Shore staff get 26 weeks of maternity leave.", []))
        response = make_runner(fake).ask(self.QUESTION)
        self.assertTrue(response.refused)
        self.assertEqual(response.trace.decision, "refused_no_citation")
        # the overridden answer is kept for the Auditor to inspect
        self.assertIn("26 weeks", response.trace.model_answer_before_checks)

    def test_answer_citing_nonexistent_source_becomes_refusal(self):
        fake = FakeLLM(answer("26 weeks.", [7]))
        response = make_runner(fake).ask(self.QUESTION)
        self.assertTrue(response.refused)
        self.assertEqual(response.trace.decision, "refused_no_citation")

    def test_citations_optional_when_contract_allows(self):
        data = agent_dict()
        data["output_contract"]["require_citations"] = False
        runner = AgentRunner(parse_agent_definition(data), llm_fn=FakeLLM(answer("24 working days.", [])))
        response = runner.ask("How many days of annual leave do shore staff get?")
        self.assertFalse(response.refused)
        self.assertEqual(response.sources, [])


class TestIrrelevant(unittest.TestCase):
    def test_no_matching_chunk_refuses_without_calling_model(self):
        fake = FakeLLM()  # no replies: any call would fail the test
        response = make_runner(fake).ask("What is the weather in Mumbai tomorrow?")
        self.assertTrue(response.refused)
        self.assertEqual(response.trace.decision, "refused_no_evidence")
        self.assertFalse(response.trace.model_called)
        self.assertEqual(response.trace.query_terms, ["weather", "mumbai", "tomorrow"])
        self.assertEqual(fake.calls, [])

    def test_weak_bm25_match_is_not_treated_as_evidence(self):
        # "staff" matches, but "pension" and "contribution" appear nowhere: coverage 1/3.
        fake = FakeLLM()
        response = make_runner(fake).ask("What is the pension contribution for staff?")
        self.assertTrue(response.trace.retrieved)          # BM25 DID return chunks...
        self.assertTrue(response.refused)                   # ...but the runner refused
        self.assertEqual(response.trace.decision, "refused_low_coverage")
        self.assertIn("pension", response.refusal_reason)
        self.assertEqual(fake.calls, [])

    def test_coverage_threshold_is_configurable(self):
        data = agent_dict()
        data["retrieval"]["min_query_coverage"] = 0.0
        fake = FakeLLM(refusal())
        AgentRunner(parse_agent_definition(data), llm_fn=fake).ask("What is the pension contribution for staff?")
        self.assertEqual(len(fake.calls), 1)  # gate disabled -> model decides


class TestMalformedModelOutput(unittest.TestCase):
    Q = "How many days of annual leave do shore staff get?"

    def test_retry_after_bad_json_succeeds(self):
        fake = FakeLLM("Annual leave is 24 days.", answer("24 working days.", [1]))
        response = make_runner(fake).ask(self.Q)
        self.assertFalse(response.refused)
        self.assertEqual(response.trace.model_attempts, 2)
        self.assertIn("could not be used", fake.calls[1]["prompt"])  # retry explains the problem

    def test_bad_output_twice_fails_closed(self):
        fake = FakeLLM("not json", '{"answer": "24"')
        response = make_runner(fake).ask(self.Q)
        self.assertTrue(response.refused)
        self.assertEqual(response.trace.decision, "refused_invalid_output")
        self.assertEqual(response.trace.model_attempts, 2)
        self.assertEqual(len(response.trace.raw_model_outputs), 2)

    def test_missing_fields_and_empty_reply(self):
        fake = FakeLLM('{"answer": "24 days"}', "")
        response = make_runner(fake).ask(self.Q)
        self.assertEqual(response.trace.decision, "refused_invalid_output")
        self.assertEqual(len(response.trace.warnings), 2)


class TestMissingDocuments(unittest.TestCase):
    def test_folder_does_not_exist(self):
        with self.assertRaises(AgentLoadError) as ctx:
            make_runner(FakeLLM(), documents={"folder": "data/no_such_folder"})
        self.assertIn("not found", str(ctx.exception))

    def test_folder_without_searchable_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "notes.docx").write_text("unsupported")
            Path(tmp, "parts.csv").write_text("a,b\n1,2\n")  # CSVs are not chunked yet
            with self.assertRaises(AgentLoadError) as ctx:
                make_runner(FakeLLM(), documents={"folder": tmp})
            self.assertIn("no searchable text", str(ctx.exception))


class TestInvalidAgentFile(unittest.TestCase):
    def test_from_file_rejects_invalid_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "agent.json")
            bad = agent_dict(tools=["weather"])  # not a known tool name
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(AgentDefinitionError):
                AgentRunner.from_file(path, llm_fn=FakeLLM())


class TestOllamaErrors(unittest.TestCase):
    """Infrastructure failures must surface as errors, never as refusals."""

    Q = "How many days of annual leave do shore staff get?"

    def test_errors_propagate(self):
        for error in (OllamaNotRunningError("down"), ModelNotFoundError("missing"), LLMTimeoutError("slow")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    make_runner(FakeLLM(error)).ask(self.Q)

    def test_gate_refusal_needs_no_model(self):
        # Even with Ollama down, an irrelevant question is refused (the gate never calls the model).
        response = make_runner(FakeLLM(OllamaNotRunningError("down"))).ask("weather in Mumbai tomorrow?")
        self.assertTrue(response.refused)


class TestGeneric(unittest.TestCase):
    def test_runner_works_for_a_different_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "planets.txt").write_text(
                "Mars has two moons, Phobos and Deimos. Jupiter has at least 95 known moons.", encoding="utf-8"
            )
            data = agent_dict(name="astronomy_helper", documents={"folder": tmp})
            data["scope"]["in_scope_topics"] = ["planets", "moons"]
            fake = FakeLLM(answer("Mars has two moons: Phobos and Deimos.", [1]))
            response = AgentRunner(parse_agent_definition(data), llm_fn=fake).ask("How many moons does Mars have?")
            self.assertFalse(response.refused)
            self.assertEqual(response.sources, ["planets.txt::p1::c0"])

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            make_runner(FakeLLM()).ask("   ")


if __name__ == "__main__":
    unittest.main()
