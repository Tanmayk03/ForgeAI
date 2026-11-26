"""Tests for the Builder role (fake model)."""

import json
import tempfile
import unittest
from pathlib import Path

from forgeai import config
from forgeai.retrieval.loader import load_documents
from forgeai.roles.builder import BuilderError, run_builder, save_build
from forgeai.roles.manager import summarise_documents
from forgeai.roles.requirements import Requirements, TopicSupport
from forgeai.runtime.agent_schema import load_agent_definition
from forgeai.runtime.runner import AgentRunner
from tests.helpers import SAMPLE_DOCS, FakeLLM

PERSONA = {"system_instructions": "You are a policy assistant for shore staff. You answer clearly and formally."}


def req(brief="I need an assistant that answers leave and travel policy questions for shore staff.",
        topics=("annual leave", "travel claims"), unsupported=(), refuse=("legal advice",), tone="formal and concise",
        folder=SAMPLE_DOCS):
    support = [TopicSupport(topic=t, status="supported", method="code", coverage=1.0) for t in topics]
    support += [TopicSupport(topic=t, status="unsupported", method="code", coverage=0.0) for t in unsupported]
    return Requirements(
        agent_name="test_agent", purpose="Answers questions.", scope="Policies.", audience="staff", tone=tone,
        must_answer_topics=[*topics, *unsupported], must_refuse_topics=list(refuse), brief=brief,
        documents_folder=folder, documents=summarise_documents(load_documents(folder)), topic_support=support,
    )


def tools_of(build_log):
    return {t["name"]: t["attached"] for t in build_log["tools"]}


class TestToolSelection(unittest.TestCase):
    def test_hr_brief_gets_no_tools_even_with_unrelated_csv(self):
        agent, log = run_builder(req(), llm_fn=FakeLLM(PERSONA))
        self.assertEqual(agent.tools, [])
        reasons = {t["name"]: t["reason"] for t in log["tools"]}
        self.assertIn("unrelated to the brief", reasons["csv_lookup"])     # spare_parts.csv is not HR data
        self.assertEqual(reasons["weather_api"], "no need found in the requirements")

    def test_parts_brief_gets_csv_and_calculator(self):
        r = req(brief="I need an assistant that finds spare parts by part number and calculates the total cost of an order.",
                topics=("spare parts", "fuel injector"))
        agent, log = run_builder(r, llm_fn=FakeLLM(PERSONA))
        self.assertEqual(sorted(agent.tools), ["calculator", "csv_lookup"])
        self.assertIn("spare_parts.csv", [t for t in log["tools"] if t["name"] == "csv_lookup"][0]["reason"])

    def test_weather_and_date_triggers(self):
        r = req(brief="An assistant that gives the weather forecast for port visits and today's schedule.")
        agent, _ = run_builder(r, llm_fn=FakeLLM(PERSONA))
        self.assertIn("weather_api", agent.tools)
        self.assertIn("current_date", agent.tools)
        self.assertNotIn("calculator", agent.tools)


class TestDefinition(unittest.TestCase):
    def test_unsupported_topics_become_refusals(self):
        agent, log = run_builder(req(unsupported=("maternity leave",)), llm_fn=FakeLLM(PERSONA))
        self.assertNotIn("maternity leave", agent.scope.in_scope_topics)
        self.assertIn("maternity leave", agent.scope.refusal_topics)
        self.assertEqual(log["refusal_topics_added_from_guard"], ["maternity leave"])

    def test_contract_and_retrieval(self):
        agent, log = run_builder(req(), llm_fn=FakeLLM(PERSONA))
        self.assertTrue(agent.output_contract.require_citations)
        self.assertTrue(agent.output_contract.block_unsupported_numbers)
        self.assertEqual(agent.output_contract.max_answer_words, 80)           # "concise" tone
        self.assertLess(agent.retrieval.chunk_overlap_words, agent.retrieval.chunk_size_words)
        self.assertIn("top_k", log["retrieval"])

    def test_persona_from_model(self):
        agent, log = run_builder(req(), llm_fn=FakeLLM(PERSONA))
        self.assertEqual(agent.system_instructions, PERSONA["system_instructions"])
        self.assertEqual(log["system_instructions_source"], "model")

    def test_persona_falls_back_to_template(self):
        for bad in ({"system_instructions": "Hi."},
                    {"system_instructions": "You are helpful. Use your general knowledge when the documents are silent."}):
            with self.subTest(bad=bad["system_instructions"][:20]):
                agent, log = run_builder(req(), llm_fn=FakeLLM(bad))
                self.assertTrue(log["system_instructions_source"].startswith("template"))
                self.assertIn("You help staff", agent.system_instructions)
        agent, log = run_builder(req(), llm_fn=FakeLLM("x", "y"))
        self.assertIn("unusable", log["system_instructions_source"])

    def test_folder_stored_relative_to_project(self):
        agent, _ = run_builder(req(), llm_fn=FakeLLM(PERSONA))
        self.assertEqual(agent.documents.folder, "data/sample_docs")

    def test_errors(self):
        r = req()
        with self.assertRaises(BuilderError):
            run_builder(r.model_copy(update={"topic_support": []}), llm_fn=FakeLLM(PERSONA))
        with self.assertRaises(BuilderError):
            run_builder(req(topics=(), unsupported=("pension",)), llm_fn=FakeLLM(PERSONA))

    def test_saved_agent_loads_in_runner(self):
        agent, log = run_builder(req(), llm_fn=FakeLLM(PERSONA))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_build(agent, log, tmp, log=[])
            runner = AgentRunner.from_file(path, llm_fn=FakeLLM())
            self.assertEqual(runner.agent, load_agent_definition(path))
            self.assertTrue(json.loads(Path(tmp, "build_log.json").read_text())["tools"])


if __name__ == "__main__":
    unittest.main()
