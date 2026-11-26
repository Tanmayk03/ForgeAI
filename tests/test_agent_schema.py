"""Tests for agent.json validation (invalid agent definitions)."""

import json
import tempfile
import unittest
from pathlib import Path

from forgeai import config
from forgeai.runtime.agent_schema import (
    AgentDefinitionError, load_agent_definition, parse_agent_definition, save_agent_definition,
)
from tests.helpers import VALID_AGENT, agent_dict


class TestValidDefinitions(unittest.TestCase):
    def test_valid_definition_loads(self):
        agent = parse_agent_definition(VALID_AGENT)
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.retrieval.top_k, 3)

    def test_sample_agent_file_is_valid(self):
        agent = load_agent_definition(config.DATA_DIR / "sample_agents" / "policy_assistant.json")
        self.assertEqual(agent.name, "policy_assistant")

    def test_defaults_fill_optional_sections(self):
        data = agent_dict()
        for key in ("retrieval", "tools", "output_contract"):
            del data[key]
        agent = parse_agent_definition(data)
        self.assertEqual(agent.retrieval.chunk_size_words, config.CHUNK_SIZE_WORDS)
        self.assertTrue(agent.output_contract.require_citations)
        self.assertEqual(agent.tools, [])

    def test_save_and_reload_is_identical(self):
        agent = parse_agent_definition(VALID_AGENT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.json"
            save_agent_definition(agent, path)
            self.assertEqual(load_agent_definition(path), agent)


class TestInvalidDefinitions(unittest.TestCase):
    def assertInvalid(self, data, expected_text):
        with self.assertRaises(AgentDefinitionError) as ctx:
            parse_agent_definition(data)
        self.assertIn(expected_text, str(ctx.exception))

    def test_missing_required_section(self):
        data = agent_dict()
        del data["scope"]
        self.assertInvalid(data, "scope")

    def test_unknown_key_is_rejected(self):
        data = agent_dict()
        data["retrieval"]["top_kk"] = 5  # typo
        self.assertInvalid(data, "top_kk")

    def test_top_k_out_of_range(self):
        data = agent_dict()
        data["retrieval"]["top_k"] = 0
        self.assertInvalid(data, "retrieval.top_k")

    def test_overlap_not_smaller_than_chunk(self):
        data = agent_dict()
        data["retrieval"]["chunk_overlap_words"] = 300
        self.assertInvalid(data, "chunk_overlap_words must be smaller")

    def test_unknown_tool(self):
        self.assertInvalid(agent_dict(tools=["calculater"]), "unknown tool")

    def test_duplicate_tool(self):
        self.assertInvalid(agent_dict(tools=["calculator", "calculator"]), "duplicates")

    def test_bad_name(self):
        self.assertInvalid(agent_dict(name="My Agent!"), "name must be")

    def test_unsupported_schema_version(self):
        self.assertInvalid(agent_dict(schema_version="9.9"), "unsupported schema_version")

    def test_empty_in_scope_topics(self):
        data = agent_dict()
        data["scope"]["in_scope_topics"] = []
        self.assertInvalid(data, "scope.in_scope_topics")

    def test_wrong_type(self):
        data = agent_dict()
        data["output_contract"]["require_citations"] = "maybe"
        self.assertInvalid(data, "output_contract.require_citations")


class TestDefinitionFiles(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(AgentDefinitionError) as ctx:
            load_agent_definition("does/not/exist.json")
        self.assertIn("not found", str(ctx.exception))

    def test_file_is_not_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.json"
            path.write_text("{ this is not json", encoding="utf-8")
            with self.assertRaises(AgentDefinitionError) as ctx:
                load_agent_definition(path)
            self.assertIn("not valid JSON", str(ctx.exception))

    def test_top_level_must_be_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.json"
            path.write_text(json.dumps([VALID_AGENT]), encoding="utf-8")
            with self.assertRaises(AgentDefinitionError):
                load_agent_definition(path)


if __name__ == "__main__":
    unittest.main()
