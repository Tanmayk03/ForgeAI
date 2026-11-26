"""Tests for the Manager role (fake model, no Ollama)."""

import json
import tempfile
import unittest
from pathlib import Path

from forgeai.llm_client import OllamaNotRunningError
from forgeai.llm_json import LLMJSONError
from forgeai.roles.manager import ManagerError, load_requirements, run_manager, save_requirements, summarise_documents
from forgeai.retrieval.loader import load_documents
from tests.helpers import SAMPLE_DOCS, FakeLLM

BRIEF = "I need an assistant that answers leave and travel policy questions for shore staff."


def draft(questions=None, **changes):
    data = {
        "agent_name": "Shore Staff Policy Assistant!",
        "purpose": "Answers leave and travel policy questions for shore staff.",
        "scope": "Annual leave, sick leave and business travel claims.",
        "audience": "shore-based employees",
        "tone": "formal and concise",
        "must_answer_topics": ["annual leave", "sick leave", "travel claims", "Annual Leave"],
        "must_refuse_topics": ["legal advice", "salary negotiation"],
        "questions": questions if questions is not None else [
            {"question": "Should it answer salary questions?", "default_answer": "No, refuse them."},
            {"question": "Formal or friendly tone?", "default_answer": "Formal."},
        ],
    }
    data.update(changes)
    return data


def final(**changes):
    data = draft()
    del data["questions"]
    data.update(changes)
    return data


class TestManager(unittest.TestCase):
    def test_auto_mode_uses_defaults_and_one_model_call(self):
        fake = FakeLLM(draft())
        req = run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=fake)
        self.assertEqual(len(fake.calls), 1)                       # no finalize call
        self.assertEqual([c.source for c in req.clarifications], ["assumed", "assumed"])
        self.assertEqual(req.agent_name, "shore_staff_policy_assistant")
        self.assertEqual(req.must_answer_topics, ["annual leave", "sick leave", "travel claims"])  # duplicate removed
        self.assertIn("old_memo.docx", req.skipped_files[0])

    def test_draft_prompt_contains_brief_and_document_overview(self):
        fake = FakeLLM(draft())
        run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=fake)
        prompt = fake.calls[0]["prompt"]
        self.assertIn(BRIEF, prompt)
        self.assertIn("leave_policy.pdf (pdf, 2 page(s)", prompt)
        self.assertIn("CSV columns: part_no,description", prompt)

    def test_interactive_answers_trigger_finalize(self):
        answers = iter(["Yes, but only general salary policy", ""])
        fake = FakeLLM(draft(), final(tone="friendly"))
        req = run_manager(BRIEF, SAMPLE_DOCS, input_fn=lambda _: next(answers), llm_fn=fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([c.source for c in req.clarifications], ["requester", "assumed"])
        self.assertIn("general salary policy", fake.calls[1]["prompt"])
        self.assertEqual(req.tone, "friendly")

    def test_at_most_five_questions_are_asked(self):
        many = [{"question": f"Q{i}?", "default_answer": "x"} for i in range(8)]
        asked = []
        req = run_manager(BRIEF, SAMPLE_DOCS, input_fn=lambda q: asked.append(q) or "", llm_fn=FakeLLM(draft(many)))
        self.assertEqual(len(asked), 5)
        self.assertEqual(len(req.clarifications), 5)
        self.assertTrue(any("only the first 5" in n for n in req.notes))

    def test_topic_in_both_lists_becomes_refuse_only(self):
        req = run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=FakeLLM(
            draft(must_refuse_topics=["sick leave", "legal advice"])))
        self.assertNotIn("sick leave", req.must_answer_topics)
        self.assertIn("sick leave", req.must_refuse_topics)
        self.assertTrue(any("both answer and refuse" in n for n in req.notes))

    def test_bad_json_is_retried(self):
        fake = FakeLLM("not json", draft())
        log = []
        run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=fake, log=log)
        self.assertEqual(len(log), 2)

    def test_unusable_output_raises(self):
        with self.assertRaises(LLMJSONError):
            run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=FakeLLM("x", '{"agent_name": 1}'))

    def test_no_must_answer_topics_raises(self):
        with self.assertRaises(ManagerError):
            run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=FakeLLM(draft(must_answer_topics=[" "])))

    def test_input_errors(self):
        with self.assertRaises(ManagerError):
            run_manager("help me", SAMPLE_DOCS, auto=True, llm_fn=FakeLLM())
        with self.assertRaises(ManagerError):
            run_manager(BRIEF, "data/no_such_folder", auto=True, llm_fn=FakeLLM())
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ManagerError):
            run_manager(BRIEF, tmp, auto=True, llm_fn=FakeLLM())

    def test_ollama_error_propagates(self):
        with self.assertRaises(OllamaNotRunningError):
            run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=FakeLLM(OllamaNotRunningError("down")))

    def test_save_and_load(self):
        log = []
        req = run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=FakeLLM(draft()), log=log)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_requirements(req, tmp, log=log)
            self.assertEqual(load_requirements(path), req)
            self.assertEqual(len(json.loads(Path(tmp, "manager_log.json").read_text())), 1)


class TestSummaries(unittest.TestCase):
    def test_document_summaries(self):
        summaries = {s.name: s for s in summarise_documents(load_documents(SAMPLE_DOCS))}
        self.assertEqual(summaries["leave_policy.pdf"].pages, 2)
        self.assertIn("leave", summaries["leave_policy.pdf"].keywords)
        self.assertFalse(summaries["spare_parts.csv"].chunked)
        self.assertIn("2 rows", summaries["spare_parts.csv"].preview)


if __name__ == "__main__":
    unittest.main()
