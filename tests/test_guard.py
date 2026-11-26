"""Tests for the groundability guard (fake model)."""

import json
import tempfile
import unittest
from pathlib import Path

from forgeai.retrieval.loader import load_documents
from forgeai.roles.guard import run_guard, save_guard_report
from forgeai.roles.manager import summarise_documents
from forgeai.roles.requirements import Requirements
from tests.helpers import SAMPLE_DOCS, FakeLLM


def requirements(must_answer, must_refuse=("legal advice",)):
    return Requirements(
        agent_name="test", purpose="p", scope="s", audience="a", tone="t",
        must_answer_topics=list(must_answer), must_refuse_topics=list(must_refuse),
        brief="test brief for the guard", documents_folder=SAMPLE_DOCS,
        documents=summarise_documents(load_documents(SAMPLE_DOCS)),
    )


def by_topic(req):
    return {s.topic: s for s in req.topic_support}


class TestGuard(unittest.TestCase):
    def test_code_decides_clear_cases_without_model(self):
        fake = FakeLLM()  # any model call fails the test
        req = run_guard(requirements(["annual leave", "pension scheme", "fuel injector"]), llm_fn=fake)
        s = by_topic(req)
        self.assertEqual((s["annual leave"].status, s["annual leave"].method), ("supported", "code"))
        self.assertIn("leave_policy.pdf::p1::c0", s["annual leave"].evidence)
        self.assertEqual((s["pension scheme"].status, s["pension scheme"].method), ("unsupported", "code"))
        self.assertEqual(s["fuel injector"].evidence, ["csv:spare_parts.csv"])   # found in the CSV
        self.assertEqual(fake.calls, [])

    def test_partial_match_verified_by_model_with_real_quote(self):
        quote = "Employees receive 12 days of paid sick leave per year."
        fake = FakeLLM({"verdict": "supported", "evidence_quote": quote})
        s = by_topic(run_guard(requirements(["sick leave allowance"]), llm_fn=fake))["sick leave allowance"]
        self.assertEqual((s.status, s.method), ("supported", "llm"))
        self.assertEqual(len(fake.calls), 1)

    def test_invented_quote_means_unsupported(self):
        # "maternity leave": "leave" matches, "maternity" never does -> model asked.
        fake = FakeLLM({"verdict": "supported", "evidence_quote": "Mothers receive 26 weeks of paid maternity leave."})
        s = by_topic(run_guard(requirements(["maternity leave"]), llm_fn=fake))["maternity leave"]
        self.assertEqual(s.status, "unsupported")
        self.assertIn("quote is not in the documents", s.reason)

    def test_model_says_unsupported(self):
        fake = FakeLLM({"verdict": "unsupported", "evidence_quote": ""})
        s = by_topic(run_guard(requirements(["maternity leave"]), llm_fn=fake))["maternity leave"]
        self.assertEqual(s.status, "unsupported")

    def test_warnings(self):
        req = run_guard(requirements(["annual leave", "pension scheme"], must_refuse=["medical", "legal advice"]),
                        llm_fn=FakeLLM())
        text = " ".join(req.guard_warnings)
        self.assertIn("refuse topic 'medical' appears in the documents", text)   # sick-leave certificate
        self.assertNotIn("'legal advice'", text)
        self.assertIn("UNSUPPORTED by the documents", text)
        self.assertIn("pension scheme", text)
        self.assertIn("not in any must-answer topic", text)
        keywords = [w for w in req.guard_warnings if "not in any must-answer topic" in w][0]
        self.assertIn("sick", keywords)          # distinctive word is reported
        self.assertNotIn(" per", keywords)       # generic word is not

    def test_save_updates_requirements_and_writes_report(self):
        req = run_guard(requirements(["annual leave"]), llm_fn=FakeLLM())
        with tempfile.TemporaryDirectory() as tmp:
            path = save_guard_report(req, tmp, log=[])
            report = json.loads(path.read_text())
            self.assertEqual(report["topic_support"][0]["status"], "supported")
            saved = Requirements.model_validate_json(Path(tmp, "requirements.json").read_text())
            self.assertEqual(saved.topic_support[0].topic, "annual leave")


if __name__ == "__main__":
    unittest.main()
