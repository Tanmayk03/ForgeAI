"""Tests for the Auditor's eval generator (scripted fake model)."""

import json
import tempfile
import unittest
from pathlib import Path

from forgeai.retrieval.loader import load_documents
from forgeai.roles.eval_generator import (
    check_answerable, fact_in_text, generate_evals, load_evals, save_evals, strengthen_fact, verify_frozen,
)
from forgeai.roles.manager import summarise_documents
from forgeai.roles.requirements import Requirements, TopicSupport
from tests.helpers import SAMPLE_DOCS


def req(brief="An assistant for leave and travel policy questions for shore staff.", unsupported=()):
    support = [TopicSupport(topic="annual leave", status="supported", method="code", coverage=1.0)]
    support += [TopicSupport(topic=t, status="unsupported", method="code", coverage=0.0) for t in unsupported]
    return Requirements(
        agent_name="t", purpose="p", scope="leave and travel policies", audience="shore staff", tone="formal",
        must_answer_topics=["annual leave", *unsupported], must_refuse_topics=["legal advice", "medical advice"],
        brief=brief, documents_folder=SAMPLE_DOCS,
        documents=summarise_documents(load_documents(SAMPLE_DOCS)), topic_support=support,
    )


# Facts that really are in each sample passage, keyed by a word only that passage contains.
PASSAGE_QA = {
    "Annual leave": [("How many paid annual leave days do shore staff get?", ["24 working days"]),
                     ("How far ahead must leave requests be submitted?", ["10 working days in advance"]),
                     ("When do carried-forward days expire?", ["31 March"]),
                     ("How much leave accrues per month?", ["2 days per completed month"])],
    "Sick leave": [("How many sick days are paid each year?", ["12 days of paid sick leave"]),
                   ("When is a medical certificate required?", ["longer than 2 consecutive days"]),
                   ("Are public holidays counted during annual leave?", ["not counted as leave days"]),
                   ("What does the policy say on sick leave pay?", ["THIS FACT IS INVENTED"])],   # must be rejected
    "Business travel": [("What is the hotel limit per night?", ["INR 6,000 per night"]),
                        ("What is the daily meal allowance?", ["INR 1,200"]),
                        ("Within how many days must claims be submitted?", ["within 30 days of returning"]),
                        ("Can travel during leave be claimed?", ["Travel undertaken during approved annual leave is personal"])],
}


class ScriptedLLM:
    """Answers each prompt type with plausible JSON; records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, system=None, json_schema=None, **kw):
        self.calls.append(prompt)
        if "Rewrite each question" in prompt:
            n = prompt.count("\n- ")
            return json.dumps({"rewrites": ["hey how many days off do i get pls"] + [f"quick q number {i} about stuff here" for i in range(n - 1)]})
        if "must REFUSE" in prompt:
            return json.dumps({"topic_questions": [
                {"topic": "legal advice", "question": "Can I sue my manager for refusing my leave?"},
                {"topic": "medical advice", "question": "What medicine should I take for a fever?"},
                {"topic": "astrology", "question": "What does my horoscope say about travel?"}],
                "unrelated_questions": ["Who won the football world cup in 2018?",
                                        "How do I bake a chocolate cake at home?",
                                        "What is the annual leave policy for shore staff?",   # NOT unrelated
                                        "How do I reverse a list in Python?"]})
        for marker, qas in PASSAGE_QA.items():
            if marker in prompt:
                return json.dumps({"cases": [{"question": q, "key_facts": f} for q, f in qas]})
        return json.dumps({"cases": []})


class TestChecks(unittest.TestCase):
    def test_fact_matching(self):
        self.assertTrue(fact_in_text("INR 6000", "hotel up to INR 6,000 per night"))
        self.assertTrue(fact_in_text("24 Working Days", "entitled to 24 working days of leave"))
        self.assertFalse(fact_in_text("4 working days", "entitled to 24 working days"))   # whole words only
        self.assertFalse(fact_in_text("", "anything"))

    def test_check_answerable(self):
        p = ["Staff get 24 working days of leave."]
        self.assertEqual(check_answerable("How many leave days do staff get?", ["24 working days"], p, []), "")
        self.assertIn("not found", check_answerable("How many leave days do staff get?", ["30 days"], p, []))
        self.assertIn("own answer", check_answerable("Do staff get 24 working days of leave?", ["24 working days"], p, []))
        self.assertIn("too short", check_answerable("Leave?", ["24 working days"], p, []))


class TestStrengthenFact(unittest.TestCase):
    P = ("Shore staff are entitled to 24 working days of paid annual leave per calendar year. "
         "Up to 5 unused days may be carried forward. 3. Sick leave. Employees receive 12 days of paid sick leave.")

    def test_bare_number_gets_its_unit(self):
        self.assertEqual(strengthen_fact("24", self.P), "24 working days")
        self.assertEqual(strengthen_fact("5", self.P), "5 unused days")

    def test_long_sentence_becomes_number_phrase(self):
        self.assertEqual(strengthen_fact("Up to 5 unused days may be carried forward to the next year", self.P),
                         "5 unused days")
        self.assertEqual(strengthen_fact("3. Sick leave. Employees receive 12 days of paid sick leave per year.",
                                         self.P), "12 days")          # list marker "3." ignored

    def test_normal_fact_unchanged(self):
        self.assertEqual(strengthen_fact("medical certificate", self.P), "medical certificate")


class TestCombinedMultiDoc(unittest.TestCase):
    def test_multi_doc_cases_combine_two_documents(self):
        evals, _ = generate_evals(req(), llm_fn=ScriptedLLM())
        multi = [c for c in evals.cases if c.type == "multi_doc"]
        self.assertEqual(len(multi), 3)
        self.assertTrue(all(c.origin.startswith("combined") for c in multi))
        self.assertTrue(all(len(c.expected_docs) == 2 for c in multi))
        self.assertTrue(all(", and " in c.question for c in multi))


class TestGenerateEvals(unittest.TestCase):
    def setUp(self):
        self.llm = ScriptedLLM()
        self.evals, self.rejected = generate_evals(req(unsupported=("maternity leave",)), llm_fn=self.llm)
        self.by_type = {}
        for c in self.evals.cases:
            self.by_type.setdefault(c.type, []).append(c)

    def test_at_least_twenty_cases_of_every_type(self):
        self.assertGreaterEqual(self.evals.counts["total"], 20, self.evals.counts)
        for t in ("answerable", "multi_doc", "ambiguous", "out_of_scope"):
            self.assertIn(t, self.by_type)

    def test_every_answer_key_is_in_the_documents(self):
        for c in self.evals.cases:
            for fact in c.key_facts:
                self.assertTrue(fact_in_text(fact, c.source_text), (c.id, fact))

    def test_invented_fact_and_leaky_or_bad_cases_rejected(self):
        reasons = " | ".join(r.reason for r in self.rejected)
        self.assertIn("THIS FACT IS INVENTED", reasons)
        self.assertIn("not one of the must-refuse topics", reasons)         # astrology
        self.assertIn("not clearly unrelated", reasons)                     # annual leave question
        self.assertNotIn("What is the annual leave policy for shore staff?",
                         [c.question for c in self.by_type["out_of_scope"]])

    def test_multi_doc_and_expected_docs(self):
        m = self.by_type["multi_doc"][0]
        self.assertEqual(m.expected_docs, ["leave_policy.pdf", "travel_policy.md"])
        self.assertEqual(len(m.key_facts), 2)
        a = [c for c in self.by_type["answerable"] if "INR 6,000 per night" in c.key_facts][0]
        self.assertEqual(a.expected_docs, ["travel_policy.md"])

    def test_refusal_cases_include_guard_gaps(self):
        origins = [c.origin for c in self.by_type["out_of_scope"]]
        self.assertIn("guard: unsupported topic 'maternity leave'", origins)
        self.assertTrue(all(c.expected_behavior == "refuse" for c in self.by_type["out_of_scope"]))

    def test_no_agent_definition_is_used(self):
        for prompt in self.llm.calls:
            self.assertNotIn("system_instructions", prompt)
            self.assertNotIn("top_k", prompt)

    def test_unrelated_csv_gives_no_csv_cases(self):
        self.assertTrue(any("spare_parts.csv: unrelated" in n for n in self.evals.notes))

    def test_freeze_save_load(self):
        self.assertTrue(verify_frozen(self.evals))
        changed = self.evals.model_copy(deep=True)
        changed.cases[0].key_facts = ["something else"]
        self.assertFalse(verify_frozen(changed))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_evals(self.evals, self.rejected, tmp, log=[])
            self.assertTrue(verify_frozen(load_evals(path)))
            self.assertTrue(json.loads(Path(tmp, "evals_rejected.json").read_text()))

    def test_deterministic(self):
        again, _ = generate_evals(req(unsupported=("maternity leave",)), llm_fn=ScriptedLLM())
        self.assertEqual(again.sha256, self.evals.sha256)


class TestCsvCases(unittest.TestCase):
    def test_related_csv_gives_code_built_cases(self):
        evals, _ = generate_evals(req(brief="An assistant to look up spare parts, fuel injector prices and suppliers."),
                                  llm_fn=ScriptedLLM())
        csv_cases = [c for c in evals.cases if c.origin.startswith("csv row")]
        self.assertTrue(csv_cases)
        self.assertEqual(csv_cases[0].expected_docs, ["spare_parts.csv"])


if __name__ == "__main__":
    unittest.main()
