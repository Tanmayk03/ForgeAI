"""Tests for Repair: classification rules, patches, keep/rollback loop."""

import json
import tempfile
import unittest

from forgeai.roles.judge import CaseResult, FactCheck, run_evals
from forgeai.roles.repair import INSTRUCTIONS, classify, propose_patch, run_repair
from forgeai.runtime.agent_schema import parse_agent_definition
from forgeai.runtime.runner import AgentRunner
from tests.helpers import FakeLLM, agent_dict, answer, refusal
from tests.test_judge import AgentLLM, case, evalset


def result(**kw):
    base = dict(id="q1", type="answerable", question="q", expected_behavior="answer", status="fail",
                expected_docs=["leave_policy.pdf"], retrieved_docs=["leave_policy.pdf"])
    base.update(kw)
    return CaseResult(**base)


class TestClassify(unittest.TestCase):
    def test_rules(self):
        facts = ["24 working days"]
        cases = [
            (result(refused=True, runner_decision="refused_low_coverage"), "wrong_refusal", "gate"),
            (result(refused=True, runner_decision="refused_no_evidence", retrieved_docs=[]), "retrieval_miss", "retrieval"),
            (result(refused=True, runner_decision="refused_by_model"), "wrong_refusal", "model"),
            (result(refused=True, runner_decision="refused_unsupported_number",
                    model_answer_before_checks="24 working days"), "prompt_defect", "citation"),
            (result(refused=True, runner_decision="refused_unsupported_number",
                    model_answer_before_checks="30 days"), "hallucination", "hallucination"),
            (result(facts=[FactCheck(fact="24 working days", found=False, method="missing")]), "prompt_defect", "incomplete"),
            (result(type="multi_doc", facts=[FactCheck(fact="x", found=False, method="missing")]), "prompt_defect", "multi_part"),
            (result(expected_docs=["travel_policy.md"]), "retrieval_miss", "retrieval"),
            (result(expected_behavior="refuse"), "prompt_defect", "refusal_rule"),
        ]
        for res, category, subtype in cases:
            with self.subTest(category=category, subtype=subtype):
                f = classify(res, facts)
                self.assertEqual((f.category, f.subtype), (category, subtype), f.evidence)


class TestPatches(unittest.TestCase):
    def setUp(self):
        self.agent = parse_agent_definition(agent_dict())

    def test_targeted_patches(self):
        data, _, field, old, new = propose_patch(self.agent, "retrieval_miss", "retrieval")
        self.assertEqual((field, old, new), ("retrieval.top_k", "3", "4"))
        data, _, field, old, new = propose_patch(self.agent, "wrong_refusal", "gate")
        self.assertEqual((field, old, new), ("retrieval.min_query_coverage", "0.5", "0.35"))
        data, _, field, _, new = propose_patch(self.agent, "prompt_defect", "citation")
        self.assertEqual(field, "system_instructions")
        self.assertTrue(new.endswith(INSTRUCTIONS["citation"]))
        self.assertTrue(new.startswith(self.agent.system_instructions))   # appended, not regenerated

    def test_used_up_patches_return_none(self):
        data = agent_dict()
        data["system_instructions"] += " " + INSTRUCTIONS["citation"]
        data["retrieval"]["min_query_coverage"] = 0.2
        agent = parse_agent_definition(data)
        self.assertIsNone(propose_patch(agent, "prompt_defect", "citation"))
        self.assertIsNone(propose_patch(agent, "wrong_refusal", "gate"))


HOTEL = case("q1", "answerable", "What is the maximum amount allowed for hotel accommodation?",
             facts=["INR 6,000 per night"], docs=["travel_policy.md"])
LEAVE = case("q2", "answerable", "How many days of annual leave do shore staff get?",
             facts=["24 working days"], docs=["leave_policy.pdf"])
REPLIES = {"hotel": answer("Up to INR 6,000 per night.", [1]), "annual": answer("24 working days.", [1])}


class TestRepairLoop(unittest.TestCase):
    def test_gate_refusal_is_fixed_and_kept(self):
        evals = evalset(HOTEL, LEAVE)
        agent = parse_agent_definition(agent_dict())
        llm = AgentLLM(REPLIES)
        baseline = run_evals(evals, AgentRunner(agent, llm_fn=llm), llm_fn=FakeLLM())
        self.assertEqual(baseline.results[0].runner_decision, "refused_low_coverage")   # like real q07
        with tempfile.TemporaryDirectory() as tmp:
            best, card, logs = run_repair(agent, evals, baseline, tmp, agent_llm_fn=llm, judge_llm_fn=FakeLLM())
            self.assertEqual(logs[0].field, "retrieval.min_query_coverage")
            self.assertEqual(logs[0].fixed_cases, ["q1"])
            self.assertTrue(logs[0].kept)
            self.assertEqual(card.pass_rate, 1.0)
            self.assertIn("no failures left", logs[1].note)
            with open(f"{tmp}/repair_log.json", encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved[0]["score_before"], "1/2 (50%)")
            self.assertEqual(saved[0]["score_after"], "2/2 (100%)")

    def test_change_that_lowers_the_score_is_rolled_back(self):
        evals = evalset(HOTEL, LEAVE)
        agent = parse_agent_definition(agent_dict())
        baseline = run_evals(evals, AgentRunner(agent, llm_fn=AgentLLM(REPLIES)), llm_fn=FakeLLM())
        broken = {"hotel": refusal(), "annual": refusal()}
        with tempfile.TemporaryDirectory() as tmp:
            best, card, logs = run_repair(agent, evals, baseline, tmp, agent_llm_fn=AgentLLM(broken),
                                          judge_llm_fn=FakeLLM())
            self.assertFalse(logs[0].kept)
            self.assertIn("q2", logs[0].broken_cases)
            self.assertEqual(best, agent)                        # original kept
            self.assertEqual(card.passed, baseline.passed)
            self.assertLessEqual(len(logs), 3)
            changes = [l.change for l in logs if l.change]
            self.assertEqual(len(changes), len(set(changes)))   # a rolled-back change is never retried

    def test_no_gain_is_rolled_back_and_not_retried(self):
        # Real run: instruction A rolled back, B kept with no gain, then A retried.
        multi = case("q3", "multi_doc", "How many days of annual leave do shore staff get, and what is the hotel limit per night?",
                     facts=["24 working days", "INR 6,000 per night"], docs=["leave_policy.pdf", "travel_policy.md"])
        evals = evalset(LEAVE, multi)
        agent = parse_agent_definition(agent_dict())
        llm = AgentLLM({"annual": answer("24 working days.", [1])})      # never fixes q3
        baseline = run_evals(evals, AgentRunner(agent, llm_fn=llm), llm_fn=FakeLLM())
        with tempfile.TemporaryDirectory() as tmp:
            best, card, logs = run_repair(agent, evals, baseline, tmp, agent_llm_fn=llm, judge_llm_fn=FakeLLM())
        self.assertTrue(all(not l.kept for l in logs if l.change))
        self.assertTrue(any("no gain" in l.note for l in logs))
        self.assertEqual(best, agent)
        changes = [l.change for l in logs if l.change]
        self.assertEqual(len(changes), len(set(changes)))


if __name__ == "__main__":
    unittest.main()
