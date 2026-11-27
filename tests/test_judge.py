"""Tests for the judge (fake agent model and fake judge model)."""

import json
import unittest

from forgeai.llm_client import OllamaNotRunningError
from forgeai.roles.eval_generator import EvalCase, EvalSet, freeze
from forgeai.roles.judge import JudgeError, check_fact_by_code, run_evals
from forgeai.runtime.agent_schema import parse_agent_definition
from forgeai.runtime.runner import AgentRunner
from tests.helpers import FakeLLM, agent_dict, answer, refusal


def case(id_, type_, q, behavior="answer", facts=(), docs=()):
    return EvalCase(id=id_, type=type_, question=q, expected_behavior=behavior, key_facts=list(facts),
                    expected_docs=list(docs), origin="test")


def evalset(*cases):
    return freeze(EvalSet(created_at="t", model="m", counts={}, cases=list(cases)))


class AgentLLM:
    """Fake agent model: reply chosen by a word in the question."""

    def __init__(self, replies):
        self.replies = replies

    def __call__(self, prompt, **kw):
        q = prompt.split("QUESTION:")[-1].lower()
        for word, reply in self.replies.items():
            if word in q:
                if isinstance(reply, Exception):
                    raise reply
                return json.dumps(reply)
        return json.dumps(refusal())


def runner(replies):
    return AgentRunner(parse_agent_definition(agent_dict()), llm_fn=AgentLLM(replies))


LEAVE = case("q1", "answerable", "How many days of annual leave do shore staff get?",
             facts=["24 working days"], docs=["leave_policy.pdf"])


class TestFactChecks(unittest.TestCase):
    def test_code_matching(self):
        self.assertEqual(check_fact_by_code("24 working days", "Staff get 24 working days a year."), "exact")
        self.assertEqual(check_fact_by_code("INR 6,000 per night", "up to INR 6000 per night"), "exact")
        self.assertEqual(check_fact_by_code("written approval from the department head",
                                            "you need written approval of the department head"), "fuzzy")
        self.assertEqual(check_fact_by_code("24 working days", "Staff get 20 working days."), "")

    def test_context_words_from_the_question(self):
        # Real false failures (now pass): the missing words were already in the question.
        self.assertEqual(check_fact_by_code("3 percent discount", "3 percent",
                                            "What is the discount rate for orders above USD 10,000?"), "context")
        self.assertEqual(check_fact_by_code("USD 400 per order", "USD 400",
                                            "How much does air freight cost per order?"), "context")
        # Real true failures (must still fail): wrong answers the quote check had caught.
        self.assertEqual(check_fact_by_code("USD 5,000", "default order quantity",
                                            "What is the minimum order quantity for orders placed online?"), "")
        self.assertEqual(check_fact_by_code("USD 180", "The maximum is INR 6,000 per night.",
                                            "What is the hotel limit per night abroad?"), "")
        # The numbers must always match, and a fact fully contained in the question proves nothing.
        self.assertEqual(check_fact_by_code("3 percent discount", "5 percent", "What is the discount rate?"), "")
        self.assertEqual(check_fact_by_code("annual leave", "Yes.", "Is there annual leave?"), "")


class TestJudge(unittest.TestCase):
    def test_pass_and_fact_missing(self):
        card = run_evals(evalset(LEAVE), runner({"annual": answer("24 working days per year.", [1])}), llm_fn=FakeLLM())
        self.assertEqual(card.results[0].status, "pass")
        self.assertTrue(card.results[0].grounded)
        self.assertEqual(card.judge_llm_calls, 0)               # decided by code

        card = run_evals(evalset(LEAVE), runner({"annual": answer("Staff get paid leave.", [1])}),
                         llm_fn=FakeLLM({"states_fact": False, "quote": ""}))
        self.assertEqual(card.results[0].status, "fail")
        self.assertIn("missing key fact", card.results[0].failure)

    def test_llm_judge_needs_a_verified_quote(self):
        claims = case("q4", "answerable", "What is required for claims submitted after 30 days?",
                      facts=["written approval from the department head"], docs=["travel_policy.md"])
        agent = runner({"claims": answer("The department head must sign off on it in writing.", [1])})
        ok = run_evals(evalset(claims), agent,
                       llm_fn=FakeLLM({"states_fact": True, "quote": "The department head must sign off on it in writing"}))
        self.assertEqual(ok.results[0].facts[0].method, "llm")
        self.assertEqual(ok.judge_llm_calls, 1)
        bad = run_evals(evalset(claims), agent,
                        llm_fn=FakeLLM({"states_fact": True, "quote": "written approval from the department head"}))
        self.assertEqual(bad.results[0].facts[0].method, "llm-rejected")   # quote is not in the answer
        self.assertEqual(bad.results[0].status, "fail")

    def test_llm_quote_must_keep_the_numbers(self):
        agent = runner({"annual": answer("You get twenty-four days off for working.", [1])})
        card = run_evals(evalset(LEAVE), agent, llm_fn=FakeLLM({"states_fact": True, "quote": "twenty-four days off"}))
        self.assertEqual(card.results[0].facts[0].method, "llm-rejected")  # spelled-out numbers are not accepted

    def test_wrong_document_is_not_grounded(self):
        # Answer cites the travel policy for a leave question.
        agent = runner({"annual": answer("24 working days.", [2])})
        card = run_evals(evalset(LEAVE), agent, llm_fn=FakeLLM())
        r = card.results[0]
        if r.cited_docs == ["leave_policy.pdf"]:
            self.skipTest("source [2] happened to be the leave policy")
        self.assertFalse(r.grounded)
        self.assertEqual(r.status, "fail")

    def test_refusal_cases(self):
        refuse = case("q2", "out_of_scope", "What is the weather in Mumbai?", behavior="refuse")
        card = run_evals(evalset(refuse), runner({}), llm_fn=FakeLLM())
        self.assertEqual(card.results[0].status, "pass")
        wrong = run_evals(evalset(LEAVE), runner({}), llm_fn=FakeLLM())   # agent refuses an answerable question
        self.assertEqual(wrong.results[0].status, "fail")
        self.assertIn("refused an answerable question", wrong.results[0].failure)

    def test_multi_doc_needs_every_document(self):
        multi = case("q3", "multi_doc", "How many days of annual leave do shore staff get, and what is the hotel limit per night?",
                     facts=["24 working days", "INR 6,000 per night"], docs=["leave_policy.pdf", "travel_policy.md"])
        agent = runner({"annual": answer("24 working days and INR 6,000 per night.", [1, 2, 3])})
        self.assertEqual(run_evals(evalset(multi), agent, llm_fn=FakeLLM()).results[0].status, "pass")

    def test_errors_are_not_scored(self):
        card = run_evals(evalset(LEAVE), runner({"annual": OllamaNotRunningError("down")}), llm_fn=FakeLLM())
        self.assertEqual((card.errors, card.passed, card.failed), (1, 0, 0))
        self.assertEqual(card.pass_rate, 0.0)

    def test_tampered_evals_rejected(self):
        evals = evalset(LEAVE)
        evals.cases[0].key_facts = ["something easier"]
        with self.assertRaises(JudgeError):
            run_evals(evals, runner({}), llm_fn=FakeLLM())

    def test_scorecard_by_type(self):
        refuse = case("q2", "out_of_scope", "What is the weather in Mumbai?", behavior="refuse")
        card = run_evals(evalset(LEAVE, refuse), runner({"annual": answer("24 working days.", [1])}), llm_fn=FakeLLM())
        self.assertEqual(card.passed, 2)
        self.assertEqual(card.pass_rate, 1.0)
        self.assertEqual(card.by_type["out_of_scope"]["pass"], 1)


if __name__ == "__main__":
    unittest.main()


class TestRejudge(unittest.TestCase):
    def test_rejudge_fixes_context_false_failure_without_model(self):
        from forgeai.roles.judge import CaseResult, FactCheck, Scorecard, rejudge
        res = CaseResult(id="q5", type="answerable", question="What is the discount rate for orders above USD 10,000?",
                         expected_behavior="answer", status="fail", answer="3 percent", grounded=True,
                         expected_docs=["terms.md"], cited_docs=["terms.md"],
                         facts=[FactCheck(fact="3 percent discount", found=False, method="llm-rejected")],
                         failure="missing key fact(s): ['3 percent discount']")
        old = Scorecard(agent_name="a", eval_sha256="x", label="iteration_0", passed=0, failed=1, errors=0, total=1,
                        pass_rate=0.0, by_type={"answerable": {"pass": 0, "fail": 1, "error": 0}},
                        judge_llm_calls=1, total_seconds=1.0, results=[res])
        new = rejudge(old)
        self.assertEqual(new.results[0].status, "pass")
        self.assertEqual(new.results[0].facts[0].method, "context")
        self.assertEqual((new.passed, new.pass_rate, new.label), (1, 1.0, "iteration_0_rejudged"))
        self.assertEqual(old.results[0].status, "fail")          # original scorecard untouched
