"""Tests for the judge-vs-human agreement measure."""

import tempfile
import unittest

from forgeai.roles.judge import CaseResult
from forgeai.roles.judge_agreement import HumanLabel, agreement, choose_sample, load_labels, save_labels


def res(id_, status):
    return CaseResult(id=id_, type="answerable", question="q", expected_behavior="answer", status=status)


def lab(id_, judge, ok):
    return HumanLabel(scorecard="iteration_0", id=id_, question="q", answer="a", judge=judge, human_ok=ok)


class TestAgreement(unittest.TestCase):
    def test_counts_and_disagreements(self):
        a = agreement([lab("q1", "pass", True), lab("q2", "fail", False), lab("q3", "pass", False), lab("q4", "fail", True)])
        self.assertEqual((a.labelled, a.agree, a.rate), (4, 2, 0.5))
        self.assertEqual(a.false_pass, ["q3"])
        self.assertEqual(a.false_fail, ["q4"])

    def test_sample_takes_all_failures_first_and_is_repeatable(self):
        results = [res(f"q{i:02d}", "fail" if i % 4 == 0 else "pass") for i in range(1, 25)] + [res("q99", "error")]
        s1, s2 = choose_sample(results, 10), choose_sample(results, 10)
        self.assertEqual([r.id for r in s1], [r.id for r in s2])
        self.assertEqual(len(s1), 10)
        self.assertTrue({"q04", "q08", "q12", "q16", "q20", "q24"} <= {r.id for r in s1})
        self.assertNotIn("q99", [r.id for r in s1])                  # errors are not labelled

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_labels([lab("q1", "pass", True)], tmp)
            self.assertEqual(load_labels(tmp)[0].id, "q1")
        self.assertEqual(load_labels("/nonexistent"), [])


if __name__ == "__main__":
    unittest.main()
