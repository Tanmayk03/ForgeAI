"""Tests for reading the model's raw reply (malformed JSON, missing fields, empty replies)."""

import unittest

from forgeai.runtime.output_parser import parse_model_output


class TestParseModelOutput(unittest.TestCase):
    def test_valid_json(self):
        out, reason = parse_model_output('{"answer": "24 days", "sources": [1], "refused": false, "refusal_reason": ""}')
        self.assertEqual(reason, "")
        self.assertEqual(out.answer, "24 days")
        self.assertEqual(out.sources, [1])

    def test_code_fence_and_extra_text(self):
        raw = 'Sure! ```json\n{"answer": "Yes", "sources": [2], "refused": false}\n``` Hope that helps.'
        out, _ = parse_model_output(raw)
        self.assertEqual(out.sources, [2])
        self.assertEqual(out.refusal_reason, "")  # missing optional field gets its default

    def test_empty_reply(self):
        for raw in ("", "   \n ", None):
            out, reason = parse_model_output(raw)
            self.assertIsNone(out)
            self.assertEqual(reason, "empty reply")

    def test_not_json(self):
        out, reason = parse_model_output("The annual leave is 24 days.")
        self.assertIsNone(out)
        self.assertIn("not valid JSON", reason)

    def test_truncated_json(self):
        out, reason = parse_model_output('{"answer": "24 days", "sources": [1')
        self.assertIsNone(out)
        self.assertIn("not valid JSON", reason)

    def test_missing_required_field(self):
        out, reason = parse_model_output('{"answer": "24 days", "sources": [1]}')
        self.assertIsNone(out)
        self.assertIn("refused", reason)

    def test_wrong_field_type(self):
        out, reason = parse_model_output('{"answer": "x", "sources": "first one", "refused": false}')
        self.assertIsNone(out)
        self.assertIn("sources", reason)

    def test_json_array_instead_of_object(self):
        out, reason = parse_model_output('[1, 2, 3]')
        self.assertIsNone(out)

    def test_empty_answer_without_refusal(self):
        out, reason = parse_model_output('{"answer": "  ", "sources": [1], "refused": false}')
        self.assertIsNone(out)
        self.assertIn("answer is empty", reason)

    def test_refusal_with_empty_answer_is_fine(self):
        out, reason = parse_model_output('{"answer": "", "sources": [], "refused": true, "refusal_reason": "no info"}')
        self.assertTrue(out.refused)
        self.assertEqual(out.refusal_reason, "no info")


if __name__ == "__main__":
    unittest.main()
