"""Tests for the code-only evidence checks."""

import unittest

from forgeai.runtime.evidence import estimate_tokens, query_coverage, unsupported_numbers


class TestQueryCoverage(unittest.TestCase):
    def test_partial_coverage(self):
        terms, covered, coverage = query_coverage(
            "Do staff get maternity leave?", ["Staff get 24 days of annual leave."]
        )
        self.assertEqual(terms, ["staff", "get", "maternity", "leave"])
        self.assertEqual(covered, ["staff", "get", "leave"])
        self.assertAlmostEqual(coverage, 0.75)

    def test_full_and_zero_coverage(self):
        self.assertEqual(query_coverage("annual leave", ["annual leave is 24 days"])[2], 1.0)
        self.assertEqual(query_coverage("weather Mumbai", ["annual leave"])[2], 0.0)

    def test_only_stopwords(self):
        self.assertEqual(query_coverage("what is the", ["anything"]), ([], [], 0.0))


class TestUnsupportedNumbers(unittest.TestCase):
    def test_numbers_present_in_source(self):
        self.assertEqual(unsupported_numbers("You get 24 days.", ["entitled to 24 working days"]), [])

    def test_invented_number_flagged(self):
        self.assertEqual(unsupported_numbers("You get 30 days.", ["entitled to 24 working days"]), ["30"])

    def test_number_formats_match(self):
        self.assertEqual(unsupported_numbers("Limit is INR 6000.", ["up to INR 6,000 per night"]), [])
        self.assertEqual(unsupported_numbers("Price 12.50", ["costs 12.5 USD"]), [])

    def test_no_numbers(self):
        self.assertEqual(unsupported_numbers("Yes, a certificate is needed.", ["text"]), [])


class TestEstimateTokens(unittest.TestCase):
    def test_estimate_grows_with_length(self):
        self.assertLess(estimate_tokens("one two"), estimate_tokens("one two three four five six"))


if __name__ == "__main__":
    unittest.main()
