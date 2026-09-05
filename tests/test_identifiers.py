"""Tests for opaque race-number strings."""

import unittest

from racesort.identifiers import normalize_number


class NormalizeNumberTests(unittest.TestCase):
    def test_preserves_zero(self):
        self.assertEqual(normalize_number(0), "0")

    def test_preserves_leading_zeros(self):
        self.assertEqual(normalize_number("007"), "007")
        self.assertEqual(normalize_number("00"), "00")

    def test_accepts_alphanumeric_identifiers(self):
        self.assertEqual(normalize_number("54A"), "54A")
        self.assertEqual(normalize_number("A12"), "A12")

    def test_normalizes_case_and_surrounding_space(self):
        self.assertEqual(normalize_number("  p7  "), "P7")

    def test_rejects_missing_or_unknown_values(self):
        for value in (None, "", "   ", "UNKNOWN", "unknown"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_number(value))

    def test_rejects_invalid_characters_and_length(self):
        for value in ("A-12", "12 3", "1234567", "7.0"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_number(value))


if __name__ == "__main__":
    unittest.main()
