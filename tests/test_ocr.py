"""Unit tests for conservative RapidOCR candidate normalization."""

from types import SimpleNamespace
import unittest

from racesort.ocr import extract_ocr_candidates


def ocr_result(*texts):
    """Create the small RapidOCR-shaped object needed by these tests."""

    return SimpleNamespace(txts=list(texts))


class ExtractOcrCandidatesTests(unittest.TestCase):
    def test_missing_or_empty_results_return_no_candidates(self):
        self.assertEqual(extract_ocr_candidates(None), [])
        self.assertEqual(extract_ocr_candidates(SimpleNamespace()), [])
        self.assertEqual(extract_ocr_candidates(ocr_result()), [])

    def test_preserves_leading_zeros_and_valid_zero(self):
        self.assertEqual(
            extract_ocr_candidates(ocr_result("007", "0")),
            ["007", "0"],
        )

    def test_normalizes_case_and_removes_spaces(self):
        self.assertEqual(
            extract_ocr_candidates(ocr_result(" a 1 2 ", "54 a")),
            ["A12", "54A"],
        )

    def test_rejects_letter_only_and_invalid_text(self):
        self.assertEqual(
            extract_ocr_candidates(
                ocr_result("YAMAHA", "A", "12-3", "1234567", "UNKNOWN")
            ),
            [],
        )

    def test_removes_duplicates_without_changing_order(self):
        self.assertEqual(
            extract_ocr_candidates(ocr_result("54", "007", "54", "0")),
            ["54", "007", "0"],
        )


if __name__ == "__main__":
    unittest.main()
