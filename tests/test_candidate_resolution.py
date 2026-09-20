"""Unit tests for model-free independent-DINO candidate policy."""

from pathlib import Path
import unittest

from racesort.candidate_resolution import (
    CandidateDisposition,
    independent_reference_paths,
    resolve_candidate_disposition,
)


class IndependentReferenceTests(unittest.TestCase):
    def test_candidate_crop_cannot_corroborate_itself(self):
        candidate = Path("event/crop-01.jpg")
        references = [candidate, Path("event/crop-02.jpg")]
        self.assertEqual(
            independent_reference_paths(candidate, references),
            [Path("event/crop-02.jpg")],
        )

    def test_all_distinct_references_preserve_order(self):
        references = [Path("ref-b.jpg"), Path("ref-a.jpg")]
        self.assertEqual(
            independent_reference_paths(Path("candidate.jpg"), references),
            references,
        )


class CandidateDispositionTests(unittest.TestCase):
    def test_similarity_at_threshold_is_corroborated(self):
        self.assertEqual(
            resolve_candidate_disposition(0.90, 2, 0.90),
            CandidateDisposition(
                "CORROBORATED",
                (
                    "KNOWN_CONFIRMED_NUMBER",
                    "STRONG_INDEPENDENT_DINO_SUPPORT",
                ),
            ),
        )

    def test_similarity_just_below_threshold_stays_in_review(self):
        self.assertEqual(
            resolve_candidate_disposition(0.8999, 1, 0.90).disposition,
            "KNOWN_NUMBER_REVIEW",
        )

    def test_known_number_without_similarity_stays_in_review(self):
        result = resolve_candidate_disposition(None, 2, 0.90)
        self.assertEqual(result.disposition, "KNOWN_NUMBER_REVIEW")
        self.assertIn("DINO_BELOW_PROMOTION_THRESHOLD", result.reasons)

    def test_no_independent_reference_is_unsupported(self):
        self.assertEqual(
            resolve_candidate_disposition(None, 0, 0.90),
            CandidateDisposition(
                "UNSUPPORTED",
                ("NO_INDEPENDENT_CONFIRMED_REFERENCE",),
            ),
        )

    def test_below_threshold_is_not_conflicting(self):
        result = resolve_candidate_disposition(0.2, 3, 0.90)
        self.assertEqual(result.disposition, "KNOWN_NUMBER_REVIEW")
        self.assertNotIn("CONFLICTING", result.disposition)


if __name__ == "__main__":
    unittest.main()
