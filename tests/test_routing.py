"""Unit tests for model-free OCR/Qwen routing policy."""

import unittest

from racesort.routing import (
    RoutingDecision,
    resolve_recognition_route,
    should_verify_ocr_candidate,
)


class VerificationPlanningTests(unittest.TestCase):
    def test_verifies_only_when_direct_read_matches_an_ocr_candidate(self):
        self.assertTrue(should_verify_ocr_candidate(["49", "007"], "007"))
        self.assertFalse(should_verify_ocr_candidate(["49"], "54"))
        self.assertFalse(should_verify_ocr_candidate(["49"], None))
        self.assertFalse(should_verify_ocr_candidate([], "49"))


class RecognitionRoutingTests(unittest.TestCase):
    def test_three_way_agreement_confirms(self):
        self.assertEqual(
            resolve_recognition_route(["007"], "007", "007"),
            RoutingDecision("007", "CONFIRMED", "OCR_DIRECT_VERIFY_AGREE"),
        )

    def test_rejected_verification_keeps_direct_read_as_candidate(self):
        self.assertEqual(
            resolve_recognition_route(["721"], "721", None),
            RoutingDecision(
                "721", "QWEN_CANDIDATE", "OCR_DIRECT_VERIFY_REJECTED"
            ),
        )

    def test_ocr_direct_conflict_never_confirms(self):
        self.assertEqual(
            resolve_recognition_route(["122"], "721"),
            RoutingDecision("721", "QWEN_CANDIDATE", "OCR_DIRECT_CONFLICT"),
        )

    def test_ocr_with_unknown_direct_read_routes_to_review(self):
        self.assertEqual(
            resolve_recognition_route(["54"], None),
            RoutingDecision(None, "REVIEW", "OCR_DIRECT_UNKNOWN"),
        )

    def test_direct_only_read_remains_a_candidate(self):
        self.assertEqual(
            resolve_recognition_route([], "A12"),
            RoutingDecision(
                "A12", "QWEN_CANDIDATE", "OCR_EMPTY_DIRECT_CANDIDATE"
            ),
        )

    def test_no_evidence_routes_to_review(self):
        self.assertEqual(
            resolve_recognition_route([], None),
            RoutingDecision(None, "REVIEW", "OCR_EMPTY_DIRECT_UNKNOWN"),
        )

    def test_zero_is_a_valid_identifier(self):
        self.assertEqual(
            resolve_recognition_route(["0"], "0", "0").final_number,
            "0",
        )


if __name__ == "__main__":
    unittest.main()
