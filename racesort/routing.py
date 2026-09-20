"""Apply RaceSort's conservative OCR/Qwen recognition policy.

This working application module contains no model calls. It receives normalized
OCR and Qwen evidence and returns the final number, decision, and route name.
Keeping policy separate from inference makes every evidence combination easy to
test, including leading-zero, zero, and alphanumeric string identifiers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    """One deterministic outcome from OCR and Qwen number evidence."""

    final_number: str | None
    decision: str
    route: str


def should_verify_ocr_candidate(ocr_candidates, direct_number):
    """Return whether anchored verification can still confirm the direct read."""

    return direct_number is not None and direct_number in ocr_candidates


def resolve_recognition_route(
    ocr_candidates,
    direct_number,
    verification_number=None,
):
    """Resolve normalized evidence without running OCR or vision models."""

    if ocr_candidates:
        if should_verify_ocr_candidate(ocr_candidates, direct_number):
            if verification_number == direct_number:
                return RoutingDecision(
                    direct_number,
                    "CONFIRMED",
                    "OCR_DIRECT_VERIFY_AGREE",
                )

            return RoutingDecision(
                direct_number,
                "QWEN_CANDIDATE",
                "OCR_DIRECT_VERIFY_REJECTED",
            )

        if direct_number is not None:
            return RoutingDecision(
                direct_number,
                "QWEN_CANDIDATE",
                "OCR_DIRECT_CONFLICT",
            )

        return RoutingDecision(None, "REVIEW", "OCR_DIRECT_UNKNOWN")

    if direct_number is not None:
        return RoutingDecision(
            direct_number,
            "QWEN_CANDIDATE",
            "OCR_EMPTY_DIRECT_CANDIDATE",
        )

    return RoutingDecision(None, "REVIEW", "OCR_EMPTY_DIRECT_UNKNOWN")
