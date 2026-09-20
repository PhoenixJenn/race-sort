"""Resolve Qwen candidates using independent DINO evidence.

This working application module contains no model calls. It filters out an
invalid self-reference and converts the best independent similarity evidence
into RaceSort's conservative candidate disposition and reason strings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateDisposition:
    """A candidate-resolution state and its explainable reasons."""

    disposition: str
    reasons: tuple[str, ...]


def independent_reference_paths(candidate_path, reference_paths):
    """Return references excluding the candidate crop itself."""

    candidate_path = Path(candidate_path).resolve()
    return [
        Path(reference_path)
        for reference_path in reference_paths
        if Path(reference_path).resolve() != candidate_path
    ]


def resolve_candidate_disposition(
    best_similarity,
    independent_reference_count,
    threshold,
):
    """Apply the current DINO promotion and review policy."""

    if best_similarity is not None and best_similarity >= threshold:
        return CandidateDisposition(
            "CORROBORATED",
            (
                "KNOWN_CONFIRMED_NUMBER",
                "STRONG_INDEPENDENT_DINO_SUPPORT",
            ),
        )

    if independent_reference_count > 0:
        return CandidateDisposition(
            "KNOWN_NUMBER_REVIEW",
            (
                "KNOWN_CONFIRMED_NUMBER",
                "DINO_BELOW_PROMOTION_THRESHOLD",
            ),
        )

    return CandidateDisposition(
        "UNSUPPORTED",
        ("NO_INDEPENDENT_CONFIRMED_REFERENCE",),
    )
