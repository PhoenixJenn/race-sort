"""Normalize RapidOCR text into conservative race-number candidates.

This working pipeline module accepts a RapidOCR result and returns unique
candidate strings in their original encounter order. It preserves leading
zeros and valid alphanumeric identifiers, while rejecting invalid values and
letter-only text that is more likely to be a sponsor or logo.
"""

from racesort.identifiers import normalize_number


def extract_ocr_candidates(result):
    """Extract unique plausible race-number candidates from RapidOCR output."""

    candidates = []

    if result is None:
        return candidates

    texts = getattr(result, "txts", None)
    if not texts:
        return candidates

    for text in texts:
        compact = str(text).strip().upper().replace(" ", "")
        candidate = normalize_number(compact)

        if candidate is None:
            continue

        if not any(character.isdigit() for character in candidate):
            continue

        candidates.append(candidate)

    return list(dict.fromkeys(candidates))
