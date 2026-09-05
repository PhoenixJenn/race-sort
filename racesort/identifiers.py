"""Race-number identifier validation and normalization."""

import re


RACE_NUMBER_PATTERN = re.compile(r"[A-Z0-9]{1,6}")


def normalize_number(value):
    """Normalize a race number while preserving its string identity.

    Race numbers are opaque identifiers. In particular, ``0`` is valid and
    leading zeros such as ``007`` must not be converted to integers.
    """

    if value is None:
        return None

    normalized = str(value).strip().upper()
    if not normalized or normalized == "UNKNOWN":
        return None

    if not RACE_NUMBER_PATTERN.fullmatch(normalized):
        return None

    return normalized
