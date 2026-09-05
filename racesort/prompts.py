"""Qwen prompts used by the RaceSort working pipeline."""


NUMBER_PROMPT_A = """
Inspect this race vehicle specifically for its race number.

Return ONLY one of these:

1. The exact race number as visibly written.
2. UNKNOWN

Rules:

- Read the race number character by character.
- Race numbers are identifiers, not quantities.
- Preserve leading zeros exactly.
- Race numbers may contain digits and letters.
- Preserve the visible character order exactly.
- If any character is ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- If no number is clearly visible, return UNKNOWN.
- Never guess.
"""


NUMBER_PROMPT_B = """
Look only for a clearly visible race number on this vehicle.

Return only the exact identifier you can actually read.

Examples of valid identifiers:

54
007
54A
A12

Rules:

- Preserve leading zeros exactly.
- Treat the race number as an identifier, not a quantity.
- Return UNKNOWN if you cannot confidently distinguish
  every visible character.
- Do not infer from colors, vehicle type, rider,
  sponsors, graphics, or context.
- Do not guess.
"""


DIRECT_NUMBER_PROMPT = """
Inspect this race vehicle specifically for its race number.

Return ONLY one of these:

1. The exact race number as visibly written.
2. UNKNOWN

Rules:

- Read the race number character by character.
- Preserve leading zeros exactly.
- Race numbers are identifiers, not quantities.
- Race numbers may contain digits and letters.
- Preserve the visible character order exactly.
- If any character is ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- If no number is clearly visible, return UNKNOWN.
- Ignore sponsor names, logos, decals, and unrelated text.
- Never guess.
"""


MOTORCYCLE_PROFILE_PROMPT = """
Analyze this race motorcycle.

Return valid JSON only.
Do not include markdown, commentary, or explanation.

Report only information visually supported by the image.

Use exactly this JSON structure:

{
  "race_number": {
    "value": null,
    "status": "unreadable"
  },
  "make": {
    "value": null
  },
  "colors": {
    "primary": []
  },
  "rider": {
    "leathers_colors": [],
    "helmet_colors": []
  },
  "number_plate": {
    "color": null,
    "visible": false,
    "appears_blank": false
  }
}

Rules:

- race_number.value:
  - Return the race number exactly as visibly written.
  - Return it as a string only if clearly readable.
  - Race numbers are identifiers, not quantities.
  - Preserve leading zeros exactly.
    Example: 007 must be returned as "007", not "7".
  - Race numbers may contain digits and may also contain letters.
    Examples: "54", "007", "54A", "A12".
  - Preserve visible character order exactly.
  - Otherwise return null.
  - Never guess.

- race_number.status must be one of:
  - "readable"
  - "unreadable"
  - "not_visible"
  - "blank"

- make.value:
  - Identify the motorcycle manufacturer only when supported
    by a visible logo, badge, or readable manufacturer name.
  - Otherwise return null.
  - Do not infer make from colors or styling.

- colors.primary:
  - List the main visually distinctive colors
    of the motorcycle.

- rider.leathers_colors:
  - List the primary colors of the rider's leathers.

- rider.helmet_colors:
  - List the primary distinctive colors
    of the rider's helmet.

- number_plate.color:
  - Give the primary plate/number-area color if visible.
  - Otherwise return null.

- number_plate.visible:
  - true only if the race-number area is visibly present.

- number_plate.appears_blank:
  - true only if the intended number area is visible
    but no race number can actually be seen.

When uncertain, prefer null, false, an empty array,
or "unreadable" rather than guessing.
"""


CAR_PROFILE_PROMPT = """
Analyze this race car.

Return valid JSON only.
Do not include markdown, commentary, or explanation.

Report only information visually supported by the image.

Use exactly this JSON structure:

{
  "race_number": {
    "value": null,
    "status": "unreadable"
  },
  "make": {
    "value": null
  },
  "model": {
    "value": null
  },
  "colors": {
    "primary": []
  },
  "number_plate": {
    "color": null,
    "visible": false,
    "appears_blank": false
  }
}

Rules:

- race_number.value:
  - Return the race number exactly as visibly written.
  - Return it as a string only if clearly readable.
  - Race numbers are identifiers, not quantities.
  - Preserve leading zeros exactly.
  - Race numbers may contain digits and letters.
    Examples: "54", "007", "54A", "A12".
  - Preserve visible character order exactly.
  - Otherwise return null.
  - Never guess.

- race_number.status must be one of:
  - "readable"
  - "unreadable"
  - "not_visible"
  - "blank"

- make.value:
  - Identify manufacturer only when visually supported.
  - Otherwise return null.

- model.value:
  - Identify model only when visually supported.
  - Otherwise return null.

- colors.primary:
  - List the main visually distinctive vehicle colors.

- number_plate.color:
  - Give the primary race-number-area color if visible.
  - Otherwise return null.

- number_plate.visible:
  - true only if a race-number area is visible.

- number_plate.appears_blank:
  - true only if the intended number area is visible
    but no number can actually be seen.

When uncertain, prefer null, false, an empty array,
or "unreadable" rather than guessing.
"""


def build_ocr_verification_prompt(candidates):
    """Build the constrained prompt for OCR candidate verification."""

    candidate_text = ", ".join(candidates)

    return f"""
Inspect this race vehicle specifically for its race number.

OCR found these possible identifiers:

{candidate_text}

Return ONLY one of these:

1. One exact identifier from the OCR candidate list above.
2. UNKNOWN

Rules:

- Only choose a candidate if that exact identifier is clearly visible
  as the vehicle's race number.
- Preserve leading zeros exactly.
- Do not invent a new number.
- Do not return anything not present in the candidate list.
- Ignore sponsor names, logos, decals, and unrelated text.
- If more than one candidate seems plausible, return UNKNOWN.
- If the race number is blurry or ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- Never guess.
"""


def get_profile_prompt(race_type):
    """Return the rich-profile prompt for a validated race type."""

    if race_type == "motorcycle":
        return MOTORCYCLE_PROFILE_PROMPT

    if race_type == "car":
        return CAR_PROFILE_PROMPT

    raise ValueError("race_type must be either 'motorcycle' or 'car'")
