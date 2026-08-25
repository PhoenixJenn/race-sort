from pathlib import Path
import time
import re

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

VISION_MODEL = "qwen3-vl:4b-instruct"


# ============================================================
# NUMBER PROMPT A
# ============================================================

NUMBER_PROMPT_A = """
Inspect this race vehicle specifically for its race number.

Return ONLY one of these:

1. The exact race number as visibly written.
2. UNKNOWN

Rules:

- Read the race number character by character.
- Preserve leading zeros exactly.
- Race numbers are identifiers, not quantities.
- Race numbers may contain digits and letters.
- If any character is ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- If no number is clearly visible, return UNKNOWN.
- Never guess.
"""


# ============================================================
# NUMBER PROMPT B
#
# Intentionally worded differently so this is a second,
# independent observation rather than an identical prompt.
# ============================================================

NUMBER_PROMPT_B = """
Look only for a clearly visible race number on this vehicle.

Return only the exact identifier you can actually read.

Examples of valid identifiers:
54
007
54A
A12

If you cannot confidently distinguish every visible character,
return exactly:

UNKNOWN

Do not infer from colors, vehicle type, rider, sponsors,
graphics, or context.
Do not guess.
"""


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_number(value):
    """
    Race numbers are opaque string identifiers.
    """

    if value is None:
        return None

    value = str(value).strip().upper()

    if value == "UNKNOWN":
        return None

    if not re.fullmatch(
        r"[A-Z0-9]{1,6}",
        value,
    ):
        return None

    return value


# ============================================================
# QWEN CALL
# ============================================================

def read_number(
    crop_path,
    prompt,
):

    start = time.perf_counter()

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [
                    str(crop_path)
                ],
            }
        ],
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    raw_result = (
        response[
            "message"
        ][
            "content"
        ]
        .strip()
    )

    result = normalize_number(
        raw_result
    )

    return (
        result,
        elapsed,
        raw_result,
    )


# ============================================================
# FIND ALL VEHICLE CROPS
# ============================================================

crop_paths = sorted(
    OUTPUT_DIR.glob(
        "GGBM*/motorcycle-*.jpg"
    )
)

print(
    f"Found {len(crop_paths)} "
    f"motorcycle crops."
)

print()


# ============================================================
# TEST
# ============================================================

total_a = 0.0
total_b = 0.0

confirmed = 0
review = 0

batch_start = (
    time.perf_counter()
)


for index, crop_path in enumerate(
    crop_paths,
    start=1,
):

    print("=" * 70)

    print(
        f"[{index}/"
        f"{len(crop_paths)}] "
        f"{crop_path}"
    )

    print("=" * 70)


 # --------------------------------------------------------
    # PASS B FIRST
    # --------------------------------------------------------

    number_b, time_b, raw_b = (
        read_number(
            crop_path,
            NUMBER_PROMPT_B,
        )
    )

    total_b += time_b


    # --------------------------------------------------------
    # PASS A SECOND
    # --------------------------------------------------------

    number_a, time_a, raw_a = (
        read_number(
            crop_path,
            NUMBER_PROMPT_A,
        )
    )

    total_a += time_a


   

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if (
        number_a is not None
        and number_b is not None
        and number_a == number_b
    ):

        decision = "CONFIRMED"
        final_number = number_a

        confirmed += 1

    else:

        decision = "REVIEW"
        final_number = None

        review += 1


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        f"Pass A: "
        f"{number_a or 'UNKNOWN'} "
        f"({time_a:.2f}s)"
    )

    print(
        f"Pass B: "
        f"{number_b or 'UNKNOWN'} "
        f"({time_b:.2f}s)"
    )

    print(
        f"Decision: "
        f"{decision}"
    )

    print(
        f"Final: "
        f"{final_number or 'UNKNOWN'}"
    )

    print()


# ============================================================
# SUMMARY
# ============================================================

batch_elapsed = (
    time.perf_counter()
    - batch_start
)

vehicle_count = len(
    crop_paths
)


print("=" * 70)
print("FAST NUMBER TEST COMPLETE")
print("=" * 70)

print(
    f"Vehicles tested: "
    f"{vehicle_count}"
)

print(
    f"Confirmed: "
    f"{confirmed}"
)

print(
    f"Review: "
    f"{review}"
)

print()

print(
    f"Pass A total: "
    f"{total_a:.2f}s"
)

print(
    f"Pass B total: "
    f"{total_b:.2f}s"
)

print(
    f"Total elapsed: "
    f"{batch_elapsed:.2f}s"
)

if vehicle_count > 0:

    print()

    print(
        f"Average Pass A: "
        f"{total_a / vehicle_count:.2f}s"
    )

    print(
        f"Average Pass B: "
        f"{total_b / vehicle_count:.2f}s"
    )

    print(
        f"Average two-pass "
        f"time/vehicle: "
        f"{(total_a + total_b) / vehicle_count:.2f}s"
    )