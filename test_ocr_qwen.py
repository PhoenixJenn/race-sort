from pathlib import Path
import csv
import re
import time

import ollama
from rapidocr import RapidOCR


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

RESULTS_PATH = (
    OUTPUT_DIR
    / "ocr-qwen-results.csv"
)

VISION_MODEL = "qwen3-vl:4b-instruct"


# ============================================================
# OCR
# ============================================================

ocr_engine = RapidOCR()


# ============================================================
# HELPERS
# ============================================================

def normalize_race_number(value):
    """
    Race numbers are opaque string identifiers.

    Preserve leading zeros and allow future letters.
    """

    if value is None:
        return None

    value = (
        str(value)
        .strip()
        .upper()
    )

    if not value:
        return None

    if value == "UNKNOWN":
        return None

    if not re.fullmatch(
        r"[A-Z0-9]{1,6}",
        value,
    ):
        return None

    return value


def extract_ocr_candidates(result):
    """
    Extract plausible race-number candidates
    from RapidOCR output.

    Candidate must contain at least one digit.
    """

    candidates = []

    if result is None:
        return candidates

    txts = getattr(
        result,
        "txts",
        None,
    )

    if not txts:
        return candidates

    for text in txts:

        cleaned = (
            str(text)
            .strip()
            .upper()
        )

        if not cleaned:
            continue

        compact = cleaned.replace(
            " ",
            "",
        )

        candidate = (
            normalize_race_number(
                compact
            )
        )

        if candidate is None:
            continue

        # Reject sponsor-only words.
        if not any(
            character.isdigit()
            for character in candidate
        ):
            continue

        candidates.append(
            candidate
        )

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(
            candidates
        )
    )


def verify_with_qwen(
    crop_path,
    candidates,
):
    """
    Ask Qwen which OCR candidate, if any,
    is actually the visible race number.
    """

    candidate_text = ", ".join(
        candidates
    )

    prompt = f"""
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
- Do not return a number that is not in the candidate list.
- Ignore sponsor names, logos, decals, and unrelated text.
- If more than one candidate seems possible, return UNKNOWN.
- If the number is blurry or ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- Never guess.
"""

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

    raw = (
        response[
            "message"
        ][
            "content"
        ]
        .strip()
    )

    normalized = (
        normalize_race_number(
            raw
        )
    )

    # Qwen is only allowed to return
    # one of the OCR candidates.
    if (
        normalized is None
        or normalized not in candidates
    ):
        return (
            None,
            elapsed,
            raw,
        )

    return (
        normalized,
        elapsed,
        raw,
    )


# ============================================================
# FIND EXISTING CROPS
# ============================================================

crop_paths = sorted(
    OUTPUT_DIR.glob(
        "GGBM*/motorcycle-*.jpg"
    )
)

print(
    f"Found {len(crop_paths)} "
    f"vehicle crops."
)

print()


# ============================================================
# BENCHMARK
# ============================================================

rows = []

batch_start = (
    time.perf_counter()
)

total_ocr_time = 0.0
total_qwen_time = 0.0

qwen_calls = 0

confirmed = 0
review = 0


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
    # OCR PASS
    # --------------------------------------------------------

    ocr_start = (
        time.perf_counter()
    )

    ocr_result = ocr_engine(
        str(crop_path)
    )

    ocr_elapsed = (
        time.perf_counter()
        - ocr_start
    )

    total_ocr_time += (
        ocr_elapsed
    )

    candidates = (
        extract_ocr_candidates(
            ocr_result
        )
    )


    print(
        f"OCR candidates: "
        f"{candidates}"
    )

    print(
        f"OCR time: "
        f"{ocr_elapsed:.3f}s"
    )


    # --------------------------------------------------------
    # NO OCR CANDIDATE
    # --------------------------------------------------------

    if not candidates:

        final_number = None
        qwen_number = None
        qwen_elapsed = 0.0
        qwen_raw = None

        decision = "REVIEW"

        review += 1

        print(
            "No OCR candidate."
        )

        print(
            "Decision: REVIEW"
        )


    # --------------------------------------------------------
    # OCR FOUND ONE OR MORE CANDIDATES
    # --------------------------------------------------------

    else:

        qwen_calls += 1

        (
            qwen_number,
            qwen_elapsed,
            qwen_raw,
        ) = verify_with_qwen(
            crop_path,
            candidates,
        )

        total_qwen_time += (
            qwen_elapsed
        )


        print(
            f"Qwen verification: "
            f"{qwen_number or 'UNKNOWN'} "
            f"({qwen_elapsed:.2f}s)"
        )


        # ----------------------------------------------------
        # CONFIRMATION
        #
        # Qwen has already been constrained to select only
        # from the OCR candidate list.
        # ----------------------------------------------------

        if qwen_number is not None:

            final_number = (
                qwen_number
            )

            decision = (
                "CONFIRMED"
            )

            confirmed += 1

        else:

            final_number = None

            decision = (
                "REVIEW"
            )

            review += 1


        print(
            f"Decision: "
            f"{decision}"
        )

        print(
            f"Final number: "
            f"{final_number or 'UNKNOWN'}"
        )


    print()


    # --------------------------------------------------------
    # SAVE ROW
    # --------------------------------------------------------

    rows.append(
        {
            "crop":
                str(crop_path),

            "ocr_candidates":
                " | ".join(
                    candidates
                ),

            "ocr_seconds":
                ocr_elapsed,

            "qwen_called":
                bool(candidates),

            "qwen_number":
                qwen_number,

            "qwen_seconds":
                qwen_elapsed,

            "final_number":
                final_number,

            "decision":
                decision,

            "qwen_raw":
                qwen_raw,
        }
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "crop",
        "ocr_candidates",
        "ocr_seconds",
        "qwen_called",
        "qwen_number",
        "qwen_seconds",
        "final_number",
        "decision",
        "qwen_raw",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


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
print("OCR + QWEN TEST COMPLETE")
print("=" * 70)

print(
    f"Vehicles tested: "
    f"{vehicle_count}"
)

print(
    f"Qwen calls made: "
    f"{qwen_calls}"
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
    f"OCR total: "
    f"{total_ocr_time:.2f}s"
)

print(
    f"Qwen total: "
    f"{total_qwen_time:.2f}s"
)

print(
    f"Batch elapsed: "
    f"{batch_elapsed:.2f}s"
)


if vehicle_count > 0:

    print()

    print(
        f"Average OCR time/vehicle: "
        f"{total_ocr_time / vehicle_count:.3f}s"
    )

    print(
        f"Average total time/vehicle: "
        f"{batch_elapsed / vehicle_count:.2f}s"
    )


if qwen_calls > 0:

    print(
        f"Average Qwen call: "
        f"{total_qwen_time / qwen_calls:.2f}s"
    )


print(
    f"Results CSV: "
    f"{RESULTS_PATH}"
)