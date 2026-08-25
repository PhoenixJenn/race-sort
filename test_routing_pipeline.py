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

PRIMARY_RESULTS_PATH = (
    OUTPUT_DIR
    / "primary-filter-results.csv"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "routing-pipeline-results.csv"
)

VISION_MODEL = "qwen3-vl:4b-instruct"


# ============================================================
# GARBAGE FILTER
#
# Human-validated safe rule:
#
# multi-bike photo
# AND relative area < 0.20
# AND relative sharpness < 0.45
# ============================================================

MAX_FILTER_AREA = 0.20
MAX_FILTER_SHARPNESS = 0.45


# ============================================================
# OCR
# ============================================================

ocr_engine = RapidOCR()


# ============================================================
# PROMPTS
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def normalize_race_number(value):
    """
    Race numbers are string identifiers.
    Preserve leading zeros.
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
    Extract plausible race-number candidates.

    A candidate must contain at least one digit.
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

        if not any(
            character.isdigit()
            for character in candidate
        ):
            continue

        candidates.append(
            candidate
        )

    return list(
        dict.fromkeys(
            candidates
        )
    )


def should_filter_out(
    vehicles_in_photo,
    relative_area,
    relative_sharpness,
):
    """
    Conservative human-validated non-primary filter.
    """

    return (
        vehicles_in_photo > 1
        and relative_area
        < MAX_FILTER_AREA
        and relative_sharpness
        < MAX_FILTER_SHARPNESS
    )


def verify_ocr_candidates(
    crop_path,
    candidates,
):
    """
    Ask Qwen to verify one of the OCR candidates.
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
- Do not return anything not present in the candidate list.
- Ignore sponsor names, logos, decals, and unrelated text.
- If more than one candidate seems plausible, return UNKNOWN.
- If the race number is blurry or ambiguous, return UNKNOWN.
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

    number = (
        normalize_race_number(
            raw
        )
    )

    if (
        number is None
        or number not in candidates
    ):
        return (
            None,
            elapsed,
            raw,
        )

    return (
        number,
        elapsed,
        raw,
    )


def direct_qwen_read(
    crop_path,
):
    """
    Ask Qwen for a direct number read when OCR
    produces nothing useful.
    """

    start = time.perf_counter()

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content":
                    DIRECT_NUMBER_PROMPT,

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

    number = (
        normalize_race_number(
            raw
        )
    )

    return (
        number,
        elapsed,
        raw,
    )


# ============================================================
# LOAD PRIMARY FILTER METRICS
# ============================================================

primary_rows = load_csv(
    PRIMARY_RESULTS_PATH
)

print(
    f"Loaded {len(primary_rows)} "
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

total_qwen_verify_time = 0.0
total_qwen_direct_time = 0.0

filtered_count = 0

ocr_candidate_count = 0
ocr_empty_count = 0

qwen_verify_calls = 0
qwen_direct_calls = 0

confirmed_count = 0
review_count = 0


for index, row in enumerate(
    primary_rows,
    start=1,
):

    crop_path = Path(
        row["crop"]
    )

    vehicles_in_photo = int(
        row[
            "vehicles_in_photo"
        ]
    )

    relative_area = float(
        row[
            "relative_area"
        ]
    )

    relative_sharpness = float(
        row[
            "relative_sharpness"
        ]
    )


    print("=" * 70)

    print(
        f"[{index}/"
        f"{len(primary_rows)}] "
        f"{crop_path}"
    )

    print("=" * 70)


    # ========================================================
    # STAGE 1 — GARBAGE FILTER
    # ========================================================

    filtered_out = (
        should_filter_out(
            vehicles_in_photo,
            relative_area,
            relative_sharpness,
        )
    )


    if filtered_out:

        filtered_count += 1

        print(
            "Route: FILTERED_OUT"
        )

        print(
            f"relative area="
            f"{relative_area:.3f}, "
            f"relative sharpness="
            f"{relative_sharpness:.3f}"
        )

        print()


        rows.append(
            {
                "crop":
                    str(crop_path),

                "route":
                    "FILTERED_OUT",

                "relative_area":
                    relative_area,

                "relative_sharpness":
                    relative_sharpness,

                "ocr_candidates":
                    "",

                "ocr_seconds":
                    0.0,

                "qwen_mode":
                    "NONE",

                "qwen_number":
                    "",

                "qwen_seconds":
                    0.0,

                "final_number":
                    "",

                "decision":
                    "FILTERED_OUT",

                "qwen_raw":
                    "",
            }
        )

        continue


    # ========================================================
    # STAGE 2 — OCR
    # ========================================================

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


    # ========================================================
    # STAGE 3A — OCR FOUND CANDIDATE(S)
    # ========================================================

    if candidates:

        ocr_candidate_count += 1
        qwen_verify_calls += 1

        (
            qwen_number,
            qwen_elapsed,
            qwen_raw,
        ) = verify_ocr_candidates(
            crop_path,
            candidates,
        )

        total_qwen_verify_time += (
            qwen_elapsed
        )

        qwen_mode = (
            "VERIFY_OCR"
        )


        if qwen_number is not None:

            final_number = (
                qwen_number
            )

            decision = (
                "CONFIRMED"
            )

            confirmed_count += 1

        else:

            final_number = None

            decision = (
                "REVIEW"
            )

            review_count += 1


    # ========================================================
    # STAGE 3B — OCR FOUND NOTHING
    # ========================================================

    else:

        ocr_empty_count += 1
        qwen_direct_calls += 1

        (
            qwen_number,
            qwen_elapsed,
            qwen_raw,
        ) = direct_qwen_read(
            crop_path
        )

        total_qwen_direct_time += (
            qwen_elapsed
        )

        qwen_mode = (
            "DIRECT_READ"
        )


        if qwen_number is not None:

            final_number = (
                qwen_number
            )

            decision = (
                "CONFIRMED"
            )

            confirmed_count += 1

        else:

            final_number = None

            decision = (
                "REVIEW"
            )

            review_count += 1


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"Qwen mode: "
        f"{qwen_mode}"
    )

    print(
        f"Qwen result: "
        f"{qwen_number or 'UNKNOWN'}"
    )

    print(
        f"Qwen time: "
        f"{qwen_elapsed:.2f}s"
    )

    print(
        f"Decision: "
        f"{decision}"
    )

    print(
        f"Final number: "
        f"{final_number or 'UNKNOWN'}"
    )

    print()


    # ========================================================
    # STORE ROW
    # ========================================================

    rows.append(
        {
            "crop":
                str(crop_path),

            "route":
                qwen_mode,

            "relative_area":
                relative_area,

            "relative_sharpness":
                relative_sharpness,

            "ocr_candidates":
                " | ".join(
                    candidates
                ),

            "ocr_seconds":
                ocr_elapsed,

            "qwen_mode":
                qwen_mode,

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
        "route",
        "relative_area",
        "relative_sharpness",
        "ocr_candidates",
        "ocr_seconds",
        "qwen_mode",
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

processed_count = (
    len(primary_rows)
    - filtered_count
)


print("=" * 70)
print("ROUTING PIPELINE TEST COMPLETE")
print("=" * 70)

print(
    f"Vehicle crops: "
    f"{len(primary_rows)}"
)

print(
    f"Filtered out: "
    f"{filtered_count}"
)

print(
    f"Processed further: "
    f"{processed_count}"
)

print()

print(
    f"OCR candidate cases: "
    f"{ocr_candidate_count}"
)

print(
    f"OCR empty cases: "
    f"{ocr_empty_count}"
)

print()

print(
    f"Qwen VERIFY calls: "
    f"{qwen_verify_calls}"
)

print(
    f"Qwen DIRECT calls: "
    f"{qwen_direct_calls}"
)

print(
    f"Total Qwen calls: "
    f"{qwen_verify_calls + qwen_direct_calls}"
)

print()

print(
    f"Confirmed: "
    f"{confirmed_count}"
)

print(
    f"Review: "
    f"{review_count}"
)

print()

print(
    f"OCR total: "
    f"{total_ocr_time:.2f}s"
)

print(
    f"Qwen VERIFY total: "
    f"{total_qwen_verify_time:.2f}s"
)

print(
    f"Qwen DIRECT total: "
    f"{total_qwen_direct_time:.2f}s"
)

print(
    f"Batch elapsed: "
    f"{batch_elapsed:.2f}s"
)


if processed_count > 0:

    print()

    print(
        f"Average total time / processed vehicle: "
        f"{batch_elapsed / processed_count:.2f}s"
    )


print()

print(
    f"Results CSV: "
    f"{RESULTS_PATH}"
)