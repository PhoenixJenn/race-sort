from pathlib import Path
import csv
import re
import time

from rapidocr import RapidOCR


OUTPUT_DIR = Path("test-output")
RESULTS_PATH = OUTPUT_DIR / "ocr-results.csv"

engine = RapidOCR()


def normalize_race_number(value):
    """
    Race numbers are identifiers, not quantities.

    Preserve leading zeros and allow future letters.
    """

    if value is None:
        return None

    value = str(value).strip().upper()

    if not value:
        return None

    if not re.fullmatch(
        r"[A-Z0-9]{1,6}",
        value,
    ):
        return None

    return value


crop_paths = sorted(
    OUTPUT_DIR.glob(
        "GGBM*/motorcycle-*.jpg"
    )
)

print(
    f"Found {len(crop_paths)} vehicle crops."
)

print()

rows = []

batch_start = time.perf_counter()
total_ocr_time = 0.0


for index, crop_path in enumerate(
    crop_paths,
    start=1,
):

    print("=" * 70)

    print(
        f"[{index}/{len(crop_paths)}] "
        f"{crop_path}"
    )

    start = time.perf_counter()

    result = engine(
        str(crop_path)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    total_ocr_time += elapsed


    # --------------------------------------------------------
    # Collect all OCR text that RapidOCR detected
    # --------------------------------------------------------

    detected_text = []

    if result is not None:

        # Current RapidOCR result objects expose text results
        # through txts when text was detected.
        txts = getattr(
            result,
            "txts",
            None,
        )

        if txts:

            for text in txts:

                cleaned = (
                    str(text)
                    .strip()
                    .upper()
                )

                if cleaned:
                    detected_text.append(
                        cleaned
                    )


    # --------------------------------------------------------
    # Find strings that could plausibly be race numbers
    # --------------------------------------------------------

    race_number_candidates = []

    for text in detected_text:

        # Remove spaces so something such as "8 66"
        # can still be evaluated as "866".
        compact = text.replace(
            " ",
            "",
        )

        candidate = normalize_race_number(
            compact
        )

        # OCR text may contain sponsor names or other words.
        # A race-number candidate must contain at least one digit.
        #
        # This still allows:
        #   54
        #   007
        #   54A
        #   A12
        #
        # But rejects:
        #   SHOEI
        #   RACING
        #   TIME



        if (
            candidate is not None
            and any(
                character.isdigit()
                for character in candidate
            )
        ):

            race_number_candidates.append(
                candidate
            )


    # Remove duplicates while preserving order.
    race_number_candidates = list(
        dict.fromkeys(
            race_number_candidates
        )
    )


    print(
        f"OCR time: "
        f"{elapsed:.3f}s"
    )

    print(
        f"Detected text: "
        f"{detected_text}"
    )

    print(
        f"Possible race numbers: "
        f"{race_number_candidates}"
    )

    print()


    rows.append(
        {
            "crop":
                str(crop_path),

            "ocr_seconds":
                elapsed,

            "detected_text":
                " | ".join(
                    detected_text
                ),

            "race_number_candidates":
                " | ".join(
                    race_number_candidates
                ),
        }
    )


batch_elapsed = (
    time.perf_counter()
    - batch_start
)


with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "crop",
            "ocr_seconds",
            "detected_text",
            "race_number_candidates",
        ],
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


print("=" * 70)
print("OCR TEST COMPLETE")
print("=" * 70)

print(
    f"Vehicles tested: "
    f"{len(crop_paths)}"
)

print(
    f"OCR total: "
    f"{total_ocr_time:.2f}s"
)

print(
    f"Batch elapsed: "
    f"{batch_elapsed:.2f}s"
)

if crop_paths:

    print(
        f"Average OCR time/vehicle: "
        f"{total_ocr_time / len(crop_paths):.3f}s"
    )

print(
    f"Results CSV: "
    f"{RESULTS_PATH}"
)