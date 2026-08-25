"""Check the current RaceSort regression output without running models."""

from collections import Counter
import json
from pathlib import Path
import re
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "test-output"

EXPECTED_PHOTOS = 19
EXPECTED_VEHICLES = 33
EXPECTED_FILTERED_NON_PRIMARY = 7
EXPECTED_FILTERED_TOO_BLURRY = 2
EXPECTED_OCR_CANDIDATE_CASES = 13
EXPECTED_OCR_EMPTY_CASES = 11
EXPECTED_QWEN_VERIFY_CALLS = 10
EXPECTED_QWEN_DIRECT_CALLS = 24
EXPECTED_REVIEW_WORKLOAD = 10

RACE_NUMBER_PATTERN = re.compile(r"[A-Z0-9]{1,6}")

AUTOMATIC_ASSIGNMENT_DECISIONS = {
    "CONFIRMED",
    "CORROBORATED",
}

HUMAN_REVIEW_DECISIONS = {
    "KNOWN_NUMBER_REVIEW",
    "CONFLICTING",
    "UNSUPPORTED",
    "REVIEW",
}


failures = []
passes = []
warnings = []


def check(condition, description):
    """Record one readable pass/failure instead of stopping early."""

    if condition:
        passes.append(description)
    else:
        failures.append(description)


def warn_if_false(condition, description):
    """Record a non-fatal operational variation."""

    if condition:
        passes.append(description)
    else:
        warnings.append(description)


def load_photo_results():
    """Load every current per-photo result file."""

    result_paths = sorted(
        OUTPUT_DIR.glob("GGBM*/photo-results.json")
    )

    results = []

    for result_path in result_paths:
        with open(
            result_path,
            "r",
            encoding="utf-8",
        ) as json_file:
            results.append(
                (result_path, json.load(json_file))
            )

    return results


def find_vehicle(photo_lookup, photo_stem, vehicle_number):
    """Return one known vehicle from the regression batch."""

    photo_result = photo_lookup[photo_stem]

    return next(
        vehicle
        for vehicle in photo_result["vehicles"]
        if vehicle["vehicle"] == vehicle_number
    )


if not OUTPUT_DIR.exists():
    print(
        f"ERROR: output directory does not exist: {OUTPUT_DIR}"
    )
    print("Run: python test_pipeline.py")
    sys.exit(1)


photo_records = load_photo_results()

if not photo_records:
    print(
        f"ERROR: no photo-results.json files found in {OUTPUT_DIR}"
    )
    print("Run: python test_pipeline.py")
    sys.exit(1)


photo_lookup = {
    result_path.parent.name: photo_result
    for result_path, photo_result in photo_records
}

vehicle_records = [
    (result_path, photo_result, vehicle)
    for result_path, photo_result in photo_records
    for vehicle in photo_result["vehicles"]
]

decision_counts = Counter(
    vehicle["decision"]
    for _, _, vehicle in vehicle_records
)


# ------------------------------------------------------------------
# Stable batch/routing invariants
# ------------------------------------------------------------------

check(
    len(photo_records) == EXPECTED_PHOTOS,
    f"photos processed == {EXPECTED_PHOTOS}",
)

check(
    len(vehicle_records) == EXPECTED_VEHICLES,
    f"vehicles detected == {EXPECTED_VEHICLES}",
)

check(
    decision_counts["FILTERED_NON_PRIMARY"]
    == EXPECTED_FILTERED_NON_PRIMARY,
    (
        "FILTERED_NON_PRIMARY == "
        f"{EXPECTED_FILTERED_NON_PRIMARY}"
    ),
)

check(
    decision_counts["FILTERED_TOO_BLURRY"]
    == EXPECTED_FILTERED_TOO_BLURRY,
    (
        "FILTERED_TOO_BLURRY == "
        f"{EXPECTED_FILTERED_TOO_BLURRY}"
    ),
)

processed_vehicles = [
    vehicle
    for _, _, vehicle in vehicle_records
    if vehicle["decision"]
    not in {
        "FILTERED_NON_PRIMARY",
        "FILTERED_TOO_BLURRY",
    }
]

ocr_candidate_cases = sum(
    1
    for vehicle in processed_vehicles
    if vehicle["routing"]["ocr_candidates"]
)

ocr_empty_cases = sum(
    1
    for vehicle in processed_vehicles
    if not vehicle["routing"]["ocr_candidates"]
)

qwen_verify_calls = sum(
    1
    for vehicle in processed_vehicles
    if vehicle["routing"]["qwen_verify_raw"] is not None
)

qwen_direct_calls = sum(
    1
    for vehicle in processed_vehicles
    if "DIRECT" in vehicle["routing"]["route"]
)

check(
    ocr_candidate_cases == EXPECTED_OCR_CANDIDATE_CASES,
    f"OCR candidate cases == {EXPECTED_OCR_CANDIDATE_CASES}",
)

check(
    ocr_empty_cases == EXPECTED_OCR_EMPTY_CASES,
    f"OCR empty cases == {EXPECTED_OCR_EMPTY_CASES}",
)

warn_if_false(
    qwen_verify_calls == EXPECTED_QWEN_VERIFY_CALLS,
    (
        "Qwen verification-call count differs from the expected "
        f"{EXPECTED_QWEN_VERIFY_CALLS}; observed {qwen_verify_calls}"
    ),
)

check(
    qwen_direct_calls == EXPECTED_QWEN_DIRECT_CALLS,
    f"Qwen direct calls == {EXPECTED_QWEN_DIRECT_CALLS}",
)

for vehicle in processed_vehicles:
    routing = vehicle["routing"]

    if routing["qwen_verify_raw"] is None:
        continue

    check(
        routing["qwen_direct_number"]
        in routing["ocr_candidates"],
        (
            "anchored verification only runs after OCR/direct "
            "candidate agreement"
        ),
    )


# ------------------------------------------------------------------
# Identifier and automatic-assignment safety
# ------------------------------------------------------------------

for result_path, photo_result in photo_records:
    expected_assignments = []

    for vehicle in photo_result["vehicles"]:
        race_number = vehicle.get("final_number")

        if race_number is not None:
            check(
                isinstance(race_number, str),
                (
                    f"{result_path.parent.name} vehicle "
                    f"{vehicle['vehicle']} number is a string"
                ),
            )

            check(
                bool(RACE_NUMBER_PATTERN.fullmatch(race_number)),
                (
                    f"{result_path.parent.name} vehicle "
                    f"{vehicle['vehicle']} number matches "
                    "[A-Z0-9]{1,6}"
                ),
            )

        if (
            vehicle["decision"]
            in AUTOMATIC_ASSIGNMENT_DECISIONS
            and race_number is not None
            and race_number not in expected_assignments
        ):
            expected_assignments.append(race_number)

    check(
        photo_result["confirmed_photo_numbers"]
        == expected_assignments,
        (
            f"{result_path.parent.name} assignments contain only "
            "CONFIRMED/CORROBORATED numbers"
        ),
    )


# ------------------------------------------------------------------
# Critical known regression cases
# ------------------------------------------------------------------

blurry_0005 = find_vehicle(
    photo_lookup,
    "GGBM0005",
    1,
)

blurry_0006 = find_vehicle(
    photo_lookup,
    "GGBM0006",
    1,
)

check(
    blurry_0005["decision"] == "FILTERED_TOO_BLURRY",
    "GGBM0005 vehicle 1 is filtered as too blurry",
)

check(
    blurry_0006["decision"] == "FILTERED_TOO_BLURRY",
    "GGBM0006 vehicle 1 is filtered as too blurry",
)

zero_vehicle = find_vehicle(
    photo_lookup,
    "GGBM0008",
    1,
)

check(
    zero_vehicle["final_number"] == "0",
    "race number 0 is preserved as the string '0'",
)

check(
    zero_vehicle["decision"] == "CONFIRMED",
    "race number 0 remains CONFIRMED",
)

recovered_54 = find_vehicle(
    photo_lookup,
    "GGBM0017",
    1,
)

check(
    recovered_54["routing"]["ocr_candidates"] == ["C42A"],
    "GGBM0017 preserves the bad OCR candidate C42A",
)

check(
    recovered_54["final_number"] == "54",
    "GGBM0017 direct fallback recovers 54",
)

check(
    recovered_54["decision"] == "CORROBORATED",
    "GGBM0017 number 54 is independently CORROBORATED",
)

recovered_721 = find_vehicle(
    photo_lookup,
    "GGBM0018",
    1,
)

check(
    recovered_721["routing"]["ocr_candidates"] == ["122"],
    "GGBM0018 preserves the bad OCR candidate 122",
)

check(
    recovered_721["final_number"] == "721",
    "GGBM0018 direct fallback recovers 721",
)

check(
    recovered_721["decision"] == "KNOWN_NUMBER_REVIEW",
    "GGBM0018 number 721 remains below automatic promotion",
)

check(
    len(photo_lookup["GGBM0018"]["vehicles"]) == 4,
    "GGBM0018 preserves all four detected vehicles",
)

for photo_stem, vehicle_number in (
    ("GGBM0012", 1),
    ("GGBM0020", 2),
):
    numberless_vehicle = find_vehicle(
        photo_lookup,
        photo_stem,
        vehicle_number,
    )

    check(
        numberless_vehicle["decision"]
        not in AUTOMATIC_ASSIGNMENT_DECISIONS,
        (
            f"{photo_stem} vehicle {vehicle_number} direct read "
            "is not automatically assigned"
        ),
    )


# ------------------------------------------------------------------
# DINO independence and threshold safety
# ------------------------------------------------------------------

for result_path, _, vehicle in vehicle_records:
    resolution = vehicle.get("candidate_resolution")

    if resolution is None:
        continue

    candidate_crop = (
        result_path.parent
        / vehicle["crop"]
    ).resolve()

    best_reference = resolution.get("best_reference")

    if best_reference is not None:
        reference_path = (
            PROJECT_DIR
            / best_reference
        ).resolve()

        check(
            candidate_crop != reference_path,
            (
                f"{result_path.parent.name} vehicle "
                f"{vehicle['vehicle']} DINO reference is independent"
            ),
        )

    if vehicle["decision"] == "CORROBORATED":
        check(
            resolution["best_dino_similarity"]
            >= resolution["threshold"],
            (
                f"{result_path.parent.name} vehicle "
                f"{vehicle['vehicle']} meets its DINO threshold"
            ),
        )


review_workload = sum(
    1
    for _, _, vehicle in vehicle_records
    if vehicle["decision"] in HUMAN_REVIEW_DECISIONS
)

warn_if_false(
    review_workload == EXPECTED_REVIEW_WORKLOAD,
    (
        "total human-review workload differs from the expected "
        f"{EXPECTED_REVIEW_WORKLOAD}; observed {review_workload}"
    ),
)


# ------------------------------------------------------------------
# Report every result, then return a shell-friendly exit status.
# ------------------------------------------------------------------

print("=" * 72)
print("RACESORT REGRESSION CHECK")
print("=" * 72)

print(f"Photos: {len(photo_records)}")
print(f"Vehicles: {len(vehicle_records)}")
print(f"Decisions: {dict(sorted(decision_counts.items()))}")
print(f"Human-review workload: {review_workload}")
print()

print(f"Passed checks: {len(passes)}")
print(f"Warnings: {len(warnings)}")
print(f"Failed checks: {len(failures)}")

if warnings:
    print()
    print("WARNINGS")

    for warning in warnings:
        print(f"- {warning}")

if failures:
    print()
    print("FAILURES")

    for failure in failures:
        print(f"- {failure}")

    sys.exit(1)

print()
print("PASS: all regression checks succeeded")
