"""Check the two-photo merged-box pipeline output without running models."""

import argparse
import json
from pathlib import Path
import sys


EXPECTED = {
    "GGBM0021": {
        "vehicles_detected": 4,
        "sources": [
            "baseline",
            "baseline",
            "merged_box_child",
            "merged_box_child",
        ],
        "split_decisions": [
            "FILTERED_NON_PRIMARY",
            "FILTERED_NON_PRIMARY",
        ],
    },
    "GGBM0082": {
        "vehicles_detected": 2,
        "sources": [
            "merged_box_child",
            "merged_box_child",
        ],
        "numbers": ["869", "215"],
        "decisions": ["UNSUPPORTED", "UNSUPPORTED"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("/tmp/racesort-merged-output"),
    )
    return parser.parse_args()


def check_equal(failures, actual, expected, description):
    if actual != expected:
        failures.append(
            f"{description}: expected {expected!r}, got {actual!r}"
        )


def main():
    args = parse_args()
    failures = []

    for photo_stem, expected in EXPECTED.items():
        result_path = (
            args.output_dir
            / photo_stem
            / "photo-results.json"
        )

        if not result_path.exists():
            failures.append(f"missing output: {result_path}")
            continue

        with result_path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)

        vehicles = result["vehicles"]
        prefix = f"{photo_stem}"

        check_equal(
            failures,
            result["vehicles_detected"],
            expected["vehicles_detected"],
            f"{prefix} vehicle count",
        )
        check_equal(
            failures,
            [vehicle.get("detection_source") for vehicle in vehicles],
            expected["sources"],
            f"{prefix} detection sources",
        )

        if "split_decisions" in expected:
            split_vehicles = [
                vehicle
                for vehicle in vehicles
                if vehicle.get("detection_source")
                == "merged_box_child"
            ]
            check_equal(
                failures,
                [vehicle["decision"] for vehicle in split_vehicles],
                expected["split_decisions"],
                f"{prefix} split decisions",
            )

        if "numbers" in expected:
            numbers = [vehicle["final_number"] for vehicle in vehicles]
            check_equal(
                failures,
                numbers,
                expected["numbers"],
                f"{prefix} race-number strings",
            )
            check_equal(
                failures,
                [type(number).__name__ for number in numbers],
                ["str"] * len(numbers),
                f"{prefix} race-number types",
            )
            check_equal(
                failures,
                [vehicle["decision"] for vehicle in vehicles],
                expected["decisions"],
                f"{prefix} routing decisions",
            )

    print("RACESORT MERGED-BOX PIPELINE CHECK")
    print(f"Output: {args.output_dir}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("PASS: recovered crops and race numbers match expectations")


if __name__ == "__main__":
    main()
