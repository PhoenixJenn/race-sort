"""Check the saved DETR merged-box experiment without rerunning models."""

import argparse
import json
from pathlib import Path
import sys


EXPECTED_SPLIT_PHOTOS = {
    "GGBM0021.JPG": 4,
    "GGBM0082.JPG": 2,
}
CHECKED_THRESHOLD = "0.275"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=Path(
            "mega-output/detr-merged-box-experiment-v2/report.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with args.report.open(encoding="utf-8") as report_file:
        report = json.load(report_file)

    records = {
        record["photo"]: record
        for record in report["photos"]
    } if "photos" in report else {
        record["photo"]: record
        for record in report["records"]
    }

    failures = []

    for photo, expected_resolved_count in EXPECTED_SPLIT_PHOTOS.items():
        record = records.get(photo)

        if record is None:
            failures.append(f"missing report record: {photo}")
            continue

        splits = record["splits"].get(CHECKED_THRESHOLD, [])
        resolved_count = record["baseline_motorcycles"] + len(splits)

        if len(splits) != 1:
            failures.append(
                f"{photo}: expected one merged-box split, got {len(splits)}"
            )

        if resolved_count != expected_resolved_count:
            failures.append(
                f"{photo}: expected {expected_resolved_count} resolved "
                f"motorcycles, got {resolved_count}"
            )

    print("RACESORT MERGED-BOX REGRESSION CHECK")
    print(f"Report: {args.report}")
    print(f"Threshold: {CHECKED_THRESHOLD}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("PASS: known merged detections resolve correctly")


if __name__ == "__main__":
    main()
