"""Build review buckets and golden references from human validation.

This is an experiment, not production routing. It never changes photographs.
Race numbers remain opaque strings throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


GOLDEN_MIN_SHARPNESS = 500.0
VALID_ROLES = {"PRIMARY", "SECONDARY"}
NO_RIDER_PATTERN = re.compile(
    r"\b(no[ -]?rider|without (?:a )?rider|empty bike)\b",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create RaceSort review buckets from a validation CSV.",
    )
    parser.add_argument(
        "validation_csv",
        type=Path,
        help="Exported racesort-human-validation-v3.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mega-output"),
        help="Directory for manifests (default: mega-output)",
    )
    return parser.parse_args()


def number_is_labeled(value):
    """Return True for a human-provided identifier, including '0'."""

    return value not in {None, "", "UNSURE"}


def sharpness(row):
    try:
        return float(row["sharpness"])
    except (KeyError, TypeError, ValueError):
        return None


def highly_unlikely_reasons(row):
    reasons = []

    if row.get("human_quality") == "TOO_BLURRY":
        reasons.append("TOO_BLURRY")

    if row.get("human_vehicle_role") == "NON_PRIMARY":
        reasons.append("DO_NOT_TAG")

    if NO_RIDER_PATTERN.search(row.get("note", "")):
        reasons.append("NO_RIDER")

    return reasons


def is_high_confidence(row):
    crop_sharpness = sharpness(row)

    return (
        row.get("human_status") == "CORRECT"
        and row.get("number_readability") == "CLEAR"
        and row.get("human_quality") == "SELLABLE"
        and row.get("human_vehicle_role") in VALID_ROLES
        and number_is_labeled(row.get("ground_truth"))
        and crop_sharpness is not None
        and crop_sharpness >= GOLDEN_MIN_SHARPNESS
    )


def classify(row):
    unlikely_reasons = highly_unlikely_reasons(row)

    if unlikely_reasons:
        return "HIGHLY_UNLIKELY", unlikely_reasons

    if is_high_confidence(row):
        return "HIGH_CONFIDENCE", ["HUMAN_VALIDATED_CLEAR_REFERENCE"]

    return "LOWER_CONFIDENCE", ["NEEDS_MORE_EVIDENCE"]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.validation_csv.open(
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    records = []

    for row in rows:
        bucket, reasons = classify(row)
        records.append(
            {
                "photo": row.get("photo", ""),
                "crop": row.get("crop", ""),
                "vehicle": row.get("vehicle", ""),
                "race_number": row.get("ground_truth", ""),
                "review_bucket": bucket,
                "bucket_reasons": reasons,
                "sharpness": sharpness(row),
                "human_status": row.get("human_status", ""),
                "number_readability": row.get("number_readability", ""),
                "human_vehicle_role": row.get("human_vehicle_role", ""),
                "human_quality": row.get("human_quality", ""),
                "current_number": row.get("current_number", ""),
                "current_decision": row.get("current_decision", ""),
                "note": row.get("note", ""),
                "golden_record": False,
            }
        )

    best_by_number = {}

    for record in records:
        if record["review_bucket"] != "HIGH_CONFIDENCE":
            continue

        race_number = record["race_number"]
        current_best = best_by_number.get(race_number)

        if (
            current_best is None
            or record["sharpness"] > current_best["sharpness"]
        ):
            best_by_number[race_number] = record

    for record in best_by_number.values():
        record["golden_record"] = True

    bucket_counts = Counter(
        record["review_bucket"]
        for record in records
    )
    reason_counts = Counter(
        reason
        for record in records
        for reason in record["bucket_reasons"]
    )

    manifest = {
        "source_validation_csv": str(args.validation_csv.resolve()),
        "policy": {
            "highly_unlikely": [
                "human_quality == TOO_BLURRY",
                "human_vehicle_role == NON_PRIMARY",
                "human note explicitly says no rider",
            ],
            "high_confidence": [
                "human_status == CORRECT",
                "number_readability == CLEAR",
                "human_quality == SELLABLE",
                "human_vehicle_role in PRIMARY or SECONDARY",
                "ground_truth is a non-empty string identifier",
                f"sharpness >= {GOLDEN_MIN_SHARPNESS}",
            ],
            "lower_confidence": "everything not assigned above",
            "golden_record": (
                "sharpest HIGH_CONFIDENCE crop per race-number string"
            ),
        },
        "counts": {
            "records": len(records),
            "buckets": dict(sorted(bucket_counts.items())),
            "reasons": dict(sorted(reason_counts.items())),
            "golden_records": len(best_by_number),
        },
        "records": records,
    }

    manifest_path = args.output_dir / "review-buckets.json"
    golden_path = args.output_dir / "golden-records.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    golden_manifest = {
        "source": str(manifest_path),
        "golden_record_definition": manifest["policy"]["golden_record"],
        "count": len(best_by_number),
        "records": sorted(
            best_by_number.values(),
            key=lambda record: record["race_number"],
        ),
    }
    golden_path.write_text(
        json.dumps(golden_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Records: {len(records)}")
    print(f"Buckets: {dict(sorted(bucket_counts.items()))}")
    print(f"Golden records: {len(best_by_number)}")
    print(f"Review manifest: {manifest_path}")
    print(f"Golden manifest: {golden_path}")


if __name__ == "__main__":
    main()
