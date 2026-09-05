"""Measure Qwen number accuracy and latency at smaller image sizes.

Generated benchmark images are written separately. Source crops and original
photographs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import ollama
import cv2
from PIL import Image


MODEL_NAME = "qwen3-vl:4b-instruct"
SIZES = (1500, 1024, 768)
CASES_PER_GROUP = 5
NUMBER_PATTERN = re.compile(r"[A-Z0-9]{1,6}")

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument(
        "--crop-root",
        type=Path,
        default=Path("resolution-label-output/labels"),
        help="Folder containing the relative crop paths in the label CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mega-output/qwen-resolution-benchmark"),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(SIZES),
        help="Maximum image dimensions to test (default: 1500 1024 768).",
    )
    parser.add_argument(
        "--all-definitive",
        action="store_true",
        help="Test every NUMBER and NONE label instead of a stress subset.",
    )
    return parser.parse_args()


def normalize_number(value):
    if value is None:
        return None

    value = str(value).strip().upper()
    if value == "UNKNOWN" or not NUMBER_PATTERN.fullmatch(value):
        return None

    return value


def numeric_value(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def resolve_crop(row, crop_root):
    crop = Path(row["crop"])
    if not crop.is_absolute():
        crop = crop_root / crop
    return crop.resolve() if crop.exists() else None


def measure_sharpness(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def supports_all_sizes(path):
    """Require 1024 and 768 to create genuinely different inputs."""

    with Image.open(path) as image:
        return max(image.size) > 1024


def select_cases(rows, crop_root, all_definitive=False):
    if all_definitive:
        selected = []
        for row in rows:
            if row.get("answer_type") not in {"NUMBER", "NONE"}:
                continue
            crop = resolve_crop(row, crop_root)
            if crop is None:
                continue
            row["_benchmark_sharpness"] = measure_sharpness(crop)
            group = (
                "NUMBER"
                if row["answer_type"] == "NUMBER"
                else "NO_NUMBER_VISIBLE"
            )
            selected.append((group, row, crop))
        return selected

    clear = [
        row
        for row in rows
        if row.get("number_readability") == "CLEAR"
        and normalize_number(row.get("ground_truth")) is not None
    ]
    blurry = [
        row
        for row in rows
        if row.get("number_readability") == "BLURRY_BUT_READABLE"
        and normalize_number(row.get("ground_truth")) is not None
    ]
    numberless = [
        row
        for row in rows
        if row.get("number_readability") == "NO_NUMBER_VISIBLE"
        and not row.get("ground_truth", "").strip()
        and row.get("answer_type") == "NONE"
    ]

    for row in clear + blurry + numberless:
        crop = resolve_crop(row, crop_root)
        row["_benchmark_sharpness"] = (
            measure_sharpness(crop) if crop is not None else 0.0
        )

    # Lowest-sharpness readable cases stress resolution loss. Highest-
    # sharpness numberless cases stress hallucination on otherwise clean crops.
    clear.sort(key=lambda row: numeric_value(row, "_benchmark_sharpness"))
    blurry.sort(key=lambda row: numeric_value(row, "_benchmark_sharpness"))
    numberless.sort(
        key=lambda row: numeric_value(row, "_benchmark_sharpness"),
        reverse=True,
    )

    selected = []
    for group, candidates in (
        ("CLEAR", clear),
        ("BLURRY_BUT_READABLE", blurry),
        ("NO_NUMBER_VISIBLE", numberless),
    ):
        usable = [
            row
            for row in candidates
            if resolve_crop(row, crop_root) is not None
            and supports_all_sizes(resolve_crop(row, crop_root))
        ]
        for row in usable[:CASES_PER_GROUP]:
            selected.append((group, row, resolve_crop(row, crop_root)))

    return selected


def create_resized_copy(source, destination, max_dimension):
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        image.save(destination, quality=95)


def summarize(records, size):
    size_records = [record for record in records if record["size"] == size]
    positive = [record for record in size_records if record["expected"]]
    numberless = [record for record in size_records if not record["expected"]]
    correct_positive = sum(
        record["observed"] == record["expected"]
        for record in positive
    )
    safe_numberless = sum(record["observed"] is None for record in numberless)
    times = [record["seconds"] for record in size_records]

    return {
        "size": size,
        "cases": len(size_records),
        "positive_exact": correct_positive,
        "positive_cases": len(positive),
        "numberless_unknown": safe_numberless,
        "numberless_cases": len(numberless),
        "total_correct": correct_positive + safe_numberless,
        "accuracy": (
            (correct_positive + safe_numberless) / len(size_records)
            if size_records
            else None
        ),
        "total_seconds": sum(times),
        "median_seconds": statistics.median(times) if times else None,
        "mean_seconds": statistics.mean(times) if times else None,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.validation_csv.open(
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    sizes = tuple(dict.fromkeys(args.sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--sizes must contain positive dimensions")

    cases = select_cases(
        rows,
        args.crop_root,
        all_definitive=args.all_definitive,
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Selected labeled crops: {len(cases)}")
    print(f"Sizes: {sizes}")

    records = []
    total_calls = len(cases) * len(sizes)
    call_number = 0

    # Rotate sizes across passes. Every pass visits all cases, the same crop is
    # never sent consecutively, and no resolution owns all early/late calls.
    for pass_index in range(len(sizes)):
        for case_index, (group, row, source) in enumerate(cases):
            size = sizes[(pass_index + case_index) % len(sizes)]
            expected = (
                normalize_number(row.get("ground_truth"))
                if group != "NO_NUMBER_VISIBLE"
                else None
            )
            call_number += 1
            destination = (
                args.output_dir
                / "resized-crops"
                / str(size)
                / f"{Path(row['photo']).stem}-v{int(row['vehicle']):02d}.jpg"
            )
            create_resized_copy(source, destination, size)

            started = time.perf_counter()
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": DIRECT_NUMBER_PROMPT,
                        "images": [str(destination)],
                    }
                ],
            )
            elapsed = time.perf_counter() - started
            raw = response["message"]["content"].strip()
            observed = normalize_number(raw)

            records.append(
                {
                    "group": group,
                    "photo": row["photo"],
                    "vehicle": row["vehicle"],
                    "source_crop": row["crop"],
                    "sharpness": numeric_value(
                        row,
                        "_benchmark_sharpness",
                    ),
                    "size": size,
                    "expected": expected,
                    "observed": observed,
                    "raw_response": raw,
                    "correct": observed == expected,
                    "seconds": elapsed,
                }
            )
            print(
                f"[{call_number:02d}/{total_calls}] {row['photo']} "
                f"v{row['vehicle']} {group} size={size}: "
                f"expected={expected or 'UNKNOWN'} "
                f"observed={observed or 'UNKNOWN'} {elapsed:.2f}s"
            )

    summaries = [summarize(records, size) for size in sizes]
    report = {
        "model": MODEL_NAME,
        "source_validation_csv": str(args.validation_csv.resolve()),
        "selection": {
            "cases_per_group": CASES_PER_GROUP,
            "all_definitive": args.all_definitive,
            "groups": dict(
                sorted(
                    defaultdict(int, {
                        group: sum(case[0] == group for case in cases)
                        for group in {case[0] for case in cases}
                    }).items()
                )
            ),
            "strategy": (
                "lowest-sharpness readable crops and highest-sharpness "
                "human-confirmed numberless crops"
            ),
        },
        "summaries": summaries,
        "records": records,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    for summary in summaries:
        print(
            f"size={summary['size']} accuracy={summary['accuracy']:.1%} "
            f"positive={summary['positive_exact']}/{summary['positive_cases']} "
            f"numberless={summary['numberless_unknown']}/"
            f"{summary['numberless_cases']} median={summary['median_seconds']:.2f}s"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
