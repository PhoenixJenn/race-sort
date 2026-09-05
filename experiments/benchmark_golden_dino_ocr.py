"""Benchmark golden-record DINO matching plus OCR agreement.

This is a read-only routing experiment. It does not modify source photographs
or the working RaceSort pipeline, and it never calls Qwen.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-small"
GOLDEN_MIN_SHARPNESS = 500.0
THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.92, 0.95)
VALID_ROLES = {"PRIMARY", "SECONDARY"}
FILTERED_DECISIONS = {
    "FILTERED_NON_PRIMARY",
    "FILTERED_TOO_BLURRY",
}
NUMBER_PATTERN = re.compile(r"[A-Z0-9]{1,6}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure fresh-batch DINO/OCR routes without Qwen.",
    )
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mega-output/golden-dino-ocr-benchmark.json"),
    )
    return parser.parse_args()


def normalize_number(value):
    """Preserve valid race-number identifiers as strings."""

    if value is None:
        return None

    normalized = str(value).strip().upper()
    if (
        not normalized
        or normalized == "UNSURE"
        or not NUMBER_PATTERN.fullmatch(normalized)
    ):
        return None

    return normalized


def parse_ocr_candidates(value):
    candidates = []

    for item in str(value or "").split(","):
        candidate = normalize_number(item)
        if candidate is not None:
            candidates.append(candidate)

    return list(dict.fromkeys(candidates))


def numeric_value(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def is_golden_candidate(row):
    return (
        row.get("human_status") == "CORRECT"
        and row.get("number_readability") == "CLEAR"
        and row.get("human_quality") == "SELLABLE"
        and row.get("human_vehicle_role") in VALID_ROLES
        and normalize_number(row.get("ground_truth")) is not None
        and numeric_value(row, "sharpness") is not None
        and numeric_value(row, "sharpness") >= GOLDEN_MIN_SHARPNESS
    )


def choose_golden_records(rows):
    best_by_number = {}

    for row in rows:
        if not is_golden_candidate(row):
            continue

        race_number = normalize_number(row["ground_truth"])
        current = best_by_number.get(race_number)

        if (
            current is None
            or numeric_value(row, "sharpness")
            > numeric_value(current, "sharpness")
        ):
            best_by_number[race_number] = row

    return best_by_number


def resolve_crop(row, validation_csv):
    crop = Path(row["crop"])
    candidates = [crop]

    if not crop.is_absolute():
        candidates.extend(
            [
                Path.cwd() / crop,
                validation_csv.parent / crop,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_embedding(path, processor, model, device, cache):
    key = str(path)

    if key in cache:
        return cache[key]

    with Image.open(path) as image:
        inputs = processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = F.normalize(
        outputs.last_hidden_state[:, 0, :],
        p=2,
        dim=1,
    ).cpu()
    cache[key] = embedding
    return embedding


def evaluation_label(row):
    """Return a trusted identifier, explicit NONE, or unknown."""

    ground_truth = normalize_number(row.get("ground_truth"))
    if ground_truth is not None:
        return ground_truth

    if (
        row.get("number_readability") == "NO_NUMBER_VISIBLE"
        and row.get("human_status") in {"CORRECT", "WRONG"}
    ):
        return "NONE"

    return None


def threshold_summary(records, threshold, require_single_ocr):
    proposed = [
        record
        for record in records
        if record["top_similarity"] >= threshold
        and record["ocr_agrees_with_top"]
        and (
            not require_single_ocr
            or len(record["ocr_candidates"]) == 1
        )
    ]
    evaluated = [
        record
        for record in proposed
        if record["evaluation_label"] is not None
    ]
    correct = [
        record
        for record in evaluated
        if record["top_identity"] == record["evaluation_label"]
    ]
    incorrect = [
        record
        for record in evaluated
        if record["top_identity"] != record["evaluation_label"]
    ]

    return {
        "ocr_policy": (
            "SINGLE_CANDIDATE"
            if require_single_ocr
            else "ANY_AGREEING_CANDIDATE"
        ),
        "threshold": threshold,
        "proposed_qwen_skips": len(proposed),
        "proposed_skip_rate": (
            len(proposed) / len(records)
            if records
            else 0.0
        ),
        "human_evaluated_proposals": len(evaluated),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "precision": (
            len(correct) / len(evaluated)
            if evaluated
            else None
        ),
        "errors": [
            {
                "photo": record["photo"],
                "crop": record["crop"],
                "expected": record["evaluation_label"],
                "predicted": record["top_identity"],
                "similarity": record["top_similarity"],
                "ocr_candidates": record["ocr_candidates"],
            }
            for record in incorrect
        ],
    }


def main():
    args = parse_args()
    started = time.perf_counter()

    with args.validation_csv.open(
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    golden_by_number = choose_golden_records(rows)
    golden_paths = {
        number: resolve_crop(row, args.validation_csv)
        for number, row in golden_by_number.items()
    }
    missing_golden = [
        number
        for number, path in golden_paths.items()
        if path is None
    ]

    if missing_golden:
        raise FileNotFoundError(
            "Missing golden crops for: " + ", ".join(missing_golden)
        )

    golden_path_set = set(golden_paths.values())
    query_rows = []
    missing_query_crops = []

    for row in rows:
        if row.get("current_decision") in FILTERED_DECISIONS:
            continue

        crop_path = resolve_crop(row, args.validation_csv)
        if crop_path is None:
            missing_query_crops.append(row.get("crop", ""))
            continue

        if crop_path in golden_path_set:
            continue

        query_rows.append((row, crop_path))

    device = resolve_device()
    print(f"Device: {device}")
    print(f"Golden identities: {len(golden_paths)}")
    print(f"Non-golden processable queries: {len(query_rows)}")
    print("Loading DINOv2...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    embedding_cache = {}
    golden_embeddings = {
        number: create_embedding(
            path,
            processor,
            model,
            device,
            embedding_cache,
        )
        for number, path in golden_paths.items()
    }

    records = []

    for index, (row, crop_path) in enumerate(query_rows, start=1):
        query_embedding = create_embedding(
            crop_path,
            processor,
            model,
            device,
            embedding_cache,
        )
        matches = sorted(
            (
                (
                    float(
                        F.cosine_similarity(
                            query_embedding,
                            reference_embedding,
                        ).item()
                    ),
                    number,
                )
                for number, reference_embedding in golden_embeddings.items()
            ),
            reverse=True,
        )
        top_similarity, top_identity = matches[0]
        second_similarity = matches[1][0] if len(matches) > 1 else None
        ocr_candidates = parse_ocr_candidates(row.get("ocr_candidates"))

        records.append(
            {
                "photo": row.get("photo", ""),
                "crop": row.get("crop", ""),
                "vehicle": row.get("vehicle", ""),
                "evaluation_label": evaluation_label(row),
                "top_identity": top_identity,
                "top_similarity": top_similarity,
                "second_similarity": second_similarity,
                "similarity_margin": (
                    top_similarity - second_similarity
                    if second_similarity is not None
                    else None
                ),
                "ocr_candidates": ocr_candidates,
                "ocr_agrees_with_top": top_identity in ocr_candidates,
            }
        )

        if index % 25 == 0 or index == len(query_rows):
            print(f"Embedded {index}/{len(query_rows)} queries")

    summaries = [
        threshold_summary(records, threshold, require_single_ocr)
        for require_single_ocr in (False, True)
        for threshold in THRESHOLDS
    ]
    label_counts = Counter(
        "POSITIVE"
        if record["evaluation_label"] not in {None, "NONE"}
        else "EXPLICIT_NONE"
        if record["evaluation_label"] == "NONE"
        else "UNEVALUATED"
        for record in records
    )

    report = {
        "model": MODEL_NAME,
        "device": str(device),
        "source_validation_csv": str(args.validation_csv.resolve()),
        "policy": {
            "golden_definition": (
                "sharpest human-CORRECT, CLEAR, SELLABLE, PRIMARY/SECONDARY "
                "crop with sharpness >= 500 per race-number string"
            ),
            "query_definition": (
                "non-golden crop not already filtered by the working pipeline"
            ),
            "proposed_skip": (
                "top DINO similarity meets threshold and OCR independently "
                "contains the same race-number string"
            ),
        },
        "counts": {
            "validation_rows": len(rows),
            "golden_identities": len(golden_paths),
            "query_crops": len(records),
            "evaluation_labels": dict(sorted(label_counts.items())),
            "missing_query_crops": len(missing_query_crops),
        },
        "threshold_summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    for summary in summaries:
        precision = summary["precision"]
        precision_text = (
            f"{precision:.1%}" if precision is not None else "n/a"
        )
        print(
            f"ocr={summary['ocr_policy']} "
            f"threshold={summary['threshold']:.2f} "
            f"skips={summary['proposed_qwen_skips']} "
            f"evaluated={summary['human_evaluated_proposals']} "
            f"errors={summary['incorrect']} "
            f"precision={precision_text}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
