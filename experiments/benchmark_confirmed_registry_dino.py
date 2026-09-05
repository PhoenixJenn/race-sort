"""Measure DINO matching against a human-confirmed event registry.

This read-only experiment chooses one golden crop per race-number string and
tests later sightings against every registry identity. It never modifies
source photographs and never calls Qwen.
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

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-small"
NUMBER_PATTERN = re.compile(r"[A-Z0-9]{1,6}")
THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.92, 0.95)
READABILITY_PRIORITY = {
    "CLEAR": 2,
    "BLURRY_BUT_READABLE": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test later sightings against confirmed golden records.",
    )
    parser.add_argument(
        "validation_csv",
        type=Path,
        nargs="+",
        help="One or more human-validation CSV files.",
    )
    parser.add_argument(
        "--crop-root",
        type=Path,
        default=Path("resolution-label-output/labels"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "resolution-label-output/confirmed-registry-dino.json"
        ),
    )
    return parser.parse_args()


def normalize_number(value):
    """Return an opaque uppercase identifier, preserving 0 and zeros."""

    normalized = str(value or "").strip().upper()
    if not NUMBER_PATTERN.fullmatch(normalized):
        return None
    return normalized


def resolve_crop(row, crop_root):
    crop = Path(row["crop"])
    candidates = [crop]
    if not crop.is_absolute():
        candidates.append(crop_root / crop)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def is_confirmed_number(row):
    """Accept explicit new labels or readable, reviewed legacy labels."""

    if row.get("answer_type"):
        return row["answer_type"] == "NUMBER"

    return (
        row.get("human_status") in {"CORRECT", "WRONG"}
        and row.get("number_readability")
        in {"CLEAR", "BLURRY_BUT_READABLE"}
    )


def measure_sharpness(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def select_registry_and_queries(rows, crop_root):
    labeled = []
    for row in rows:
        if not is_confirmed_number(row):
            continue
        number = normalize_number(row.get("ground_truth"))
        crop = resolve_crop(row, crop_root)
        if number is None or crop is None:
            continue
        labeled.append(
            {
                "row": row,
                "number": number,
                "crop": crop,
                "sharpness": measure_sharpness(crop),
            }
        )

    golden_by_number = {}
    for item in labeled:
        current = golden_by_number.get(item["number"])
        item_score = (
            READABILITY_PRIORITY.get(
                item["row"].get("number_readability", ""),
                0,
            ),
            item["sharpness"],
        )
        current_score = (
            READABILITY_PRIORITY.get(
                current["row"].get("number_readability", ""),
                0,
            ),
            current["sharpness"],
        ) if current else None
        if current is None or item_score > current_score:
            golden_by_number[item["number"]] = item

    golden_crops = {item["crop"] for item in golden_by_number.values()}
    queries = [item for item in labeled if item["crop"] not in golden_crops]
    return golden_by_number, queries


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


def summarize_threshold(records, threshold):
    accepted = [r for r in records if r["top_similarity"] >= threshold]
    correct = [r for r in accepted if r["top_identity"] == r["expected"]]
    errors = [r for r in accepted if r["top_identity"] != r["expected"]]
    return {
        "threshold": threshold,
        "accepted": len(accepted),
        "coverage": len(accepted) / len(records) if records else 0.0,
        "correct": len(correct),
        "incorrect": len(errors),
        "precision": len(correct) / len(accepted) if accepted else None,
        "errors": [
            {
                "photo": r["photo"],
                "vehicle": r["vehicle"],
                "expected": r["expected"],
                "predicted": r["top_identity"],
                "similarity": r["top_similarity"],
            }
            for r in errors
        ],
    }


def main():
    args = parse_args()
    started = time.perf_counter()
    rows = []
    for validation_csv in args.validation_csv:
        with validation_csv.open(
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            for row in csv.DictReader(csv_file):
                row["_source_validation_csv"] = str(
                    validation_csv.resolve()
                )
                rows.append(row)

    golden_by_number, queries = select_registry_and_queries(
        rows,
        args.crop_root,
    )
    device = resolve_device()
    print(f"Device: {device}")
    print(f"Confirmed registry identities: {len(golden_by_number)}")
    print(f"Later-sighting queries: {len(queries)}")
    print("Loading DINOv2 from the local model cache...")

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

    cache = {}
    golden_embeddings = {
        number: create_embedding(
            item["crop"],
            processor,
            model,
            device,
            cache,
        )
        for number, item in golden_by_number.items()
    }

    records = []
    for index, query in enumerate(queries, start=1):
        query_embedding = create_embedding(
            query["crop"],
            processor,
            model,
            device,
            cache,
        )
        matches = sorted(
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
        )
        matches.reverse()
        expected_rank = next(
            rank
            for rank, (_, number) in enumerate(matches, start=1)
            if number == query["number"]
        )
        top_similarity, top_identity = matches[0]
        second_similarity = matches[1][0] if len(matches) > 1 else None
        row = query["row"]
        records.append(
            {
                "photo": row["photo"],
                "vehicle": row["vehicle"],
                "crop": row["crop"],
                "expected": query["number"],
                "top_identity": top_identity,
                "top_similarity": top_similarity,
                "second_similarity": second_similarity,
                "similarity_margin": (
                    top_similarity - second_similarity
                    if second_similarity is not None
                    else None
                ),
                "expected_rank": expected_rank,
                "top_five": [
                    {"identity": number, "similarity": similarity}
                    for similarity, number in matches[:5]
                ],
            }
        )
        print(
            f"[{index:02d}/{len(queries)}] {row['photo']} v{row['vehicle']} "
            f"expected={query['number']} top={top_identity} "
            f"similarity={top_similarity:.4f} rank={expected_rank}"
        )

    top_one_correct = sum(r["expected_rank"] == 1 for r in records)
    top_three_correct = sum(r["expected_rank"] <= 3 for r in records)
    threshold_summaries = [
        summarize_threshold(records, threshold) for threshold in THRESHOLDS
    ]
    report = {
        "model": MODEL_NAME,
        "device": str(device),
        "source_validation_csvs": [
            str(path.resolve()) for path in args.validation_csv
        ],
        "policy": {
            "golden_record": (
                "one human-confirmed NUMBER crop per identifier; prefer CLEAR "
                "readability, then measured sharpness"
            ),
            "query": "every additional human-labeled sighting",
            "self_match_allowed": False,
            "identifier_type": "string",
        },
        "counts": {
            "validation_rows": len(rows),
            "registry_identities": len(golden_by_number),
            "query_crops": len(records),
            "query_identities": len(Counter(r["expected"] for r in records)),
        },
        "ranking": {
            "top_one_correct": top_one_correct,
            "top_one_accuracy": (
                top_one_correct / len(records) if records else None
            ),
            "top_three_correct": top_three_correct,
            "top_three_accuracy": (
                top_three_correct / len(records) if records else None
            ),
        },
        "threshold_summaries": threshold_summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "golden_records": {
            number: {
                "photo": item["row"]["photo"],
                "vehicle": item["row"]["vehicle"],
                "crop": item["row"]["crop"],
                "readability": item["row"].get("number_readability", ""),
                "sharpness": item["sharpness"],
            }
            for number, item in sorted(golden_by_number.items())
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    print(f"Top-1: {top_one_correct}/{len(records)}")
    print(f"Top-3: {top_three_correct}/{len(records)}")
    for summary in threshold_summaries:
        precision = summary["precision"]
        precision_text = f"{precision:.1%}" if precision is not None else "n/a"
        print(
            f"threshold={summary['threshold']:.2f} "
            f"accepted={summary['accepted']} "
            f"errors={summary['incorrect']} precision={precision_text}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
