"""Measure whether low-confidence DETR boxes can split merged motorcycles.

This experiment is read-only with respect to source photographs. It writes
diagnostic overlays and a JSON report to a separate output directory. It does
not change the production detector threshold or the working pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor


MODEL_NAME = "facebook/detr-resnet-50"
BASELINE_THRESHOLD = 0.70
CHILD_THRESHOLDS = (0.30, 0.275, 0.25)
MIN_CHILD_CONTAINMENT = 0.80
MIN_CHILD_AREA_RATIO = 0.12
MAX_CHILD_AREA_RATIO = 0.80
MAX_CHILD_PAIR_IOU = 0.55
MIN_CHILD_AREA_BALANCE = 0.50
MIN_HORIZONTAL_CENTER_SEPARATION = 0.33
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DETR merged-box splitting without changing RaceSort.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test-photos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mega-output/detr-merged-box-experiment"),
    )
    return parser.parse_args()


def area(box):
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def intersection_area(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def containment(child, parent):
    child_area = area(child)
    if child_area == 0:
        return 0.0
    return intersection_area(child, parent) / child_area


def iou(first, second):
    intersection = intersection_area(first, second)
    union = area(first) + area(second) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def horizontal_center(box):
    return (box[0] + box[2]) / 2


def best_child_pair(parent, detections, threshold):
    parent_area = area(parent["box"])
    candidates = []

    for detection in detections:
        if detection is parent or detection["score"] < threshold:
            continue

        area_ratio = area(detection["box"]) / parent_area
        contained = containment(detection["box"], parent["box"])

        if (
            MIN_CHILD_AREA_RATIO <= area_ratio <= MAX_CHILD_AREA_RATIO
            and contained >= MIN_CHILD_CONTAINMENT
        ):
            candidates.append(detection)

    pairs = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            pair_iou = iou(first["box"], second["box"])
            smaller_area = min(area(first["box"]), area(second["box"]))
            larger_area = max(area(first["box"]), area(second["box"]))
            area_balance = smaller_area / larger_area
            parent_width = parent["box"][2] - parent["box"][0]
            center_separation = abs(
                horizontal_center(first["box"])
                - horizontal_center(second["box"])
            ) / parent_width

            if (
                pair_iou <= MAX_CHILD_PAIR_IOU
                and area_balance >= MIN_CHILD_AREA_BALANCE
                and center_separation
                >= MIN_HORIZONTAL_CENTER_SEPARATION
            ):
                pairs.append(
                    (
                        first["score"] + second["score"],
                        pair_iou,
                        area_balance,
                        center_separation,
                        first,
                        second,
                    )
                )

    if not pairs:
        return None

    _, pair_iou, area_balance, center_separation, first, second = max(
        pairs,
        key=lambda item: item[0],
    )
    return {
        "parent": parent,
        "children": [first, second],
        "child_iou": pair_iou,
        "child_area_balance": area_balance,
        "horizontal_center_separation": center_separation,
    }


def find_splits(detections, threshold):
    splits = []
    for parent in detections:
        if parent["score"] < BASELINE_THRESHOLD:
            continue
        split = best_child_pair(parent, detections, threshold)
        if split is not None:
            splits.append(split)
    return splits


def serializable_detection(detection):
    return {
        "score": detection["score"],
        "box": detection["box"],
    }


def serializable_split(split):
    return {
        "parent": serializable_detection(split["parent"]),
        "children": [
            serializable_detection(child)
            for child in split["children"]
        ],
        "child_iou": split["child_iou"],
        "child_area_balance": split["child_area_balance"],
        "horizontal_center_separation": split[
            "horizontal_center_separation"
        ],
    }


def draw_overlay(image, detections, splits, output_path):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for detection in detections:
        if detection["score"] >= BASELINE_THRESHOLD:
            draw.rectangle(detection["box"], outline="white", width=8)
            draw.text(
                (detection["box"][0] + 8, detection["box"][1] + 8),
                f"baseline {detection['score']:.3f}",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )

    colors = ("lime", "cyan")
    for split in splits:
        for color, child in zip(colors, split["children"]):
            draw.rectangle(child["box"], outline=color, width=10)
            draw.text(
                (child["box"][0] + 8, child["box"][1] + 8),
                f"child {child['score']:.3f}",
                fill=color,
                stroke_width=2,
                stroke_fill="black",
            )

    overlay.thumbnail((1800, 1800))
    overlay.save(output_path, quality=92)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print("Loading DETR offline...")
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"Photos: {len(image_paths)}")

    records = []
    started = time.perf_counter()

    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),
            threshold=min(CHILD_THRESHOLDS),
        )[0]

        detections = []
        for score, label, box in zip(
            result["scores"],
            result["labels"],
            result["boxes"],
        ):
            if model.config.id2label[label.item()] != "motorcycle":
                continue
            detections.append(
                {
                    "score": float(score),
                    "box": [float(value) for value in box],
                }
            )

        splits_by_threshold = {
            str(threshold): find_splits(detections, threshold)
            for threshold in CHILD_THRESHOLDS
        }
        record = {
            "photo": image_path.name,
            "baseline_motorcycles": sum(
                detection["score"] >= BASELINE_THRESHOLD
                for detection in detections
            ),
            "candidate_motorcycles": len(detections),
            "splits": {
                threshold: [
                    serializable_split(split)
                    for split in splits
                ]
                for threshold, splits in splits_by_threshold.items()
            },
        }
        records.append(record)

        splits_at_025 = splits_by_threshold[str(0.25)]
        if splits_at_025:
            draw_overlay(
                image,
                detections,
                splits_at_025,
                args.output_dir / f"{image_path.stem}-split.jpg",
            )

        split_marker = " SPLIT" if splits_at_025 else ""
        print(
            f"[{index:03d}/{len(image_paths)}] {image_path.name}: "
            f"baseline={record['baseline_motorcycles']} "
            f"candidates={record['candidate_motorcycles']}"
            f"{split_marker}"
        )

    summary = {}
    for threshold in CHILD_THRESHOLDS:
        key = str(threshold)
        affected = [
            record["photo"]
            for record in records
            if record["splits"][key]
        ]
        summary[key] = {
            "photos_with_proposed_split": len(affected),
            "photos": affected,
        }

    report = {
        "model": MODEL_NAME,
        "policy": {
            "baseline_threshold": BASELINE_THRESHOLD,
            "child_thresholds": CHILD_THRESHOLDS,
            "minimum_child_containment": MIN_CHILD_CONTAINMENT,
            "child_area_ratio_range": [
                MIN_CHILD_AREA_RATIO,
                MAX_CHILD_AREA_RATIO,
            ],
            "maximum_child_pair_iou": MAX_CHILD_PAIR_IOU,
            "minimum_child_area_balance": MIN_CHILD_AREA_BALANCE,
            "minimum_horizontal_center_separation": (
                MIN_HORIZONTAL_CENTER_SEPARATION
            ),
            "description": (
                "Propose replacing one strong parent box only when it contains "
                "two distinct smaller motorcycle boxes."
            ),
        },
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
