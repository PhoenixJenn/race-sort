"""Create non-destructive DETR crops for the resolution benchmark labeler."""

import argparse
import json
from pathlib import Path

from PIL import Image


BASELINE_THRESHOLD = 0.70
SPLIT_THRESHOLD = "0.275"
MAX_CROP_SIZE = 1500
MAX_PHOTO_PROXY_SIZE = 1400


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("detection_report", type=Path)
    parser.add_argument("photo_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def box_key(box):
    return tuple(round(float(value), 4) for value in box)


def resolve_boxes(record):
    detections = record["detections"]
    detection_lookup = {
        box_key(detection["box"]): detection
        for detection in detections
    }
    baseline = [
        detection
        for detection in detections
        if detection["score"] >= BASELINE_THRESHOLD
    ]
    replacements = {}
    used_children = set()

    for split in sorted(
        record["splits"].get(SPLIT_THRESHOLD, []),
        key=lambda item: item["parent"]["score"],
        reverse=True,
    ):
        parent_key = box_key(split["parent"]["box"])
        child_keys = [box_key(child["box"]) for child in split["children"]]

        if any(key in used_children for key in child_keys):
            continue

        replacements[parent_key] = child_keys
        used_children.update(child_keys)

    resolved = []
    resolved_keys = set()

    for detection in baseline:
        detection_key = box_key(detection["box"])
        child_keys = replacements.get(detection_key)

        if child_keys is None:
            if detection_key not in resolved_keys:
                resolved.append(
                    {**detection, "detection_source": "baseline"}
                )
                resolved_keys.add(detection_key)
            continue

        for child_key in child_keys:
            if child_key not in resolved_keys:
                resolved.append(
                    {
                        **detection_lookup[child_key],
                        "detection_source": "merged_box_child",
                    }
                )
                resolved_keys.add(child_key)

    return sorted(resolved, key=lambda item: item["box"][0])


def main():
    args = parse_args()
    report = json.loads(args.detection_report.read_text())
    crops_dir = args.output_dir / "crops"
    photos_dir = args.output_dir / "photos"
    crops_dir.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for photo_record in report["records"]:
        photo_path = args.photo_dir / photo_record["photo"]

        with Image.open(photo_path) as source:
            source = source.convert("RGB")
            photo_proxy = source.copy()
            photo_proxy.thumbnail(
                (MAX_PHOTO_PROXY_SIZE, MAX_PHOTO_PROXY_SIZE)
            )
            proxy_name = f"{photo_path.stem}.jpg"
            photo_proxy.save(photos_dir / proxy_name, quality=90)

            for vehicle_index, detection in enumerate(
                resolve_boxes(photo_record),
                start=1,
            ):
                crop = source.crop(
                    tuple(int(value) for value in detection["box"])
                )
                original_crop_size = crop.size
                crop.thumbnail((MAX_CROP_SIZE, MAX_CROP_SIZE))
                crop_name = f"{photo_path.stem}-v{vehicle_index:02d}.jpg"
                crop.save(crops_dir / crop_name, quality=95)
                records.append(
                    {
                        "photo": photo_record["photo"],
                        "vehicle": vehicle_index,
                        "crop": f"crops/{crop_name}",
                        "photo_proxy": f"photos/{proxy_name}",
                        "detr_confidence": detection["score"],
                        "detection_source": detection["detection_source"],
                        "box": detection["box"],
                        "original_crop_size": list(original_crop_size),
                        "benchmark_crop_size": list(crop.size),
                    }
                )

    manifest = {
        "source_detection_report": str(args.detection_report.resolve()),
        "source_photo_directory": str(args.photo_dir.resolve()),
        "count": len(records),
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Photos: {len(report['records'])}")
    print(f"Crops: {len(records)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
