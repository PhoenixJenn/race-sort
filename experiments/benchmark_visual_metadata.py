"""Extract number-blind motorcycle/rider metadata for matching experiments.

The human race-number label is used only to score later analysis and is never
included in the model prompt. Source crops and original photographs are never
modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import ollama
from PIL import Image


MODEL_NAME = "qwen3-vl:4b-instruct"
MAX_DIMENSION = 1024
NUMBER_PATTERN = re.compile(r"[A-Z0-9]{1,6}")

METADATA_PROMPT = """
Analyze the visible motorcycle and rider for visual matching.

Do not output or transcribe any race number. Visible text and logos may help
identify a manufacturer, but do not include digits or race-number characters
anywhere in the response.

Return valid JSON only, using exactly this structure:

{
  "motorcycle": {
    "make": null,
    "primary_colors": [],
    "secondary_colors": [],
    "distinctive_patterns": []
  },
  "rider": {
    "leathers_colors": [],
    "leathers_patterns": [],
    "helmet_colors": [],
    "helmet_patterns": []
  },
  "number_plate": {
    "background_color": null
  },
  "view": {
    "angle": "unknown",
    "occlusion": "none"
  }
}

Rules:

- Report every obvious color and pattern clearly supported by the image.
- Do not leave color lists empty when the corresponding motorcycle, leathers,
  or helmet is visibly colored.
- Use short, ordinary color names.
- motorcycle.make must be null unless a manufacturer badge or name is visible.
- view.angle must be one of: front, front_left, left, rear_left, rear,
  rear_right, right, front_right, unknown.
- view.occlusion must be one of: none, partial, heavy.
- Use null or an empty list when uncertain. Never guess.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument(
        "--crop-root",
        type=Path,
        default=Path("resolution-label-output/labels"),
    )
    parser.add_argument(
        "--numbers",
        nargs="*",
        help="Optional race-number strings to include.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resolution-label-output/visual-metadata"),
    )
    return parser.parse_args()


def normalize_number(value):
    normalized = str(value or "").strip().upper()
    return normalized if NUMBER_PATTERN.fullmatch(normalized) else None


def resolve_crop(row, crop_root):
    crop = Path(row["crop"])
    candidates = [crop]
    if not crop.is_absolute():
        candidates.append(crop_root / crop)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def prepare_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
        image.save(destination, quality=95)


def response_cache_key(source):
    digest = hashlib.sha256()
    digest.update(MODEL_NAME.encode())
    digest.update(METADATA_PROMPT.encode())
    with source.open("rb") as crop_file:
        for chunk in iter(lambda: crop_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_profile(source, prepared, cache_dir):
    cache_path = cache_dir / f"{response_cache_key(source)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), 0.0, True

    started = time.perf_counter()
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": METADATA_PROMPT,
                "images": [str(prepared)],
            }
        ],
        format="json",
    )
    elapsed = time.perf_counter() - started
    raw = response["message"]["content"].strip()
    profile = json.loads(raw)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    temporary.replace(cache_path)
    return profile, elapsed, False


def main():
    args = parse_args()
    with args.validation_csv.open(
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    requested_numbers = {
        normalize_number(number) for number in (args.numbers or [])
    }
    requested_numbers.discard(None)
    cases = []
    for row in rows:
        if row.get("answer_type") != "NUMBER":
            continue
        number = normalize_number(row.get("ground_truth"))
        if number is None:
            continue
        if requested_numbers and number not in requested_numbers:
            continue
        crop = resolve_crop(row, args.crop_root)
        if crop is not None:
            cases.append((row, number, crop))
    if args.limit is not None:
        cases = cases[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (row, number, crop) in enumerate(cases, start=1):
        prepared = (
            args.output_dir
            / "prepared"
            / f"{Path(row['photo']).stem}-v{int(row['vehicle']):02d}.jpg"
        )
        prepare_image(crop, prepared)
        try:
            profile, elapsed, cache_hit = load_or_create_profile(
                crop,
                prepared,
                args.output_dir / "cache",
            )
            error = None
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            profile = None
            elapsed = 0.0
            cache_hit = False
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "photo": row["photo"],
                "vehicle": row["vehicle"],
                "crop": row["crop"],
                "human_number": number,
                "profile": profile,
                "seconds": elapsed,
                "cache_hit": cache_hit,
                "error": error,
            }
        )
        print(
            f"[{index:02d}/{len(cases)}] {row['photo']} v{row['vehicle']} "
            f"human={number} {elapsed:.2f}s "
            f"{'CACHE' if cache_hit else error or 'OK'}"
        )

    report = {
        "model": MODEL_NAME,
        "maximum_dimension": MAX_DIMENSION,
        "source_validation_csv": str(args.validation_csv.resolve()),
        "policy": {
            "number_blind_prompt": True,
            "human_number_used_in_prompt": False,
            "source_images_modified": False,
        },
        "counts": {
            "cases": len(records),
            "numbers": dict(
                sorted(Counter(r["human_number"] for r in records).items())
            ),
            "api_errors": sum(r["error"] is not None for r in records),
            "cache_hits": sum(r["cache_hit"] for r in records),
        },
        "total_model_seconds": sum(r["seconds"] for r in records),
        "records": records,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(f"Cases: {len(records)}")
    print(f"Numbers: {report['counts']['numbers']}")
    print(f"API errors: {report['counts']['api_errors']}")
    print(f"Model time: {report['total_model_seconds']:.2f}s")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
