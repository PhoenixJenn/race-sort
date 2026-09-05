"""Benchmark a cheap pre-recognition keep/reject vision gate.

This experiment reads crops only. It never changes source photographs or
pipeline outputs. Ambiguous model responses are treated as KEEP so the gate
fails safely.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import ollama


DEFAULT_MODEL = "qwen3-vl:4b-instruct"

GATE_PROMPT = """
Decide whether this detected race-motorcycle crop is worth keeping for
race-number recognition and rider/vehicle matching.

Return exactly one label:

KEEP
REJECT

Return REJECT only when at least one condition is visually clear:

- The motorcycle or rider is too blurry or out of focus for a customer-quality
  photograph, even if you can recognize that it is a motorcycle.
- A significant part of the motorcycle or rider is hidden behind another
  vehicle or person.
- Too little of the motorcycle or rider is inside the crop for the photograph
  to be sellable or useful for tagging.
- The motorcycle is a tiny or background subject rather than a photographable
  subject.
- No rider is present or meaningfully visible. This event sells rider photos,
  so a riderless or nearly riderless motorcycle crop is not useful.

Important rules:

- A missing, hidden, blank, or unreadable race number is NOT by itself a
  reason to reject. That crop may still be useful for later visual matching.
- Motion in the wheels or background is NOT a reason to reject when the main
  motorcycle and rider remain complete, useful, and reasonably sharp.
- If uncertain, return KEEP.
- Do not identify the race number.
- Do not return an explanation.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the RaceSort pre-recognition review gate.",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to review-buckets.json",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=27,
        help="Maximum examples per human class (default: 27)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mega-output/review-gate-benchmark.json"),
    )
    parser.add_argument(
        "--include-already-filtered",
        action="store_true",
        help="Also test crops already removed by deterministic filters.",
    )
    return parser.parse_args()


def select_balanced_unlikely(records, count):
    """Round-robin across exclusion reasons for a less biased sample."""

    groups = defaultdict(deque)

    for record in records:
        primary_reason = record["bucket_reasons"][0]
        groups[primary_reason].append(record)

    chosen = []
    group_names = sorted(groups)

    while len(chosen) < count and any(groups.values()):
        for name in group_names:
            if groups[name] and len(chosen) < count:
                chosen.append(groups[name].popleft())

    return chosen


def normalize_gate_response(raw):
    value = raw.strip().upper()

    if value == "REJECT":
        return "REJECT"

    # KEEP is the deliberate fail-safe for ambiguity or extra text.
    return "KEEP"


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]

    golden = [
        record
        for record in records
        if record["golden_record"]
    ][: args.per_class]
    unlikely_pool = [
        record
        for record in records
        if record["review_bucket"] == "HIGHLY_UNLIKELY"
        and (
            args.include_already_filtered
            or not record["current_decision"].startswith("FILTERED_")
        )
    ]
    unlikely = select_balanced_unlikely(
        unlikely_pool,
        min(args.per_class, len(golden)),
    )
    cases = [
        (record, "KEEP")
        for record in golden
    ] + [
        (record, "REJECT")
        for record in unlikely
    ]

    results = []
    started = time.perf_counter()

    print(f"Model: {args.model}")
    print(f"Golden KEEP cases: {len(golden)}")
    print(f"Highly Unlikely REJECT cases: {len(unlikely)}")
    print()

    for index, (record, expected) in enumerate(cases, start=1):
        crop = Path(record["crop"])
        call_started = time.perf_counter()

        try:
            response = ollama.chat(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": GATE_PROMPT,
                        "images": [str(crop)],
                    }
                ],
                options={"temperature": 0},
            )
            raw = response["message"]["content"]
            predicted = normalize_gate_response(raw)
            error = None
        except Exception as exc:  # Preserve the case and fail safely.
            raw = ""
            predicted = "KEEP"
            error = str(exc)

        seconds = time.perf_counter() - call_started
        correct = predicted == expected
        results.append(
            {
                "crop": record["crop"],
                "race_number": record["race_number"],
                "human_bucket": record["review_bucket"],
                "human_reasons": record["bucket_reasons"],
                "expected": expected,
                "predicted": predicted,
                "raw": raw,
                "correct": correct,
                "seconds": seconds,
                "error": error,
            }
        )
        print(
            f"[{index:02d}/{len(cases)}] "
            f"expected={expected:<6} predicted={predicted:<6} "
            f"{seconds:.2f}s {record['crop']}"
        )

    confusion = Counter(
        (result["expected"], result["predicted"])
        for result in results
    )
    false_rejections = sum(
        result["expected"] == "KEEP"
        and result["predicted"] == "REJECT"
        for result in results
    )
    missed_rejections = sum(
        result["expected"] == "REJECT"
        and result["predicted"] == "KEEP"
        for result in results
    )
    report = {
        "model": args.model,
        "prompt": GATE_PROMPT,
        "policy": "Ambiguous responses and API errors become KEEP.",
        "sample_policy": (
            "Includes deterministic-filtered crops."
            if args.include_already_filtered
            else "Excludes crops already removed by deterministic filters."
        ),
        "counts": {
            "cases": len(results),
            "golden_keep": len(golden),
            "highly_unlikely_reject": len(unlikely),
            "false_rejections": false_rejections,
            "missed_rejections": missed_rejections,
            "api_errors": sum(bool(result["error"]) for result in results),
        },
        "confusion": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
        "total_seconds": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"False rejection of golden records: {false_rejections}")
    print(f"Missed Highly Unlikely rejections: {missed_rejections}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
