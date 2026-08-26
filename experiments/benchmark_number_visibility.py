"""Evaluate a conservative Qwen race-number visibility gate."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import ollama


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "test-output"
HUMAN_LABELS_PATH = (
    PROJECT_DIR
    / "racesort-verbose-human-validation.csv"
)
DEFAULT_REPORT_PATH = (
    OUTPUT_DIR
    / "number-visibility-benchmark.json"
)

DEFAULT_MODEL = "qwen3-vl:4b-instruct"

FILTERED_DECISIONS = {
    "FILTERED_NON_PRIMARY",
    "FILTERED_TOO_BLURRY",
}

VALID_CLASSIFICATIONS = {
    "VISIBLE_NUMBER",
    "NO_NUMBER_VISIBLE",
    "UNCLEAR",
}

VISIBILITY_PROMPT = """
Classify whether actual race-number characters are visibly present on
this race vehicle.

Return ONLY one of these exact labels:

VISIBLE_NUMBER
NO_NUMBER_VISIBLE
UNCLEAR

Definitions:

- VISIBLE_NUMBER: One or more characters belonging to the vehicle's
  actual race number are visibly present. You do not need to read the
  complete identifier.
- NO_NUMBER_VISIBLE: The vehicle is clear enough to inspect, but no
  actual race-number characters are visible. A blank number area is
  NO_NUMBER_VISIBLE.
- UNCLEAR: Blur, angle, obstruction, crop quality, or ambiguity prevents
  a safe visibility decision.

Rules:

- Judge only whether race-number characters are visible.
- Ignore sponsor text, logos, manufacturer names, and unrelated decals.
- Do not invent or read a race number.
- If uncertain between VISIBLE_NUMBER and NO_NUMBER_VISIBLE, return
  UNCLEAR.
- Never guess.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Test a three-state Qwen number-visibility gate against "
            "the current human-labeled RaceSort crops."
        )
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Calls per crop (default: 1).",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama vision model (default: {DEFAULT_MODEL}).",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON report destination.",
    )

    return parser.parse_args()


def load_current_decisions():
    """Map human-label row keys to current pipeline decisions."""

    decisions = {}

    for result_path in sorted(
        OUTPUT_DIR.glob("GGBM*/photo-results.json")
    ):
        with open(
            result_path,
            "r",
            encoding="utf-8",
        ) as json_file:
            photo_result = json.load(json_file)

        photo_stem = result_path.parent.name

        for vehicle in photo_result["vehicles"]:
            decisions[
                (photo_stem, str(vehicle["vehicle"]))
            ] = vehicle["decision"]

    return decisions


def load_cases():
    """Load labeled crops that currently reach recognition."""

    current_decisions = load_current_decisions()

    if not current_decisions:
        raise RuntimeError(
            "No current photo results found. Run test_pipeline.py first."
        )

    with open(
        HUMAN_LABELS_PATH,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        human_rows = list(csv.DictReader(csv_file))

    cases = []

    for row in human_rows:
        key = (row["photo"], row["vehicle"])
        decision = current_decisions.get(key)

        if decision is None:
            continue

        if decision in FILTERED_DECISIONS:
            continue

        crop_path = (
            PROJECT_DIR
            / row["crop"]
        ).resolve()

        if not crop_path.exists():
            raise FileNotFoundError(
                f"Labeled crop not found: {crop_path}"
            )

        readability = (
            row["number_readability"].strip()
            or "UNLABELED"
        )

        cases.append(
            {
                "photo": row["photo"],
                "vehicle": int(row["vehicle"]),
                "crop": row["crop"],
                "crop_path": crop_path,
                "human_readability": readability,
                "ground_truth": row["ground_truth"],
            }
        )

    return cases


def normalize_classification(raw):
    """Accept only one exact visibility label."""

    value = str(raw).strip().upper()

    if value in VALID_CLASSIFICATIONS:
        return value

    return "UNCLEAR"


def classify_case(case, model, round_number):
    """Run one visibility-only Qwen call."""

    started = time.perf_counter()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": VISIBILITY_PROMPT,
                    "images": [str(case["crop_path"])],
                }
            ],
        )

        raw = response["message"]["content"].strip()
        classification = normalize_classification(raw)
        error = None

    except Exception as caught_error:
        raw = None
        classification = "UNCLEAR"
        error = (
            f"{type(caught_error).__name__}: "
            f"{caught_error}"
        )

    elapsed = time.perf_counter() - started

    return {
        "round": round_number,
        "photo": case["photo"],
        "vehicle": case["vehicle"],
        "crop": case["crop"],
        "human_readability": case["human_readability"],
        "ground_truth": case["ground_truth"],
        "raw": raw,
        "classification": classification,
        "seconds": elapsed,
        "error": error,
    }


def main():
    args = parse_arguments()

    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")

    cases = load_cases()
    report_path = args.report.resolve()

    try:
        ollama.ps()
    except Exception as error:
        print("ERROR: Ollama is not running or accessible.")
        print(f"Details: {type(error).__name__}: {error}")
        print("Start Ollama, verify it with `ollama list`, then retry.")
        return 1

    label_counts = {}

    for case in cases:
        label = case["human_readability"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("=" * 72)
    print("RACESORT NUMBER-VISIBILITY BENCHMARK")
    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"Processable labeled crops: {len(cases)}")
    print(f"Human labels: {label_counts}")
    print(f"Rounds: {args.rounds}")
    print()

    results = []
    benchmark_started = time.perf_counter()

    for round_number in range(1, args.rounds + 1):
        print(f"Round {round_number}/{args.rounds}")

        for case_number, case in enumerate(cases, start=1):
            result = classify_case(
                case,
                args.model,
                round_number,
            )

            results.append(result)

            print(
                f"  [{case_number:02d}/{len(cases)}] "
                f"{case['photo']} vehicle {case['vehicle']}: "
                f"human={case['human_readability']} "
                f"model={result['classification']} "
                f"({result['seconds']:.2f}s)"
            )

    elapsed = time.perf_counter() - benchmark_started

    errors = [
        result
        for result in results
        if result["error"] is not None
    ]

    false_rejections = [
        result
        for result in results
        if result["human_readability"] == "CLEAR"
        and result["classification"] == "NO_NUMBER_VISIBLE"
    ]

    numberless_results = [
        result
        for result in results
        if result["human_readability"] == "NO_NUMBER_VISIBLE"
    ]

    numberless_caught = [
        result
        for result in numberless_results
        if result["classification"] == "NO_NUMBER_VISIBLE"
    ]

    unclear_or_unlabeled = [
        result
        for result in results
        if result["human_readability"]
        in {
            "NOT_READABLE",
            "UNLABELED",
        }
    ]

    unsafe_uncertain_rejections = [
        result
        for result in unclear_or_unlabeled
        if result["classification"] == "NO_NUMBER_VISIBLE"
    ]

    classification_counts = {}

    for result in results:
        classification = result["classification"]
        classification_counts[classification] = (
            classification_counts.get(classification, 0)
            + 1
        )

    call_times = [
        result["seconds"]
        for result in results
    ]

    report = {
        "model": args.model,
        "rounds": args.rounds,
        "processable_labeled_crops": len(cases),
        "human_label_counts": label_counts,
        "classification_counts": classification_counts,
        "false_rejections_of_clear_numbers": false_rejections,
        "unsafe_rejections_of_uncertain_crops": (
            unsafe_uncertain_rejections
        ),
        "numberless_total": len(numberless_results),
        "numberless_caught": len(numberless_caught),
        "numberless_recall": (
            len(numberless_caught) / len(numberless_results)
            if numberless_results
            else None
        ),
        "api_errors": errors,
        "timing": {
            "total_seconds": elapsed,
            "median_call_seconds": statistics.median(call_times),
            "minimum_call_seconds": min(call_times),
            "maximum_call_seconds": max(call_times),
        },
        "results": results,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(report, json_file, indent=2)

    print()
    print("SUMMARY")
    print(f"Classifications: {classification_counts}")
    print(
        "Clear-number false rejections: "
        f"{len(false_rejections)}"
    )
    print(
        "Uncertain/unlabeled rejected as numberless: "
        f"{len(unsafe_uncertain_rejections)}"
    )
    print(
        f"Numberless caught: {len(numberless_caught)}/"
        f"{len(numberless_results)}"
    )
    print(
        "Numberless recall: "
        f"{report['numberless_recall']:.1%}"
    )
    print(f"API errors: {len(errors)}")
    print(f"Total time: {elapsed:.2f}s")
    print(
        "Median call time: "
        f"{report['timing']['median_call_seconds']:.2f}s"
    )
    print(f"Report: {report_path}")

    if errors:
        print()
        print("FAIL: one or more Ollama calls failed")
        return 1

    if false_rejections or unsafe_uncertain_rejections:
        print()
        print(
            "FAIL: visibility gate produced an unsafe rejection"
        )
        return 1

    print()
    print(
        "PASS: no clear or uncertain crop was rejected as numberless"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
