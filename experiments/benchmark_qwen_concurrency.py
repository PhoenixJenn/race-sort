"""Compare serial and two-call Qwen vision throughput safely."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import statistics
import time

import ollama


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = (
    PROJECT_DIR
    / "test-output"
    / "qwen-concurrency-benchmark.json"
)

DEFAULT_MODEL = "qwen3-vl:4b-instruct"

RACE_NUMBER_PATTERN = re.compile(
    r"[A-Z0-9]{1,6}"
)

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

BENCHMARK_CASES = [
    {
        "name": "clear_49",
        "crop": "test-output/GGBM0001/motorcycle-01.jpg",
        "expected": "49",
    },
    {
        "name": "valid_zero",
        "crop": "test-output/GGBM0008/motorcycle-01.jpg",
        "expected": "0",
    },
    {
        "name": "recovery_54",
        "crop": "test-output/GGBM0017/motorcycle-01.jpg",
        "expected": "54",
    },
    {
        "name": "numberless_motorcycle",
        "crop": "test-output/GGBM0012/motorcycle-01.jpg",
        "expected": None,
    },
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark serial versus two simultaneous Qwen vision "
            "calls without changing the RaceSort pipeline."
        )
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Serial and parallel rounds to alternate (default: 2).",
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


def normalize_race_number(value):
    """Apply the same identifier policy as the main pipeline."""

    if value is None:
        return None

    value = str(value).strip().upper()

    if not value or value == "UNKNOWN":
        return None

    if not RACE_NUMBER_PATTERN.fullmatch(value):
        return None

    return value


def serialize_ollama_state():
    """Capture available model and VRAM state without requiring it."""

    try:
        state = ollama.ps()
    except Exception as error:  # The benchmark can still run without ps().
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }

    if hasattr(state, "model_dump"):
        data = state.model_dump(mode="json")
    else:
        data = str(state)

    return {
        "available": True,
        "state": data,
    }


def prepare_cases():
    """Resolve paths and verify the fixed benchmark crops exist."""

    cases = []

    for case in BENCHMARK_CASES:
        crop_path = (
            PROJECT_DIR
            / case["crop"]
        ).resolve()

        if not crop_path.exists():
            raise FileNotFoundError(
                f"Benchmark crop not found: {crop_path}\n"
                "Run test_pipeline.py before this experiment."
            )

        cases.append(
            {
                **case,
                "crop_path": crop_path,
            }
        )

    return cases


def run_case(case, model):
    """Run one unanchored Qwen read and preserve all evidence."""

    started = time.perf_counter()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": DIRECT_NUMBER_PROMPT,
                    "images": [str(case["crop_path"])],
                }
            ],
        )

        raw = response["message"]["content"].strip()
        normalized = normalize_race_number(raw)
        error = None

    except Exception as caught_error:
        raw = None
        normalized = None
        error = (
            f"{type(caught_error).__name__}: "
            f"{caught_error}"
        )

    elapsed = time.perf_counter() - started

    return {
        "name": case["name"],
        "crop": case["crop"],
        "expected": case["expected"],
        "raw": raw,
        "normalized": normalized,
        "matches_expected": (
            error is None
            and normalized == case["expected"]
        ),
        "seconds": elapsed,
        "error": error,
    }


def run_mode(cases, model, workers):
    """Run all fixed cases serially or with two caller threads."""

    started = time.perf_counter()

    if workers == 1:
        results = [
            run_case(case, model)
            for case in cases
        ]

    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=(
                f"racesort-qwen-{workers}"
            ),
        ) as executor:
            results = list(
                executor.map(
                    lambda case: run_case(case, model),
                    cases,
                )
            )

    wall_seconds = time.perf_counter() - started
    summed_call_seconds = sum(
        result["seconds"]
        for result in results
    )

    return {
        "workers": workers,
        "wall_seconds": wall_seconds,
        "summed_call_seconds": summed_call_seconds,
        "overlap_ratio": (
            summed_call_seconds / wall_seconds
            if wall_seconds > 0
            else None
        ),
        "results": results,
    }


def result_signature(mode_result):
    """Return case names and normalized outputs for comparison."""

    return [
        (result["name"], result["normalized"])
        for result in mode_result["results"]
    ]


def main():
    args = parse_arguments()

    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")

    cases = prepare_cases()
    report_path = args.report.resolve()

    try:
        ollama.ps()
    except Exception as error:
        print("ERROR: Ollama is not running or accessible.")
        print(f"Details: {type(error).__name__}: {error}")
        print("Start Ollama, verify it with `ollama list`, then retry.")
        return 1

    print("=" * 72)
    print("RACESORT QWEN CONCURRENCY BENCHMARK")
    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"Fixed crops: {len(cases)}")
    print(f"Alternating rounds per mode: {args.rounds}")
    print()

    state_before = serialize_ollama_state()

    print("Warming every fixed case with untimed serial calls...")
    warmup = [
        run_case(case, args.model)
        for case in cases
    ]

    warmup_errors = [
        result
        for result in warmup
        if result["error"] is not None
    ]

    if warmup_errors:
        print(
            f"ERROR: warmup failed: {warmup_errors}"
        )
        return 1

    rounds = []

    for round_number in range(1, args.rounds + 1):
        if round_number % 2 == 1:
            execution_order = ["serial", "parallel"]
        else:
            execution_order = ["parallel", "serial"]

        mode_results = {}

        for mode in execution_order:
            mode_results[mode] = run_mode(
                cases,
                args.model,
                workers=(
                    1
                    if mode == "serial"
                    else 2
                ),
            )

        serial = mode_results["serial"]
        parallel = mode_results["parallel"]

        rounds.append(
            {
                "round": round_number,
                "execution_order": execution_order,
                "serial": serial,
                "parallel": parallel,
                "responses_equivalent": (
                    result_signature(serial)
                    == result_signature(parallel)
                ),
            }
        )

    state_after = serialize_ollama_state()

    serial_times = [
        round_result["serial"]["wall_seconds"]
        for round_result in rounds
    ]

    parallel_times = [
        round_result["parallel"]["wall_seconds"]
        for round_result in rounds
    ]

    serial_median = statistics.median(serial_times)
    parallel_median = statistics.median(parallel_times)

    speedup = (
        serial_median / parallel_median
        if parallel_median > 0
        else None
    )

    all_results = [
        result
        for round_result in rounds
        for mode_result in (
            round_result["serial"],
            round_result["parallel"],
        )
        for result in mode_result["results"]
    ]

    errors = [
        result
        for result in all_results
        if result["error"] is not None
    ]

    expected_matches = sum(
        1
        for result in all_results
        if result["matches_expected"]
    )

    report = {
        "model": args.model,
        "cases": [
            {
                "name": case["name"],
                "crop": case["crop"],
                "expected": case["expected"],
            }
            for case in cases
        ],
        "warmup": warmup,
        "ollama_state_before": state_before,
        "ollama_state_after": state_after,
        "rounds": rounds,
        "serial_median_seconds": serial_median,
        "parallel_median_seconds": parallel_median,
        "median_speedup": speedup,
        "responses_equivalent_by_round": [
            round_result["responses_equivalent"]
            for round_result in rounds
        ],
        "expected_matches": expected_matches,
        "total_timed_results": len(all_results),
        "errors": errors,
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

    for round_result in rounds:
        serial = round_result["serial"]
        parallel = round_result["parallel"]

        print(
            f"Round {round_result['round']}: "
            f"order={round_result['execution_order']}, "
            f"serial={serial['wall_seconds']:.2f}s, "
            f"two-call={parallel['wall_seconds']:.2f}s, "
            "equivalent="
            f"{round_result['responses_equivalent']}"
        )

        print(
            "  serial outputs: "
            f"{result_signature(serial)}"
        )

        print(
            "  two-call outputs: "
            f"{result_signature(parallel)}"
        )

    print()
    print(f"Serial median: {serial_median:.2f}s")
    print(f"Two-call median: {parallel_median:.2f}s")
    print(f"Median speedup: {speedup:.2f}x")
    print(
        "Expected outcomes: "
        f"{expected_matches}/{len(all_results)}"
    )
    print(f"API errors: {len(errors)}")
    print(f"Report: {report_path}")

    if errors:
        print()
        print("FAIL: one or more Ollama calls failed")
        return 1

    print()
    print(
        "COMPLETE: inspect timing and response variation before "
        "considering pipeline concurrency"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
