"""Compare serial and two-worker RapidOCR throughput safely."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import statistics
import threading
import time

from rapidocr import RapidOCR


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "test-output"
DEFAULT_REPORT_PATH = (
    DEFAULT_OUTPUT_DIR
    / "ocr-worker-benchmark.json"
)

FILTERED_DECISIONS = {
    "FILTERED_NON_PRIMARY",
    "FILTERED_TOO_BLURRY",
}

RACE_NUMBER_PATTERN = re.compile(
    r"[A-Z0-9]{1,6}"
)

thread_state = threading.local()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one versus two persistent RapidOCR workers "
            "on the current RaceSort processable crops."
        )
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Timed rounds for each worker count (default: 3).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing current photo-results.json files.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON report destination.",
    )

    return parser.parse_args()


def normalize_race_number(value):
    """Apply the same string normalization as the main pipeline."""

    if value is None:
        return None

    value = str(value).strip().upper()

    if not value or value == "UNKNOWN":
        return None

    if not RACE_NUMBER_PATTERN.fullmatch(value):
        return None

    return value


def extract_ocr_candidates(result):
    """Apply the same OCR candidate policy as the main pipeline."""

    candidates = []

    if result is None:
        return candidates

    texts = getattr(result, "txts", None)

    if not texts:
        return candidates

    for text in texts:
        compact = (
            str(text)
            .strip()
            .upper()
            .replace(" ", "")
        )

        candidate = normalize_race_number(compact)

        if candidate is None:
            continue

        if not any(
            character.isdigit()
            for character in candidate
        ):
            continue

        candidates.append(candidate)

    return list(dict.fromkeys(candidates))


def find_processable_crops(output_dir):
    """Read current results and return crops that reached OCR."""

    crops = []

    for result_path in sorted(
        output_dir.glob("GGBM*/photo-results.json")
    ):
        with open(
            result_path,
            "r",
            encoding="utf-8",
        ) as json_file:
            photo_result = json.load(json_file)

        for vehicle in photo_result["vehicles"]:
            if vehicle["decision"] in FILTERED_DECISIONS:
                continue

            crop_path = result_path.parent / vehicle["crop"]

            if not crop_path.exists():
                raise FileNotFoundError(
                    f"Crop not found: {crop_path}"
                )

            crops.append(crop_path)

    return crops


def initialize_worker():
    """Create one OCR engine owned by one executor thread."""

    thread_state.ocr_engine = RapidOCR()


def warm_worker(crop_path, barrier):
    """Load each worker before timed rounds begin."""

    barrier.wait()
    thread_state.ocr_engine(str(crop_path))
    return threading.get_ident()


def read_crop(crop_path):
    """Run OCR using only the engine owned by this thread."""

    result = thread_state.ocr_engine(str(crop_path))

    texts = list(
        getattr(result, "txts", None)
        or []
    )

    return {
        "crop": str(crop_path.relative_to(PROJECT_DIR)),
        "texts": texts,
        "candidates": extract_ocr_candidates(result),
    }


def benchmark_worker_count(crop_paths, workers, rounds):
    """Warm persistent workers, then run repeated timed rounds."""

    round_results = []

    with ThreadPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        thread_name_prefix=f"racesort-ocr-{workers}",
    ) as executor:

        barrier = threading.Barrier(workers)

        warm_futures = [
            executor.submit(
                warm_worker,
                crop_paths[index % len(crop_paths)],
                barrier,
            )
            for index in range(workers)
        ]

        worker_thread_ids = sorted(
            future.result()
            for future in warm_futures
        )

        for round_number in range(1, rounds + 1):
            start = time.perf_counter()

            results = list(
                executor.map(
                    read_crop,
                    crop_paths,
                )
            )

            elapsed = time.perf_counter() - start

            round_results.append(
                {
                    "round": round_number,
                    "seconds": elapsed,
                    "results": results,
                }
            )

    seconds = [
        item["seconds"]
        for item in round_results
    ]

    return {
        "workers": workers,
        "worker_thread_ids": worker_thread_ids,
        "rounds": round_results,
        "median_seconds": statistics.median(seconds),
        "minimum_seconds": min(seconds),
        "maximum_seconds": max(seconds),
    }


def comparable_results(mode_result):
    """Return only recognition evidence, excluding timing."""

    return [
        round_result["results"]
        for round_result in mode_result["rounds"]
    ]


def main():
    args = parse_arguments()

    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")

    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()

    crop_paths = find_processable_crops(output_dir)

    if not crop_paths:
        raise RuntimeError(
            "No processable crops found. Run test_pipeline.py first."
        )

    # ONNX Runtime may create a telemetry session file in the
    # current directory. Keep generated runtime artifacts inside
    # the already ignored output directory, never the project root.
    os.chdir(output_dir)

    print("=" * 72)
    print("RACESORT RAPIDOCR WORKER BENCHMARK")
    print("=" * 72)
    print(f"Processable crops: {len(crop_paths)}")
    print(f"Rounds per mode: {args.rounds}")
    print()

    serial = benchmark_worker_count(
        crop_paths,
        workers=1,
        rounds=args.rounds,
    )

    parallel = benchmark_worker_count(
        crop_paths,
        workers=2,
        rounds=args.rounds,
    )

    serial_rounds = comparable_results(serial)
    parallel_rounds = comparable_results(parallel)

    reference_results = serial_rounds[0]

    equivalent = all(
        round_results == reference_results
        for round_results in (
            serial_rounds
            + parallel_rounds
        )
    )

    speedup = (
        serial["median_seconds"]
        / parallel["median_seconds"]
        if parallel["median_seconds"] > 0
        else None
    )

    report = {
        "processable_crops": len(crop_paths),
        "rounds_per_mode": args.rounds,
        "results_equivalent": equivalent,
        "serial": serial,
        "parallel": parallel,
        "median_speedup": speedup,
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

    print(
        "Serial round times: "
        + ", ".join(
            f"{item['seconds']:.3f}s"
            for item in serial["rounds"]
        )
    )

    print(
        "Two-worker round times: "
        + ", ".join(
            f"{item['seconds']:.3f}s"
            for item in parallel["rounds"]
        )
    )

    print(
        f"Serial median: {serial['median_seconds']:.3f}s"
    )

    print(
        "Two-worker median: "
        f"{parallel['median_seconds']:.3f}s"
    )

    print(f"Median speedup: {speedup:.2f}x")
    print(f"OCR results equivalent: {equivalent}")
    print(f"Report: {report_path}")

    if not equivalent:
        print()
        print(
            "FAIL: worker modes produced different OCR evidence"
        )
        return 1

    print()
    print("PASS: worker modes produced identical OCR evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
