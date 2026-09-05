"""Verify that Qwen cache hits preserve cold-run pipeline results."""

import argparse
import json
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cold_output", type=Path)
    parser.add_argument("warm_output", type=Path)
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def semantic_vehicle_result(vehicle):
    """Exclude timing while retaining recognition and routing evidence."""

    return {
        "vehicle": vehicle["vehicle"],
        "detection_source": vehicle.get("detection_source"),
        "profile_number": vehicle["profile_number"],
        "verification_number": vehicle["verification_number"],
        "final_number": vehicle["final_number"],
        "decision": vehicle["decision"],
        "profile_type": vehicle["profile_type"],
        "routing": vehicle["routing"],
        "quality": vehicle["quality"],
    }


def main():
    args = parse_args()
    failures = []

    cold_summary = load_json(args.cold_output / "run-summary.json")
    warm_summary = load_json(args.warm_output / "run-summary.json")

    cold_results = sorted(args.cold_output.glob("*/photo-results.json"))

    if not cold_results:
        failures.append("cold output contains no photo results")

    for cold_path in cold_results:
        relative_path = cold_path.relative_to(args.cold_output)
        warm_path = args.warm_output / relative_path

        if not warm_path.exists():
            failures.append(f"warm output is missing {relative_path}")
            continue

        cold = load_json(cold_path)
        warm = load_json(warm_path)
        cold_semantic = [
            semantic_vehicle_result(vehicle)
            for vehicle in cold["vehicles"]
        ]
        warm_semantic = [
            semantic_vehicle_result(vehicle)
            for vehicle in warm["vehicles"]
        ]

        if cold_semantic != warm_semantic:
            failures.append(f"semantic results changed for {relative_path}")

    cold_counts = cold_summary["counts"]
    warm_counts = warm_summary["counts"]

    if cold_counts["qwen_cache_hits"] != 0:
        failures.append("cold run should have zero cache hits")

    if cold_counts["qwen_cache_misses"] <= 0:
        failures.append("cold run should have at least one cache miss")

    if warm_counts["qwen_cache_misses"] != 0:
        failures.append("warm run should have zero cache misses")

    if (
        warm_counts["qwen_cache_hits"]
        != cold_counts["qwen_cache_misses"]
    ):
        failures.append("warm hits should equal cold misses")

    warm_qwen_time = (
        warm_summary["timing_seconds"]["qwen_verify_total"]
        + warm_summary["timing_seconds"]["qwen_direct_total"]
    )

    if warm_qwen_time != 0.0:
        failures.append("warm run should spend zero seconds in Qwen")

    print("RACESORT QWEN CACHE REGRESSION CHECK")
    print(f"Cold output: {args.cold_output}")
    print(f"Warm output: {args.warm_output}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("PASS: cache hits preserved recognition and routing results")


if __name__ == "__main__":
    main()
