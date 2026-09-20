"""Import a first-cycle confirmation CSV and atomically save its registry."""

import argparse
from pathlib import Path
import sys

from racesort.confirmation_import import import_confirmation_csv
from racesort.registry import EventRegistry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import first-cycle RaceSort human confirmations."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--event-id")
    parser.add_argument("--event-date")
    parser.add_argument("--race-type", choices=("motorcycle", "car"))
    return parser.parse_args()


def load_or_create_registry(args):
    if args.registry.exists():
        return EventRegistry.load(args.registry)
    if not args.event_id or not args.race_type:
        raise ValueError(
            "--event-id and --race-type are required for a new registry"
        )
    return EventRegistry(args.event_id, args.race_type, args.event_date)


def main():
    args = parse_args()
    try:
        registry = load_or_create_registry(args)
        result = import_confirmation_csv(registry, args.csv)
    except (OSError, ValueError) as exc:
        print(f"IMPORT ERROR: {exc}", file=sys.stderr)
        return 1

    for outcome in result.outcomes:
        identity = ""
        if outcome.race_number is not None:
            identity = f" {outcome.race_number}/{outcome.variant_id}"
        print(
            f"row {outcome.row_number}: {outcome.status}{identity} "
            f"- {outcome.message}"
        )

    print(f"Summary: {result.status_counts()}")
    if not result.safe_to_save:
        print("NOT SAVED: correct INVALID rows and import again.", file=sys.stderr)
        return 1

    result.registry.save(args.registry)
    print(f"Saved registry: {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
