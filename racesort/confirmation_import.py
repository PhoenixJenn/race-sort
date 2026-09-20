"""Import first-cycle human confirmations into a staged event registry.

This model-free application module reads an explicit CSV contract, validates
every row, and applies safe rows to a copy of an ``EventRegistry``. Callers save
the staged registry only when ``safe_to_save`` is true, so a partially invalid
file cannot corrupt the last valid registry.
"""

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from racesort.registry import (
    EventRegistry,
    HumanConfirmation,
    apply_human_confirmation,
)


REQUIRED_COLUMNS = {
    "source_photo",
    "crop",
    "number_action",
    "variant_action",
    "variant_id",
}


@dataclass(frozen=True)
class ImportOutcome:
    """One visible result for one CSV data row."""

    row_number: int
    status: str
    message: str
    race_number: str | None = None
    variant_id: str | None = None


@dataclass(frozen=True)
class ConfirmationImportResult:
    """A staged registry plus complete row-level import outcomes."""

    registry: EventRegistry
    outcomes: tuple[ImportOutcome, ...]

    @property
    def safe_to_save(self):
        return not any(outcome.status == "INVALID" for outcome in self.outcomes)

    def status_counts(self):
        counts = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return counts


def optional_text(row, name):
    """Return stripped CSV text or None for an empty cell."""

    value = row.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def optional_cycle(row):
    """Parse an optional integer cycle from one CSV row."""

    value = optional_text(row, "cycle")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("cycle must be an integer") from exc


def json_object(row, name):
    """Parse an optional JSON object cell."""

    value = optional_text(row, name)
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return parsed


def reference_identity(registry, source_photo, crop):
    """Return the existing number/variant for a crop, if already registered."""

    reference_key = source_photo, crop
    for number, entry in registry.entries.items():
        for variant_id, variant in entry.variants.items():
            if any(
                reference.reference_key == reference_key
                for reference in variant.references
            ):
                return number, variant_id
    return None


def confirmation_from_row(row):
    """Convert one CSV dictionary into validated import objects."""

    confirmation = HumanConfirmation(
        source_photo=optional_text(row, "source_photo"),
        crop=optional_text(row, "crop"),
        number_action=(optional_text(row, "number_action") or "").upper(),
        proposed_number=optional_text(row, "proposed_number"),
        corrected_number=optional_text(row, "corrected_number"),
        variant_action=(
            (optional_text(row, "variant_action") or "").upper() or None
        ),
        variant_id=optional_text(row, "variant_id"),
        group=(optional_text(row, "group") or "").upper() or None,
        cycle=optional_cycle(row),
        session_id=optional_text(row, "session_id"),
        metadata=json_object(row, "metadata_json"),
    )
    return confirmation, json_object(row, "variant_metadata_json")


def import_confirmation_rows(registry, rows, *, first_row_number=2):
    """Apply rows transactionally to a registry copy and report every outcome."""

    staged_registry = EventRegistry.from_dict(registry.to_dict())
    outcomes = []

    for offset, row in enumerate(rows):
        row_number = first_row_number + offset
        try:
            confirmation, variant_metadata = confirmation_from_row(row)
            race_number = confirmation.resolved_number()

            if race_number is None:
                outcomes.append(
                    ImportOutcome(row_number, "REJECTED", "human rejected proposal")
                )
                continue

            existing_identity = reference_identity(
                staged_registry,
                confirmation.source_photo,
                confirmation.crop,
            )
            target_identity = race_number, confirmation.variant_id
            if existing_identity == target_identity:
                outcomes.append(
                    ImportOutcome(
                        row_number,
                        "DUPLICATE",
                        "reference already belongs to this variant",
                        race_number,
                        confirmation.variant_id,
                    )
                )
                continue

            apply_human_confirmation(
                staged_registry,
                confirmation,
                variant_metadata=variant_metadata,
            )
            status = (
                "CORRECTED"
                if confirmation.number_action == "CORRECT"
                else "ACCEPTED"
            )
            outcomes.append(
                ImportOutcome(
                    row_number,
                    status,
                    "confirmation applied to staged registry",
                    race_number,
                    confirmation.variant_id,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            outcomes.append(
                ImportOutcome(row_number, "INVALID", str(exc))
            )

    return ConfirmationImportResult(staged_registry, tuple(outcomes))


def import_confirmation_csv(registry, csv_path):
    """Read the confirmation CSV contract and return a staged import result."""

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                "confirmation CSV is missing columns: " + ", ".join(missing)
            )
        return import_confirmation_rows(registry, reader)
