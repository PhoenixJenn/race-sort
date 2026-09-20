"""Unit tests for transactional first-cycle confirmation imports."""

import csv
from pathlib import Path
import tempfile
import unittest

from racesort.confirmation_import import (
    import_confirmation_csv,
    import_confirmation_rows,
)
from racesort.registry import ConfirmedReference, EventRegistry


def row(**overrides):
    values = {
        "source_photo": "photo-1.jpg",
        "crop": "crop-1.jpg",
        "number_action": "ACCEPT",
        "proposed_number": "69",
        "corrected_number": "",
        "variant_action": "MATCH_EXISTING",
        "variant_id": "69-a",
        "group": "A",
        "cycle": "1",
        "session_id": "cycle-1-a",
        "metadata_json": "{}",
        "variant_metadata_json": "{}",
    }
    values.update(overrides)
    return values


class ConfirmationRowImportTests(unittest.TestCase):
    def setUp(self):
        self.registry = EventRegistry("event-1", "motorcycle", "2026-09-19")
        self.registry.add_variant("69", "69-a")

    def test_reports_accepted_corrected_rejected_and_duplicate(self):
        rows = [
            row(),
            row(
                source_photo="photo-2.jpg",
                crop="crop-2.jpg",
                number_action="CORRECT",
                proposed_number="68",
                corrected_number="007",
                variant_action="CREATE_NEW",
                variant_id="007-a",
                variant_metadata_json='{"bike_colors": ["blue"]}',
            ),
            row(
                source_photo="photo-3.jpg",
                crop="crop-3.jpg",
                number_action="REJECT",
                proposed_number="88",
                variant_action="",
                variant_id="",
            ),
            row(),
        ]

        result = import_confirmation_rows(self.registry, rows)

        self.assertTrue(result.safe_to_save)
        self.assertEqual(
            [outcome.status for outcome in result.outcomes],
            ["ACCEPTED", "CORRECTED", "REJECTED", "DUPLICATE"],
        )
        self.assertEqual(result.outcomes[0].row_number, 2)
        self.assertIn("007", result.registry.entries)
        self.assertNotIn("88", result.registry.entries)

    def test_valid_zero_and_alphanumeric_values_remain_strings(self):
        result = import_confirmation_rows(
            self.registry,
            [
                row(
                    source_photo="zero.jpg",
                    crop="zero-crop.jpg",
                    proposed_number="0",
                    variant_action="CREATE_NEW",
                    variant_id="zero-bike",
                ),
                row(
                    source_photo="alpha.jpg",
                    crop="alpha-crop.jpg",
                    proposed_number="a12",
                    variant_action="CREATE_NEW",
                    variant_id="a12-bike",
                ),
            ],
        )
        self.assertTrue(result.safe_to_save)
        self.assertIn("0", result.registry.entries)
        self.assertIn("A12", result.registry.entries)

    def test_invalid_rows_are_reported_and_original_registry_is_unchanged(self):
        result = import_confirmation_rows(
            self.registry,
            [
                row(),
                row(source_photo="", cycle="not-a-number"),
            ],
        )

        self.assertFalse(result.safe_to_save)
        self.assertEqual(result.outcomes[1].status, "INVALID")
        self.assertIn("cycle", result.outcomes[1].message)
        self.assertEqual(
            self.registry.entries["69"].variants["69-a"].references,
            [],
        )
        self.assertEqual(
            len(result.registry.entries["69"].variants["69-a"].references),
            1,
        )

    def test_conflicting_reference_assignment_is_invalid(self):
        self.registry.add_variant("69", "69-b")
        self.registry.add_reference(
            "69",
            "69-a",
            ConfirmedReference("photo-1.jpg", "crop-1.jpg", "human"),
        )
        result = import_confirmation_rows(
            self.registry,
            [row(variant_id="69-b")],
        )
        self.assertEqual(result.outcomes[0].status, "INVALID")
        self.assertIn("multiple registry variants", result.outcomes[0].message)

    def test_invalid_json_metadata_is_reported(self):
        result = import_confirmation_rows(
            self.registry,
            [row(metadata_json="not-json")],
        )
        self.assertEqual(result.outcomes[0].status, "INVALID")
        self.assertIn("metadata_json", result.outcomes[0].message)


class ConfirmationCsvImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.registry = EventRegistry("event-1", "motorcycle")
        self.registry.add_variant("69", "69-a")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_csv(self, path, fieldnames, rows):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_csv_import_can_be_saved_and_loaded(self):
        csv_path = self.root / "confirmations.csv"
        values = row(metadata_json='{"helmet_colors": ["white"]}')
        self.write_csv(csv_path, list(values), [values])

        result = import_confirmation_csv(self.registry, csv_path)
        registry_path = self.root / "event-registry.json"
        self.assertTrue(result.safe_to_save)
        result.registry.save(registry_path)

        loaded = EventRegistry.load(registry_path)
        saved_reference = loaded.entries["69"].variants["69-a"].references[0]
        self.assertEqual(saved_reference.metadata["helmet_colors"], ["white"])

    def test_missing_required_csv_column_is_rejected_before_import(self):
        csv_path = self.root / "missing-column.csv"
        values = row()
        fieldnames = [name for name in values if name != "variant_id"]
        self.write_csv(csv_path, fieldnames, [{k: values[k] for k in fieldnames}])

        with self.assertRaisesRegex(ValueError, "variant_id"):
            import_confirmation_csv(self.registry, csv_path)
        self.assertEqual(self.registry.entries["69"].variants["69-a"].references, [])


if __name__ == "__main__":
    unittest.main()
