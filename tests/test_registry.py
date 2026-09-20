"""Unit tests for the event-scoped multi-variant registry contract."""

import unittest

from racesort.registry import (
    ConfirmedReference,
    EventRegistry,
    HumanConfirmation,
    apply_human_confirmation,
)


def reference(photo, crop, **context):
    return ConfirmedReference(
        source_photo=photo,
        crop=crop,
        confirmation_source="human",
        **context,
    )


class EventRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = EventRegistry(
            event_id="track-day-2026-09-19",
            event_date="2026-09-19",
            race_type="motorcycle",
        )

    def test_same_number_can_have_multiple_visual_variants(self):
        self.registry.add_variant("69", "69-red-bike")
        self.registry.add_variant("69", "69-blue-bike")

        variants = self.registry.entries["69"].variants
        self.assertEqual(list(variants), ["69-red-bike", "69-blue-bike"])

    def test_variant_can_have_multiple_confirmed_viewpoints(self):
        self.registry.add_variant("007", "007-a")
        self.registry.add_reference(
            "007",
            "007-a",
            reference("GGBM0001.JPG", "motorcycle-01.jpg", group="A", cycle=1),
        )
        self.registry.add_reference(
            "007",
            "007-a",
            reference("GGBM0051.JPG", "motorcycle-02.jpg", group="A", cycle=2),
        )

        references = self.registry.entries["007"].variants["007-a"].references
        self.assertEqual(len(references), 2)
        self.assertEqual(references[0].group, "A")
        self.assertEqual(references[1].cycle, 2)

    def test_zero_and_alphanumeric_numbers_remain_strings(self):
        self.registry.add_variant("0", "zero-bike")
        self.registry.add_variant("a12", "a12-bike")
        self.assertIn("0", self.registry.entries)
        self.assertIn("A12", self.registry.entries)

    def test_numeric_python_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            self.registry.add_variant(7, "bike-seven")

    def test_invalid_event_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "race_type"):
            EventRegistry("event-2", "boat")
        with self.assertRaisesRegex(ValueError, "event_date"):
            EventRegistry("event-2", "car", event_date="09/19/2026")

    def test_invalid_reference_session_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "group"):
            reference("photo.jpg", "crop.jpg", group="D")
        with self.assertRaisesRegex(ValueError, "cycle"):
            reference("photo.jpg", "crop.jpg", cycle=6)

    def test_duplicate_reference_is_idempotent_for_same_variant(self):
        self.registry.add_variant("54", "54-a")
        confirmed = reference("GGBM0017.JPG", "motorcycle-01.jpg")
        self.assertTrue(self.registry.add_reference("54", "54-a", confirmed))
        self.assertFalse(self.registry.add_reference("54", "54-a", confirmed))

    def test_reference_cannot_be_reassigned_to_another_variant(self):
        self.registry.add_variant("69", "69-a")
        self.registry.add_variant("69", "69-b")
        confirmed = reference("GGBM0009.JPG", "motorcycle-01.jpg")
        self.registry.add_reference("69", "69-a", confirmed)

        with self.assertRaisesRegex(ValueError, "multiple registry variants"):
            self.registry.add_reference("69", "69-b", confirmed)

    def test_json_contract_preserves_event_variant_and_provenance(self):
        self.registry.add_variant("957", "957-a", {"bike_colors": ["blue"]})
        self.registry.add_reference(
            "957",
            "957-a",
            reference(
                "GGBM0100.JPG",
                "motorcycle-01.jpg",
                group="C",
                cycle=1,
                session_id="cycle-1-c",
                metadata={"helmet_colors": ["white"]},
            ),
        )

        data = self.registry.to_dict()
        variant = data["numbers"]["957"]["variants"]["957-a"]
        self.assertEqual(data["event"]["event_id"], "track-day-2026-09-19")
        self.assertEqual(variant["metadata"]["bike_colors"], ["blue"])
        self.assertEqual(variant["references"][0]["session_id"], "cycle-1-c")

    def test_unknown_variant_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unknown variant"):
            self.registry.add_reference(
                "49", "missing", reference("photo.jpg", "crop.jpg")
            )


class HumanConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.registry = EventRegistry("event-1", "motorcycle")
        self.registry.add_variant("69", "69-a")

    def test_accepts_proposed_number_into_existing_variant(self):
        decision = HumanConfirmation(
            source_photo="photo-1.jpg",
            crop="crop-1.jpg",
            number_action="ACCEPT",
            proposed_number="69",
            variant_action="MATCH_EXISTING",
            variant_id="69-a",
            group="B",
            cycle=1,
        )
        self.assertEqual(
            apply_human_confirmation(self.registry, decision),
            ("69", "69-a"),
        )

    def test_corrects_number_and_creates_distinct_variant(self):
        decision = HumanConfirmation(
            source_photo="photo-2.jpg",
            crop="crop-2.jpg",
            number_action="CORRECT",
            proposed_number="68",
            corrected_number="069",
            variant_action="CREATE_NEW",
            variant_id="069-blue-bike",
        )
        result = apply_human_confirmation(
            self.registry,
            decision,
            variant_metadata={"bike_colors": ["blue"]},
        )
        self.assertEqual(result, ("069", "069-blue-bike"))
        self.assertIn("069", self.registry.entries)

    def test_rejection_does_not_change_registry(self):
        decision = HumanConfirmation(
            source_photo="photo-3.jpg",
            crop="crop-3.jpg",
            number_action="REJECT",
            proposed_number="88",
        )
        self.assertIsNone(apply_human_confirmation(self.registry, decision))
        self.assertEqual(list(self.registry.entries), ["69"])

    def test_rejection_cannot_assign_a_variant(self):
        with self.assertRaisesRegex(ValueError, "cannot assign"):
            HumanConfirmation(
                source_photo="photo-4.jpg",
                crop="crop-4.jpg",
                number_action="REJECT",
                variant_action="MATCH_EXISTING",
                variant_id="69-a",
            )


if __name__ == "__main__":
    unittest.main()
