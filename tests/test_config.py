"""Tests for RaceSort environment configuration."""

import unittest
from pathlib import Path

from racesort.config import RaceSortConfig, parse_bool


class ConfigDefaultsTests(unittest.TestCase):
    def test_defaults_match_the_working_pipeline(self):
        config = RaceSortConfig.from_environ({})
        self.assertEqual(config.input_dir, Path("test-photos"))
        self.assertEqual(config.output_dir, Path("test-output"))
        self.assertEqual(config.detector_model, "facebook/detr-resnet-50")
        self.assertEqual(config.vision_model, "qwen3-vl:4b-instruct")
        self.assertEqual(config.dino_model, "facebook/dinov2-small")
        self.assertEqual(config.detection_threshold, 0.70)
        self.assertEqual(config.max_crop_size, 1500)
        self.assertFalse(config.enable_qwen_cache)
        self.assertFalse(config.enable_merged_box_split)
        self.assertEqual(config.merged_box_child_threshold, 0.275)
        self.assertEqual(
            config.merged_box_criteria.minimum_child_containment,
            0.80,
        )
        self.assertEqual(
            config.merged_box_criteria.minimum_child_area_ratio,
            0.12,
        )
        self.assertEqual(
            config.merged_box_criteria.maximum_child_area_ratio,
            0.80,
        )
        self.assertEqual(config.merged_box_criteria.maximum_child_iou, 0.55)
        self.assertEqual(
            config.merged_box_criteria.minimum_child_area_balance,
            0.50,
        )
        self.assertEqual(
            config.merged_box_criteria.minimum_horizontal_separation,
            0.33,
        )
        self.assertEqual(config.max_filter_area, 0.20)
        self.assertEqual(config.max_filter_relative_sharpness, 0.45)
        self.assertEqual(config.max_blur_sharpness, 150.0)
        self.assertEqual(config.dino_corroboration_threshold, 0.90)
        self.assertEqual(config.race_type, "motorcycle")
        self.assertEqual(
            config.event_context(),
            {
                "event_id": None,
                "event_date": None,
                "group": None,
                "cycle": None,
                "session_id": None,
            },
        )


class ConfigOverrideTests(unittest.TestCase):
    def test_event_context_is_normalized_and_separate(self):
        config = RaceSortConfig.from_environ(
            {
                "RACESORT_EVENT_ID": " weekend-1 ",
                "RACESORT_EVENT_DATE": "2026-09-04",
                "RACESORT_GROUP": " b ",
                "RACESORT_CYCLE": "3",
                "RACESORT_SESSION_ID": " B-cycle-3 ",
            }
        )
        self.assertEqual(
            config.event_context(),
            {
                "event_id": "weekend-1",
                "event_date": "2026-09-04",
                "group": "B",
                "cycle": 3,
                "session_id": "B-cycle-3",
            },
        )

    def test_paths_remain_path_objects(self):
        config = RaceSortConfig.from_environ(
            {
                "RACESORT_INPUT_DIR": r"C:\RaceSort\input",
                "RACESORT_OUTPUT_DIR": r"D:\RaceSort\output",
            }
        )
        self.assertIsInstance(config.input_dir, Path)
        self.assertIsInstance(config.output_dir, Path)
        self.assertEqual(str(config.input_dir), r"C:\RaceSort\input")
        self.assertEqual(str(config.output_dir), r"D:\RaceSort\output")

    def test_boolean_spellings_are_consistent(self):
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value):
                self.assertTrue(parse_bool(value, "SETTING"))
        for value in ("0", "false", "NO", "off"):
            with self.subTest(value=value):
                self.assertFalse(parse_bool(value, "SETTING"))


class ConfigValidationTests(unittest.TestCase):
    def assert_invalid(self, environment):
        with self.assertRaises(ValueError):
            RaceSortConfig.from_environ(environment)

    def test_rejects_invalid_race_type(self):
        self.assert_invalid({"RACESORT_RACE_TYPE": "boat"})

    def test_rejects_invalid_group(self):
        self.assert_invalid({"RACESORT_GROUP": "D"})

    def test_rejects_invalid_cycle(self):
        for value in ("0", "6", "one"):
            with self.subTest(value=value):
                self.assert_invalid({"RACESORT_CYCLE": value})

    def test_rejects_invalid_event_date(self):
        self.assert_invalid({"RACESORT_EVENT_DATE": "09/04/2026"})

    def test_rejects_invalid_boolean(self):
        self.assert_invalid({"RACESORT_ENABLE_QWEN_CACHE": "sometimes"})

    def test_rejects_invalid_threshold(self):
        self.assert_invalid({"RACESORT_DETECTION_THRESHOLD": "1.1"})

    def test_rejects_invalid_merged_box_criteria(self):
        self.assert_invalid(
            {"RACESORT_MERGED_BOX_MAX_CHILD_IOU": "-0.1"}
        )
        self.assert_invalid(
            {
                "RACESORT_MERGED_BOX_MIN_CHILD_AREA_RATIO": "0.8",
                "RACESORT_MERGED_BOX_MAX_CHILD_AREA_RATIO": "0.2",
            }
        )

    def test_rejects_empty_paths(self):
        self.assert_invalid({"RACESORT_INPUT_DIR": ""})
        self.assert_invalid({"RACESORT_OUTPUT_DIR": "   "})


if __name__ == "__main__":
    unittest.main()
