"""Tests for validated RaceSort quality rules."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from racesort.quality import (
    measure_sharpness,
    should_filter_non_primary,
    should_filter_too_blurry,
)


MAX_AREA = 0.20
MAX_RELATIVE_SHARPNESS = 0.45
MAX_BLUR_SHARPNESS = 150.0


class NonPrimaryFilterTests(unittest.TestCase):
    def test_filters_only_when_every_condition_is_below_threshold(self):
        self.assertTrue(
            should_filter_non_primary(
                2,
                0.19,
                0.44,
                MAX_AREA,
                MAX_RELATIVE_SHARPNESS,
            )
        )

    def test_does_not_filter_the_only_vehicle(self):
        self.assertFalse(
            should_filter_non_primary(
                1,
                0.01,
                0.01,
                MAX_AREA,
                MAX_RELATIVE_SHARPNESS,
            )
        )

    def test_area_threshold_is_exclusive(self):
        self.assertFalse(
            should_filter_non_primary(
                2,
                MAX_AREA,
                0.44,
                MAX_AREA,
                MAX_RELATIVE_SHARPNESS,
            )
        )

    def test_relative_sharpness_threshold_is_exclusive(self):
        self.assertFalse(
            should_filter_non_primary(
                2,
                0.19,
                MAX_RELATIVE_SHARPNESS,
                MAX_AREA,
                MAX_RELATIVE_SHARPNESS,
            )
        )


class BlurFilterTests(unittest.TestCase):
    def test_filters_just_below_threshold(self):
        self.assertTrue(
            should_filter_too_blurry(
                MAX_BLUR_SHARPNESS - 0.001,
                MAX_BLUR_SHARPNESS,
            )
        )

    def test_threshold_is_exclusive(self):
        self.assertFalse(
            should_filter_too_blurry(
                MAX_BLUR_SHARPNESS,
                MAX_BLUR_SHARPNESS,
            )
        )

    def test_does_not_filter_above_threshold(self):
        self.assertFalse(
            should_filter_too_blurry(
                MAX_BLUR_SHARPNESS + 0.001,
                MAX_BLUR_SHARPNESS,
            )
        )


class SharpnessMeasurementTests(unittest.TestCase):
    def test_edges_score_higher_than_a_uniform_image(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            uniform_path = directory / "uniform.jpg"
            edges_path = directory / "edges.jpg"
            Image.new("RGB", (200, 200), "gray").save(uniform_path)
            edges = Image.new("RGB", (200, 200), "white")
            draw = ImageDraw.Draw(edges)
            for offset in range(0, 200, 10):
                draw.rectangle((offset, 0, offset + 4, 199), fill="black")
            edges.save(edges_path)

            uniform_score = measure_sharpness(uniform_path)
            edge_score = measure_sharpness(edges_path)

            self.assertGreaterEqual(uniform_score, 0.0)
            self.assertGreater(edge_score, uniform_score)


if __name__ == "__main__":
    unittest.main()
