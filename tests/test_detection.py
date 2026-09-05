"""Tests for detector-box geometry and merged-box recovery."""

import unittest

from racesort.detection import (
    MergedBoxCriteria,
    box_area,
    box_containment,
    box_intersection_area,
    box_iou,
    resolve_merged_vehicle_boxes,
)


CRITERIA = MergedBoxCriteria(
    minimum_child_containment=0.80,
    minimum_child_area_ratio=0.12,
    maximum_child_area_ratio=0.80,
    maximum_child_iou=0.55,
    minimum_child_area_balance=0.50,
    minimum_horizontal_separation=0.33,
)


def detection(score, box):
    return {
        "score": score,
        "box": box,
        "detection_source": "baseline",
    }


class BoxGeometryTests(unittest.TestCase):
    def test_area_is_non_negative(self):
        self.assertEqual(box_area([0, 0, 10, 5]), 50)
        self.assertEqual(box_area([10, 5, 0, 0]), 0)

    def test_intersection_containment_and_iou(self):
        parent = [0, 0, 10, 10]
        child = [0, 0, 5, 10]
        self.assertEqual(box_intersection_area(parent, child), 50)
        self.assertEqual(box_containment(child, parent), 1.0)
        self.assertEqual(box_iou(parent, child), 0.5)

    def test_zero_area_boxes_are_safe(self):
        zero = [0, 0, 0, 10]
        self.assertEqual(box_containment(zero, zero), 0.0)
        self.assertEqual(box_iou(zero, zero), 0.0)


class MergedBoxResolutionTests(unittest.TestCase):
    def setUp(self):
        self.parent = detection(0.99, [0, 0, 100, 100])
        self.left = detection(0.40, [5, 5, 45, 95])
        self.right = detection(0.38, [55, 5, 95, 95])

    def resolve(self, detections, enabled=True, detection_class="motorcycle"):
        return resolve_merged_vehicle_boxes(
            detections,
            detection_threshold=0.70,
            enabled=enabled,
            detection_class=detection_class,
            criteria=CRITERIA,
        )

    def test_disabled_feature_returns_only_baseline_detections(self):
        self.assertEqual(
            self.resolve(
                [self.parent, self.left, self.right],
                enabled=False,
            ),
            [self.parent],
        )

    def test_non_motorcycle_class_does_not_split(self):
        self.assertEqual(
            self.resolve(
                [self.parent, self.left, self.right],
                detection_class="car",
            ),
            [self.parent],
        )

    def test_valid_children_replace_the_merged_parent(self):
        resolved = self.resolve([self.parent, self.left, self.right])
        self.assertEqual(resolved, [self.left, self.right])
        self.assertTrue(
            all(
                child["detection_source"] == "merged_box_child"
                for child in resolved
            )
        )

    def test_unbalanced_children_do_not_replace_parent(self):
        tiny = detection(0.45, [55, 5, 74, 95])
        resolved = self.resolve([self.parent, self.left, tiny])
        self.assertEqual(resolved, [self.parent])

    def test_close_children_do_not_replace_parent(self):
        close_left = detection(0.40, [20, 5, 60, 95])
        close_right = detection(0.38, [40, 5, 80, 95])
        resolved = self.resolve([self.parent, close_left, close_right])
        self.assertEqual(resolved, [self.parent])


if __name__ == "__main__":
    unittest.main()
