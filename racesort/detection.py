"""Detection-box geometry and conservative merged-motorcycle recovery."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MergedBoxCriteria:
    """Validated geometric limits for replacing one merged parent box."""

    minimum_child_containment: float
    minimum_child_area_ratio: float
    maximum_child_area_ratio: float
    maximum_child_iou: float
    minimum_child_area_balance: float
    minimum_horizontal_separation: float


def box_area(box):
    """Return the non-negative area of one detector box."""

    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def box_intersection_area(first, second):
    """Return the shared area of two detector boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def box_containment(child, parent):
    """Return the fraction of a child box inside its parent."""

    child_area = box_area(child)
    if child_area == 0:
        return 0.0
    return box_intersection_area(child, parent) / child_area


def box_iou(first, second):
    """Return intersection-over-union for two detector boxes."""

    intersection = box_intersection_area(first, second)
    union = box_area(first) + box_area(second) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def find_merged_box_children(parent, detections, criteria):
    """Find two conservative low-confidence children for one parent."""

    parent_area = box_area(parent["box"])
    parent_width = parent["box"][2] - parent["box"][0]
    if parent_area == 0 or parent_width <= 0:
        return None

    candidates = []
    for detection in detections:
        if detection is parent:
            continue
        area_ratio = box_area(detection["box"]) / parent_area
        if (
            criteria.minimum_child_area_ratio
            <= area_ratio
            <= criteria.maximum_child_area_ratio
            and box_containment(detection["box"], parent["box"])
            >= criteria.minimum_child_containment
        ):
            candidates.append(detection)

    valid_pairs = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            first_area = box_area(first["box"])
            second_area = box_area(second["box"])
            area_balance = min(first_area, second_area) / max(
                first_area,
                second_area,
            )
            first_center = (first["box"][0] + first["box"][2]) / 2
            second_center = (second["box"][0] + second["box"][2]) / 2
            horizontal_separation = (
                abs(first_center - second_center) / parent_width
            )
            if (
                box_iou(first["box"], second["box"])
                <= criteria.maximum_child_iou
                and area_balance >= criteria.minimum_child_area_balance
                and horizontal_separation
                >= criteria.minimum_horizontal_separation
            ):
                valid_pairs.append(
                    (first["score"] + second["score"], first, second)
                )

    if not valid_pairs:
        return None
    _, first, second = max(valid_pairs, key=lambda item: item[0])
    return [first, second]


def resolve_merged_vehicle_boxes(
    detections,
    detection_threshold,
    enabled,
    detection_class,
    criteria,
):
    """Replace a strong merged parent with two validated child boxes."""

    baseline = [
        detection
        for detection in detections
        if detection["score"] >= detection_threshold
    ]
    if not enabled or detection_class != "motorcycle":
        return baseline

    replacements = {}
    used_children = set()
    for parent in sorted(
        baseline,
        key=lambda item: item["score"],
        reverse=True,
    ):
        children = find_merged_box_children(parent, detections, criteria)
        if (
            children is None
            or any(id(child) in used_children for child in children)
        ):
            continue
        replacements[id(parent)] = children
        used_children.update(id(child) for child in children)

    resolved = []
    resolved_ids = set()
    for detection in baseline:
        children = replacements.get(id(detection))
        if children is None:
            if id(detection) not in resolved_ids:
                resolved.append(detection)
                resolved_ids.add(id(detection))
            continue
        for child in children:
            child["detection_source"] = "merged_box_child"
            if id(child) not in resolved_ids:
                resolved.append(child)
                resolved_ids.add(id(child))
    return resolved
