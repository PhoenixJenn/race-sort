"""Validated, cross-platform configuration for RaceSort pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

from racesort.detection import MergedBoxCriteria


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
SUPPORTED_RACE_TYPES = {"motorcycle", "car"}
SUPPORTED_GROUPS = {"A", "B", "C"}
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def parse_bool(value, name):
    """Parse an explicit environment boolean or raise a useful error."""

    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of: "
        "1, 0, true, false, yes, no, on, off"
    )


def optional_text(environ, name):
    value = str(environ.get(name, "")).strip()
    return value or None


def float_setting(environ, name, default):
    try:
        return float(environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def int_setting(environ, name, default):
    try:
        return int(environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def path_setting(environ, name, default):
    value = str(environ.get(name, default)).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return Path(value)


@dataclass(frozen=True)
class RaceSortConfig:
    input_dir: Path
    output_dir: Path
    enable_qwen_cache: bool
    qwen_cache_dir: Path
    qwen_cache_schema_version: int
    detector_model: str
    vision_model: str
    dino_model: str
    detection_threshold: float
    max_crop_size: int
    enable_merged_box_split: bool
    merged_box_child_threshold: float
    merged_box_criteria: MergedBoxCriteria
    max_filter_area: float
    max_filter_relative_sharpness: float
    max_blur_sharpness: float
    dino_corroboration_threshold: float
    race_type: str
    event_id: str | None
    event_date: str | None
    group: str | None
    cycle: int | None
    session_id: str | None
    supported_extensions: frozenset[str] = SUPPORTED_EXTENSIONS

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None):
        if environ is None:
            import os

            environ = os.environ

        race_type = str(
            environ.get("RACESORT_RACE_TYPE", "motorcycle")
        ).strip().lower()
        if race_type not in SUPPORTED_RACE_TYPES:
            raise ValueError(
                "RACESORT_RACE_TYPE must be either 'motorcycle' or 'car'"
            )

        group = optional_text(environ, "RACESORT_GROUP")
        if group is not None:
            group = group.upper()
            if group not in SUPPORTED_GROUPS:
                raise ValueError("RACESORT_GROUP must be A, B, or C")

        cycle_text = optional_text(environ, "RACESORT_CYCLE")
        cycle = None
        if cycle_text is not None:
            try:
                cycle = int(cycle_text)
            except ValueError as exc:
                raise ValueError("RACESORT_CYCLE must be an integer") from exc
            if cycle not in range(1, 6):
                raise ValueError("RACESORT_CYCLE must be from 1 through 5")

        event_date = optional_text(environ, "RACESORT_EVENT_DATE")
        if event_date is not None:
            try:
                date.fromisoformat(event_date)
            except ValueError as exc:
                raise ValueError(
                    "RACESORT_EVENT_DATE must use YYYY-MM-DD"
                ) from exc

        config = cls(
            input_dir=path_setting(
                environ,
                "RACESORT_INPUT_DIR",
                "test-photos",
            ),
            output_dir=path_setting(
                environ,
                "RACESORT_OUTPUT_DIR",
                "test-output",
            ),
            enable_qwen_cache=parse_bool(
                environ.get("RACESORT_ENABLE_QWEN_CACHE", "0"),
                "RACESORT_ENABLE_QWEN_CACHE",
            ),
            qwen_cache_dir=path_setting(
                environ,
                "RACESORT_QWEN_CACHE_DIR",
                ".racesort-cache/qwen",
            ),
            qwen_cache_schema_version=int_setting(
                environ,
                "RACESORT_QWEN_CACHE_SCHEMA_VERSION",
                1,
            ),
            detector_model=str(
                environ.get(
                    "RACESORT_DETECTOR_MODEL",
                    "facebook/detr-resnet-50",
                )
            ),
            vision_model=str(
                environ.get(
                    "RACESORT_VISION_MODEL",
                    "qwen3-vl:4b-instruct",
                )
            ),
            dino_model=str(
                environ.get(
                    "RACESORT_DINO_MODEL",
                    "facebook/dinov2-small",
                )
            ),
            detection_threshold=float_setting(
                environ,
                "RACESORT_DETECTION_THRESHOLD",
                0.70,
            ),
            max_crop_size=int_setting(
                environ,
                "RACESORT_MAX_CROP_SIZE",
                1500,
            ),
            enable_merged_box_split=parse_bool(
                environ.get("RACESORT_ENABLE_MERGED_BOX_SPLIT", "0"),
                "RACESORT_ENABLE_MERGED_BOX_SPLIT",
            ),
            merged_box_child_threshold=float_setting(
                environ,
                "RACESORT_MERGED_BOX_CHILD_THRESHOLD",
                0.275,
            ),
            merged_box_criteria=MergedBoxCriteria(
                minimum_child_containment=float_setting(
                    environ,
                    "RACESORT_MERGED_BOX_MIN_CHILD_CONTAINMENT",
                    0.80,
                ),
                minimum_child_area_ratio=float_setting(
                    environ,
                    "RACESORT_MERGED_BOX_MIN_CHILD_AREA_RATIO",
                    0.12,
                ),
                maximum_child_area_ratio=float_setting(
                    environ,
                    "RACESORT_MERGED_BOX_MAX_CHILD_AREA_RATIO",
                    0.80,
                ),
                maximum_child_iou=float_setting(
                    environ,
                    "RACESORT_MERGED_BOX_MAX_CHILD_IOU",
                    0.55,
                ),
                minimum_child_area_balance=float_setting(
                    environ,
                    "RACESORT_MERGED_BOX_MIN_CHILD_AREA_BALANCE",
                    0.50,
                ),
                minimum_horizontal_separation=float_setting(
                    environ,
                    "RACESORT_MERGED_BOX_MIN_HORIZONTAL_SEPARATION",
                    0.33,
                ),
            ),
            max_filter_area=float_setting(
                environ,
                "RACESORT_MAX_FILTER_AREA",
                0.20,
            ),
            max_filter_relative_sharpness=float_setting(
                environ,
                "RACESORT_MAX_FILTER_RELATIVE_SHARPNESS",
                0.45,
            ),
            max_blur_sharpness=float_setting(
                environ,
                "RACESORT_MAX_BLUR_SHARPNESS",
                150.0,
            ),
            dino_corroboration_threshold=float_setting(
                environ,
                "RACESORT_DINO_CORROBORATION_THRESHOLD",
                0.90,
            ),
            race_type=race_type,
            event_id=optional_text(environ, "RACESORT_EVENT_ID"),
            event_date=event_date,
            group=group,
            cycle=cycle,
            session_id=optional_text(environ, "RACESORT_SESSION_ID"),
        )
        config.validate()
        return config

    def validate(self):
        unit_interval = {
            "RACESORT_DETECTION_THRESHOLD": self.detection_threshold,
            "RACESORT_MERGED_BOX_CHILD_THRESHOLD": (
                self.merged_box_child_threshold
            ),
            "RACESORT_MAX_FILTER_AREA": self.max_filter_area,
            "RACESORT_MAX_FILTER_RELATIVE_SHARPNESS": (
                self.max_filter_relative_sharpness
            ),
            "RACESORT_DINO_CORROBORATION_THRESHOLD": (
                self.dino_corroboration_threshold
            ),
            "RACESORT_MERGED_BOX_MIN_CHILD_CONTAINMENT": (
                self.merged_box_criteria.minimum_child_containment
            ),
            "RACESORT_MERGED_BOX_MIN_CHILD_AREA_RATIO": (
                self.merged_box_criteria.minimum_child_area_ratio
            ),
            "RACESORT_MERGED_BOX_MAX_CHILD_AREA_RATIO": (
                self.merged_box_criteria.maximum_child_area_ratio
            ),
            "RACESORT_MERGED_BOX_MAX_CHILD_IOU": (
                self.merged_box_criteria.maximum_child_iou
            ),
            "RACESORT_MERGED_BOX_MIN_CHILD_AREA_BALANCE": (
                self.merged_box_criteria.minimum_child_area_balance
            ),
            "RACESORT_MERGED_BOX_MIN_HORIZONTAL_SEPARATION": (
                self.merged_box_criteria.minimum_horizontal_separation
            ),
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_crop_size <= 0:
            raise ValueError("RACESORT_MAX_CROP_SIZE must be positive")
        if (
            self.merged_box_criteria.minimum_child_area_ratio
            > self.merged_box_criteria.maximum_child_area_ratio
        ):
            raise ValueError(
                "merged-box minimum child area ratio must not exceed maximum"
            )
        if self.max_blur_sharpness < 0:
            raise ValueError(
                "RACESORT_MAX_BLUR_SHARPNESS must not be negative"
            )
        if self.qwen_cache_schema_version <= 0:
            raise ValueError(
                "RACESORT_QWEN_CACHE_SCHEMA_VERSION must be positive"
            )

    def event_context(self):
        """Return run context without mixing it into race-number strings."""

        return {
            "event_id": self.event_id,
            "event_date": self.event_date,
            "group": self.group,
            "cycle": self.cycle,
            "session_id": self.session_id,
        }
