from pathlib import Path
import csv
import json
import os
import re
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import ollama
import torch
import torch.nn.functional as F
from rapidocr import RapidOCR
from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageStat,
)
from transformers import (
    AutoImageProcessor,
    AutoModel,
    DetrImageProcessor,
    DetrForObjectDetection,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    os.environ.get(
        "RACESORT_INPUT_DIR",
        "test-photos",
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "RACESORT_OUTPUT_DIR",
        "test-output",
    )
)

DETECTOR_MODEL = "facebook/detr-resnet-50"
VISION_MODEL = "qwen3-vl:4b-instruct"
DINO_MODEL = "facebook/dinov2-small"

DETECTION_THRESHOLD = 0.70
MAX_CROP_SIZE = 1500

# Experimental, opt-in recovery for rare DETR boxes that contain two
# distinct motorcycles. The normal 0.70 behavior remains the default.
ENABLE_MERGED_BOX_SPLIT = (
    os.environ.get(
        "RACESORT_ENABLE_MERGED_BOX_SPLIT",
        "0",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)
MERGED_BOX_CHILD_THRESHOLD = 0.275
MERGED_BOX_MIN_CHILD_CONTAINMENT = 0.80
MERGED_BOX_MIN_CHILD_AREA_RATIO = 0.12
MERGED_BOX_MAX_CHILD_AREA_RATIO = 0.80
MERGED_BOX_MAX_CHILD_IOU = 0.55
MERGED_BOX_MIN_CHILD_AREA_BALANCE = 0.50
MERGED_BOX_MIN_HORIZONTAL_SEPARATION = 0.33

# Human-validated conservative quality filters.
MAX_FILTER_AREA = 0.20
MAX_FILTER_RELATIVE_SHARPNESS = 0.45
MAX_BLUR_SHARPNESS = 150.0
DINO_CORROBORATION_THRESHOLD = 0.90

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ------------------------------------------------------------
# EVENT TYPE
#
# Set this before processing an event.
#
# Allowed:
#   "motorcycle"
#   "car"
# ------------------------------------------------------------

RACE_TYPE = "motorcycle"

if RACE_TYPE not in {
    "motorcycle",
    "car",
}:
    raise ValueError(
        "RACE_TYPE must be either "
        "'motorcycle' or 'car'"
    )


# ============================================================
# FAST NUMBER PROMPTS
# ============================================================

NUMBER_PROMPT_A = """
Inspect this race vehicle specifically for its race number.

Return ONLY one of these:

1. The exact race number as visibly written.
2. UNKNOWN

Rules:

- Read the race number character by character.
- Race numbers are identifiers, not quantities.
- Preserve leading zeros exactly.
- Race numbers may contain digits and letters.
- Preserve the visible character order exactly.
- If any character is ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- If no number is clearly visible, return UNKNOWN.
- Never guess.
"""


NUMBER_PROMPT_B = """
Look only for a clearly visible race number on this vehicle.

Return only the exact identifier you can actually read.

Examples of valid identifiers:

54
007
54A
A12

Rules:

- Preserve leading zeros exactly.
- Treat the race number as an identifier, not a quantity.
- Return UNKNOWN if you cannot confidently distinguish
  every visible character.
- Do not infer from colors, vehicle type, rider,
  sponsors, graphics, or context.
- Do not guess.
"""


DIRECT_NUMBER_PROMPT = """
Inspect this race vehicle specifically for its race number.

Return ONLY one of these:

1. The exact race number as visibly written.
2. UNKNOWN

Rules:

- Read the race number character by character.
- Preserve leading zeros exactly.
- Race numbers are identifiers, not quantities.
- Race numbers may contain digits and letters.
- Preserve the visible character order exactly.
- If any character is ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- If no number is clearly visible, return UNKNOWN.
- Ignore sponsor names, logos, decals, and unrelated text.
- Never guess.
"""


# ============================================================
# RICH PROFILE PROMPTS
# ============================================================

MOTORCYCLE_PROFILE_PROMPT = """
Analyze this race motorcycle.

Return valid JSON only.
Do not include markdown, commentary, or explanation.

Report only information visually supported by the image.

Use exactly this JSON structure:

{
  "race_number": {
    "value": null,
    "status": "unreadable"
  },
  "make": {
    "value": null
  },
  "colors": {
    "primary": []
  },
  "rider": {
    "leathers_colors": [],
    "helmet_colors": []
  },
  "number_plate": {
    "color": null,
    "visible": false,
    "appears_blank": false
  }
}

Rules:

- race_number.value:
  - Return the race number exactly as visibly written.
  - Return it as a string only if clearly readable.
  - Race numbers are identifiers, not quantities.
  - Preserve leading zeros exactly.
    Example: 007 must be returned as "007", not "7".
  - Race numbers may contain digits and may also contain letters.
    Examples: "54", "007", "54A", "A12".
  - Preserve visible character order exactly.
  - Otherwise return null.
  - Never guess.

- race_number.status must be one of:
  - "readable"
  - "unreadable"
  - "not_visible"
  - "blank"

- make.value:
  - Identify the motorcycle manufacturer only when supported
    by a visible logo, badge, or readable manufacturer name.
  - Otherwise return null.
  - Do not infer make from colors or styling.

- colors.primary:
  - List the main visually distinctive colors
    of the motorcycle.

- rider.leathers_colors:
  - List the primary colors of the rider's leathers.

- rider.helmet_colors:
  - List the primary distinctive colors
    of the rider's helmet.

- number_plate.color:
  - Give the primary plate/number-area color if visible.
  - Otherwise return null.

- number_plate.visible:
  - true only if the race-number area is visibly present.

- number_plate.appears_blank:
  - true only if the intended number area is visible
    but no race number can actually be seen.

When uncertain, prefer null, false, an empty array,
or "unreadable" rather than guessing.
"""


CAR_PROFILE_PROMPT = """
Analyze this race car.

Return valid JSON only.
Do not include markdown, commentary, or explanation.

Report only information visually supported by the image.

Use exactly this JSON structure:

{
  "race_number": {
    "value": null,
    "status": "unreadable"
  },
  "make": {
    "value": null
  },
  "model": {
    "value": null
  },
  "colors": {
    "primary": []
  },
  "number_plate": {
    "color": null,
    "visible": false,
    "appears_blank": false
  }
}

Rules:

- race_number.value:
  - Return the race number exactly as visibly written.
  - Return it as a string only if clearly readable.
  - Race numbers are identifiers, not quantities.
  - Preserve leading zeros exactly.
  - Race numbers may contain digits and letters.
    Examples: "54", "007", "54A", "A12".
  - Preserve visible character order exactly.
  - Otherwise return null.
  - Never guess.

- race_number.status must be one of:
  - "readable"
  - "unreadable"
  - "not_visible"
  - "blank"

- make.value:
  - Identify manufacturer only when visually supported.
  - Otherwise return null.

- model.value:
  - Identify model only when visually supported.
  - Otherwise return null.

- colors.primary:
  - List the main visually distinctive vehicle colors.

- number_plate.color:
  - Give the primary race-number-area color if visible.
  - Otherwise return null.

- number_plate.visible:
  - true only if a race-number area is visible.

- number_plate.appears_blank:
  - true only if the intended number area is visible
    but no number can actually be seen.

When uncertain, prefer null, false, an empty array,
or "unreadable" rather than guessing.
"""


# ============================================================
# HELPERS
# ============================================================

def is_source_photo(path):
    """
    Return True only for original source images.
    """

    if not path.is_file():
        return False

    if (
        path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):
        return False

    lower_name = path.name.lower()

    if "-small" in lower_name:
        return False

    if "-detected" in lower_name:
        return False

    return True


def box_area(box):
    """Return the non-negative area of one DETR box."""

    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def box_intersection_area(first, second):
    """Return the shared area of two DETR boxes."""

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
    """Return intersection-over-union for two DETR boxes."""

    intersection = box_intersection_area(first, second)
    union = box_area(first) + box_area(second) - intersection
    if union == 0:
        return 0.0

    return intersection / union


def find_merged_box_children(parent, detections):
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
            MERGED_BOX_MIN_CHILD_AREA_RATIO
            <= area_ratio
            <= MERGED_BOX_MAX_CHILD_AREA_RATIO
            and box_containment(detection["box"], parent["box"])
            >= MERGED_BOX_MIN_CHILD_CONTAINMENT
        ):
            candidates.append(detection)

    valid_pairs = []

    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            first_area = box_area(first["box"])
            second_area = box_area(second["box"])
            area_balance = (
                min(first_area, second_area)
                / max(first_area, second_area)
            )
            first_center = (first["box"][0] + first["box"][2]) / 2
            second_center = (second["box"][0] + second["box"][2]) / 2
            horizontal_separation = (
                abs(first_center - second_center)
                / parent_width
            )

            if (
                box_iou(first["box"], second["box"])
                <= MERGED_BOX_MAX_CHILD_IOU
                and area_balance
                >= MERGED_BOX_MIN_CHILD_AREA_BALANCE
                and horizontal_separation
                >= MERGED_BOX_MIN_HORIZONTAL_SEPARATION
            ):
                valid_pairs.append(
                    (
                        first["score"] + second["score"],
                        first,
                        second,
                    )
                )

    if not valid_pairs:
        return None

    _, first, second = max(
        valid_pairs,
        key=lambda item: item[0],
    )
    return [first, second]


def resolve_merged_vehicle_boxes(detections):
    """Replace a strong merged parent with two validated child boxes."""

    baseline = [
        detection
        for detection in detections
        if detection["score"] >= DETECTION_THRESHOLD
    ]

    if (
        not ENABLE_MERGED_BOX_SPLIT
        or DETECTION_CLASS != "motorcycle"
    ):
        return baseline

    replacements = {}
    used_children = set()

    for parent in sorted(
        baseline,
        key=lambda item: item["score"],
        reverse=True,
    ):
        children = find_merged_box_children(parent, detections)

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


def normalize_number(value):
    """
    Race numbers are opaque string identifiers.

    Examples:

        "007"  -> "007"
        54     -> "54"
        "866"  -> "866"
        "54A"  -> "54A"
        "A12"  -> "A12"

    Leading zeros are preserved.
    """

    if value is None:
        return None

    value = (
        str(value)
        .strip()
        .upper()
    )

    if not value:
        return None

    if value == "UNKNOWN":
        return None

    if not re.fullmatch(
        r"[A-Z0-9]{1,6}",
        value,
    ):
        return None

    return value


def measure_sharpness(image_path):
    """
    Return the same edge-variance sharpness score used by
    the validated primary-filter experiment.
    """

    image = Image.open(
        image_path
    ).convert("L")

    image.thumbnail(
        (800, 800)
    )

    edges = image.filter(
        ImageFilter.FIND_EDGES
    )

    stats = ImageStat.Stat(
        edges
    )

    return stats.var[0]


def should_filter_non_primary(
    vehicles_in_photo,
    relative_area,
    relative_sharpness,
):
    """Apply the validated conservative non-primary rule."""

    return (
        vehicles_in_photo > 1
        and relative_area < MAX_FILTER_AREA
        and relative_sharpness
        < MAX_FILTER_RELATIVE_SHARPNESS
    )


def should_filter_too_blurry(sharpness):
    """Apply the validated conservative blur rule."""

    return sharpness < MAX_BLUR_SHARPNESS


def get_profile_prompt():
    if RACE_TYPE == "motorcycle":
        return MOTORCYCLE_PROFILE_PROMPT

    return CAR_PROFILE_PROMPT


def get_detection_class():
    if RACE_TYPE == "motorcycle":
        return "motorcycle"

    return "car"


def read_fast_number(
    crop_path,
    prompt,
):
    """
    Perform one fast number-only Qwen read.

    Returns:
        normalized_number
        elapsed_seconds
        raw_response
    """

    start = time.perf_counter()

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [
                    str(crop_path)
                ],
            }
        ],
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    raw_response = (
        response["message"]["content"]
        .strip()
    )

    number = normalize_number(
        raw_response
    )

    return (
        number,
        elapsed,
        raw_response,
    )


def extract_ocr_candidates(result):
    """Extract unique plausible race-number candidates."""

    candidates = []

    if result is None:
        return candidates

    texts = getattr(
        result,
        "txts",
        None,
    )

    if not texts:
        return candidates

    for text in texts:

        compact = (
            str(text)
            .strip()
            .upper()
            .replace(" ", "")
        )

        candidate = normalize_number(
            compact
        )

        if candidate is None:
            continue

        if not any(
            character.isdigit()
            for character in candidate
        ):
            continue

        candidates.append(candidate)

    return list(
        dict.fromkeys(candidates)
    )


def verify_ocr_candidates(
    crop_path,
    candidates,
):
    """Ask Qwen to verify exactly one OCR candidate."""

    candidate_text = ", ".join(candidates)

    prompt = f"""
Inspect this race vehicle specifically for its race number.

OCR found these possible identifiers:

{candidate_text}

Return ONLY one of these:

1. One exact identifier from the OCR candidate list above.
2. UNKNOWN

Rules:

- Only choose a candidate if that exact identifier is clearly visible
  as the vehicle's race number.
- Preserve leading zeros exactly.
- Do not invent a new number.
- Do not return anything not present in the candidate list.
- Ignore sponsor names, logos, decals, and unrelated text.
- If more than one candidate seems plausible, return UNKNOWN.
- If the race number is blurry or ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- Never guess.
"""

    number, elapsed, raw = read_fast_number(
        crop_path,
        prompt,
    )

    if number not in candidates:
        number = None

    return number, elapsed, raw


def direct_qwen_read(crop_path):
    """Return a direct Qwen read as candidate evidence."""

    return read_fast_number(
        crop_path,
        DIRECT_NUMBER_PROMPT,
    )


def resolve_dino_device():
    """Choose CUDA, Apple MPS, or CPU without platform forks."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_dino_embedding(
    image_path,
    processor,
    model,
    device,
    cache,
):
    """Create and cache one normalized DINO embedding."""

    image_path = Path(image_path)
    cache_key = str(image_path.resolve())

    if cache_key in cache:
        return cache[cache_key]

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = F.normalize(
        outputs.last_hidden_state[:, 0, :],
        p=2,
        dim=1,
    ).cpu()

    cache[cache_key] = embedding

    return embedding


def ask_qwen_for_profile(
    crop_path,
):
    """
    Expensive rich metadata pass.

    Phase 3B intentionally uses this only when needed.
    """

    start = time.perf_counter()

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content":
                    get_profile_prompt(),
                "images": [
                    str(crop_path)
                ],
            }
        ],
        format="json",
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    raw_content = (
        response["message"]["content"]
        .strip()
    )

    try:
        profile = json.loads(
            raw_content
        )

    except json.JSONDecodeError:

        print(
            "WARNING: Qwen returned "
            "invalid profile JSON."
        )

        return (
            None,
            raw_content,
            elapsed,
        )

    return (
        profile,
        raw_content,
        elapsed,
    )


def create_minimal_confirmed_profile(
    race_number,
):
    """
    Create a schema-compatible profile when we do not
    run the expensive rich profile.

    IMPORTANT:
    This is NOT pretending we observed make/colors/etc.

    It records only the directly confirmed number.
    """

    if RACE_TYPE == "motorcycle":

        return {
            "race_number": {
                "value":
                    race_number,

                "status":
                    "readable",
            },

            "make": {
                "value":
                    None,
            },

            "colors": {
                "primary": [],
            },

            "rider": {
                "leathers_colors": [],
                "helmet_colors": [],
            },

            "number_plate": {
                "color":
                    None,

                # A race number was successfully read.
                "visible":
                    True,

                "appears_blank":
                    False,
            },

            "_racesort": {
                "profile_type":
                    "minimal_confirmed",

                "rich_profile_generated":
                    False,
            },
        }

    return {
        "race_number": {
            "value":
                race_number,

            "status":
                "readable",
        },

        "make": {
            "value":
                None,
        },

        "model": {
            "value":
                None,
        },

        "colors": {
            "primary": [],
        },

        "number_plate": {
            "color":
                None,

            "visible":
                True,

            "appears_blank":
                False,
        },

        "_racesort": {
            "profile_type":
                "minimal_confirmed",

            "rich_profile_generated":
                False,
        },
    }


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    exist_ok=True
)

print(
    "Loading RapidOCR..."
)

ocr_engine = RapidOCR()

print(
    "RapidOCR loaded."
)

print()

print(
    "Loading DETR model..."
)

processor = (
    DetrImageProcessor
    .from_pretrained(
        DETECTOR_MODEL
    )
)

detector = (
    DetrForObjectDetection
    .from_pretrained(
        DETECTOR_MODEL
    )
)

detector.eval()

print(
    "DETR loaded."
)

print()

DETECTION_CLASS = (
    get_detection_class()
)

print(
    f"Race type: "
    f"{RACE_TYPE}"
)

print(
    f"Input directory: "
    f"{INPUT_DIR}"
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)

print(
    f"DETR class: "
    f"{DETECTION_CLASS}"
)

print(
    "Merged-box recovery: "
    + ("enabled" if ENABLE_MERGED_BOX_SPLIT else "disabled")
)

print()


# ============================================================
# FIND PHOTOS
# ============================================================

image_paths = sorted(
    path
    for path
    in INPUT_DIR.iterdir()
    if is_source_photo(path)
)

print(
    f"Found {len(image_paths)} "
    f"test photos."
)

print()


# ============================================================
# BATCH RESULTS
# ============================================================

batch_rows = []


# ============================================================
# PERFORMANCE METRICS
# ============================================================

batch_start_time = (
    time.perf_counter()
)

total_detr_time = 0.0

total_fast_a_time = 0.0
total_fast_b_time = 0.0

total_profile_time = 0.0

total_ocr_time = 0.0
total_qwen_verify_time = 0.0
total_qwen_direct_time = 0.0

total_vehicle_time = 0.0

total_vehicles_processed = 0

rich_profiles_generated = 0

fast_confirmed_count = 0

review_count_total = 0

ocr_candidate_count = 0
ocr_empty_count = 0
qwen_verify_calls = 0
qwen_direct_calls = 0
qwen_candidate_count = 0

filtered_non_primary_count = 0
filtered_too_blurry_count = 0


# ============================================================
# PROFILE TRACKING
#
# We generate a rich profile for the first confirmed
# sighting of each race number.
#
# Repeated confirmed sightings get a minimal profile.
#
# REVIEW vehicles always get a rich profile.
# ============================================================

rich_profiled_numbers = set()


# ============================================================
# PROCESS PHOTOS
# ============================================================

for photo_number, image_path in enumerate(
    image_paths,
    start=1,
):

    photo_start_time = (
        time.perf_counter()
    )

    print("=" * 70)

    print(
        f"[{photo_number}/"
        f"{len(image_paths)}] "
        f"Processing "
        f"{image_path.name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    photo_output_dir = (
        OUTPUT_DIR
        / image_path.stem
    )

    photo_output_dir.mkdir(
        exist_ok=True
    )


    # --------------------------------------------------------
    # Open original image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # ========================================================
    # DETECT VEHICLES
    # ========================================================

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    detr_start = (
        time.perf_counter()
    )

    with torch.no_grad():
        outputs = detector(
            **inputs
        )

    detr_elapsed = (
        time.perf_counter()
        - detr_start
    )

    total_detr_time += (
        detr_elapsed
    )


    target_sizes = torch.tensor(
        [
            image.size[::-1]
        ]
    )

    results = (
        processor
        .post_process_object_detection(
            outputs,
            target_sizes=
                target_sizes,
            threshold=
                (
                    MERGED_BOX_CHILD_THRESHOLD
                    if ENABLE_MERGED_BOX_SPLIT
                    else DETECTION_THRESHOLD
                ),
        )[0]
    )


    detection_candidates = []

    for (
        score,
        label,
        box,
    ) in zip(
        results["scores"],
        results["labels"],
        results["boxes"],
    ):

        class_name = (
            detector
            .config
            .id2label[
                label.item()
            ]
        )

        if (
            class_name
            == DETECTION_CLASS
        ):

            detection_candidates.append(
                {
                    "score":
                        score.item(),

                    "box":
                        box.tolist(),

                    "detection_source":
                        "baseline",
                }
            )


    vehicles = resolve_merged_vehicle_boxes(
        detection_candidates
    )


    # Left-to-right ordering.
    vehicles.sort(
        key=lambda item:
            item["box"][0]
    )


    print(
        f"Vehicles detected: "
        f"{len(vehicles)}"
    )

    print(
        f"DETR time: "
        f"{detr_elapsed:.2f}s"
    )


    # ========================================================
    # DEBUG IMAGE
    # ========================================================

    debug_image = image.copy()

    draw = ImageDraw.Draw(
        debug_image
    )

    for index, vehicle in enumerate(
        vehicles,
        start=1,
    ):

        (
            x1,
            y1,
            x2,
            y2,
        ) = vehicle["box"]

        draw.rectangle(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            width=8,
        )

        draw.text(
            (
                x1 + 10,
                y1 + 10,
            ),
            f"Vehicle {index}",
        )


    detected_path = (
        photo_output_dir
        / "detected.jpg"
    )

    debug_image.save(
        detected_path,
        quality=95,
    )


    # ========================================================
    # ANALYZE VEHICLES
    # ========================================================

    photo_numbers = []

    vehicle_results = []


    # ========================================================
    # CREATE AND MEASURE ALL CROPS
    #
    # Relative measurements require every vehicle crop in the
    # photo to be measured before routing begins.
    # ========================================================

    for index, vehicle in enumerate(
        vehicles,
        start=1,
    ):

        (
            x1,
            y1,
            x2,
            y2,
        ) = vehicle["box"]

        crop = image.crop(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )
        )

        crop.thumbnail(
            (
                MAX_CROP_SIZE,
                MAX_CROP_SIZE,
            )
        )

        crop_path = (
            photo_output_dir
            / f"{RACE_TYPE}-{index:02d}.jpg"
        )

        crop.save(
            crop_path,
            quality=95,
        )

        width, height = crop.size

        vehicle["crop_path"] = crop_path
        vehicle["pixel_area"] = width * height
        vehicle["sharpness"] = measure_sharpness(
            crop_path
        )


    largest_area = max(
        (
            vehicle["pixel_area"]
            for vehicle in vehicles
        ),
        default=0,
    )

    sharpest_score = max(
        (
            vehicle["sharpness"]
            for vehicle in vehicles
        ),
        default=0,
    )


    for vehicle in vehicles:

        vehicle["relative_area"] = (
            vehicle["pixel_area"] / largest_area
            if largest_area > 0
            else 0
        )

        vehicle["relative_sharpness"] = (
            vehicle["sharpness"] / sharpest_score
            if sharpest_score > 0
            else 0
        )


    for index, vehicle in enumerate(
        vehicles,
        start=1,
    ):

        vehicle_start = (
            time.perf_counter()
        )


        crop_path = vehicle["crop_path"]


        print()

        print(
            f"Vehicle {index}: "
            f"DETR confidence="
            f"{vehicle['score']:.2f}"
        )

        print(
            f"Quality: relative area="
            f"{vehicle['relative_area']:.2f}, "
            f"sharpness={vehicle['sharpness']:.2f}, "
            f"relative sharpness="
            f"{vehicle['relative_sharpness']:.2f}"
        )


        # ====================================================
        # VALIDATED QUALITY FILTERS
        # ====================================================

        if should_filter_non_primary(
            len(vehicles),
            vehicle["relative_area"],
            vehicle["relative_sharpness"],
        ):

            decision = "FILTERED_NON_PRIMARY"
            filtered_non_primary_count += 1

        elif should_filter_too_blurry(
            vehicle["sharpness"]
        ):

            decision = "FILTERED_TOO_BLURRY"
            filtered_too_blurry_count += 1

        else:

            decision = None


        if decision is not None:

            print(
                f"Quality decision: {decision}"
            )

            profile_path = (
                photo_output_dir
                / f"{RACE_TYPE}-{index:02d}.json"
            )

            with open(
                profile_path,
                "w",
                encoding="utf-8",
            ) as json_file:

                json.dump(
                    {
                        "_racesort": {
                            "profile_type": "filtered",
                            "rich_profile_generated": False,
                            "filter_decision": decision,
                        }
                    },
                    json_file,
                    indent=2,
                )

            vehicle_results.append(
                {
                    "vehicle": index,
                    "detr_confidence": vehicle["score"],
                    "detection_source": vehicle[
                        "detection_source"
                    ],
                    "profile_number": None,
                    "verification_number": None,
                    "final_number": None,
                    "decision": decision,
                    "crop": crop_path.name,
                    "profile": profile_path.name,
                    "profile_type": "filtered",
                    "routing": {
                        "route": decision,
                        "ocr_candidates": [],
                        "qwen_verify_number": None,
                        "qwen_direct_number": None,
                        "qwen_verify_raw": None,
                        "qwen_direct_raw": None,
                    },
                    "quality": {
                        "pixel_area": vehicle["pixel_area"],
                        "relative_area": vehicle["relative_area"],
                        "sharpness": vehicle["sharpness"],
                        "relative_sharpness": (
                            vehicle["relative_sharpness"]
                        ),
                    },
                    "timing": {
                        "fast_pass_b_seconds": 0.0,
                        "fast_pass_a_seconds": 0.0,
                        "rich_profile_seconds": 0.0,
                    },
                }
            )

            vehicle_elapsed = (
                time.perf_counter()
                - vehicle_start
            )

            total_vehicle_time += vehicle_elapsed
            total_vehicles_processed += 1

            print(
                f"Vehicle total: "
                f"{vehicle_elapsed:.2f}s"
            )

            continue


        # ====================================================
        # RAPIDOCR ROUTING
        # ====================================================

        ocr_start = time.perf_counter()

        ocr_result = ocr_engine(
            str(crop_path)
        )

        ocr_elapsed = (
            time.perf_counter()
            - ocr_start
        )

        total_ocr_time += ocr_elapsed

        ocr_candidates = extract_ocr_candidates(
            ocr_result
        )

        qwen_verify_number = None
        qwen_direct_number = None
        qwen_verify_raw = None
        qwen_direct_raw = None
        verify_elapsed = 0.0
        direct_elapsed = 0.0


        if ocr_candidates:

            ocr_candidate_count += 1
            qwen_direct_calls += 1

            (
                qwen_direct_number,
                direct_elapsed,
                qwen_direct_raw,
            ) = direct_qwen_read(
                crop_path
            )

            total_qwen_direct_time += direct_elapsed


            # Anchored verification can only contribute to a
            # three-way confirmation when the independent
            # direct read already matches an OCR candidate.
            if (
                qwen_direct_number is not None
                and qwen_direct_number in ocr_candidates
            ):

                qwen_verify_calls += 1

                (
                    qwen_verify_number,
                    verify_elapsed,
                    qwen_verify_raw,
                ) = verify_ocr_candidates(
                    crop_path,
                    ocr_candidates,
                )

                total_qwen_verify_time += verify_elapsed


                if (
                    qwen_verify_number
                    == qwen_direct_number
                ):
                    final_number = qwen_direct_number
                    decision = "CONFIRMED"
                    route = "OCR_DIRECT_VERIFY_AGREE"
                    fast_confirmed_count += 1

                else:
                    final_number = qwen_direct_number
                    decision = "QWEN_CANDIDATE"
                    route = "OCR_DIRECT_VERIFY_REJECTED"
                    qwen_candidate_count += 1


            elif qwen_direct_number is not None:
                final_number = qwen_direct_number
                decision = "QWEN_CANDIDATE"
                route = "OCR_DIRECT_CONFLICT"
                qwen_candidate_count += 1


            else:
                final_number = None
                decision = "REVIEW"
                route = "OCR_DIRECT_UNKNOWN"
                review_count_total += 1


        else:

            ocr_empty_count += 1
            qwen_direct_calls += 1

            (
                qwen_direct_number,
                direct_elapsed,
                qwen_direct_raw,
            ) = direct_qwen_read(
                crop_path
            )

            total_qwen_direct_time += direct_elapsed

            if qwen_direct_number is not None:
                final_number = qwen_direct_number
                decision = "QWEN_CANDIDATE"
                route = "OCR_EMPTY_DIRECT_CANDIDATE"
                qwen_candidate_count += 1

            else:
                final_number = None
                decision = "REVIEW"
                route = "OCR_EMPTY_DIRECT_UNKNOWN"
                review_count_total += 1


        print(
            f"OCR candidates: {ocr_candidates} "
            f"({ocr_elapsed:.2f}s)"
        )

        print(
            f"Route: {route}"
        )

        print(
            f"Decision: {decision}"
        )


        # Keep the existing profile-file contract for older
        # registry utilities without making rich Qwen calls.
        if decision == "CONFIRMED":
            profile = create_minimal_confirmed_profile(
                final_number
            )
            profile_type = "minimal_confirmed"

        else:
            profile = {
                "_racesort": {
                    "profile_type": "routing_evidence",
                    "rich_profile_generated": False,
                }
            }
            profile_type = "routing_evidence"

        raw_profile = None
        profile_elapsed = 0.0

        # Compatibility aliases for existing output fields.
        number_a = qwen_direct_number
        number_b = qwen_verify_number
        raw_a = qwen_direct_raw
        raw_b = qwen_verify_raw
        time_a = direct_elapsed
        time_b = verify_elapsed


        # ====================================================
        # SAVE PROFILE
        # ====================================================

        profile_path = (
            photo_output_dir
            / f"{RACE_TYPE}-{index:02d}.json"
        )


        if profile is not None:

            # Preserve internal RaceSort provenance.
            if "_racesort" not in profile:
                profile["_racesort"] = {}

            profile["_racesort"][
                "profile_type"
            ] = profile_type

            profile["_racesort"][
                "rich_profile_generated"
            ] = (
                profile_elapsed > 0
            )


            with open(
                profile_path,
                "w",
                encoding="utf-8",
            ) as json_file:

                json.dump(
                    profile,
                    json_file,
                    indent=2,
                )


        else:

            with open(
                profile_path,
                "w",
                encoding="utf-8",
            ) as json_file:

                json_file.write(
                    raw_profile
                    or "{}"
                )


        # ====================================================
        # STORE RESULT
        #
        # Keep familiar field names so existing downstream
        # scripts remain as compatible as possible.
        #
        # profile_number = Fast Pass A
        # verification_number = Fast Pass B
        # ====================================================

        vehicle_result = {
            "vehicle":
                index,

            "detr_confidence":
                vehicle["score"],

            "detection_source":
                vehicle["detection_source"],

            "profile_number":
                number_a,

            "verification_number":
                number_b,

            "final_number":
                final_number,

            "decision":
                decision,

            "crop":
                crop_path.name,

            "profile":
                profile_path.name,

            "profile_type":
                profile_type,

            "routing": {
                "route": route,
                "ocr_candidates": ocr_candidates,
                "qwen_verify_number": qwen_verify_number,
                "qwen_direct_number": qwen_direct_number,
                "qwen_verify_raw": qwen_verify_raw,
                "qwen_direct_raw": qwen_direct_raw,
            },

            "quality": {
                "pixel_area":
                    vehicle["pixel_area"],

                "relative_area":
                    vehicle["relative_area"],

                "sharpness":
                    vehicle["sharpness"],

                "relative_sharpness":
                    vehicle["relative_sharpness"],
            },

            "timing": {
                "ocr_seconds":
                    ocr_elapsed,

                "qwen_verify_seconds":
                    verify_elapsed,

                "qwen_direct_seconds":
                    direct_elapsed,

                "fast_pass_b_seconds":
                    time_b,

                "fast_pass_a_seconds":
                    time_a,

                "rich_profile_seconds":
                    profile_elapsed,
            },
        }


        vehicle_results.append(
            vehicle_result
        )


        if (
            decision == "CONFIRMED"
            and final_number is not None
        ):

            photo_numbers.append(
                final_number
            )


        # ====================================================
        # VEHICLE TIMING
        # ====================================================

        vehicle_elapsed = (
            time.perf_counter()
            - vehicle_start
        )

        total_vehicle_time += (
            vehicle_elapsed
        )

        total_vehicles_processed += 1


        print(
            f"Vehicle total: "
            f"{vehicle_elapsed:.2f}s"
        )


    # ========================================================
    # REMOVE DUPLICATE PHOTO NUMBERS
    # ========================================================

    photo_numbers = list(
        dict.fromkeys(
            photo_numbers
        )
    )


    # ========================================================
    # SAVE STRUCTURED PHOTO RESULTS
    # ========================================================

    structured_result = {
        "photo":
            image_path.name,

        "race_type":
            RACE_TYPE,

        "vehicles_detected":
            len(vehicles),

        "confirmed_photo_numbers":
            photo_numbers,

        "vehicles":
            vehicle_results,
    }


    structured_results_path = (
        photo_output_dir
        / "photo-results.json"
    )


    with open(
        structured_results_path,
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            structured_result,
            json_file,
            indent=2,
        )


    print(
        f"Saved structured result: "
        f"{structured_results_path}"
    )


    # ========================================================
    # HUMAN-READABLE RESULTS
    # ========================================================

    results_path = (
        photo_output_dir
        / "results.txt"
    )


    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Photo: "
            f"{image_path.name}\n"
        )

        file.write(
            f"Race type: "
            f"{RACE_TYPE}\n"
        )

        file.write(
            f"Vehicles detected: "
            f"{len(vehicles)}\n\n"
        )


        for result in vehicle_results:

            file.write(
                f"Vehicle "
                f"{result['vehicle']}\n"
            )

            file.write(
                f"  DETR confidence: "
                f"{result['detr_confidence']:.2f}\n"
            )

            file.write(
                f"  Route: "
                f"{result['routing']['route']}\n"
            )

            file.write(
                f"  OCR candidates: "
                f"{result['routing']['ocr_candidates']}\n"
            )

            file.write(
                f"  Qwen verify: "
                f"{result['routing']['qwen_verify_number'] or 'UNKNOWN'}\n"
            )

            file.write(
                f"  Qwen direct: "
                f"{result['routing']['qwen_direct_number'] or 'UNKNOWN'}\n"
            )

            file.write(
                f"  Final number: "
                f"{result['final_number'] or 'UNKNOWN'}\n"
            )

            file.write(
                f"  Decision: "
                f"{result['decision']}\n"
            )

            file.write(
                f"  Profile type: "
                f"{result['profile_type']}\n"
            )

            file.write(
                f"  Crop: "
                f"{result['crop']}\n"
            )

            file.write(
                f"  Profile: "
                f"{result['profile']}\n"
            )

            file.write(
                f"  OCR time: "
                f"{result['timing'].get('ocr_seconds', 0.0):.2f}s\n"
            )

            file.write(
                f"  Qwen verify time: "
                f"{result['timing'].get('qwen_verify_seconds', 0.0):.2f}s\n"
            )

            file.write(
                f"  Qwen direct time: "
                f"{result['timing'].get('qwen_direct_seconds', 0.0):.2f}s\n"
            )

            file.write(
                f"  Rich profile time: "
                f"{result['timing']['rich_profile_seconds']:.2f}s\n\n"
            )


        file.write(
            "Confirmed photo race numbers: "
        )

        if photo_numbers:

            file.write(
                ", ".join(
                    photo_numbers
                )
            )

        else:

            file.write(
                "NONE"
            )

        file.write("\n")


    # ========================================================
    # PHOTO TIMING
    # ========================================================

    photo_elapsed = (
        time.perf_counter()
        - photo_start_time
    )


    photo_review_count = sum(
        1
        for result
        in vehicle_results
        if result["decision"]
        in {
            "QWEN_CANDIDATE",
            "REVIEW",
        }
    )


    batch_rows.append(
        {
            "photo":
                image_path.name,

            "vehicles_detected":
                len(vehicles),

            "confirmed_numbers":
                ", ".join(
                    photo_numbers
                ),

            "vehicles_for_review":
                photo_review_count,

            "photo_seconds":
                photo_elapsed,
        }
    )


    print()

    print(
        f"PHOTO RESULT: "
        f"{image_path.name} "
        f"→ {photo_numbers}"
    )

    print(
        f"Vehicles needing review: "
        f"{photo_review_count}"
    )

    print(
        f"Photo processing time: "
        f"{photo_elapsed:.2f}s"
    )

    print()


# ============================================================
# RESOLVE QWEN CANDIDATES WITH INDEPENDENT DINO EVIDENCE
# ============================================================

dino_start = time.perf_counter()

confirmed_references = {}
candidate_records = []
photo_result_records = []

for result_path in sorted(
    OUTPUT_DIR.glob("*/photo-results.json")
):

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as json_file:
        photo_result = json.load(json_file)

    photo_result_records.append(
        (result_path, photo_result)
    )

    for vehicle_result in photo_result["vehicles"]:

        crop_path = (
            result_path.parent
            / vehicle_result["crop"]
        )

        race_number = vehicle_result.get(
            "final_number"
        )

        if (
            vehicle_result["decision"] == "CONFIRMED"
            and race_number is not None
        ):

            confirmed_references.setdefault(
                race_number,
                [],
            ).append(crop_path)

        elif (
            vehicle_result["decision"]
            == "QWEN_CANDIDATE"
        ):

            candidate_records.append(
                (
                    result_path,
                    photo_result,
                    vehicle_result,
                    crop_path,
                )
            )


comparable_candidates = [
    record
    for record in candidate_records
    if confirmed_references.get(
        record[2].get("final_number")
    )
]

dino_processor = None
dino_model = None
dino_device = None
dino_embedding_cache = {}

if comparable_candidates:

    dino_device = resolve_dino_device()

    print(
        f"Loading DINOv2 on {dino_device}..."
    )

    dino_processor = (
        AutoImageProcessor.from_pretrained(
            DINO_MODEL,
            local_files_only=True,
        )
    )

    dino_model = AutoModel.from_pretrained(
        DINO_MODEL,
        local_files_only=True,
    )

    dino_model.to(dino_device)
    dino_model.eval()


dino_resolution_rows = []

for (
    result_path,
    photo_result,
    vehicle_result,
    candidate_crop_path,
) in candidate_records:

    candidate_number = vehicle_result[
        "final_number"
    ]

    reference_paths = []

    for reference_path in confirmed_references.get(
        candidate_number,
        [],
    ):

        # A crop can never corroborate itself.
        if (
            reference_path.resolve()
            == candidate_crop_path.resolve()
        ):
            continue

        reference_paths.append(reference_path)


    similarities = []

    if reference_paths:

        candidate_embedding = create_dino_embedding(
            candidate_crop_path,
            dino_processor,
            dino_model,
            dino_device,
            dino_embedding_cache,
        )

        for reference_path in reference_paths:

            reference_embedding = create_dino_embedding(
                reference_path,
                dino_processor,
                dino_model,
                dino_device,
                dino_embedding_cache,
            )

            score = F.cosine_similarity(
                candidate_embedding,
                reference_embedding,
            ).item()

            similarities.append(
                (score, reference_path)
            )

        similarities.sort(
            key=lambda item: item[0],
            reverse=True,
        )


    best_similarity = (
        similarities[0][0]
        if similarities
        else None
    )

    best_reference = (
        similarities[0][1]
        if similarities
        else None
    )


    if (
        best_similarity is not None
        and best_similarity
        >= DINO_CORROBORATION_THRESHOLD
    ):
        disposition = "CORROBORATED"
        reasons = [
            "KNOWN_CONFIRMED_NUMBER",
            "STRONG_INDEPENDENT_DINO_SUPPORT",
        ]

    elif reference_paths:
        disposition = "KNOWN_NUMBER_REVIEW"
        reasons = [
            "KNOWN_CONFIRMED_NUMBER",
            "DINO_BELOW_PROMOTION_THRESHOLD",
        ]

    else:
        disposition = "UNSUPPORTED"
        reasons = [
            "NO_INDEPENDENT_CONFIRMED_REFERENCE",
        ]


    vehicle_result["decision"] = disposition
    vehicle_result["candidate_resolution"] = {
        "candidate_number": candidate_number,
        "independent_reference_count": len(
            reference_paths
        ),
        "best_dino_similarity": best_similarity,
        "best_reference": (
            str(best_reference)
            if best_reference is not None
            else None
        ),
        "threshold": DINO_CORROBORATION_THRESHOLD,
        "disposition": disposition,
        "reasons": reasons,
    }


    if disposition == "CORROBORATED":

        confirmed_numbers = photo_result[
            "confirmed_photo_numbers"
        ]

        if candidate_number not in confirmed_numbers:
            confirmed_numbers.append(candidate_number)


    dino_resolution_rows.append(
        {
            "crop": str(candidate_crop_path),
            "candidate_number": candidate_number,
            "independent_reference_count": len(
                reference_paths
            ),
            "best_dino_similarity": best_similarity,
            "best_reference": (
                str(best_reference)
                if best_reference is not None
                else ""
            ),
            "disposition": disposition,
            "reasons": " | ".join(reasons),
        }
    )


for result_path, photo_result in photo_result_records:

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            photo_result,
            json_file,
            indent=2,
        )


dino_results_path = (
    OUTPUT_DIR
    / "dino-candidate-resolution-results.csv"
)

with open(
    dino_results_path,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "crop",
            "candidate_number",
            "independent_reference_count",
            "best_dino_similarity",
            "best_reference",
            "disposition",
            "reasons",
        ],
    )

    writer.writeheader()
    writer.writerows(dino_resolution_rows)


total_dino_time = (
    time.perf_counter()
    - dino_start
)


for batch_row in batch_rows:

    matching_photo = next(
        photo_result
        for _, photo_result
        in photo_result_records
        if photo_result["photo"]
        == batch_row["photo"]
    )

    batch_row["confirmed_numbers"] = ", ".join(
        matching_photo["confirmed_photo_numbers"]
    )

    batch_row["vehicles_for_review"] = sum(
        1
        for vehicle_result
        in matching_photo["vehicles"]
        if vehicle_result["decision"]
        in {
            "KNOWN_NUMBER_REVIEW",
            "CONFLICTING",
            "UNSUPPORTED",
            "REVIEW",
        }
    )


# ============================================================
# SAVE BATCH CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "batch-results.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "photo",
            "vehicles_detected",
            "confirmed_numbers",
            "vehicles_for_review",
            "photo_seconds",
        ],
    )

    writer.writeheader()

    writer.writerows(
        batch_rows
    )


# ============================================================
# FINAL PERFORMANCE SUMMARY
# ============================================================

batch_elapsed = (
    time.perf_counter()
    - batch_start_time
)

corroborated_count = sum(
    1
    for row in dino_resolution_rows
    if row["disposition"] == "CORROBORATED"
)

known_number_review_count = sum(
    1
    for row in dino_resolution_rows
    if row["disposition"] == "KNOWN_NUMBER_REVIEW"
)

conflicting_count = sum(
    1
    for row in dino_resolution_rows
    if row["disposition"] == "CONFLICTING"
)

unsupported_count = sum(
    1
    for row in dino_resolution_rows
    if row["disposition"] == "UNSUPPORTED"
)

total_human_review_workload = (
    review_count_total
    + known_number_review_count
    + conflicting_count
    + unsupported_count
)

average_vehicle_time = (
    total_vehicle_time / total_vehicles_processed
    if total_vehicles_processed > 0
    else 0.0
)

average_photo_time = (
    batch_elapsed / len(image_paths)
    if image_paths
    else 0.0
)

projected_1000_photo_minutes = (
    average_photo_time
    * 1000
    / 60
)


# ============================================================
# SAVE MACHINE-READABLE RUN SUMMARY
# ============================================================

run_summary = {
    "models": {
        "detector": DETECTOR_MODEL,
        "vision": VISION_MODEL,
        "dino": DINO_MODEL,
    },
    "thresholds": {
        "detection": DETECTION_THRESHOLD,
        "merged_box_split_enabled": ENABLE_MERGED_BOX_SPLIT,
        "merged_box_child_detection": (
            MERGED_BOX_CHILD_THRESHOLD
        ),
        "non_primary_max_relative_area": MAX_FILTER_AREA,
        "non_primary_max_relative_sharpness": (
            MAX_FILTER_RELATIVE_SHARPNESS
        ),
        "too_blurry_max_sharpness": MAX_BLUR_SHARPNESS,
        "dino_corroboration": DINO_CORROBORATION_THRESHOLD,
    },
    "counts": {
        "photos_processed": len(image_paths),
        "vehicles_processed": total_vehicles_processed,
        "confirmed": fast_confirmed_count,
        "qwen_candidates_routed": qwen_candidate_count,
        "corroborated": corroborated_count,
        "known_number_review": known_number_review_count,
        "conflicting": conflicting_count,
        "unsupported": unsupported_count,
        "filtered_non_primary": filtered_non_primary_count,
        "filtered_too_blurry": filtered_too_blurry_count,
        "review": review_count_total,
        "total_human_review_workload": (
            total_human_review_workload
        ),
        "ocr_candidate_cases": ocr_candidate_count,
        "ocr_empty_cases": ocr_empty_count,
        "qwen_verify_calls": qwen_verify_calls,
        "qwen_direct_calls": qwen_direct_calls,
    },
    "timing_seconds": {
        "detr_total": total_detr_time,
        "ocr_total": total_ocr_time,
        "qwen_verify_total": total_qwen_verify_time,
        "qwen_direct_total": total_qwen_direct_time,
        "dino_total": total_dino_time,
        "batch_total": batch_elapsed,
        "average_vehicle": average_vehicle_time,
        "average_photo": average_photo_time,
    },
    "projection": {
        "photos": 1000,
        "minutes": projected_1000_photo_minutes,
    },
}

run_summary_path = (
    OUTPUT_DIR
    / "run-summary.json"
)

with open(
    run_summary_path,
    "w",
    encoding="utf-8",
) as json_file:

    json.dump(
        run_summary,
        json_file,
        indent=2,
    )


print("=" * 70)
print("PHASE 3B BATCH COMPLETE")
print("=" * 70)

print(
    f"Photos processed: "
    f"{len(image_paths)}"
)

print(
    f"Vehicles processed: "
    f"{total_vehicles_processed}"
)

print()

print(
    f"CONFIRMED: "
    f"{fast_confirmed_count}"
)

print(
    f"Qwen candidates routed: "
    f"{qwen_candidate_count}"
)

print(
    f"CORROBORATED: "
    f"{corroborated_count}"
)

print(
    f"KNOWN_NUMBER_REVIEW: "
    f"{known_number_review_count}"
)

print(
    f"CONFLICTING: "
    f"{conflicting_count}"
)

print(
    f"UNSUPPORTED: "
    f"{unsupported_count}"
)

print(
    f"Filtered non-primary: "
    f"{filtered_non_primary_count}"
)

print(
    f"Filtered too blurry: "
    f"{filtered_too_blurry_count}"
)

print(
    f"Vehicles for REVIEW: "
    f"{review_count_total}"
)

print(
    f"Total human-review workload: "
    f"{total_human_review_workload}"
)

print()

print(
    f"OCR candidate cases: "
    f"{ocr_candidate_count}"
)

print(
    f"OCR empty cases: "
    f"{ocr_empty_count}"
)

print(
    f"Qwen VERIFY calls: "
    f"{qwen_verify_calls}"
)

print(
    f"Qwen DIRECT calls: "
    f"{qwen_direct_calls}"
)

print()

print(
    f"Total batch time: "
    f"{batch_elapsed:.2f}s"
)

print()

print(
    f"DETR total: "
    f"{total_detr_time:.2f}s"
)

print(
    f"OCR total: "
    f"{total_ocr_time:.2f}s"
)

print(
    f"Qwen VERIFY total: "
    f"{total_qwen_verify_time:.2f}s"
)

print(
    f"Qwen DIRECT total: "
    f"{total_qwen_direct_time:.2f}s"
)

print(
    f"DINO total: "
    f"{total_dino_time:.2f}s"
)


if total_vehicles_processed > 0:

    print()

    print(
        f"Average vehicle total: "
        f"{average_vehicle_time:.2f}s"
    )


if image_paths:

    print()

    print(
        f"Average seconds/photo: "
        f"{average_photo_time:.2f}"
    )

    print(
        f"Projected 1,000-photo time: "
        f"{projected_1000_photo_minutes:.1f} minutes"
    )


print()

print(
    f"Batch CSV: "
    f"{csv_path}"
)

print(
    f"DINO resolution CSV: "
    f"{dino_results_path}"
)

print(
    f"Run summary JSON: "
    f"{run_summary_path}"
)
