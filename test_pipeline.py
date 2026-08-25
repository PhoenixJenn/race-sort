from pathlib import Path
import csv
import json
import re
import time

import ollama
import torch
from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageStat,
)
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("test-photos")
OUTPUT_DIR = Path("test-output")

DETECTOR_MODEL = "facebook/detr-resnet-50"
VISION_MODEL = "qwen3-vl:4b-instruct"

DETECTION_THRESHOLD = 0.70
MAX_CROP_SIZE = 1500

# Human-validated conservative quality filters.
MAX_FILTER_AREA = 0.20
MAX_FILTER_RELATIVE_SHARPNESS = 0.45
MAX_BLUR_SHARPNESS = 150.0

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
    f"DETR class: "
    f"{DETECTION_CLASS}"
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

total_vehicle_time = 0.0

total_vehicles_processed = 0

rich_profiles_generated = 0

fast_confirmed_count = 0

review_count_total = 0

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
                DETECTION_THRESHOLD,
        )[0]
    )


    vehicles = []

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

            vehicles.append(
                {
                    "score":
                        score.item(),

                    "box":
                        box.tolist(),
                }
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

            vehicle_results.append(
                {
                    "vehicle": index,
                    "detr_confidence": vehicle["score"],
                    "profile_number": None,
                    "verification_number": None,
                    "final_number": None,
                    "decision": decision,
                    "crop": crop_path.name,
                    "profile": None,
                    "profile_type": "not_generated",
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
        # FAST PASS B FIRST
        # ====================================================

        (
            number_b,
            time_b,
            raw_b,
        ) = read_fast_number(
            crop_path,
            NUMBER_PROMPT_B,
        )

        total_fast_b_time += (
            time_b
        )


        # ====================================================
        # FAST PASS A SECOND
        # ====================================================

        (
            number_a,
            time_a,
            raw_a,
        ) = read_fast_number(
            crop_path,
            NUMBER_PROMPT_A,
        )

        total_fast_a_time += (
            time_a
        )


        print(
            f"Fast B: "
            f"{number_b or 'UNKNOWN'} "
            f"({time_b:.2f}s)"
        )

        print(
            f"Fast A: "
            f"{number_a or 'UNKNOWN'} "
            f"({time_a:.2f}s)"
        )


        # ====================================================
        # FAST CONFIRMATION DECISION
        # ====================================================

        if (
            number_a is not None
            and number_b is not None
            and number_a == number_b
        ):

            final_number = (
                number_a
            )

            decision = (
                "CONFIRMED"
            )

            fast_confirmed_count += 1


            print(
                f"Fast decision: "
                f"CONFIRMED "
                f"#{final_number}"
            )


            # ------------------------------------------------
            # Rich profile policy
            #
            # Generate full metadata only for the first
            # confirmed sighting of this number in this batch.
            # ------------------------------------------------

            if (
                final_number
                not in rich_profiled_numbers
            ):

                print(
                    "Generating rich profile "
                    "for first confirmed sighting..."
                )

                (
                    profile,
                    raw_profile,
                    profile_elapsed,
                ) = ask_qwen_for_profile(
                    crop_path
                )

                total_profile_time += (
                    profile_elapsed
                )

                rich_profiles_generated += 1

                rich_profiled_numbers.add(
                    final_number
                )

                profile_type = (
                    "rich_confirmed"
                )


            else:

                profile = (
                    create_minimal_confirmed_profile(
                        final_number
                    )
                )

                raw_profile = None

                profile_elapsed = 0.0

                profile_type = (
                    "minimal_confirmed"
                )


        # ====================================================
        # REVIEW CASE
        # ====================================================

        else:

            final_number = None

            decision = "REVIEW"

            review_count_total += 1


            print(
                "Fast decision: REVIEW"
            )

            print(
                "Generating rich profile "
                "for second-pass matching..."
            )


            (
                profile,
                raw_profile,
                profile_elapsed,
            ) = ask_qwen_for_profile(
                crop_path
            )

            total_profile_time += (
                profile_elapsed
            )

            rich_profiles_generated += 1

            profile_type = (
                "rich_review"
            )


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


        if final_number is not None:

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
                f"  Fast Pass A: "
                f"{result['profile_number'] or 'UNKNOWN'}\n"
            )

            file.write(
                f"  Fast Pass B: "
                f"{result['verification_number'] or 'UNKNOWN'}\n"
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
                f"  Fast B time: "
                f"{result['timing']['fast_pass_b_seconds']:.2f}s\n"
            )

            file.write(
                f"  Fast A time: "
                f"{result['timing']['fast_pass_a_seconds']:.2f}s\n"
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
        == "REVIEW"
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
    f"Fast-confirmed vehicles: "
    f"{fast_confirmed_count}"
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
    f"Rich profiles generated: "
    f"{rich_profiles_generated}"
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
    f"Fast Pass B total: "
    f"{total_fast_b_time:.2f}s"
)

print(
    f"Fast Pass A total: "
    f"{total_fast_a_time:.2f}s"
)

print(
    f"Rich profile total: "
    f"{total_profile_time:.2f}s"
)


if total_vehicles_processed > 0:

    print()

    print(
        f"Average Fast Pass B: "
        f"{total_fast_b_time / total_vehicles_processed:.2f}s"
    )

    print(
        f"Average Fast Pass A: "
        f"{total_fast_a_time / total_vehicles_processed:.2f}s"
    )

    print(
        f"Average fast two-pass: "
        f"{(total_fast_a_time + total_fast_b_time) / total_vehicles_processed:.2f}s"
    )

    print(
        f"Average vehicle total: "
        f"{total_vehicle_time / total_vehicles_processed:.2f}s"
    )


if len(image_paths) > 0:

    average_photo_time = (
        batch_elapsed
        / len(image_paths)
    )

    projected_minutes = (
        average_photo_time
        * 1000
        / 60
    )

    print()

    print(
        f"Average seconds/photo: "
        f"{average_photo_time:.2f}"
    )

    print(
        f"Projected 1,000-photo time: "
        f"{projected_minutes:.1f} minutes"
    )


print()

print(
    f"Batch CSV: "
    f"{csv_path}"
)
