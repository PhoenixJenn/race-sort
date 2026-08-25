from pathlib import Path
import csv
import json
import re
import time

import ollama
import torch
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("test-photos")
OUTPUT_DIR = Path("test-output")

DETECTOR_MODEL = "facebook/detr-resnet-50"
VISION_MODEL = "qwen3-vl:4b-instruct"

DETECTION_THRESHOLD = 0.70
MAX_CROP_SIZE = 1500

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

if RACE_TYPE not in {"motorcycle", "car"}:
    raise ValueError(
        "RACE_TYPE must be either 'motorcycle' or 'car'"
    )


# ============================================================
# PROMPTS
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
  - Preserve the visible character order exactly.
  - Otherwise return null.
  - Never guess.

- race_number.status must be one of:
  - "readable"
  - "unreadable"
  - "not_visible"
  - "blank"

- make.value:
  - Identify the motorcycle manufacturer only when supported by a
    visible logo, badge, or readable manufacturer name.
  - Otherwise return null.
  - Do not infer make from colors or styling.

- colors.primary:
  - List the main visually distinctive colors of the motorcycle.

- rider.leathers_colors:
  - List the primary colors of the rider's leathers.

- rider.helmet_colors:
  - List the primary distinctive colors of the rider's helmet.

- number_plate.color:
  - Give the primary plate/number-area color if visible.
  - Otherwise return null.

- number_plate.visible:
  - true only if the race-number area is visibly present.

- number_plate.appears_blank:
  - true only if the intended number area is visible but no number
    can actually be seen.

When uncertain, prefer null, false, an empty array, or "unreadable"
rather than guessing.
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
    Example: 007 must be returned as "007", not "7".
  - Race numbers may contain digits and may also contain letters.
    Examples: "54", "007", "54A", "A12".
  - Preserve the visible character order exactly.
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
  - true only if the intended number area is visible but no number
    can actually be seen.

When uncertain, prefer null, false, an empty array, or "unreadable"
rather than guessing.
"""



NUMBER_VERIFICATION_PROMPT = """
Inspect this race vehicle specifically for its race number.

This is an independent verification pass.

Return ONLY one of these:

1. The exact race number as it visibly appears.
2. UNKNOWN

Rules:

- Read the actual visible race number character by character.
- Race numbers are identifiers, not quantities.
- Preserve leading zeros exactly.
  Example: if the number is 007, return 007, not 7.
- Race numbers may contain digits and may also contain letters.
  Examples: 54, 007, 54A, A12.
- Preserve the visible character order exactly.
- Stylized or unusual racing fonts may make characters ambiguous.
- Do not infer a number from the vehicle, rider, graphics, sponsors,
  colors, context, or what a race number would normally look like.
- If any character is ambiguous, return UNKNOWN.
- If the number area is blank, return UNKNOWN.
- If the number is not clearly visible, return UNKNOWN.
- Never guess.
"""

# ============================================================
# HELPERS
# ============================================================

def is_source_photo(path):
    """Return True only for original test images."""

    if not path.is_file():
        return False

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    lower_name = path.name.lower()

    if "-small" in lower_name:
        return False

    if "-detected" in lower_name:
        return False

    return True


# def normalize_number(value):
#     """
#     Convert a possible race number into a clean digit string.

#     Anything suspicious becomes None.
#     """

#     if value is None:
#         return None

#     value = str(value).strip()

#     if not re.fullmatch(r"\d{1,4}", value):
#         return None

#     return value


def normalize_number(value):
    """
    Normalize a race number while preserving it as an identifier.

    Examples:
        "007"  -> "007"
        54     -> "54"
        "866"  -> "866"
        "54A"  -> "54A"
        "A12"  -> "A12"
        " 49 " -> "49"

    Returns None for invalid or missing values.
    """

    if value is None:
        return None

    value = str(value).strip().upper()

    if not value:
        return None

    # Race numbers are identifiers, not quantities.
    # Preserve leading zeros and allow future alphanumeric IDs.
    #
    # Examples:
    #   007
    #   54
    #   54A
    #   A12
    #
    # Limit length so arbitrary model prose is rejected.
    if not re.fullmatch(
        r"[A-Z0-9]{1,6}",
        value,
    ):
        return None

    return value

def get_profile_prompt():
    if RACE_TYPE == "motorcycle":
        return MOTORCYCLE_PROFILE_PROMPT

    return CAR_PROFILE_PROMPT


def get_detection_class():
    if RACE_TYPE == "motorcycle":
        return "motorcycle"

    return "car"


def ask_qwen_for_profile(crop_path):
    """
    First AI pass.

    Ask Qwen for structured metadata about the vehicle.
    """

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": get_profile_prompt(),
                "images": [str(crop_path)],
            }
        ],
        format="json",
    )

    raw_content = response["message"]["content"].strip()

    try:
        profile = json.loads(raw_content)
    except json.JSONDecodeError:
        print("WARNING: Qwen returned invalid JSON.")
        return None, raw_content

    return profile, raw_content


def ask_qwen_to_verify_number(crop_path):
    """
    Second independent AI pass.

    This pass knows nothing about the number returned by the profile
    request. It simply tries to read the race number again.
    """

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": NUMBER_VERIFICATION_PROMPT,
                "images": [str(crop_path)],
            }
        ],
    )

    result = response["message"]["content"].strip()

    if result.upper() == "UNKNOWN":
        return None

    return normalize_number(result)


def validate_profile_number(profile):
    """
    Apply deterministic RaceSort rules to the JSON profile.

    This is important: Qwen is not allowed to overrule these rules.
    """

    if not profile:
        return None

    race_number = profile.get("race_number", {})
    number_plate = profile.get("number_plate", {})

    candidate = normalize_number(
        race_number.get("value")
    )

    status = race_number.get(
        "status",
        "unreadable",
    )

    plate_visible = number_plate.get(
        "visible",
        False,
    )

    appears_blank = number_plate.get(
        "appears_blank",
        False,
    )

    # Explicitly reject contradictory cases.
    if appears_blank:
        return None

    if status != "readable":
        return None

    if not plate_visible:
        return None

    return candidate


def decide_final_number(
    profile_number,
    verification_number,
):
    """
    RaceSort's first validation rule:

    BOTH independent passes must agree.

    Otherwise the vehicle goes to REVIEW.
    """

    if profile_number is None:
        return None, "REVIEW"

    if verification_number is None:
        return None, "REVIEW"

    if profile_number != verification_number:
        return None, "REVIEW"

    return profile_number, "CONFIRMED"


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading DETR model...")

processor = DetrImageProcessor.from_pretrained(
    DETECTOR_MODEL
)

detector = DetrForObjectDetection.from_pretrained(
    DETECTOR_MODEL
)

detector.eval()

print("DETR loaded.")
print()

DETECTION_CLASS = get_detection_class()

print(f"Race type: {RACE_TYPE}")
print(f"DETR class: {DETECTION_CLASS}")
print()


# ============================================================
# FIND TEST PHOTOS
# ============================================================

image_paths = sorted(
    path
    for path in INPUT_DIR.iterdir()
    if is_source_photo(path)
)

print(
    f"Found {len(image_paths)} test photos."
)
print()


# ============================================================
# BATCH RESULTS
# ============================================================

batch_rows = []





# ============================================================
# PERFORMANCE METRICS
# ============================================================

batch_start_time = time.perf_counter()

total_detr_time = 0.0
total_profile_time = 0.0
total_verification_time = 0.0
total_vehicle_time = 0.0

total_vehicles_processed = 0

photo_timings = []



# ============================================================
# PROCESS PHOTOS
# ============================================================

for photo_number, image_path in enumerate(
    image_paths,
    start=1,
):
    photo_start_time = time.perf_counter()

    print("=" * 60)

    print(
        f"[{photo_number}/{len(image_paths)}] "
        f"Processing {image_path.name}"
    )

    print("=" * 60)

    photo_output_dir = (
        OUTPUT_DIR / image_path.stem
    )

    photo_output_dir.mkdir(
        exist_ok=True
    )

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # DETECT VEHICLES
    # --------------------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    detr_start_time = time.perf_counter()

    with torch.no_grad():
        outputs = detector(**inputs)

    detr_elapsed = (
        time.perf_counter()
        - detr_start_time
    )

    total_detr_time += detr_elapsed

    target_sizes = torch.tensor(
        [image.size[::-1]]
    )

    results = (
        processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=DETECTION_THRESHOLD,
        )[0]
    )

    vehicles = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"],
    ):

        class_name = (
            detector.config.id2label[
                label.item()
            ]
        )

        if class_name == DETECTION_CLASS:
            vehicles.append(
                {
                    "score": score.item(),
                    "box": box.tolist(),
                }
            )

    # Left-to-right ordering.
    vehicles.sort(
        key=lambda item: item["box"][0]
    )

    print(
        f"Vehicles detected: "
        f"{len(vehicles)}"
    )


    # --------------------------------------------------------
    # DEBUG IMAGE
    # --------------------------------------------------------

    debug_image = image.copy()

    draw = ImageDraw.Draw(
        debug_image
    )

    for index, vehicle in enumerate(
        vehicles,
        start=1,
    ):
        vehicle_start_time = time.perf_counter()

        x1, y1, x2, y2 = vehicle["box"]

        draw.rectangle(
            (x1, y1, x2, y2),
            width=8,
        )

        draw.text(
            (x1 + 10, y1 + 10),
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


    # --------------------------------------------------------
    # ANALYZE EACH VEHICLE
    # --------------------------------------------------------

    photo_numbers = []
    vehicle_results = []

    for index, vehicle in enumerate(
        vehicles,
        start=1,
    ):

        x1, y1, x2, y2 = vehicle["box"]

        crop_box = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )

        crop = image.crop(
            crop_box
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

        print()
        print(
            f"Vehicle {index}: "
            f"DETR confidence="
            f"{vehicle['score']:.2f}"
        )


        # ----------------------------------------------------
        # PASS 1 — VEHICLE PROFILE
        # ----------------------------------------------------

        profile_start_time = time.perf_counter()

        profile, raw_profile = (
            ask_qwen_for_profile(
                crop_path
            )
        )

        profile_elapsed = (
            time.perf_counter()
            - profile_start_time
        )

        total_profile_time += profile_elapsed

        profile_path = (
            photo_output_dir
            / f"{RACE_TYPE}-{index:02d}.json"
        )

        if profile is not None:
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
                )


        # ----------------------------------------------------
        # VALIDATE PROFILE NUMBER
        # ----------------------------------------------------

        profile_number = (
            validate_profile_number(
                profile
            )
        )

        print(
            f"Profile number: "
            f"{profile_number or 'UNKNOWN'}"
        )


        # ----------------------------------------------------
        # PASS 2 — INDEPENDENT NUMBER VERIFICATION
        # ----------------------------------------------------

        verification_start_time = time.perf_counter()

        verification_number = (
            ask_qwen_to_verify_number(
                crop_path
            )
        )

        verification_elapsed = (
            time.perf_counter()
            - verification_start_time
        )

        total_verification_time += verification_elapsed

        print(
            f"Verification number: "
            f"{verification_number or 'UNKNOWN'}"
        )


        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        final_number, decision = (
            decide_final_number(
                profile_number,
                verification_number,
            )
        )

        print(
            f"RaceSort decision: "
            f"{decision}"
        )

        print(
            f"Final number: "
            f"{final_number or 'UNKNOWN'}"
        )

        vehicle_elapsed = (
            time.perf_counter()
            - vehicle_start_time
        )

        total_vehicle_time += vehicle_elapsed
        total_vehicles_processed += 1

        print(
            f"Timing: profile="
            f"{profile_elapsed:.2f}s, "
            f"verification="
            f"{verification_elapsed:.2f}s, "
            f"vehicle total="
            f"{vehicle_elapsed:.2f}s"
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        vehicle_result = {
            "vehicle": index,
            "detr_confidence":
                vehicle["score"],
            "profile_number":
                profile_number,
            "verification_number":
                verification_number,
            "final_number":
                final_number,
            "decision":
                decision,
            "crop":
                crop_path.name,
            "profile":
                profile_path.name,
        }

        vehicle_results.append(
            vehicle_result
        )

        if final_number is not None:
            photo_numbers.append(
                final_number
            )



    # --------------------------------------------------------
    # REMOVE DUPLICATE PHOTO NUMBERS
    # --------------------------------------------------------

    photo_numbers = list(
        dict.fromkeys(
            photo_numbers
        )
    )


    # --------------------------------------------------------
    # SAVE STRUCTURED PHOTO RESULTS
    # --------------------------------------------------------

    structured_result = {
        "photo": image_path.name,
        "race_type": RACE_TYPE,
        "vehicles_detected": len(vehicles),
        "confirmed_photo_numbers": photo_numbers,
        "vehicles": vehicle_results,
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

    # --------------------------------------------------------
    # SAVE HUMAN-READABLE RESULTS
    # --------------------------------------------------------

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
            f"Photo: {image_path.name}\n"
        )

        file.write(
            f"Race type: {RACE_TYPE}\n"
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
                f"  Profile number: "
                f"{result['profile_number'] or 'UNKNOWN'}\n"
            )

            file.write(
                f"  Verification number: "
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
                f"  Crop: "
                f"{result['crop']}\n"
            )

            file.write(
                f"  Profile: "
                f"{result['profile']}\n\n"
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
            file.write("NONE")

        file.write("\n")







    # --------------------------------------------------------
    # BATCH SUMMARY
    # --------------------------------------------------------

    review_count = sum(
        1
        for result in vehicle_results
        if result["decision"] == "REVIEW"
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
                review_count,
        }
    )
    
    photo_elapsed = (
        time.perf_counter()
        - photo_start_time
    )

    photo_timings.append(
        {
            "photo": image_path.name,
            "vehicles": len(vehicles),
            "detr_seconds": detr_elapsed,
            "photo_seconds": photo_elapsed,
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
        f"{review_count}"
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
        ],
    )

    writer.writeheader()
    writer.writerows(
        batch_rows
    )


# ============================================================
# FINISHED
# ============================================================

print("=" * 60)
print("BATCH COMPLETE")
print("=" * 60)

print(
    f"Photos processed: "
    f"{len(image_paths)}"
)

print(
    f"Results saved in: "
    f"{OUTPUT_DIR}"
)

print(
    f"Summary CSV: "
    f"{csv_path}"
)

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

batch_elapsed = (
    time.perf_counter()
    - batch_start_time
)

print()
print("=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

print(
    f"Total batch time: "
    f"{batch_elapsed:.2f}s"
)

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
    f"DETR total: "
    f"{total_detr_time:.2f}s"
)

print(
    f"Qwen profile total: "
    f"{total_profile_time:.2f}s"
)

print(
    f"Qwen verification total: "
    f"{total_verification_time:.2f}s"
)

print()

if len(image_paths) > 0:

    average_photo_time = (
        batch_elapsed
        / len(image_paths)
    )

    print(
        f"Average seconds/photo: "
        f"{average_photo_time:.2f}"
    )

    projected_1000_seconds = (
        average_photo_time
        * 1000
    )

    projected_1000_minutes = (
        projected_1000_seconds
        / 60
    )

    print(
        f"Projected 1,000-photo time: "
        f"{projected_1000_minutes:.1f} minutes"
    )


if total_vehicles_processed > 0:

    print()

    print(
        f"Average profile call: "
        f"{total_profile_time / total_vehicles_processed:.2f}s"
    )

    print(
        f"Average verification call: "
        f"{total_verification_time / total_vehicles_processed:.2f}s"
    )