from pathlib import Path
import csv

from PIL import Image, ImageFilter, ImageStat


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

RESULTS_PATH = (
    OUTPUT_DIR
    / "primary-filter-results.csv"
)


# ============================================================
# EXPERIMENTAL THRESHOLDS
#
# These are NOT production rules.
#
# We will run them on the current test set, visually inspect
# the results, and then adjust them based on your judgment.
# ============================================================

SECONDARY_MIN_AREA_RATIO = 0.45
SECONDARY_MIN_SHARPNESS_RATIO = 0.35


# ============================================================
# SHARPNESS MEASUREMENT
# ============================================================

def measure_sharpness(image_path):
    """
    Return a simple edge-variance sharpness score.

    Higher generally means more visible edge detail.
    Lower generally means a softer / blurrier image.

    This is only a proxy for sharpness.
    """

    image = Image.open(
        image_path
    ).convert("L")

    # Reduce very large images for faster measurement.
    image.thumbnail(
        (800, 800)
    )

    edges = image.filter(
        ImageFilter.FIND_EDGES
    )

    stats = ImageStat.Stat(
        edges
    )

    variance = stats.var[0]

    return variance


# ============================================================
# FIND CROPS
# ============================================================

crop_paths = sorted(
    OUTPUT_DIR.glob(
        "GGBM*/motorcycle-*.jpg"
    )
)

print(
    f"Found {len(crop_paths)} "
    f"motorcycle crops."
)

print()


# ============================================================
# GROUP CROPS BY ORIGINAL PHOTO
# ============================================================

photos = {}

for crop_path in crop_paths:

    photo_name = (
        crop_path.parent.name
    )

    if photo_name not in photos:
        photos[photo_name] = []

    photos[photo_name].append(
        crop_path
    )


# ============================================================
# ANALYZE EACH PHOTO
# ============================================================

rows = []


for photo_index, (
    photo_name,
    photo_crops,
) in enumerate(
    sorted(photos.items()),
    start=1,
):

    print("=" * 70)

    print(
        f"[{photo_index}/"
        f"{len(photos)}] "
        f"{photo_name}"
    )

    print("=" * 70)


    measurements = []


    # --------------------------------------------------------
    # Measure each detected vehicle
    # --------------------------------------------------------

    for crop_path in photo_crops:

        with Image.open(
            crop_path
        ) as image:

            width, height = (
                image.size
            )

        area = (
            width
            * height
        )

        sharpness = (
            measure_sharpness(
                crop_path
            )
        )

        measurements.append(
            {
                "crop_path":
                    crop_path,

                "crop":
                    crop_path.name,

                "width":
                    width,

                "height":
                    height,

                "area":
                    area,

                "sharpness":
                    sharpness,
            }
        )


    # --------------------------------------------------------
    # Determine relative measurements within THIS photo
    # --------------------------------------------------------

    largest_area = max(
        item["area"]
        for item in measurements
    )

    sharpest_score = max(
        item["sharpness"]
        for item in measurements
    )


    for item in measurements:

        area_ratio = (
            item["area"]
            / largest_area
            if largest_area > 0
            else 0
        )

        sharpness_ratio = (
            item["sharpness"]
            / sharpest_score
            if sharpest_score > 0
            else 0
        )


        # ====================================================
        # EXPERIMENTAL CLASSIFICATION
        # ====================================================

        if len(measurements) == 1:

            # A single detected vehicle is provisionally
            # considered the main subject.
            classification = (
                "PRIMARY_CANDIDATE"
            )


        elif area_ratio >= 0.90:

            # One of the largest vehicles in the frame.
            classification = (
                "PRIMARY_CANDIDATE"
            )


        elif (
            area_ratio
            >= SECONDARY_MIN_AREA_RATIO
            and sharpness_ratio
            >= SECONDARY_MIN_SHARPNESS_RATIO
        ):

            classification = (
                "SECONDARY_CANDIDATE"
            )


        else:

            classification = (
                "NON_PRIMARY_CANDIDATE"
            )


        item[
            "area_ratio"
        ] = area_ratio

        item[
            "sharpness_ratio"
        ] = sharpness_ratio

        item[
            "classification"
        ] = classification


        # ----------------------------------------------------
        # Terminal display
        # ----------------------------------------------------

        print(
            f"{item['crop']}"
        )

        print(
            f"  size: "
            f"{item['width']} x "
            f"{item['height']}"
        )

        print(
            f"  relative area: "
            f"{area_ratio:.2f}"
        )

        print(
            f"  sharpness: "
            f"{item['sharpness']:.2f}"
        )

        print(
            f"  relative sharpness: "
            f"{sharpness_ratio:.2f}"
        )

        print(
            f"  classification: "
            f"{classification}"
        )

        print()


        # ----------------------------------------------------
        # CSV row
        # ----------------------------------------------------

        rows.append(
            {
                "photo":
                    photo_name,

                "crop":
                    str(
                        item[
                            "crop_path"
                        ]
                    ),

                "vehicles_in_photo":
                    len(measurements),

                "width":
                    item["width"],

                "height":
                    item["height"],

                "pixel_area":
                    item["area"],

                "relative_area":
                    area_ratio,

                "sharpness":
                    item["sharpness"],

                "relative_sharpness":
                    sharpness_ratio,

                "classification":
                    classification,
            }
        )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "photo",
        "crop",
        "vehicles_in_photo",
        "width",
        "height",
        "pixel_area",
        "relative_area",
        "sharpness",
        "relative_sharpness",
        "classification",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

primary_count = sum(
    1
    for row in rows
    if row["classification"]
    == "PRIMARY_CANDIDATE"
)

secondary_count = sum(
    1
    for row in rows
    if row["classification"]
    == "SECONDARY_CANDIDATE"
)

non_primary_count = sum(
    1
    for row in rows
    if row["classification"]
    == "NON_PRIMARY_CANDIDATE"
)


print("=" * 70)
print("PRIMARY FILTER TEST COMPLETE")
print("=" * 70)

print(
    f"Vehicles analyzed: "
    f"{len(rows)}"
)

print(
    f"Primary candidates: "
    f"{primary_count}"
)

print(
    f"Secondary candidates: "
    f"{secondary_count}"
)

print(
    f"Non-primary candidates: "
    f"{non_primary_count}"
)

print()

print(
    f"Results CSV: "
    f"{RESULTS_PATH}"
)