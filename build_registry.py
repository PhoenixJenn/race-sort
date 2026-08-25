from pathlib import Path
from collections import Counter
import json


OUTPUT_DIR = Path("test-output")
REGISTRY_PATH = OUTPUT_DIR / "vehicle-registry.json"


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def most_common_value(values):
    """
    Return the most frequently observed non-empty value.
    """

    clean_values = [
        value
        for value in values
        if value
    ]

    if not clean_values:
        return None

    return Counter(
        clean_values
    ).most_common(1)[0][0]


registry = {}

review_observations = []


# ------------------------------------------------------------
# Find every photo-results.json file
# ------------------------------------------------------------

result_files = sorted(
    OUTPUT_DIR.glob(
        "*/photo-results.json"
    )
)

print(
    f"Found {len(result_files)} "
    f"photo result files."
)


# ------------------------------------------------------------
# Process every photograph
# ------------------------------------------------------------

for result_path in result_files:

    photo_result = load_json(
        result_path
    )

    photo_name = photo_result["photo"]
    race_type = photo_result["race_type"]

    photo_directory = (
        result_path.parent
    )

    for vehicle in photo_result["vehicles"]:

        profile_path = (
            photo_directory
            / vehicle["profile"]
        )

        profile = load_json(
            profile_path
        )

        final_number = (
            vehicle["final_number"]
        )

        decision = (
            vehicle["decision"]
        )


        # ----------------------------------------------------
        # CONFIRMED VEHICLE
        # ----------------------------------------------------

        if (
            decision == "CONFIRMED"
            and final_number
        ):

            if final_number not in registry:

                registry[final_number] = {
                    "race_number": final_number,
                    "vehicle_type": race_type,

                    "confirmed_sightings": 0,

                    "makes_seen": [],
                    "colors_seen": [],
                    "leathers_colors_seen": [],
                    "helmet_colors_seen": [],

                    "number_plate_colors_seen": [],

                    "photos": [],
                    "observations": [],
                }


            entry = registry[
                final_number
            ]

            entry[
                "confirmed_sightings"
            ] += 1

            entry["photos"].append(
                photo_name
            )


            # ------------------------------------------------
            # Make
            # ------------------------------------------------

            make_info = profile.get(
                "make"
            )

            make_value = None

            if isinstance(
                make_info,
                dict,
            ):
                make_value = (
                    make_info.get(
                        "value"
                    )
                )

            elif isinstance(
                make_info,
                str,
            ):
                make_value = make_info


            if make_value:
                entry[
                    "makes_seen"
                ].append(
                    make_value
                )


            # ------------------------------------------------
            # Vehicle colors
            # ------------------------------------------------

            colors = (
                profile
                .get(
                    "colors",
                    {},
                )
                .get(
                    "primary",
                    [],
                )
            )

            entry[
                "colors_seen"
            ].extend(
                colors
            )


            # ------------------------------------------------
            # Motorcycle rider metadata
            # ------------------------------------------------

            rider = profile.get(
                "rider",
                {},
            )

            entry[
                "leathers_colors_seen"
            ].extend(
                rider.get(
                    "leathers_colors",
                    [],
                )
            )

            entry[
                "helmet_colors_seen"
            ].extend(
                rider.get(
                    "helmet_colors",
                    [],
                )
            )


            # ------------------------------------------------
            # Number plate
            # ------------------------------------------------

            number_plate = (
                profile.get(
                    "number_plate",
                    {},
                )
            )

            plate_color = (
                number_plate.get(
                    "color"
                )
            )

            if plate_color:
                entry[
                    "number_plate_colors_seen"
                ].append(
                    plate_color
                )


            # ------------------------------------------------
            # Preserve complete observation
            # ------------------------------------------------

            entry[
                "observations"
            ].append(
                {
                    "photo": photo_name,

                    "crop": vehicle[
                        "crop"
                    ],

                    "detr_confidence":
                        vehicle[
                            "detr_confidence"
                        ],

                    "profile":
                        profile,
                }
            )


        # ----------------------------------------------------
        # REVIEW / UNKNOWN VEHICLE
        # ----------------------------------------------------

        else:

            review_observations.append(
                {
                    "photo":
                        photo_name,

                    "vehicle":
                        vehicle[
                            "vehicle"
                        ],

                    "crop":
                        vehicle[
                            "crop"
                        ],

                    "profile_number":
                        vehicle[
                            "profile_number"
                        ],

                    "verification_number":
                        vehicle[
                            "verification_number"
                        ],

                    "decision":
                        decision,

                    "profile":
                        profile,
                }
            )


# ------------------------------------------------------------
# Create summary metadata
# ------------------------------------------------------------

for number, entry in registry.items():

    entry["most_likely_make"] = (
        most_common_value(
            entry[
                "makes_seen"
            ]
        )
    )

    entry["common_colors"] = [
        color
        for color, count
        in Counter(
            entry[
                "colors_seen"
            ]
        ).most_common()
    ]

    entry[
        "common_leathers_colors"
    ] = [
        color
        for color, count
        in Counter(
            entry[
                "leathers_colors_seen"
            ]
        ).most_common()
    ]

    entry[
        "common_helmet_colors"
    ] = [
        color
        for color, count
        in Counter(
            entry[
                "helmet_colors_seen"
            ]
        ).most_common()
    ]


# ------------------------------------------------------------
# Final registry structure
# ------------------------------------------------------------

output = {
    "vehicles": registry,
    "review_observations":
        review_observations,
}


# ------------------------------------------------------------
# Save registry
# ------------------------------------------------------------

with open(
    REGISTRY_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


# ------------------------------------------------------------
# Terminal summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("VEHICLE REGISTRY BUILT")
print("=" * 60)

print(
    f"Confirmed vehicles: "
    f"{len(registry)}"
)

print(
    f"Review observations: "
    f"{len(review_observations)}"
)

print(
    f"Saved registry: "
    f"{REGISTRY_PATH}"
)

print()

for number, entry in sorted(
    registry.items()
):

    print(
        f"#{number}: "
        f"{entry['confirmed_sightings']} "
        f"confirmed sightings"
    )

    print(
        f"  Make: "
        f"{entry['most_likely_make']}"
    )

    print(
        f"  Colors: "
        f"{entry['common_colors']}"
    )