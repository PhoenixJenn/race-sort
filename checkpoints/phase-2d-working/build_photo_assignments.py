from pathlib import Path
import csv
import json


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

CANDIDATE_EVALUATIONS_PATH = (
    OUTPUT_DIR
    / "candidate-evaluations.json"
)

ASSIGNMENTS_JSON_PATH = (
    OUTPUT_DIR
    / "photo-assignments.json"
)

ASSIGNMENTS_CSV_PATH = (
    OUTPUT_DIR
    / "photo-assignments.csv"
)


# ============================================================
# CURRENT POLICY
#
# Only STRONG_CANDIDATE second-pass matches are allowed to
# become INFERRED assignments.
#
# POSSIBLE_CANDIDATE remains REVIEW.
# ============================================================

INFERRED_CLASSIFICATIONS = {
    "STRONG_CANDIDATE",
}


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_race_number(value):
    """
    Race numbers are identifiers, not quantities.

    Preserve:
        007
        54
        54A
        A12

    Never convert them to integers.
    """

    if value is None:
        return None

    value = str(value).strip().upper()

    if not value:
        return None

    return value


# ============================================================
# LOAD SECOND-PASS EVALUATIONS
# ============================================================

print(
    f"Loading candidate evaluations: "
    f"{CANDIDATE_EVALUATIONS_PATH}"
)

candidate_data = load_json(
    CANDIDATE_EVALUATIONS_PATH
)

evaluations = candidate_data[
    "evaluations"
]


# ------------------------------------------------------------
# Lookup by:
#
#   (photo filename, vehicle number)
#
# Example:
#
#   ("GGBM0017.JPG", 1)
# ------------------------------------------------------------

evaluation_lookup = {}

for evaluation in evaluations:

    key = (
        evaluation["photo"],
        evaluation["vehicle"],
    )

    evaluation_lookup[key] = evaluation


print(
    f"Second-pass evaluations loaded: "
    f"{len(evaluation_lookup)}"
)

print()


# ============================================================
# FIND ALL PER-PHOTO RESULT FILES
# ============================================================

photo_result_paths = sorted(
    OUTPUT_DIR.glob(
        "*/photo-results.json"
    )
)

print(
    f"Photo result files found: "
    f"{len(photo_result_paths)}"
)

print()


# ============================================================
# BUILD PHOTO ASSIGNMENTS
# ============================================================

photo_assignments = []

csv_rows = []


for photo_index, result_path in enumerate(
    photo_result_paths,
    start=1,
):

    photo_result = load_json(
        result_path
    )

    photo_name = (
        photo_result["photo"]
    )

    race_type = (
        photo_result["race_type"]
    )

    vehicles = (
        photo_result["vehicles"]
    )


    print("=" * 70)

    print(
        f"[{photo_index}/"
        f"{len(photo_result_paths)}] "
        f"{photo_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # We temporarily collect assignments by race number.
    #
    # This prevents the same original photo from receiving
    # duplicate #52 assignments if multiple observations point
    # to the same vehicle identity.
    # --------------------------------------------------------

    assignment_by_number = {}

    unresolved_vehicles = []


    # ========================================================
    # PROCESS EACH DETECTED VEHICLE
    # ========================================================

    for vehicle in vehicles:

        vehicle_index = (
            vehicle["vehicle"]
        )

        decision = (
            vehicle["decision"]
        )

        final_number = (
            normalize_race_number(
                vehicle.get(
                    "final_number"
                )
            )
        )


        # ----------------------------------------------------
        # CASE 1:
        # Directly confirmed number.
        #
        # This is READ evidence.
        # ----------------------------------------------------

        if (
            decision == "CONFIRMED"
            and final_number
        ):

            print(
                f"Vehicle {vehicle_index}: "
                f"READ #{final_number}"
            )

            assignment = {
                "race_number":
                    final_number,

                "source":
                    "READ",

                "classification":
                    "CONFIRMED",

                "vehicle_indices": [
                    vehicle_index
                ],

                "evidence": [
                    {
                        "vehicle":
                            vehicle_index,

                        "source":
                            "READ",

                        "profile_number":
                            normalize_race_number(
                                vehicle.get(
                                    "profile_number"
                                )
                            ),

                        "verification_number":
                            normalize_race_number(
                                vehicle.get(
                                    "verification_number"
                                )
                            ),

                        "detr_confidence":
                            vehicle.get(
                                "detr_confidence"
                            ),
                    }
                ],
            }


            # ------------------------------------------------
            # If this number already exists, merge evidence.
            #
            # READ always has priority over INFERRED.
            # ------------------------------------------------

            if final_number in assignment_by_number:

                existing = (
                    assignment_by_number[
                        final_number
                    ]
                )

                if (
                    vehicle_index
                    not in existing[
                        "vehicle_indices"
                    ]
                ):
                    existing[
                        "vehicle_indices"
                    ].append(
                        vehicle_index
                    )

                existing[
                    "evidence"
                ].extend(
                    assignment[
                        "evidence"
                    ]
                )

                existing[
                    "source"
                ] = "READ"

                existing[
                    "classification"
                ] = "CONFIRMED"

            else:

                assignment_by_number[
                    final_number
                ] = assignment


            continue


        # ----------------------------------------------------
        # CASE 2:
        # First pass left this vehicle in REVIEW.
        #
        # Look for Phase 2D evidence.
        # ----------------------------------------------------

        evaluation_key = (
            photo_name,
            vehicle_index,
        )

        evaluation = (
            evaluation_lookup.get(
                evaluation_key
            )
        )


        # ----------------------------------------------------
        # No second-pass evaluation exists.
        # ----------------------------------------------------

        if evaluation is None:

            print(
                f"Vehicle {vehicle_index}: "
                f"REVIEW — no second-pass evaluation"
            )

            unresolved_vehicles.append(
                {
                    "vehicle":
                        vehicle_index,

                    "reason":
                        "NO_SECOND_PASS_EVALUATION",

                    "profile_number":
                        normalize_race_number(
                            vehicle.get(
                                "profile_number"
                            )
                        ),

                    "verification_number":
                        normalize_race_number(
                            vehicle.get(
                                "verification_number"
                            )
                        ),
                }
            )

            continue


        recommended_number = (
            normalize_race_number(
                evaluation.get(
                    "recommended_candidate"
                )
            )
        )

        classification = (
            evaluation.get(
                "recommended_classification"
            )
        )


        # ----------------------------------------------------
        # CASE 2A:
        # STRONG second-pass candidate.
        #
        # Record as INFERRED.
        # ----------------------------------------------------

        if (
            classification
            in INFERRED_CLASSIFICATIONS
            and recommended_number
        ):

            print(
                f"Vehicle {vehicle_index}: "
                f"INFERRED #{recommended_number} "
                f"({classification})"
            )

            inferred_evidence = {
                "vehicle":
                    vehicle_index,

                "source":
                    "INFERRED",

                "classification":
                    classification,

                "profile_number":
                    normalize_race_number(
                        evaluation.get(
                            "profile_number"
                        )
                    ),

                "verification_number":
                    normalize_race_number(
                        evaluation.get(
                            "verification_number"
                        )
                    ),

                "recommended_candidate":
                    recommended_number,
            }


            # ------------------------------------------------
            # Pull the matching candidate's evidence details.
            # ------------------------------------------------

            for candidate in evaluation.get(
                "candidates",
                [],
            ):

                candidate_number = (
                    normalize_race_number(
                        candidate.get(
                            "race_number"
                        )
                    )
                )

                if (
                    candidate_number
                    == recommended_number
                ):

                    inferred_evidence[
                        "best_similarity"
                    ] = candidate.get(
                        "best_similarity"
                    )

                    inferred_evidence[
                        "mean_similarity"
                    ] = candidate.get(
                        "mean_similarity"
                    )

                    inferred_evidence[
                        "supporting_sightings"
                    ] = candidate.get(
                        "supporting_sightings"
                    )

                    inferred_evidence[
                        "make_status"
                    ] = candidate.get(
                        "make_status"
                    )

                    inferred_evidence[
                        "vehicle_color_overlap"
                    ] = candidate.get(
                        "vehicle_color_overlap"
                    )

                    inferred_evidence[
                        "leathers_overlap"
                    ] = candidate.get(
                        "leathers_overlap"
                    )

                    inferred_evidence[
                        "helmet_overlap"
                    ] = candidate.get(
                        "helmet_overlap"
                    )

                    inferred_evidence[
                        "reasons"
                    ] = candidate.get(
                        "reasons",
                        [],
                    )

                    break


            if (
                recommended_number
                in assignment_by_number
            ):

                existing = (
                    assignment_by_number[
                        recommended_number
                    ]
                )

                if (
                    vehicle_index
                    not in existing[
                        "vehicle_indices"
                    ]
                ):
                    existing[
                        "vehicle_indices"
                    ].append(
                        vehicle_index
                    )

                existing[
                    "evidence"
                ].append(
                    inferred_evidence
                )

                # READ evidence wins if we already
                # confirmed this number directly.
                if (
                    existing["source"]
                    != "READ"
                ):
                    existing[
                        "source"
                    ] = "INFERRED"

                    existing[
                        "classification"
                    ] = (
                        classification
                    )

            else:

                assignment_by_number[
                    recommended_number
                ] = {
                    "race_number":
                        recommended_number,

                    "source":
                        "INFERRED",

                    "classification":
                        classification,

                    "vehicle_indices": [
                        vehicle_index
                    ],

                    "evidence": [
                        inferred_evidence
                    ],
                }


            continue


        # ----------------------------------------------------
        # CASE 2B:
        #
        # POSSIBLE / CONFLICT / INSUFFICIENT
        #
        # Remain unresolved.
        # ----------------------------------------------------

        print(
            f"Vehicle {vehicle_index}: "
            f"REVIEW "
            f"({classification})"
        )

        unresolved_vehicles.append(
            {
                "vehicle":
                    vehicle_index,

                "reason":
                    classification,

                "recommended_candidate":
                    recommended_number,

                "profile_number":
                    normalize_race_number(
                        evaluation.get(
                            "profile_number"
                        )
                    ),

                "verification_number":
                    normalize_race_number(
                        evaluation.get(
                            "verification_number"
                        )
                    ),
            }
        )


    # ========================================================
    # FINALIZE PHOTO
    # ========================================================

    assignments = list(
        assignment_by_number.values()
    )


    # Sort by race-number string for predictable output.
    assignments.sort(
        key=lambda item:
            item["race_number"]
    )


    assigned_numbers = [
        assignment["race_number"]
        for assignment in assignments
    ]


    read_numbers = [
        assignment["race_number"]
        for assignment in assignments
        if assignment["source"] == "READ"
    ]


    inferred_numbers = [
        assignment["race_number"]
        for assignment in assignments
        if assignment["source"] == "INFERRED"
    ]


    # --------------------------------------------------------
    # Human review is still required if ANY detected vehicle
    # remains unresolved.
    # --------------------------------------------------------

    needs_review = (
        len(unresolved_vehicles) > 0
    )


    photo_assignment = {
        "photo":
            photo_name,

        "race_type":
            race_type,

        "vehicles_detected":
            len(vehicles),

        "assignments":
            assignments,

        "assigned_numbers":
            assigned_numbers,

        "read_numbers":
            read_numbers,

        "inferred_numbers":
            inferred_numbers,

        "unresolved_vehicles":
            unresolved_vehicles,

        "needs_review":
            needs_review,
    }


    photo_assignments.append(
        photo_assignment
    )


    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print()

    print(
        f"READ numbers: "
        f"{read_numbers}"
    )

    print(
        f"INFERRED numbers: "
        f"{inferred_numbers}"
    )

    print(
        f"Unresolved vehicles: "
        f"{len(unresolved_vehicles)}"
    )

    print(
        f"Needs human review: "
        f"{needs_review}"
    )

    print()


    # --------------------------------------------------------
    # CSV SUMMARY
    # --------------------------------------------------------

    csv_rows.append(
        {
            "photo":
                photo_name,

            "vehicles_detected":
                len(vehicles),

            "assigned_numbers":
                ", ".join(
                    assigned_numbers
                ),

            "read_numbers":
                ", ".join(
                    read_numbers
                ),

            "inferred_numbers":
                ", ".join(
                    inferred_numbers
                ),

            "unresolved_vehicle_count":
                len(
                    unresolved_vehicles
                ),

            "needs_review":
                needs_review,
        }
    )


# ============================================================
# SAVE JSON
# ============================================================

with open(
    ASSIGNMENTS_JSON_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "assignment_policy": {
                "direct_read":
                    "CONFIRMED",

                "inferred_allowed":
                    sorted(
                        INFERRED_CLASSIFICATIONS
                    ),

                "possible_candidate":
                    "REVIEW",

                "conflict":
                    "REVIEW",

                "insufficient_evidence":
                    "REVIEW",
            },

            "photos":
                photo_assignments,
        },
        file,
        indent=2,
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    ASSIGNMENTS_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "photo",
        "vehicles_detected",
        "assigned_numbers",
        "read_numbers",
        "inferred_numbers",
        "unresolved_vehicle_count",
        "needs_review",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        csv_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_photos = len(
    photo_assignments
)

photos_with_assignments = sum(
    1
    for photo in photo_assignments
    if photo["assignments"]
)

photos_needing_review = sum(
    1
    for photo in photo_assignments
    if photo["needs_review"]
)

fully_resolved_photos = (
    total_photos
    - photos_needing_review
)

total_read_assignments = sum(
    len(
        photo["read_numbers"]
    )
    for photo in photo_assignments
)

total_inferred_assignments = sum(
    len(
        photo["inferred_numbers"]
    )
    for photo in photo_assignments
)


print("=" * 70)
print("PHOTO ASSIGNMENT BUILD COMPLETE")
print("=" * 70)

print(
    f"Photos processed: "
    f"{total_photos}"
)

print(
    f"Photos with at least one assignment: "
    f"{photos_with_assignments}"
)

print(
    f"Fully resolved photos: "
    f"{fully_resolved_photos}"
)

print(
    f"Photos still needing review: "
    f"{photos_needing_review}"
)

print(
    f"Direct READ assignments: "
    f"{total_read_assignments}"
)

print(
    f"INFERRED assignments: "
    f"{total_inferred_assignments}"
)

print()

print(
    f"JSON output: "
    f"{ASSIGNMENTS_JSON_PATH}"
)

print(
    f"CSV output: "
    f"{ASSIGNMENTS_CSV_PATH}"
)