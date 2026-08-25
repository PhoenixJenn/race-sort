from pathlib import Path
import csv
import json


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

REGISTRY_PATH = (
    OUTPUT_DIR
    / "vehicle-registry.json"
)

MATCHES_PATH = (
    OUTPUT_DIR
    / "review-matches.json"
)

EVALUATION_JSON_PATH = (
    OUTPUT_DIR
    / "candidate-evaluations.json"
)

EVALUATION_CSV_PATH = (
    OUTPUT_DIR
    / "candidate-evaluations.csv"
)


# ============================================================
# RULE THRESHOLDS
#
# These are intentionally conservative and experimental.
# We are NOT auto-assigning identities yet.
# ============================================================

VERY_LOW_VISUAL = 0.40
MODERATE_VISUAL = 0.75
HIGH_VISUAL = 0.90

SMALL_MARGIN = 0.02
GOOD_MARGIN = 0.05


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


def normalize_text(value):
    if value is None:
        return None

    return (
        str(value)
        .strip()
        .lower()
    )


def normalize_color_list(values):
    if not values:
        return set()

    return {
        normalize_text(value)
        for value in values
        if value
    }


def extract_make(profile):
    """
    Handle both:
        "make": {"value": "BMW"}

    and:
        "make": "BMW"
    """

    if not profile:
        return None

    make = profile.get("make")

    if isinstance(make, dict):
        return normalize_text(
            make.get("value")
        )

    if isinstance(make, str):
        return normalize_text(make)

    return None


def extract_vehicle_colors(profile):
    if not profile:
        return set()

    colors = (
        profile
        .get("colors", {})
        .get("primary", [])
    )

    return normalize_color_list(
        colors
    )


def extract_leathers_colors(profile):
    if not profile:
        return set()

    colors = (
        profile
        .get("rider", {})
        .get(
            "leathers_colors",
            [],
        )
    )

    return normalize_color_list(
        colors
    )


def extract_helmet_colors(profile):
    if not profile:
        return set()

    colors = (
        profile
        .get("rider", {})
        .get(
            "helmet_colors",
            [],
        )
    )

    return normalize_color_list(
        colors
    )


def color_overlap(
    review_colors,
    registry_colors,
):
    """
    Return:
        None if there is not enough information
        otherwise a 0.0–1.0 overlap ratio
    """

    if (
        not review_colors
        or not registry_colors
    ):
        return None

    intersection = (
        review_colors
        & registry_colors
    )

    union = (
        review_colors
        | registry_colors
    )

    if not union:
        return None

    return (
        len(intersection)
        / len(union)
    )


def make_compatibility(
    review_make,
    candidate_make,
):
    """
    Returns:
        "MATCH"
        "CONFLICT"
        "UNKNOWN"
    """

    if (
        review_make is None
        or candidate_make is None
    ):
        return "UNKNOWN"

    if review_make == candidate_make:
        return "MATCH"

    return "CONFLICT"


def number_evidence(
    candidate_number,
    profile_number,
    verification_number,
):
    """
    Explain how direct number observations relate
    to this candidate.
    """

    candidate_number = str(
        candidate_number
    )

    evidence = []

    if profile_number:
        if str(profile_number) == candidate_number:
            evidence.append(
                "PROFILE_MATCH"
            )
        else:
            evidence.append(
                "PROFILE_CONFLICT"
            )

    if verification_number:
        if (
            str(verification_number)
            == candidate_number
        ):
            evidence.append(
                "VERIFICATION_MATCH"
            )
        else:
            evidence.append(
                "VERIFICATION_CONFLICT"
            )

    if not evidence:
        evidence.append(
            "NO_NUMBER_EVIDENCE"
        )

    return evidence


def classify_candidate(
    *,
    candidate_number,
    visual_rank,
    best_similarity,
    mean_similarity,
    support,
    best_margin,
    profile_number,
    verification_number,
    make_status,
    vehicle_color_overlap,
    leathers_overlap,
    helmet_overlap,
):
    """
    Rule-based evidence classification.

    Returns:
        label
        reasons
    """

    reasons = []

    number_flags = number_evidence(
        candidate_number,
        profile_number,
        verification_number,
    )

    reasons.extend(
        number_flags
    )


    # --------------------------------------------------------
    # Hard conflict from direct number evidence
    # --------------------------------------------------------

    direct_conflict = any(
        flag in {
            "PROFILE_CONFLICT",
            "VERIFICATION_CONFLICT",
        }
        for flag in number_flags
    )

    direct_match = any(
        flag in {
            "PROFILE_MATCH",
            "VERIFICATION_MATCH",
        }
        for flag in number_flags
    )


    # --------------------------------------------------------
    # Very weak visual evidence
    # --------------------------------------------------------

    if best_similarity < VERY_LOW_VISUAL:
        reasons.append(
            "VERY_LOW_VISUAL_SIMILARITY"
        )

        return (
            "INSUFFICIENT_EVIDENCE",
            reasons,
        )


    # --------------------------------------------------------
    # Make conflict matters
    # --------------------------------------------------------

    if make_status == "CONFLICT":
        reasons.append(
            "MAKE_CONFLICT"
        )


    # --------------------------------------------------------
    # Visual observations
    # --------------------------------------------------------

    if best_similarity >= HIGH_VISUAL:
        reasons.append(
            "HIGH_VISUAL_SIMILARITY"
        )

    elif (
        best_similarity
        >= MODERATE_VISUAL
    ):
        reasons.append(
            "MODERATE_VISUAL_SIMILARITY"
        )

    else:
        reasons.append(
            "LOW_VISUAL_SIMILARITY"
        )


    if visual_rank == 1:
        reasons.append(
            "TOP_VISUAL_CANDIDATE"
        )


    if best_margin is not None:

        if best_margin >= GOOD_MARGIN:
            reasons.append(
                "GOOD_VISUAL_MARGIN"
            )

        elif best_margin <= SMALL_MARGIN:
            reasons.append(
                "SMALL_VISUAL_MARGIN"
            )


    if support >= 2:
        reasons.append(
            "MULTIPLE_CONFIRMED_REFERENCES"
        )


    # --------------------------------------------------------
    # Metadata evidence
    # --------------------------------------------------------

    if make_status == "MATCH":
        reasons.append(
            "MAKE_MATCH"
        )

    if (
        vehicle_color_overlap
        is not None
    ):
        if vehicle_color_overlap >= 0.50:
            reasons.append(
                "VEHICLE_COLORS_COMPATIBLE"
            )

    if (
        leathers_overlap
        is not None
    ):
        if leathers_overlap >= 0.50:
            reasons.append(
                "LEATHERS_COMPATIBLE"
            )

    if (
        helmet_overlap
        is not None
    ):
        if helmet_overlap >= 0.50:
            reasons.append(
                "HELMET_COMPATIBLE"
            )


    # ========================================================
    # DECISION RULES
    # ========================================================

    # --------------------------------------------------------
    # Direct number evidence conflicts with this candidate.
    # --------------------------------------------------------

    if direct_conflict:
        return (
            "CONFLICT",
            reasons,
        )


    # --------------------------------------------------------
    # Strongest case:
    #
    # direct number evidence agrees
    # + candidate is visually plausible
    # --------------------------------------------------------

    if (
        direct_match
        and visual_rank == 1
        and best_similarity
        >= MODERATE_VISUAL
    ):

        reasons.append(
            "NUMBER_AND_VISUAL_AGREE"
        )

        return (
            "STRONG_CANDIDATE",
            reasons,
        )


    # --------------------------------------------------------
    # Direct number evidence agrees but visual rank is not #1.
    #
    # Example:
    # blurry #54 where verifier reads 54,
    # but DINO ranks #721 first.
    #
    # Keep this possible, not strong.
    # --------------------------------------------------------

    if (
        direct_match
        and best_similarity
        >= MODERATE_VISUAL
    ):

        reasons.append(
            "NUMBER_MATCH_WITH_NON_TOP_VISUAL"
        )

        return (
            "POSSIBLE_CANDIDATE",
            reasons,
        )


    # --------------------------------------------------------
    # No number evidence:
    #
    # require a strong visual result AND useful separation.
    # --------------------------------------------------------

    if (
        not direct_match
        and not direct_conflict
        and visual_rank == 1
        and best_similarity >= HIGH_VISUAL
        and best_margin is not None
        and best_margin >= GOOD_MARGIN
    ):

        reasons.append(
            "STRONG_VISUAL_ONLY_EVIDENCE"
        )

        return (
            "POSSIBLE_CANDIDATE",
            reasons,
        )


    # --------------------------------------------------------
    # High visual score but poor separation.
    #
    # This is common in our motorcycle set.
    # --------------------------------------------------------

    if (
        visual_rank == 1
        and best_similarity >= HIGH_VISUAL
    ):

        reasons.append(
            "HIGH_VISUAL_BUT_AMBIGUOUS"
        )

        return (
            "POSSIBLE_CANDIDATE",
            reasons,
        )


    # --------------------------------------------------------
    # Moderate match only.
    # --------------------------------------------------------

    if (
        visual_rank == 1
        and best_similarity
        >= MODERATE_VISUAL
    ):

        return (
            "POSSIBLE_CANDIDATE",
            reasons,
        )


    return (
        "INSUFFICIENT_EVIDENCE",
        reasons,
    )


# ============================================================
# LOAD INPUT FILES
# ============================================================

print(
    f"Loading registry: "
    f"{REGISTRY_PATH}"
)

registry_data = load_json(
    REGISTRY_PATH
)

print(
    f"Loading review matches: "
    f"{MATCHES_PATH}"
)

matches_data = load_json(
    MATCHES_PATH
)

confirmed_vehicles = (
    registry_data["vehicles"]
)

review_matches = (
    matches_data["review_matches"]
)

print(
    f"Confirmed identities: "
    f"{len(confirmed_vehicles)}"
)

print(
    f"Review observations: "
    f"{len(review_matches)}"
)

print()


# ============================================================
# BUILD LOOKUP FOR REVIEW PROFILES
# ============================================================

review_profile_lookup = {}

for observation in (
    registry_data[
        "review_observations"
    ]
):

    key = (
        observation["photo"],
        observation["vehicle"],
    )

    review_profile_lookup[
        key
    ] = observation[
        "profile"
    ]


# ============================================================
# EVALUATE REVIEW OBSERVATIONS
# ============================================================

evaluation_results = []

csv_rows = []


for review_index, review in enumerate(
    review_matches,
    start=1,
):

    photo = review["photo"]
    vehicle_index = review["vehicle"]

    key = (
        photo,
        vehicle_index,
    )

    review_profile = (
        review_profile_lookup
        .get(
            key,
            {},
        )
    )

    review_make = extract_make(
        review_profile
    )

    review_vehicle_colors = (
        extract_vehicle_colors(
            review_profile
        )
    )

    review_leathers = (
        extract_leathers_colors(
            review_profile
        )
    )

    review_helmet = (
        extract_helmet_colors(
            review_profile
        )
    )

    profile_number = (
        str(review["profile_number"])
        if review.get("profile_number") is not None
        else None
    )

    verification_number = (
        str(review["verification_number"])
        if review.get("verification_number") is not None
        else None
    )


    candidates = (
        review["candidates"]
    )


    best_margin = (
        review.get(
            "best_similarity_margin"
        )
    )


    print("=" * 72)

    print(
        f"[{review_index}/"
        f"{len(review_matches)}] "
        f"{photo} "
        f"vehicle {vehicle_index}"
    )

    print("=" * 72)

    print(
        f"Profile number: "
        f"{profile_number}"
    )

    print(
        f"Verification number: "
        f"{verification_number}"
    )

    print(
        f"Review make: "
        f"{review_make}"
    )

    print()


    evaluated_candidates = []


    # --------------------------------------------------------
    # Evaluate only the candidates already ranked by DINO.
    # --------------------------------------------------------

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        

        race_number = str(
            candidate[
                "race_number"
            ]
        )
        

        confirmed = (
            confirmed_vehicles[
                race_number
            ]
        )


        # ----------------------------------------------------
        # Candidate metadata
        # ----------------------------------------------------

        candidate_make = (
            normalize_text(
                confirmed.get(
                    "most_likely_make"
                )
            )
        )

        candidate_vehicle_colors = (
            normalize_color_list(
                confirmed.get(
                    "common_colors",
                    [],
                )
            )
        )

        candidate_leathers = (
            normalize_color_list(
                confirmed.get(
                    "common_leathers_colors",
                    [],
                )
            )
        )

        candidate_helmet = (
            normalize_color_list(
                confirmed.get(
                    "common_helmet_colors",
                    [],
                )
            )
        )


        # ----------------------------------------------------
        # Metadata comparisons
        # ----------------------------------------------------

        make_status = (
            make_compatibility(
                review_make,
                candidate_make,
            )
        )

        vehicle_color_score = (
            color_overlap(
                review_vehicle_colors,
                candidate_vehicle_colors,
            )
        )

        leathers_score = (
            color_overlap(
                review_leathers,
                candidate_leathers,
            )
        )

        helmet_score = (
            color_overlap(
                review_helmet,
                candidate_helmet,
            )
        )


        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        label, reasons = (
            classify_candidate(
                candidate_number=
                    race_number,

                visual_rank=
                    rank,

                best_similarity=
                    candidate[
                        "best_similarity"
                    ],

                mean_similarity=
                    candidate[
                        "mean_similarity"
                    ],

                support=
                    candidate[
                        "supporting_sightings"
                    ],

                best_margin=
                    best_margin,

                profile_number=
                    profile_number,

                verification_number=
                    verification_number,

                make_status=
                    make_status,

                vehicle_color_overlap=
                    vehicle_color_score,

                leathers_overlap=
                    leathers_score,

                helmet_overlap=
                    helmet_score,
            )
        )


        evaluated = {
            "rank":
                rank,

            "race_number":
                race_number,

            "classification":
                label,

            "best_similarity":
                candidate[
                    "best_similarity"
                ],

            "mean_similarity":
                candidate[
                    "mean_similarity"
                ],

            "supporting_sightings":
                candidate[
                    "supporting_sightings"
                ],

            "make_status":
                make_status,

            "vehicle_color_overlap":
                vehicle_color_score,

            "leathers_overlap":
                leathers_score,

            "helmet_overlap":
                helmet_score,

            "reasons":
                reasons,
        }

        evaluated_candidates.append(
            evaluated
        )


        # ----------------------------------------------------
        # Terminal display
        # ----------------------------------------------------

        print(
            f"{rank}. "
            f"#{race_number} "
            f"→ {label}"
        )

        print(
            f"   DINO best: "
            f"{candidate['best_similarity']:.4f}"
        )

        print(
            f"   DINO mean: "
            f"{candidate['mean_similarity']:.4f}"
        )

        print(
            f"   references: "
            f"{candidate['supporting_sightings']}"
        )

        print(
            f"   make: "
            f"{make_status}"
        )

        print(
            f"   vehicle color overlap: "
            f"{vehicle_color_score}"
        )

        print(
            f"   leathers overlap: "
            f"{leathers_score}"
        )

        print(
            f"   helmet overlap: "
            f"{helmet_score}"
        )

        print(
            f"   evidence: "
            f"{', '.join(reasons)}"
        )

        print()


    # --------------------------------------------------------
    # Pick the highest-priority recommendation.
    #
    # Still recommendation only — not assignment.
    # --------------------------------------------------------

    priority = {
        "STRONG_CANDIDATE": 4,
        "POSSIBLE_CANDIDATE": 3,
        "INSUFFICIENT_EVIDENCE": 2,
        "CONFLICT": 1,
    }

    recommendation = max(
        evaluated_candidates,
        key=lambda item: (
            priority[
                item[
                    "classification"
                ]
            ],
            item[
                "best_similarity"
            ],
        ),
    )


    result = {
        "photo":
            photo,

        "vehicle":
            vehicle_index,

        "crop":
            review["crop"],

        "profile_number":
            profile_number,

        "verification_number":
            verification_number,

        "recommended_candidate":
            recommendation[
                "race_number"
            ],

        "recommended_classification":
            recommendation[
                "classification"
            ],

        "candidates":
            evaluated_candidates,
    }

    evaluation_results.append(
        result
    )


    print(
        f"RECOMMENDATION: "
        f"#{recommendation['race_number']} "
        f"{recommendation['classification']}"
    )

    print()


    # --------------------------------------------------------
    # CSV row
    # --------------------------------------------------------

    csv_rows.append(
        {
            "photo":
                photo,

            "vehicle":
                vehicle_index,

            "crop":
                review["crop"],

            "profile_number":
                profile_number,

            "verification_number":
                verification_number,

            "recommended_candidate":
                recommendation[
                    "race_number"
                ],

            "classification":
                recommendation[
                    "classification"
                ],

            "best_similarity":
                recommendation[
                    "best_similarity"
                ],

            "mean_similarity":
                recommendation[
                    "mean_similarity"
                ],

            "supporting_sightings":
                recommendation[
                    "supporting_sightings"
                ],

            "make_status":
                recommendation[
                    "make_status"
                ],

            "vehicle_color_overlap":
                recommendation[
                    "vehicle_color_overlap"
                ],

            "leathers_overlap":
                recommendation[
                    "leathers_overlap"
                ],

            "helmet_overlap":
                recommendation[
                    "helmet_overlap"
                ],

            "reasons":
                " | ".join(
                    recommendation[
                        "reasons"
                    ]
                ),
        }
    )


# ============================================================
# SAVE JSON
# ============================================================

with open(
    EVALUATION_JSON_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "rules": {
                "very_low_visual":
                    VERY_LOW_VISUAL,

                "moderate_visual":
                    MODERATE_VISUAL,

                "high_visual":
                    HIGH_VISUAL,

                "small_margin":
                    SMALL_MARGIN,

                "good_margin":
                    GOOD_MARGIN,
            },

            "evaluations":
                evaluation_results,
        },
        file,
        indent=2,
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    EVALUATION_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "photo",
        "vehicle",
        "crop",
        "profile_number",
        "verification_number",
        "recommended_candidate",
        "classification",
        "best_similarity",
        "mean_similarity",
        "supporting_sightings",
        "make_status",
        "vehicle_color_overlap",
        "leathers_overlap",
        "helmet_overlap",
        "reasons",
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
# FINISHED
# ============================================================

print("=" * 72)
print("CANDIDATE EVALUATION COMPLETE")
print("=" * 72)

print(
    f"Review observations evaluated: "
    f"{len(evaluation_results)}"
)

print(
    f"JSON output: "
    f"{EVALUATION_JSON_PATH}"
)

print(
    f"CSV output: "
    f"{EVALUATION_CSV_PATH}"
)