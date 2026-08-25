from pathlib import Path
import csv
import json


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

ROUTING_RESULTS_PATH = (
    OUTPUT_DIR
    / "routing-pipeline-v2-results.csv"
)

REGISTRY_PATH = (
    OUTPUT_DIR
    / "vehicle-registry.json"
)

REVIEW_MATCHES_PATH = (
    OUTPUT_DIR
    / "review-matches.json"
)

CANDIDATE_EVALUATIONS_PATH = (
    OUTPUT_DIR
    / "candidate-evaluations.json"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "candidate-resolution-results.csv"
)


# ============================================================
# EXPERIMENTAL THRESHOLDS
#
# These are intentionally conservative.
# ============================================================

STRONG_VISUAL_SIMILARITY = 0.90
MODERATE_VISUAL_SIMILARITY = 0.75


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def normalize_number(value):
    """
    Race numbers are identifiers, not quantities.
    """

    if value is None:
        return None

    value = str(value).strip().upper()

    if not value:
        return None

    return value


def photo_from_crop(crop_path):
    """
    Example:

    test-output/GGBM0017/motorcycle-01.jpg

    returns:

    GGBM0017
    """

    return (
        Path(crop_path)
        .parent
        .name
    )


def crop_name(crop_path):
    return (
        Path(crop_path)
        .name
    )


def vehicle_index_from_crop(crop_path):
    """
    motorcycle-01.jpg -> 1
    motorcycle-02.jpg -> 2
    """

    stem = (
        Path(crop_path)
        .stem
    )

    try:
        return int(
            stem.split("-")[-1]
        )

    except ValueError:
        return None


# ============================================================
# LOAD DATA
# ============================================================

print(
    f"Loading routing results: "
    f"{ROUTING_RESULTS_PATH}"
)

routing_rows = load_csv(
    ROUTING_RESULTS_PATH
)


print(
    f"Loading registry: "
    f"{REGISTRY_PATH}"
)

registry = load_json(
    REGISTRY_PATH
)


print(
    f"Loading DINO review matches: "
    f"{REVIEW_MATCHES_PATH}"
)

review_matches_data = load_json(
    REVIEW_MATCHES_PATH
)


print(
    f"Loading candidate evaluations: "
    f"{CANDIDATE_EVALUATIONS_PATH}"
)

candidate_evaluations_data = (
    load_json(
        CANDIDATE_EVALUATIONS_PATH
    )
)

print()


# ============================================================
# PREPARE LOOKUPS
# ============================================================

known_numbers = set(
    str(number)
    for number
    in registry[
        "vehicles"
    ].keys()
)


review_match_lookup = {}

for item in review_matches_data[
    "review_matches"
]:

    key = (
        Path(
            item["photo"]
        ).stem,
        int(
            item["vehicle"]
        ),
    )

    review_match_lookup[
        key
    ] = item


evaluation_lookup = {}

for item in candidate_evaluations_data[
    "evaluations"
]:

    key = (
        Path(
            item["photo"]
        ).stem,
        int(
            item["vehicle"]
        ),
    )

    evaluation_lookup[
        key
    ] = item


# ============================================================
# FIND QWEN CANDIDATES
# ============================================================

qwen_candidate_rows = [
    row
    for row in routing_rows
    if row[
        "decision"
    ] == "QWEN_CANDIDATE"
]


print(
    f"QWEN candidates found: "
    f"{len(qwen_candidate_rows)}"
)

print()


# ============================================================
# RESOLVE CANDIDATES
# ============================================================

results = []


for index, row in enumerate(
    qwen_candidate_rows,
    start=1,
):

    crop = row["crop"]

    photo = photo_from_crop(
        crop
    )

    vehicle_index = (
        vehicle_index_from_crop(
            crop
        )
    )

    candidate_number = (
        normalize_number(
            row[
                "final_number"
            ]
        )
    )


    print("=" * 72)

    print(
        f"[{index}/"
        f"{len(qwen_candidate_rows)}] "
        f"{crop}"
    )

    print("=" * 72)

    print(
        f"Qwen candidate: "
        f"{candidate_number}"
    )


    # --------------------------------------------------------
    # REGISTRY EVIDENCE
    # --------------------------------------------------------

    candidate_exists_in_registry = (
        candidate_number
        in known_numbers
    )

    print(
        f"Exists in registry: "
        f"{candidate_exists_in_registry}"
    )


    # --------------------------------------------------------
    # DINO EVIDENCE
    # --------------------------------------------------------

    key = (
        photo,
        vehicle_index,
    )

    review_match = (
        review_match_lookup.get(
            key
        )
    )

    dino_candidate_found = False
    dino_best_similarity = None
    dino_rank = None


    if review_match is not None:

        for rank, candidate in enumerate(
            review_match.get(
                "candidates",
                []
            ),
            start=1,
        ):

            if (
                normalize_number(
                    candidate.get(
                        "race_number"
                    )
                )
                == candidate_number
            ):

                dino_candidate_found = True

                dino_best_similarity = (
                    candidate.get(
                        "best_similarity",
                        candidate.get(
                            "similarity"
                        ),
                    )
                )

                dino_rank = rank

                break


    print(
        f"DINO candidate found: "
        f"{dino_candidate_found}"
    )

    print(
        f"DINO rank: "
        f"{dino_rank}"
    )

    print(
        f"DINO similarity: "
        f"{dino_best_similarity}"
    )


    # --------------------------------------------------------
    # EVIDENCE FUSION
    # --------------------------------------------------------

    evaluation = (
        evaluation_lookup.get(
            key
        )
    )

    fusion_candidate = None
    fusion_classification = None


    if evaluation is not None:

        fusion_candidate = (
            normalize_number(
                evaluation.get(
                    "recommended_candidate"
                )
            )
        )

        fusion_classification = (
            evaluation.get(
                "recommended_classification"
            )
        )


    fusion_agrees = (
        fusion_candidate
        == candidate_number
    )


    print(
        f"Fusion candidate: "
        f"{fusion_candidate}"
    )

    print(
        f"Fusion classification: "
        f"{fusion_classification}"
    )

    print(
        f"Fusion agrees: "
        f"{fusion_agrees}"
    )


    # ========================================================
    # DISPOSITION RULES
    # ========================================================

    reasons = []


    if candidate_exists_in_registry:

        reasons.append(
            "CANDIDATE_EXISTS_IN_REGISTRY"
        )


    if (
        dino_candidate_found
        and dino_best_similarity
        is not None
    ):

        if (
            dino_best_similarity
            >= STRONG_VISUAL_SIMILARITY
        ):

            reasons.append(
                "STRONG_DINO_SUPPORT"
            )

        elif (
            dino_best_similarity
            >= MODERATE_VISUAL_SIMILARITY
        ):

            reasons.append(
                "MODERATE_DINO_SUPPORT"
            )


    if fusion_agrees:

        reasons.append(
            "FUSION_AGREES"
        )


    # --------------------------------------------------------
    # CORROBORATED
    #
    # Strong cases:
    #
    # 1. Candidate exists in registry
    #    AND strong DINO support
    #
    # OR
    #
    # 2. Evidence fusion explicitly agrees
    #    with STRONG_CANDIDATE
    # --------------------------------------------------------

    if (
        candidate_exists_in_registry
        and dino_best_similarity
        is not None
        and dino_best_similarity
        >= STRONG_VISUAL_SIMILARITY
    ):

        disposition = (
            "CORROBORATED"
        )


    elif (
        fusion_agrees
        and fusion_classification
        == "STRONG_CANDIDATE"
    ):

        disposition = (
            "CORROBORATED"
        )


    # --------------------------------------------------------
    # SUPPORTED BUT NOT STRONG ENOUGH
    # --------------------------------------------------------

    elif (
        candidate_exists_in_registry
        and dino_best_similarity
        is not None
        and dino_best_similarity
        >= MODERATE_VISUAL_SIMILARITY
    ):

        disposition = (
            "SUPPORTED_REVIEW"
        )


    elif (
        fusion_agrees
        and fusion_classification
        == "POSSIBLE_CANDIDATE"
    ):

        disposition = (
            "SUPPORTED_REVIEW"
        )

    # --------------------------------------------------------
    # CONFLICTING
    #
    # Another evidence layer actively recommends
    # a different identity.
    # --------------------------------------------------------

    elif (
        fusion_candidate is not None
        and fusion_candidate
        != candidate_number
    ):

        disposition = (
            "CONFLICTING"
        )

        reasons.append(
            "FUSION_RECOMMENDS_DIFFERENT_NUMBER"
        )


    # --------------------------------------------------------
    # KNOWN NUMBER, BUT NOT ENOUGH CORROBORATION
    #
    # Important distinction:
    #
    # The Qwen candidate already exists in the registry,
    # but this crop does not currently have sufficient
    # DINO / fusion evidence to promote it.
    #
    # This is NOT the same as unsupported.
    # --------------------------------------------------------

    elif candidate_exists_in_registry:

        disposition = (
            "KNOWN_NUMBER_REVIEW"
        )

        reasons.append(
            "KNOWN_REGISTRY_NUMBER"
        )

        reasons.append(
            "INSUFFICIENT_INDEPENDENT_CORROBORATION"
        )


    # --------------------------------------------------------
    # UNSUPPORTED
    #
    # Candidate is not known in the registry and has no
    # meaningful corroborating evidence.
    # --------------------------------------------------------

    else:

        disposition = (
            "UNSUPPORTED"
        )







    print()

    print(
        f"DISPOSITION: "
        f"{disposition}"
    )

    print(
        f"Reasons: "
        f"{reasons}"
    )

    print()


    results.append(
        {
            "crop":
                crop,

            "photo":
                photo,

            "vehicle":
                vehicle_index,

            "qwen_candidate":
                candidate_number,

            "candidate_exists_in_registry":
                candidate_exists_in_registry,

            "dino_rank":
                dino_rank,

            "dino_best_similarity":
                dino_best_similarity,

            "fusion_candidate":
                fusion_candidate,

            "fusion_classification":
                fusion_classification,

            "fusion_agrees":
                fusion_agrees,

            "disposition":
                disposition,

            "reasons":
                " | ".join(
                    reasons
                ),
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
        "crop",
        "photo",
        "vehicle",
        "qwen_candidate",
        "candidate_exists_in_registry",
        "dino_rank",
        "dino_best_similarity",
        "fusion_candidate",
        "fusion_classification",
        "fusion_agrees",
        "disposition",
        "reasons",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

corroborated = sum(
    1
    for row in results
    if row[
        "disposition"
    ] == "CORROBORATED"
)

supported_review = sum(
    1
    for row in results
    if row[
        "disposition"
    ] == "SUPPORTED_REVIEW"
)

conflicting = sum(
    1
    for row in results
    if row[
        "disposition"
    ] == "CONFLICTING"
)

known_number_review = sum(
    1
    for row in results
    if row[
        "disposition"
    ] == "KNOWN_NUMBER_REVIEW"
)


unsupported = sum(
    1
    for row in results
    if row[
        "disposition"
    ] == "UNSUPPORTED"
)


print("=" * 72)
print("CANDIDATE RESOLUTION TEST COMPLETE")
print("=" * 72)

print(
    f"Qwen candidates evaluated: "
    f"{len(results)}"
)

print(
    f"CORROBORATED: "
    f"{corroborated}"
)

print(
    f"SUPPORTED_REVIEW: "
    f"{supported_review}"
)

print(
    f"KNOWN_NUMBER_REVIEW: "
    f"{known_number_review}"
)

print(
    f"CONFLICTING: "
    f"{conflicting}"
)


print(
    f"UNSUPPORTED: "
    f"{unsupported}"
)

print()

print(
    f"Results CSV: "
    f"{RESULTS_PATH}"
)