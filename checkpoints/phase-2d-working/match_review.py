from pathlib import Path
import csv
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

REGISTRY_PATH = (
    OUTPUT_DIR
    / "vehicle-registry.json"
)

MATCHES_JSON_PATH = (
    OUTPUT_DIR
    / "review-matches.json"
)

MATCHES_CSV_PATH = (
    OUTPUT_DIR
    / "review-matches.csv"
)

MODEL_NAME = "facebook/dinov2-small"

TOP_MATCHES = 3


# ============================================================
# OFFLINE MODE
#
# DINOv2 should already be cached locally.
# These settings prevent Hugging Face access at the track.
# ============================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD REGISTRY
# ============================================================

print(
    f"Loading registry: "
    f"{REGISTRY_PATH}"
)

with open(
    REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    registry_data = json.load(file)


confirmed_vehicles = (
    registry_data["vehicles"]
)

review_observations = (
    registry_data["review_observations"]
)

print(
    f"Confirmed identities: "
    f"{len(confirmed_vehicles)}"
)

print(
    f"Review observations: "
    f"{len(review_observations)}"
)

print()


# ============================================================
# LOAD DINOV2
# ============================================================

print("Loading DINOv2...")

processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
)

model = (
    AutoModel
    .from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
)

model.to(DEVICE)
model.eval()

print("DINOv2 loaded.")
print()


# ============================================================
# EMBEDDING CACHE
# ============================================================

embedding_cache = {}


# ============================================================
# HELPERS
# ============================================================

def photo_folder(photo_name):
    """
    Convert:
        GGBM0005.JPG

    into:
        test-output/GGBM0005
    """

    return (
        OUTPUT_DIR
        / Path(photo_name).stem
    )


def crop_path(
    photo_name,
    crop_name,
):
    """
    Build the complete path to a vehicle crop.
    """

    return (
        photo_folder(photo_name)
        / crop_name
    )


def create_embedding(image_path):
    """
    Create one normalized DINOv2 embedding.

    Results are cached so the same image is only
    processed once per run.
    """

    image_path = Path(image_path)

    cache_key = str(
        image_path.resolve()
    )

    if cache_key in embedding_cache:
        return embedding_cache[
            cache_key
        ]

    if not image_path.exists():
        raise FileNotFoundError(
            f"Crop not found: "
            f"{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value
        in inputs.items()
    }

    with torch.no_grad():
        outputs = model(
            **inputs
        )

    # CLS token = overall visual representation
    embedding = (
        outputs
        .last_hidden_state[
            :, 0, :
        ]
    )

    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    embedding = (
        embedding
        .cpu()
    )

    embedding_cache[
        cache_key
    ] = embedding

    return embedding


def cosine_similarity(
    embedding_a,
    embedding_b,
):
    """
    Compare two normalized embeddings.
    """

    return (
        F.cosine_similarity(
            embedding_a,
            embedding_b,
        )
        .item()
    )


def summarize_candidate(
    race_number,
    comparisons,
):
    """
    Given all confirmed-sighting comparisons for one
    race number, calculate:

    - best similarity
    - mean similarity
    - support count
    - best supporting photo/crop
    """

    similarities = [
        item["similarity"]
        for item in comparisons
    ]

    best_match = max(
        comparisons,
        key=lambda item:
            item["similarity"],
    )

    mean_similarity = (
        sum(similarities)
        / len(similarities)
    )

    return {
        "race_number":
            race_number,

        "best_similarity":
            best_match[
                "similarity"
            ],

        "mean_similarity":
            mean_similarity,

        "supporting_sightings":
            len(comparisons),

        "best_confirmed_photo":
            best_match[
                "confirmed_photo"
            ],

        "best_confirmed_crop":
            best_match[
                "confirmed_crop"
            ],

        "all_similarities":
            similarities,
    }


# ============================================================
# BUILD CONFIRMED CROP LIST
# ============================================================

confirmed_crops = []

for race_number, vehicle in (
    confirmed_vehicles.items()
):

    for observation in vehicle[
        "observations"
    ]:

        path = crop_path(
            observation["photo"],
            observation["crop"],
        )

        confirmed_crops.append(
            {
                "race_number":
                    race_number,

                "photo":
                    observation["photo"],

                "crop":
                    observation["crop"],

                "path":
                    path,
            }
        )


print(
    f"Confirmed crop sightings: "
    f"{len(confirmed_crops)}"
)

print()


# ============================================================
# PRECOMPUTE CONFIRMED EMBEDDINGS
# ============================================================

print(
    "Embedding confirmed sightings..."
)

for index, confirmed in enumerate(
    confirmed_crops,
    start=1,
):

    print(
        f"  [{index}/"
        f"{len(confirmed_crops)}] "
        f"#{confirmed['race_number']} "
        f"{confirmed['photo']}"
    )

    confirmed[
        "embedding"
    ] = create_embedding(
        confirmed["path"]
    )

print()


# ============================================================
# MATCH REVIEW OBSERVATIONS
# ============================================================

all_match_results = []

csv_rows = []


for review_index, review in enumerate(
    review_observations,
    start=1,
):

    review_photo = (
        review["photo"]
    )

    review_crop = (
        review["crop"]
    )

    review_path = crop_path(
        review_photo,
        review_crop,
    )

    print("=" * 70)

    print(
        f"[{review_index}/"
        f"{len(review_observations)}] "
        f"{review_photo} "
        f"{review_crop}"
    )

    print("=" * 70)

    review_embedding = (
        create_embedding(
            review_path
        )
    )


    # --------------------------------------------------------
    # COMPARE REVIEW CROP TO EVERY CONFIRMED CROP
    # --------------------------------------------------------

    comparisons_by_number = {}

    for confirmed in confirmed_crops:

        similarity = (
            cosine_similarity(
                review_embedding,
                confirmed[
                    "embedding"
                ],
            )
        )

        race_number = (
            confirmed[
                "race_number"
            ]
        )

        comparison = {
            "race_number":
                race_number,

            "similarity":
                similarity,

            "confirmed_photo":
                confirmed[
                    "photo"
                ],

            "confirmed_crop":
                confirmed[
                    "crop"
                ],
        }

        if race_number not in (
            comparisons_by_number
        ):
            comparisons_by_number[
                race_number
            ] = []

        comparisons_by_number[
            race_number
        ].append(
            comparison
        )


    # --------------------------------------------------------
    # SUMMARIZE EACH RACE NUMBER
    # --------------------------------------------------------

    candidate_summaries = []

    for (
        race_number,
        comparisons
    ) in comparisons_by_number.items():

        summary = summarize_candidate(
            race_number,
            comparisons,
        )

        candidate_summaries.append(
            summary
        )


    # --------------------------------------------------------
    # RANK CANDIDATES
    #
    # For now, ranking still uses BEST similarity.
    # We are reporting mean/support alongside it so we can
    # inspect whether they help separate identities.
    # --------------------------------------------------------

    ranked = sorted(
        candidate_summaries,
        key=lambda item:
            item[
                "best_similarity"
            ],
        reverse=True,
    )

    all_candidates = ranked

    top_candidates = (
        ranked[
            :TOP_MATCHES
        ]
    )


    # --------------------------------------------------------
    # DISPLAY CANDIDATES
    # --------------------------------------------------------

    print(
        f"Profile number observation: "
        f"{review.get('profile_number')}"
    )

    print(
        f"Verification observation: "
        f"{review.get('verification_number')}"
    )

    print()

    for rank, candidate in enumerate(
        top_candidates,
        start=1,
    ):

        print(
            f"{rank}. "
            f"#{candidate['race_number']}"
        )

        print(
            f"   best similarity: "
            f"{candidate['best_similarity']:.4f}"
        )

        print(
            f"   mean similarity: "
            f"{candidate['mean_similarity']:.4f}"
        )

        print(
            f"   supporting sightings: "
            f"{candidate['supporting_sightings']}"
        )

        print(
            f"   best match: "
            f"{candidate['best_confirmed_photo']} "
            f"{candidate['best_confirmed_crop']}"
        )

        print()


    # --------------------------------------------------------
    # TOP-CANDIDATE MARGINS
    # --------------------------------------------------------

    best_margin = None
    mean_margin = None

    if len(top_candidates) >= 2:

        best_margin = (
            top_candidates[0][
                "best_similarity"
            ]
            -
            top_candidates[1][
                "best_similarity"
            ]
        )

        mean_margin = (
            top_candidates[0][
                "mean_similarity"
            ]
            -
            top_candidates[1][
                "mean_similarity"
            ]
        )


    # --------------------------------------------------------
    # STORE FULL RESULT
    # --------------------------------------------------------

    result = {
        "photo":
            review_photo,

        "vehicle":
            review[
                "vehicle"
            ],

        "crop":
            review_crop,

        "profile_number":
            review.get(
                "profile_number"
            ),

        "verification_number":
            review.get(
                "verification_number"
            ),

        "candidates":
            all_candidates,

        "displayed_top_candidates":
            top_candidates,

        "best_similarity_margin":
            best_margin,

        "mean_similarity_margin":
            mean_margin,
    }

    all_match_results.append(
        result
    )


    # --------------------------------------------------------
    # CSV SUMMARY
    # --------------------------------------------------------

    row = {
        "photo":
            review_photo,

        "vehicle":
            review[
                "vehicle"
            ],

        "crop":
            review_crop,

        "profile_number":
            review.get(
                "profile_number"
            ),

        "verification_number":
            review.get(
                "verification_number"
            ),

        "top_candidate":
            None,

        "top_best_similarity":
            None,

        "top_mean_similarity":
            None,

        "top_supporting_sightings":
            None,

        "second_candidate":
            None,

        "second_best_similarity":
            None,

        "second_mean_similarity":
            None,

        "second_supporting_sightings":
            None,

        "best_similarity_margin":
            best_margin,

        "mean_similarity_margin":
            mean_margin,
    }


    if len(top_candidates) >= 1:

        top = (
            top_candidates[0]
        )

        row[
            "top_candidate"
        ] = (
            top[
                "race_number"
            ]
        )

        row[
            "top_best_similarity"
        ] = (
            top[
                "best_similarity"
            ]
        )

        row[
            "top_mean_similarity"
        ] = (
            top[
                "mean_similarity"
            ]
        )

        row[
            "top_supporting_sightings"
        ] = (
            top[
                "supporting_sightings"
            ]
        )


    if len(top_candidates) >= 2:

        second = (
            top_candidates[1]
        )

        row[
            "second_candidate"
        ] = (
            second[
                "race_number"
            ]
        )

        row[
            "second_best_similarity"
        ] = (
            second[
                "best_similarity"
            ]
        )

        row[
            "second_mean_similarity"
        ] = (
            second[
                "mean_similarity"
            ]
        )

        row[
            "second_supporting_sightings"
        ] = (
            second[
                "supporting_sightings"
            ]
        )


    csv_rows.append(
        row
    )


# ============================================================
# SAVE JSON
# ============================================================

with open(
    MATCHES_JSON_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "model":
                MODEL_NAME,

            "review_matches":
                all_match_results,
        },
        file,
        indent=2,
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    MATCHES_CSV_PATH,
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

        "top_candidate",
        "top_best_similarity",
        "top_mean_similarity",
        "top_supporting_sightings",

        "second_candidate",
        "second_best_similarity",
        "second_mean_similarity",
        "second_supporting_sightings",

        "best_similarity_margin",
        "mean_similarity_margin",
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

print("=" * 70)
print("REVIEW MATCHING COMPLETE")
print("=" * 70)

print(
    f"Review observations: "
    f"{len(review_observations)}"
)

print(
    f"Embeddings computed/cached: "
    f"{len(embedding_cache)}"
)

print(
    f"JSON results: "
    f"{MATCHES_JSON_PATH}"
)

print(
    f"CSV results: "
    f"{MATCHES_CSV_PATH}"
)