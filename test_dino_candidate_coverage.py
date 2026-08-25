from pathlib import Path
import csv
import json
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
)


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

RESULTS_PATH = (
    OUTPUT_DIR
    / "dino-candidate-coverage-results.csv"
)

MODEL_NAME = "facebook/dinov2-small"


# ============================================================
# OFFLINE MODE
# ============================================================

os.environ[
    "HF_HUB_OFFLINE"
] = "1"

os.environ[
    "TRANSFORMERS_OFFLINE"
] = "1"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print(
    f"Using device: "
    f"{DEVICE}"
)

print()


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

        return json.load(
            file
        )


def normalize_number(value):

    if value is None:
        return None

    value = (
        str(value)
        .strip()
        .upper()
    )

    if not value:
        return None

    return value


def photo_folder(
    photo_name,
):
    """
    Convert:

        GGBM0005.JPG

    into:

        test-output/GGBM0005
    """

    return (
        OUTPUT_DIR
        / Path(
            photo_name
        ).stem
    )


def registry_crop_path(
    observation,
):
    """
    Build full crop path from one registry observation.
    """

    return (
        photo_folder(
            observation[
                "photo"
            ]
        )
        / observation[
            "crop"
        ]
    )


# ============================================================
# LOAD DATA
# ============================================================

routing_rows = load_csv(
    ROUTING_RESULTS_PATH
)

registry = load_json(
    REGISTRY_PATH
)


known_vehicles = (
    registry[
        "vehicles"
    ]
)


# ============================================================
# FIND KNOWN QWEN CANDIDATES
# ============================================================

candidate_rows = []


for row in routing_rows:

    if (
        row[
            "decision"
        ]
        !=
        "QWEN_CANDIDATE"
    ):
        continue


    candidate_number = (
        normalize_number(
            row[
                "final_number"
            ]
        )
    )


    if (
        candidate_number
        not in known_vehicles
    ):
        continue


    candidate_rows.append(
        row
    )


print(
    f"Known Qwen candidates "
    f"to evaluate: "
    f"{len(candidate_rows)}"
)

print()


# ============================================================
# LOAD DINOV2
# ============================================================

print(
    "Loading DINOv2..."
)

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

model.to(
    DEVICE
)

model.eval()

print(
    "DINOv2 loaded."
)

print()


# ============================================================
# EMBEDDING CACHE
# ============================================================

embedding_cache = {}


def create_embedding(
    image_path,
):
    """
    Create one normalized DINOv2 embedding.

    Cache results so every image is encoded only once.
    """

    image_path = Path(
        image_path
    )

    key = str(
        image_path.resolve()
    )


    if key in embedding_cache:

        return (
            embedding_cache[
                key
            ]
        )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Crop not found: "
            f"{image_path}"
        )


    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    inputs = processor(
        images=image,
        return_tensors="pt",
    )


    inputs = {
        key:
            value.to(
                DEVICE
            )

        for key, value
        in inputs.items()
    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    embedding = (
        outputs
        .last_hidden_state[
            :, 0, :
        ]
    )


    embedding = (
        F.normalize(
            embedding,
            p=2,
            dim=1,
        )
    )


    embedding = (
        embedding.cpu()
    )


    embedding_cache[
        key
    ] = embedding


    return embedding


def similarity(
    embedding_a,
    embedding_b,
):

    return (
        F.cosine_similarity(
            embedding_a,
            embedding_b,
        )
        .item()
    )


# ============================================================
# EVALUATE CANDIDATES
# ============================================================

results = []

batch_start = (
    time.perf_counter()
)


for index, row in enumerate(
    candidate_rows,
    start=1,
):

    crop_path = Path(
        row[
            "crop"
        ]
    )


    candidate_number = (
        normalize_number(
            row[
                "final_number"
            ]
        )
    )


    vehicle = (
        known_vehicles[
            candidate_number
        ]
    )


    observations = (
        vehicle[
            "observations"
        ]
    )


    print("=" * 72)

    print(
        f"[{index}/"
        f"{len(candidate_rows)}] "
        f"{crop_path}"
    )

    print("=" * 72)

    print(
        f"Qwen candidate: "
        f"#{candidate_number}"
    )

    print(
        f"Registry references: "
        f"{len(observations)}"
    )

    print()


    # --------------------------------------------------------
    # Embed candidate crop
    # --------------------------------------------------------

    candidate_embedding = (
        create_embedding(
            crop_path
        )
    )


    similarities = []


    # --------------------------------------------------------
    # Compare against every confirmed sighting
    # of this exact race number.
    # --------------------------------------------------------

    for observation in observations:

        reference_path = (
            registry_crop_path(
                observation
            )
        )

        # ----------------------------------------------------
        # Never use the candidate crop as its own reference.
        #
        # A self-match produces similarity ~1.0 but provides
        # zero independent identity evidence.
        # ----------------------------------------------------

        if (
            reference_path.resolve()
            == crop_path.resolve()
        ):

            print(
                f"Skipping self-reference: "
                f"{reference_path}"
            )

            continue


        reference_embedding = (
            create_embedding(
                reference_path
            )
        )


        score = similarity(
            candidate_embedding,
            reference_embedding,
        )


        similarities.append(
            {
                "score":
                    score,

                "photo":
                    observation[
                        "photo"
                    ],

                "crop":
                    observation[
                        "crop"
                    ],

                "path":
                    str(
                        reference_path
                    ),
            }
        )


    # --------------------------------------------------------
    # Sort best-first
    # --------------------------------------------------------

    similarities.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )


    best = (
        similarities[0]
        if similarities
        else None
    )


    mean_similarity = (
        sum(
            item["score"]
            for item
            in similarities
        )
        /
        len(similarities)

        if similarities
        else None
    )


    # --------------------------------------------------------
    # Display all supporting references
    # --------------------------------------------------------

    for rank, item in enumerate(
        similarities,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{item['photo']} "
            f"{item['crop']}"
        )

        print(
            f"   similarity="
            f"{item['score']:.4f}"
        )


    print()


    if best is not None:

        print(
            f"Best similarity: "
            f"{best['score']:.4f}"
        )

        print(
            f"Mean similarity: "
            f"{mean_similarity:.4f}"
        )

        print(
            f"Best reference: "
            f"{best['photo']} "
            f"{best['crop']}"
        )


    print()


    results.append(
        {
            "crop":
                str(
                    crop_path
                ),

            "qwen_candidate":
                candidate_number,

            "registry_reference_count":
                len(
                    observations
                ),

            "best_similarity":
                (
                    best[
                        "score"
                    ]
                    if best
                    else None
                ),

            "mean_similarity":
                mean_similarity,

            "best_reference_photo":
                (
                    best[
                        "photo"
                    ]
                    if best
                    else None
                ),

            "best_reference_crop":
                (
                    best[
                        "crop"
                    ]
                    if best
                    else None
                ),

            "all_similarities":
                " | ".join(
                    f"{item['score']:.4f}"
                    for item
                    in similarities
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
        "qwen_candidate",
        "registry_reference_count",
        "best_similarity",
        "mean_similarity",
        "best_reference_photo",
        "best_reference_crop",
        "all_similarities",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=
            fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

batch_elapsed = (
    time.perf_counter()
    - batch_start
)


print("=" * 72)

print(
    "DINO CANDIDATE COVERAGE "
    "TEST COMPLETE"
)

print("=" * 72)


print(
    f"Known Qwen candidates tested: "
    f"{len(results)}"
)

print(
    f"Embeddings computed: "
    f"{len(embedding_cache)}"
)

print(
    f"Batch elapsed: "
    f"{batch_elapsed:.2f}s"
)

print()

print(
    f"Results CSV: "
    f"{RESULTS_PATH}"
)