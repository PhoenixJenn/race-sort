from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/dinov2-small"
OUTPUT_DIR = Path("test-output")


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("DINOv2 loaded.")
print()


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def create_embedding(image_path):
    """
    Convert one vehicle crop into a normalized DINOv2
    visual embedding.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token = overall visual representation
    embedding = outputs.last_hidden_state[:, 0, :]

    # Normalize so cosine similarity is straightforward.
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return embedding.cpu()


# ============================================================
# SIMILARITY
# ============================================================

def compare_images(path_a, path_b):

    embedding_a = create_embedding(
        path_a
    )

    embedding_b = create_embedding(
        path_b
    )

    similarity = F.cosine_similarity(
        embedding_a,
        embedding_b,
    ).item()

    print()
    print(f"A: {path_a}")
    print(f"B: {path_b}")

    print(
        f"Similarity: "
        f"{similarity:.4f}"
    )

    return similarity


# ============================================================
# FIRST TEST
# ============================================================

# Change these paths to two KNOWN sightings
# of the same motorcycle.

IMAGE_A = (
    OUTPUT_DIR
    / "GGBM0007"
    / "motorcycle-01.jpg"
)

IMAGE_B = (
    OUTPUT_DIR
    / "GGBM0017"
    / "motorcycle-01.jpg"
)



compare_images(
    IMAGE_A,
    IMAGE_B,
)