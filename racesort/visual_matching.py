"""DINOv2 device, embedding, and similarity helpers for RaceSort."""

from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F


def resolve_dino_device():
    """Choose CUDA, Apple MPS, or CPU without platform-specific forks."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_dino_embedding(image_path, processor, model, device, cache):
    """Create and cache one normalized DINO CLS embedding."""

    image_path = Path(image_path)
    cache_key = str(image_path.resolve())
    if cache_key in cache:
        return cache[cache_key]

    with Image.open(image_path) as image:
        inputs = processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = F.normalize(
        outputs.last_hidden_state[:, 0, :],
        p=2,
        dim=1,
    ).cpu()
    cache[cache_key] = embedding
    return embedding


def cosine_similarity_score(first_embedding, second_embedding):
    """Return one scalar cosine-similarity score."""

    return F.cosine_similarity(first_embedding, second_embedding).item()
