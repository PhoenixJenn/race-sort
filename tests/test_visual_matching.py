from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from racesort.visual_matching import (
    cosine_similarity_score,
    create_dino_embedding,
    resolve_dino_device,
)


class DeviceSelectionTests(unittest.TestCase):
    @patch("racesort.visual_matching.torch.backends.mps.is_available")
    @patch("racesort.visual_matching.torch.cuda.is_available")
    def test_cuda_has_first_priority(self, cuda_available, mps_available):
        cuda_available.return_value = True
        mps_available.return_value = True
        self.assertEqual(resolve_dino_device().type, "cuda")

    @patch("racesort.visual_matching.torch.backends.mps.is_available")
    @patch("racesort.visual_matching.torch.cuda.is_available")
    def test_mps_is_used_when_cuda_is_unavailable(
        self, cuda_available, mps_available
    ):
        cuda_available.return_value = False
        mps_available.return_value = True
        self.assertEqual(resolve_dino_device().type, "mps")

    @patch("racesort.visual_matching.torch.backends.mps.is_available")
    @patch("racesort.visual_matching.torch.cuda.is_available")
    def test_cpu_is_the_portable_fallback(
        self, cuda_available, mps_available
    ):
        cuda_available.return_value = False
        mps_available.return_value = False
        self.assertEqual(resolve_dino_device().type, "cpu")


class FakeProcessor:
    def __init__(self):
        self.image_modes = []

    def __call__(self, images, return_tensors):
        self.image_modes.append(images.mode)
        return {"pixel_values": torch.ones((1, 3, 2, 2))}


class FakeOutputs:
    last_hidden_state = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, **inputs):
        self.calls += 1
        return FakeOutputs()


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "crop.png"
        Image.new("L", (2, 2), color=128).save(self.image_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_embedding_is_rgb_normalized_and_cached(self):
        processor = FakeProcessor()
        model = FakeModel()
        cache = {}

        first = create_dino_embedding(
            self.image_path, processor, model, torch.device("cpu"), cache
        )
        second = create_dino_embedding(
            self.image_path, processor, model, torch.device("cpu"), cache
        )

        self.assertIs(first, second)
        self.assertEqual(model.calls, 1)
        self.assertEqual(processor.image_modes, ["RGB"])
        self.assertTrue(torch.allclose(first, torch.tensor([[0.6, 0.8]])))

    def test_cosine_similarity_returns_scalar(self):
        first = torch.tensor([[1.0, 0.0]])
        second = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(cosine_similarity_score(first, second), 0.0)


if __name__ == "__main__":
    unittest.main()
