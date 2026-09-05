import json
from pathlib import Path
import tempfile
import unittest

from racesort.qwen import QwenClient, QwenResponseCache, hash_file


class FakeChat:
    def __init__(self, response="  007  "):
        self.response = response
        self.calls = []

    def __call__(self, **arguments):
        self.calls.append(arguments)
        return {"message": {"content": self.response}}


class QwenResponseCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image = self.root / "crop.jpg"
        self.image.write_bytes(b"original crop bytes")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_hash_uses_image_content(self):
        first_hash = hash_file(self.image)
        self.image.write_bytes(b"different crop bytes")
        self.assertNotEqual(first_hash, hash_file(self.image))

    def test_round_trip_preserves_string_response(self):
        cache = QwenResponseCache(self.root / "cache", enabled=True)
        path, identity = cache.locate(self.image, "model", "prompt")
        cache.save(path, identity, "007")

        raw, loaded_path, loaded_identity = cache.load(
            self.image, "model", "prompt"
        )
        self.assertEqual(raw, "007")
        self.assertEqual(loaded_path, path)
        self.assertEqual(loaded_identity, identity)

    def test_invalid_cache_file_is_a_safe_miss(self):
        cache = QwenResponseCache(self.root / "cache", enabled=True)
        path, _ = cache.locate(self.image, "model", "prompt")
        path.parent.mkdir(parents=True)
        path.write_text("not JSON", encoding="utf-8")

        raw, _, _ = cache.load(self.image, "model", "prompt")
        self.assertIsNone(raw)


class QwenClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image = self.root / "crop.jpg"
        self.image.write_bytes(b"crop")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_cache_miss_then_hit_calls_model_once(self):
        chat = FakeChat()
        cache = QwenResponseCache(self.root / "cache", enabled=True)
        client = QwenClient("qwen-model", chat, cache)

        first = client.ask(self.image, "read it", use_cache=True)
        second = client.ask(self.image, "read it", use_cache=True)

        self.assertTrue(first.cache_miss)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.elapsed_seconds, 0.0)
        self.assertEqual(second.raw_response, "007")
        self.assertEqual(len(chat.calls), 1)

    def test_json_format_is_forwarded_without_enabling_cache(self):
        chat = FakeChat('{"race_number": {"value": "0"}}')
        client = QwenClient("qwen-model", chat)

        result = client.ask(self.image, "profile", json_format=True)

        self.assertEqual(json.loads(result.raw_response)["race_number"]["value"], "0")
        self.assertEqual(chat.calls[0]["format"], "json")
        self.assertFalse(result.cache_hit)
        self.assertFalse(result.cache_miss)


if __name__ == "__main__":
    unittest.main()
