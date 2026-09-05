"""Low-level Qwen calls and optional response caching for RaceSort."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time


def hash_file(path):
    """Return a SHA-256 digest without changing the source file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QwenResponseCache:
    """Store raw responses by model, prompt, schema, and image content."""

    def __init__(self, directory, schema_version=1, enabled=False):
        self.directory = Path(directory)
        self.schema_version = schema_version
        self.enabled = enabled

    def identity(self, image_path, model, prompt):
        return {
            "schema_version": self.schema_version,
            "model": model,
            "prompt": prompt,
            "crop_sha256": hash_file(image_path),
        }

    def locate(self, image_path, model, prompt):
        identity = self.identity(image_path, model, prompt)
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        return self.directory / f"{key}.json", identity

    def load(self, image_path, model, prompt):
        """Return raw response and cache metadata, or None on a miss."""

        if not self.enabled:
            return None, None, None

        cache_path, identity = self.locate(image_path, model, prompt)
        if not cache_path.exists():
            return None, cache_path, identity

        try:
            with cache_path.open(encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
        except (OSError, json.JSONDecodeError):
            return None, cache_path, identity

        if (
            cached.get("identity") != identity
            or not isinstance(cached.get("raw_response"), str)
        ):
            return None, cache_path, identity

        return cached["raw_response"], cache_path, identity

    def save(self, cache_path, identity, raw_response):
        """Atomically save one successful raw response."""

        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(
                {"identity": identity, "raw_response": raw_response},
                cache_file,
                indent=2,
            )
        temporary_path.replace(cache_path)


@dataclass(frozen=True)
class QwenResult:
    raw_response: str
    elapsed_seconds: float
    cache_hit: bool = False
    cache_miss: bool = False


class QwenClient:
    """Call a Qwen-compatible chat function with optional raw caching."""

    def __init__(self, model, chat, cache=None):
        self.model = model
        self.chat = chat
        self.cache = cache

    def ask(self, image_path, prompt, *, json_format=False, use_cache=False):
        cache_path = None
        identity = None

        if use_cache and self.cache is not None:
            cached, cache_path, identity = self.cache.load(
                image_path,
                self.model,
                prompt,
            )
            if cached is not None:
                return QwenResult(cached, 0.0, cache_hit=True)

        start = time.perf_counter()
        arguments = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }],
        }
        if json_format:
            arguments["format"] = "json"

        response = self.chat(**arguments)
        elapsed = time.perf_counter() - start
        raw_response = response["message"]["content"].strip()

        if use_cache and self.cache is not None:
            self.cache.save(cache_path, identity, raw_response)

        return QwenResult(
            raw_response,
            elapsed,
            cache_miss=(
                use_cache
                and self.cache is not None
                and self.cache.enabled
            ),
        )
