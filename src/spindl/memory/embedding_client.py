"""
EmbeddingClient — HTTP client for OpenAI-compatible /v1/embeddings endpoint.

Backend-agnostic: works with llama.cpp (--embedding mode), Ollama, or any
OpenAI-compatible embedding API. The `model` field in requests is informational
only — llama.cpp ignores it and serves whatever model is loaded.

Part of NANO-043 Phase 1.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """HTTP client for embedding text via /v1/embeddings endpoint."""

    def __init__(self, base_url: str, timeout: float = 10.0, model: str = "default"):
        """
        Args:
            base_url: Base URL of embedding server (e.g., "http://127.0.0.1:5559").
                      Trailing slash is stripped if present.
            timeout: HTTP request timeout in seconds.
            model: Model name sent in requests. Informational only for llama.cpp.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._model = model
        self._endpoint = f"{self._base_url}/v1/embeddings"

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If the server request fails or returns unexpected data.
        """
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single request.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text. Order matches input.

        Raises:
            EmbeddingError: If the server request fails or returns unexpected data.
        """
        if not texts:
            return []

        payload = {
            "input": texts,
            "model": self._model,
        }

        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError as e:
            raise EmbeddingError(
                f"Cannot connect to embedding server at {self._base_url}. "
                f"Is the server running? Error: {e}"
            ) from e
        except requests.Timeout as e:
            raise EmbeddingError(
                f"Embedding request timed out after {self._timeout}s. "
                f"Try increasing timeout or reducing batch size."
            ) from e
        except requests.HTTPError as e:
            raise EmbeddingError(
                f"Embedding server returned HTTP {resp.status_code}: {resp.text}"
            ) from e

        data = resp.json()

        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        try:
            items = sorted(data["data"], key=lambda x: x["index"])
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as e:
            raise EmbeddingError(
                f"Unexpected response format from embedding server: {e}"
            ) from e

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, got {len(vectors)}"
            )

        return vectors

    def health_check(self) -> bool:
        """
        Check if the embedding server is reachable.

        Returns:
            True if the server's /health endpoint responds with status "ok".
        """
        try:
            resp = requests.get(
                f"{self._base_url}/health",
                timeout=self._timeout,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    @property
    def base_url(self) -> str:
        """Base URL of the embedding server."""
        return self._base_url

    @property
    def dimension(self) -> Optional[int]:
        """
        Probe the embedding dimension by embedding a test string.

        Returns:
            Embedding dimension, or None if the server is unreachable.
        """
        try:
            vec = self.embed("dimension probe")
            return len(vec)
        except EmbeddingError:
            return None


class EmbeddingError(Exception):
    """Raised when an embedding operation fails."""
    pass
