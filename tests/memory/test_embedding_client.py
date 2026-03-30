"""Tests for EmbeddingClient — /v1/embeddings HTTP client."""

import json
from unittest.mock import patch, MagicMock

import pytest
import requests

from spindl.memory.embedding_client import EmbeddingClient, EmbeddingError


class TestEmbeddingClientInit:
    """Tests for EmbeddingClient initialization."""

    def test_strips_trailing_slash(self) -> None:
        """Trailing slash on base_url is stripped."""
        client = EmbeddingClient("http://localhost:5559/")
        assert client.base_url == "http://localhost:5559"

    def test_preserves_clean_url(self) -> None:
        """Clean base_url is preserved as-is."""
        client = EmbeddingClient("http://localhost:5559")
        assert client.base_url == "http://localhost:5559"


class TestEmbedBatch:
    """Tests for embed_batch method."""

    def test_empty_input_returns_empty(self) -> None:
        """Empty input list returns empty output without making a request."""
        client = EmbeddingClient("http://localhost:5559")
        result = client.embed_batch([])
        assert result == []

    @patch("spindl.memory.embedding_client.requests.post")
    def test_single_text_embedding(self, mock_post: MagicMock) -> None:
        """Single text is embedded correctly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test",
        }
        mock_post.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        result = client.embed_batch(["hello"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

    @patch("spindl.memory.embedding_client.requests.post")
    def test_batch_embedding_preserves_order(self, mock_post: MagicMock) -> None:
        """Multiple texts are returned in correct order regardless of response order."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Response arrives out of order
        mock_resp.json.return_value = {
            "data": [
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.7, 0.8, 0.9], "index": 2},
            ],
            "model": "test",
        }
        mock_post.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        result = client.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]  # index 0
        assert result[1] == [0.4, 0.5, 0.6]  # index 1
        assert result[2] == [0.7, 0.8, 0.9]  # index 2

    @patch("spindl.memory.embedding_client.requests.post")
    def test_sends_correct_payload(self, mock_post: MagicMock) -> None:
        """Request payload matches OpenAI embedding format."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1], "index": 0}],
        }
        mock_post.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559", model="snowflake")
        client.embed_batch(["test"])

        call_kwargs = mock_post.call_args
        assert call_kwargs.args[0] == "http://localhost:5559/v1/embeddings"
        payload = call_kwargs.kwargs["json"]
        assert payload["input"] == ["test"]
        assert payload["model"] == "snowflake"


class TestEmbedSingle:
    """Tests for embed (single text) method."""

    @patch("spindl.memory.embedding_client.requests.post")
    def test_returns_single_vector(self, mock_post: MagicMock) -> None:
        """embed() returns a flat list, not nested."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        }
        mock_post.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        result = client.embed("hello")

        assert result == [0.1, 0.2, 0.3]
        assert isinstance(result, list)
        assert not isinstance(result[0], list)


class TestEmbeddingErrors:
    """Tests for error handling."""

    @patch("spindl.memory.embedding_client.requests.post")
    def test_connection_error_raises_embedding_error(self, mock_post: MagicMock) -> None:
        """Connection failure raises EmbeddingError with helpful message."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        client = EmbeddingClient("http://localhost:5559")
        with pytest.raises(EmbeddingError, match="Cannot connect"):
            client.embed("hello")

    @patch("spindl.memory.embedding_client.requests.post")
    def test_timeout_raises_embedding_error(self, mock_post: MagicMock) -> None:
        """Timeout raises EmbeddingError with helpful message."""
        mock_post.side_effect = requests.Timeout("Timed out")

        client = EmbeddingClient("http://localhost:5559", timeout=5.0)
        with pytest.raises(EmbeddingError, match="timed out"):
            client.embed("hello")

    @patch("spindl.memory.embedding_client.requests.post")
    def test_http_error_raises_embedding_error(self, mock_post: MagicMock) -> None:
        """HTTP error status raises EmbeddingError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_post.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        with pytest.raises(EmbeddingError, match="HTTP 500"):
            client.embed("hello")

    @patch("spindl.memory.embedding_client.requests.post")
    def test_malformed_response_raises_embedding_error(self, mock_post: MagicMock) -> None:
        """Unexpected JSON structure raises EmbeddingError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unexpected": "format"}
        mock_post.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        with pytest.raises(EmbeddingError, match="Unexpected response"):
            client.embed("hello")

    @patch("spindl.memory.embedding_client.requests.post")
    def test_count_mismatch_raises_embedding_error(self, mock_post: MagicMock) -> None:
        """Mismatched embedding count raises EmbeddingError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1], "index": 0}],
        }
        mock_post.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        with pytest.raises(EmbeddingError, match="Expected 2 embeddings, got 1"):
            client.embed_batch(["a", "b"])


class TestHealthCheck:
    """Tests for health_check method."""

    @patch("spindl.memory.embedding_client.requests.get")
    def test_healthy_server(self, mock_get: MagicMock) -> None:
        """Returns True when server is healthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        assert client.health_check() is True

    @patch("spindl.memory.embedding_client.requests.get")
    def test_unreachable_server(self, mock_get: MagicMock) -> None:
        """Returns False when server is unreachable."""
        mock_get.side_effect = requests.ConnectionError()

        client = EmbeddingClient("http://localhost:5559")
        assert client.health_check() is False

    @patch("spindl.memory.embedding_client.requests.get")
    def test_unhealthy_server(self, mock_get: MagicMock) -> None:
        """Returns False when server returns non-200."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_get.return_value = mock_resp

        client = EmbeddingClient("http://localhost:5559")
        assert client.health_check() is False
