"""Tests for NANO-102: CurationClient — OpenRouter LLM judge with function calling."""

import json
from unittest.mock import MagicMock, patch

import pytest

from spindl.memory.curation_client import CurationClient, CurationDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openrouter_response(
    action: str,
    target_id: str = None,
    merged_text: str = None,
) -> dict:
    """Build a mock OpenRouter response with a tool call."""
    args = {"action": action}
    if target_id is not None:
        args["target_id"] = target_id
    if merged_text is not None:
        args["merged_text"] = merged_text

    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_test_001",
                            "function": {
                                "name": "memory_decision",
                                "arguments": json.dumps(args),
                            },
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
    }


def _make_text_response(text: str) -> dict:
    """Build a mock OpenRouter response with free text (no tool call)."""
    return {
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": "stop",
            }
        ],
    }


EXISTING = [
    {"id": "abc123", "content": "User likes Monster Energy drinks", "distance": 0.15},
]


@pytest.fixture
def client() -> CurationClient:
    return CurationClient(
        api_key="sk-or-v1-test-key",
        model="anthropic/claude-haiku-4-5",
        timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Happy path: valid tool call responses
# ---------------------------------------------------------------------------


class TestClassifyHappyPath:

    @patch("spindl.memory.curation_client.requests.post")
    def test_classify_skip(self, mock_post, client):
        """SKIP: model says memory is already captured."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response("SKIP")
        mock_post.return_value = mock_resp

        decision = client.classify("User's favorite drink is Monster", EXISTING)
        assert decision.action == "SKIP"
        assert decision.target_id is None
        assert decision.merged_text is None

    @patch("spindl.memory.curation_client.requests.post")
    def test_classify_add(self, mock_post, client):
        """ADD: model says this is genuinely new information."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response("ADD")
        mock_post.return_value = mock_resp

        decision = client.classify("User also drinks C4 Performance", EXISTING)
        assert decision.action == "ADD"

    @patch("spindl.memory.curation_client.requests.post")
    def test_classify_update(self, mock_post, client):
        """UPDATE: model merges new info into existing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "UPDATE",
            target_id="abc123",
            merged_text="User likes Monster Energy and C4 Performance drinks",
        )
        mock_post.return_value = mock_resp

        decision = client.classify("User also likes C4 Performance", EXISTING)
        assert decision.action == "UPDATE"
        assert decision.target_id == "abc123"
        assert "Monster Energy" in decision.merged_text
        assert "C4 Performance" in decision.merged_text

    @patch("spindl.memory.curation_client.requests.post")
    def test_classify_delete(self, mock_post, client):
        """DELETE: model says new memory contradicts existing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "DELETE",
            target_id="abc123",
        )
        mock_post.return_value = mock_resp

        decision = client.classify("User has quit energy drinks entirely", EXISTING)
        assert decision.action == "DELETE"
        assert decision.target_id == "abc123"


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestClassifyFallbacks:

    @patch("spindl.memory.curation_client.requests.post")
    def test_fallback_on_timeout(self, mock_post, client):
        """API timeout → ADD (permissive)."""
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout("timed out")

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "ADD"
        assert decision.raw_response == "TIMEOUT"

    @patch("spindl.memory.curation_client.requests.post")
    def test_fallback_on_connection_error(self, mock_post, client):
        """API unreachable → ADD (permissive)."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("unreachable")

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "ADD"
        assert decision.raw_response == "CONNECTION_ERROR"

    @patch("spindl.memory.curation_client.requests.post")
    def test_fallback_on_http_error(self, mock_post, client):
        """Non-200 status → ADD."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "ADD"

    @patch("spindl.memory.curation_client.requests.post")
    def test_fallback_on_free_text(self, mock_post, client):
        """Model returns text instead of tool call → ADD."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_text_response(
            "I think this is a duplicate, you should skip it."
        )
        mock_post.return_value = mock_resp

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "ADD"


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


class TestValidationGuards:

    @patch("spindl.memory.curation_client.requests.post")
    def test_rejects_hallucinated_id(self, mock_post, client):
        """UPDATE with target_id not in existing list → ADD."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "UPDATE",
            target_id="hallucinated_999",
            merged_text="Some merged content here.",
        )
        mock_post.return_value = mock_resp

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "ADD"

    @patch("spindl.memory.curation_client.requests.post")
    def test_rejects_empty_merged_text(self, mock_post, client):
        """UPDATE with empty merged_text → SKIP."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "UPDATE",
            target_id="abc123",
            merged_text="",
        )
        mock_post.return_value = mock_resp

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "SKIP"

    @patch("spindl.memory.curation_client.requests.post")
    def test_rejects_short_merged_text(self, mock_post, client):
        """UPDATE with merged_text < 10 chars → SKIP."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "UPDATE",
            target_id="abc123",
            merged_text="short",
        )
        mock_post.return_value = mock_resp

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "SKIP"

    @patch("spindl.memory.curation_client.requests.post")
    def test_rejects_identical_merged_text_candidate(self, mock_post, client):
        """UPDATE where merged_text equals candidate → SKIP."""
        candidate = "User likes Monster Energy drinks a lot"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "UPDATE",
            target_id="abc123",
            merged_text=candidate,
        )
        mock_post.return_value = mock_resp

        decision = client.classify(candidate, EXISTING)
        assert decision.action == "SKIP"

    @patch("spindl.memory.curation_client.requests.post")
    def test_rejects_identical_merged_text_existing(self, mock_post, client):
        """UPDATE where merged_text equals existing content → SKIP."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "UPDATE",
            target_id="abc123",
            merged_text="User likes Monster Energy drinks",
        )
        mock_post.return_value = mock_resp

        decision = client.classify("Some different candidate", EXISTING)
        assert decision.action == "SKIP"

    @patch("spindl.memory.curation_client.requests.post")
    def test_rejects_unknown_action(self, mock_post, client):
        """Unknown action string → ADD."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response("MERGE")
        mock_post.return_value = mock_resp

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "ADD"

    @patch("spindl.memory.curation_client.requests.post")
    def test_delete_with_hallucinated_id(self, mock_post, client):
        """DELETE with invalid target_id → ADD."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response(
            "DELETE",
            target_id="nonexistent_id",
        )
        mock_post.return_value = mock_resp

        decision = client.classify("Some candidate", EXISTING)
        assert decision.action == "ADD"


# ---------------------------------------------------------------------------
# Request formatting
# ---------------------------------------------------------------------------


class TestRequestFormatting:

    @patch("spindl.memory.curation_client.requests.post")
    def test_sends_correct_payload(self, mock_post, client):
        """Verify the request payload structure."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response("SKIP")
        mock_post.return_value = mock_resp

        client.classify("New memory text", EXISTING)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert payload["model"] == "anthropic/claude-haiku-4-5"
        assert payload["temperature"] == 0
        assert payload["stream"] is False
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "memory_decision"
        assert payload["tool_choice"]["function"]["name"] == "memory_decision"

        # Messages: system + user
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert "NEW:" in payload["messages"][1]["content"]
        assert "EXISTING:" in payload["messages"][1]["content"]

    @patch("spindl.memory.curation_client.requests.post")
    def test_uses_custom_system_prompt(self, mock_post):
        """Custom system prompt overrides default."""
        custom_prompt = "You are a strict dedup judge. Never allow duplicates."
        custom_client = CurationClient(
            api_key="sk-test",
            system_prompt=custom_prompt,
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_openrouter_response("SKIP")
        mock_post.return_value = mock_resp

        custom_client.classify("Test", EXISTING)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["messages"][0]["content"] == custom_prompt

    def test_env_var_resolution(self):
        """API key with ${VAR} pattern resolves from environment."""
        import os
        os.environ["TEST_CURATION_KEY"] = "sk-or-v1-resolved"
        try:
            c = CurationClient(api_key="${TEST_CURATION_KEY}")
            assert c._api_key == "sk-or-v1-resolved"
        finally:
            del os.environ["TEST_CURATION_KEY"]

    def test_env_var_missing_raises(self):
        """Missing env var raises ValueError."""
        with pytest.raises(ValueError, match="not set"):
            CurationClient(api_key="${MISSING_VAR_12345}")
