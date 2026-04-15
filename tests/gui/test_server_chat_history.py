"""Tests for request_chat_history handler (NANO-073a + NANO-075).

Tests cover:
- Returns empty array when no orchestrator
- Returns empty array when no session file
- Returns empty array when session file doesn't exist
- Returns visible turns mapped to frontend format
- Filters out summary and hidden turns
- Caps at 200 messages
- Handles read errors gracefully
- NANO-075: Forwards metadata fields (reasoning, stimulus_source, codex, memories)
- NANO-075: Pre-075 JSONL without metadata fields still works (backward compat)
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.gui.server import GUIServer


# ============================================================================
# Helpers
# ============================================================================


def _make_server(with_orchestrator=False, session_file=None):
    """Create a GUIServer with mocked internals, capturing registered handlers."""
    server = GUIServer.__new__(GUIServer)
    server.sio = MagicMock()
    server.sio.emit = AsyncMock()
    server._config_path = "/tmp/test_config.yaml"
    server._event_loop = asyncio.new_event_loop()
    server._launch_in_progress = False
    server._shutdown_in_progress = False
    server._conversations_dir = None
    server._personas_dir = None
    server._prompt_blocks_config = None
    server._tools_config_cache = None
    server._llm_config_cache = None
    server._vlm_config_cache = None
    server._clients = set()
    server._service_runner = None
    server._log_aggregator = None
    server._launched_services = set()
    server._on_services_ready = None
    server._uvicorn_server = None

    if with_orchestrator:
        server._orchestrator = MagicMock()
        type(server._orchestrator).session_file = PropertyMock(
            return_value=session_file
        )
    else:
        server._orchestrator = None

    # Capture handlers registered via @sio.event
    server._handlers = {}

    def capture_event(fn):
        server._handlers[fn.__name__] = fn
        return fn

    server.sio.event = capture_event
    server._register_handlers()

    return server


def _write_jsonl(filepath: Path, turns: list[dict]) -> None:
    """Write turns to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


# ============================================================================
# Tests
# ============================================================================


class TestRequestChatHistory:
    """Tests for the request_chat_history socket handler."""

    @pytest.mark.asyncio
    async def test_handler_registered(self) -> None:
        """request_chat_history handler is registered via sio.event."""
        server = _make_server()
        assert "request_chat_history" in server._handlers

    @pytest.mark.asyncio
    async def test_returns_empty_without_orchestrator(self) -> None:
        """Returns empty turns when no orchestrator."""
        server = _make_server(with_orchestrator=False)
        handler = server._handlers["request_chat_history"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "chat_history",
            {"turns": []},
            to="test-sid",
        )

    @pytest.mark.asyncio
    async def test_returns_empty_without_session_file(self) -> None:
        """Returns empty turns when orchestrator has no session file."""
        server = _make_server(with_orchestrator=True, session_file=None)
        handler = server._handlers["request_chat_history"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "chat_history",
            {"turns": []},
            to="test-sid",
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_file_missing(self) -> None:
        """Returns empty turns when session file path doesn't exist on disk."""
        server = _make_server(
            with_orchestrator=True,
            session_file=Path("/tmp/nonexistent_session.jsonl"),
        )
        handler = server._handlers["request_chat_history"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "chat_history",
            {"turns": []},
            to="test-sid",
        )

    @pytest.mark.asyncio
    async def test_returns_visible_turns(self, tmp_path) -> None:
        """Returns only visible user/assistant turns in frontend format."""
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00Z", "hidden": False},
            {"turn_id": 2, "role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T00:00:01Z", "hidden": False},
            {"turn_id": 3, "role": "user", "content": "How are you?", "timestamp": "2026-01-01T00:00:02Z", "hidden": False},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "chat_history",
            {
                "turns": [
                    {"role": "user", "text": "Hello", "timestamp": "2026-01-01T00:00:00Z"},
                    {"role": "assistant", "text": "Hi there!", "timestamp": "2026-01-01T00:00:01Z"},
                    {"role": "user", "text": "How are you?", "timestamp": "2026-01-01T00:00:02Z"},
                ],
            },
            to="test-sid",
        )

    @pytest.mark.asyncio
    async def test_filters_hidden_and_summary_turns(self, tmp_path) -> None:
        """Hidden turns and summary turns are excluded."""
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "Old msg", "timestamp": "T1", "hidden": True},
            {"turn_id": 2, "role": "assistant", "content": "Old reply", "timestamp": "T2", "hidden": True},
            {"turn_id": 3, "role": "summary", "content": "Summary of T1-T2", "timestamp": "T3", "hidden": False},
            {"turn_id": 4, "role": "user", "content": "New msg", "timestamp": "T4", "hidden": False},
            {"turn_id": 5, "role": "assistant", "content": "New reply", "timestamp": "T5", "hidden": False},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]

        await handler("test-sid", {})

        call_args = server.sio.emit.call_args
        turns = call_args[0][1]["turns"]
        # Summary should be filtered (not user/assistant role), hidden should be filtered by read_visible_turns
        assert len(turns) == 2
        assert turns[0]["text"] == "New msg"
        assert turns[1]["text"] == "New reply"

    @pytest.mark.asyncio
    async def test_caps_at_200_messages(self, tmp_path) -> None:
        """Large histories are capped at the most recent 200 messages."""
        session_file = tmp_path / "test_session.jsonl"
        turns = []
        for i in range(250):
            role = "user" if i % 2 == 0 else "assistant"
            turns.append({
                "turn_id": i + 1,
                "role": role,
                "content": f"Message {i}",
                "timestamp": f"T{i}",
                "hidden": False,
            })
        _write_jsonl(session_file, turns)

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]

        await handler("test-sid", {})

        call_args = server.sio.emit.call_args
        result_turns = call_args[0][1]["turns"]
        assert len(result_turns) == 200
        # Should be the LAST 200 messages (most recent)
        assert result_turns[0]["text"] == "Message 50"
        assert result_turns[-1]["text"] == "Message 249"

    # ========================================================================
    # NANO-075: Metadata hydration tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_forwards_reasoning_on_assistant_turn(self, tmp_path) -> None:
        """reasoning field is forwarded from JSONL to frontend."""
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "Think", "timestamp": "T1", "hidden": False},
            {"turn_id": 2, "role": "assistant", "content": "Done", "timestamp": "T2", "hidden": False,
             "reasoning": "I considered the options..."},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]
        await handler("test-sid", {})

        turns = server.sio.emit.call_args[0][1]["turns"]
        assert turns[1]["reasoning"] == "I considered the options..."
        # User turn should NOT have reasoning
        assert "reasoning" not in turns[0]

    @pytest.mark.asyncio
    async def test_forwards_stimulus_source(self, tmp_path) -> None:
        """stimulus_source is forwarded on assistant turns."""
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "prompt", "timestamp": "T1", "hidden": False,
             "input_modality": "stimulus"},
            {"turn_id": 2, "role": "assistant", "content": "response", "timestamp": "T2", "hidden": False,
             "stimulus_source": "patience"},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]
        await handler("test-sid", {})

        turns = server.sio.emit.call_args[0][1]["turns"]
        assert turns[0]["input_modality"] == "stimulus"
        assert turns[1]["stimulus_source"] == "patience"

    @pytest.mark.asyncio
    async def test_forwards_codex_and_memories(self, tmp_path) -> None:
        """activated_codex_entries and retrieved_memories forwarded."""
        codex = [{"name": "lore", "keys": ["dragon"], "activation_method": "keyword"}]
        memories = [{"content_preview": "Dragons exist", "collection": "general", "distance": 0.1}]
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "Tell me", "timestamp": "T1", "hidden": False},
            {"turn_id": 2, "role": "assistant", "content": "Here", "timestamp": "T2", "hidden": False,
             "activated_codex_entries": codex, "retrieved_memories": memories},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]
        await handler("test-sid", {})

        turns = server.sio.emit.call_args[0][1]["turns"]
        assert turns[1]["activated_codex_entries"] == codex
        assert turns[1]["retrieved_memories"] == memories

    @pytest.mark.asyncio
    async def test_pre_075_jsonl_backward_compat(self, tmp_path) -> None:
        """Pre-075 JSONL without metadata fields produces clean turns (no KeyError)."""
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "Old", "timestamp": "T1", "hidden": False},
            {"turn_id": 2, "role": "assistant", "content": "Reply", "timestamp": "T2", "hidden": False},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]
        await handler("test-sid", {})

        turns = server.sio.emit.call_args[0][1]["turns"]
        assert len(turns) == 2
        # Base fields present
        assert turns[0]["role"] == "user"
        assert turns[0]["text"] == "Old"
        # No metadata keys
        assert "reasoning" not in turns[1]
        assert "stimulus_source" not in turns[1]
        assert "activated_codex_entries" not in turns[1]
        assert "retrieved_memories" not in turns[1]
        assert "barge_in_truncated" not in turns[1]

    @pytest.mark.asyncio
    async def test_barge_in_truncated_turn_shows_truncated_content(self, tmp_path) -> None:
        """NANO-111 Phase 2.5 / Session 639: when a turn is flagged
        barge_in_truncated, display text is `content` (what user heard),
        NOT `display_content` (full pre-barge generation).
        """
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "Tell us something.",
             "timestamp": "T1", "hidden": False},
            # Assistant turn that was barge-in-truncated: `content` is the
            # spoken portion; `display_content` preserves the full generation.
            {"turn_id": 2, "role": "assistant",
             "content": "Oh cool! Running tests sounds like a good time.",
             "display_content": (
                 "Oh cool! Running tests sounds like a good time. "
                 "Let me know if there's anything specific you need from me."
                 "Anything in particular you're testing today?"
             ),
             "barge_in_truncated": True,
             "timestamp": "T2", "hidden": False},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]
        await handler("test-sid", {})

        turns = server.sio.emit.call_args[0][1]["turns"]
        assert len(turns) == 2
        # Display text is the truncated content, NOT the full generation
        assert turns[1]["text"] == "Oh cool! Running tests sounds like a good time."
        assert "Anything in particular" not in turns[1]["text"]
        # Flag is propagated to frontend
        assert turns[1]["barge_in_truncated"] is True

    @pytest.mark.asyncio
    async def test_non_truncated_turn_prefers_display_content(self, tmp_path) -> None:
        """NANO-109 behavior is unchanged: without barge_in_truncated,
        display_content is preferred over content (formatting vs. LLM-clean)."""
        session_file = tmp_path / "test_session.jsonl"
        _write_jsonl(session_file, [
            {"turn_id": 1, "role": "user", "content": "hey", "timestamp": "T1", "hidden": False},
            # Non-truncated turn: display_content has formatting, content is cleaned
            {"turn_id": 2, "role": "assistant",
             "content": "Hey there! How are you?",
             "display_content": "Hey there! 🎉 How are you?",
             "timestamp": "T2", "hidden": False},
        ])

        server = _make_server(with_orchestrator=True, session_file=session_file)
        handler = server._handlers["request_chat_history"]
        await handler("test-sid", {})

        turns = server.sio.emit.call_args[0][1]["turns"]
        # Still prefers display_content (NANO-109 unchanged)
        assert turns[1]["text"] == "Hey there! 🎉 How are you?"
        assert "barge_in_truncated" not in turns[1]
        assert "input_modality" not in turns[0]
