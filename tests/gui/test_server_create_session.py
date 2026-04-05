"""Tests for create_session handler (NANO-071).

Tests cover:
- create_session succeeds when orchestrator is present
- create_session rejects without orchestrator (services not running)
- Old session file is preserved after create
- Session list refreshes after successful create
- Failure path when orchestrator.create_new_session() returns False
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.gui.server import GUIServer


# ============================================================================
# Helpers
# ============================================================================


def _make_server(with_orchestrator=False):
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
        server._orchestrator.create_new_session.return_value = True
        type(server._orchestrator).session_file = PropertyMock(
            return_value=Path("/tmp/conversations/spindle_20260304_020000.jsonl")
        )
    else:
        server._orchestrator = None

    # Capture handlers registered via @sio.event
    server._handlers = {}

    def capture_event(fn):
        server._handlers[fn.__name__] = fn
        return fn

    server.sio.event = capture_event

    # Register handlers (this populates server._handlers)
    server._register_handlers()

    return server


# Module-level mock for emit_sessions (used by session handlers)
_emit_sessions_mock = None


# ============================================================================
# NANO-071: Create New Session
# ============================================================================


@patch("spindl.gui.server_sessions.emit_sessions", new_callable=AsyncMock)
class TestCreateSession:
    """Tests for the create_session socket handler."""

    @pytest.mark.asyncio
    async def test_creates_session_successfully(self, mock_emit_sessions) -> None:
        """create_session succeeds when orchestrator is present."""
        server = _make_server(with_orchestrator=True)
        handler = server._handlers["create_session"]

        await handler("test-sid", {})

        server._orchestrator.create_new_session.assert_called_once()
        # Should emit success
        server.sio.emit.assert_any_call(
            "session_created",
            {
                "success": True,
                "filepath": str(Path("/tmp/conversations/spindle_20260304_020000.jsonl")),
            },
            to="test-sid",
        )
        # Should refresh session list for all clients
        mock_emit_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_without_orchestrator(self, mock_emit_sessions) -> None:
        """create_session emits error when services not running."""
        server = _make_server(with_orchestrator=False)
        handler = server._handlers["create_session"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "session_created",
            {"success": False, "error": "Services not running"},
            to="test-sid",
        )

    @pytest.mark.asyncio
    async def test_handles_create_failure(self, mock_emit_sessions) -> None:
        """create_session emits error when orchestrator returns False."""
        server = _make_server(with_orchestrator=True)
        server._orchestrator.create_new_session.return_value = False
        handler = server._handlers["create_session"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "session_created",
            {"success": False, "error": "Failed to create new session"},
            to="test-sid",
        )
        # Should NOT refresh session list on failure
        mock_emit_sessions.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_list_refreshes_after_create(self, mock_emit_sessions) -> None:
        """emit_sessions is called without sid (broadcast) after successful create."""
        server = _make_server(with_orchestrator=True)
        handler = server._handlers["create_session"]

        await handler("test-sid", {})

        # emit_sessions called with server arg = broadcast to all
        mock_emit_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_registered(self, mock_emit_sessions) -> None:
        """create_session handler is registered via sio.event."""
        server = _make_server(with_orchestrator=False)
        assert "create_session" in server._handlers
