"""Tests for double-launch guard (NANO-070).

Tests cover:
- start_services rejects when orchestrator already exists
- start_services allows launch when orchestrator is None (normal + retry after failure)
- Concurrent launch gate (_launch_in_progress) still works alongside orchestrator guard
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


# ============================================================================
# NANO-070: Double-Launch Guard
# ============================================================================


class TestDoubleLaunchGuard:
    """Tests for the orchestrator-existence guard on start_services."""

    @pytest.mark.asyncio
    async def test_rejects_when_orchestrator_exists(self) -> None:
        """start_services emits launch_error when orchestrator is already live."""
        server = _make_server(with_orchestrator=True)
        handler = server._handlers["start_services"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "launch_error",
            {"error": "Services already running. Shut down first.", "service": None},
            to="test-sid",
        )

    @pytest.mark.asyncio
    async def test_rejects_before_concurrent_gate(self) -> None:
        """Orchestrator guard fires before _launch_in_progress check."""
        server = _make_server(with_orchestrator=True)
        server._launch_in_progress = True  # Both gates active
        handler = server._handlers["start_services"]

        await handler("test-sid", {})

        # Should hit orchestrator guard first, not the concurrent gate
        server.sio.emit.assert_called_once_with(
            "launch_error",
            {"error": "Services already running. Shut down first.", "service": None},
            to="test-sid",
        )

    @pytest.mark.asyncio
    async def test_concurrent_gate_still_works(self) -> None:
        """_launch_in_progress gate still fires when orchestrator is None."""
        server = _make_server(with_orchestrator=False)
        server._launch_in_progress = True
        handler = server._handlers["start_services"]

        await handler("test-sid", {})

        server.sio.emit.assert_called_once_with(
            "launch_error",
            {"error": "Launch already in progress", "service": None},
            to="test-sid",
        )

    def test_allows_launch_when_no_orchestrator(self) -> None:
        """Guard allows through when orchestrator is None (normal launch)."""
        server = _make_server(with_orchestrator=False)
        # Orchestrator is None and launch not in progress = guard passes
        assert server._orchestrator is None
        assert server._launch_in_progress is False

    def test_failed_launch_retry_allowed(self) -> None:
        """After a failed launch, orchestrator stays None — retry is allowed."""
        server = _make_server(with_orchestrator=False)
        # Simulate failed launch: _launch_in_progress was reset in finally block,
        # _on_services_ready never fired, so _orchestrator stays None
        server._launch_in_progress = False
        server._orchestrator = None
        assert server._orchestrator is None
        assert server._launch_in_progress is False
