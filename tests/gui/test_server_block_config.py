"""
Tests for NANO-045c-1 — Block Config socket handlers.

Tests cover the 3 block config socket handlers in GUIServer:
- request_block_config
- set_block_config
- reset_block_config
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from spindl.gui.server import GUIServer


# =============================================================================
# Fixtures
# =============================================================================


SAMPLE_BLOCK_CONFIG = {
    "order": [
        "persona_name", "persona_appearance", "persona_personality",
        "modality_context", "voice_state", "codex_context",
        "rag_context", "persona_rules", "modality_rules",
        "conversation_summary", "recent_history", "closing_instruction",
    ],
    "disabled": ["voice_state"],
    "overrides": {"persona_appearance": "Custom appearance text"},
    "blocks": [
        {
            "id": "persona_name",
            "label": "Agent Name",
            "order": 0,
            "enabled": True,
            "is_static": False,
            "section_header": "Agent",
            "has_override": False,
            "content_wrapper": "You are {content}.",
        },
        {
            "id": "voice_state",
            "label": "Voice State",
            "order": 4,
            "enabled": False,
            "is_static": False,
            "section_header": None,
            "has_override": False,
            "content_wrapper": None,
        },
    ],
}


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock OrchestratorConfig with prompt_blocks."""
    config = MagicMock()
    config.prompt_blocks = {}
    config.save_block_config_to_yaml = MagicMock()
    return config


@pytest.fixture
def mock_orchestrator(mock_config: MagicMock) -> MagicMock:
    """Create a mock orchestrator with block config methods."""
    orch = MagicMock()
    orch._config = mock_config
    orch.get_block_config.return_value = SAMPLE_BLOCK_CONFIG.copy()
    orch.update_block_config = MagicMock()
    orch.reset_block_config = MagicMock()
    return orch


@pytest.fixture
def server() -> GUIServer:
    """Create a GUIServer instance for testing."""
    return GUIServer(host="127.0.0.1", port=0)


@pytest.fixture
def server_with_orchestrator(
    server: GUIServer, mock_orchestrator: MagicMock
) -> GUIServer:
    """Create a GUIServer with an attached orchestrator."""
    server._orchestrator = mock_orchestrator
    server._config_path = "/tmp/test_spindl.yaml"
    return server


# =============================================================================
# Helper to dispatch socket events
# =============================================================================


async def _dispatch(server: GUIServer, event_name: str, data: dict) -> list[tuple]:
    """
    Simulate a socket.io event dispatch.

    Patches sio.emit to capture emitted responses, then invokes
    the registered handler directly.

    Returns list of (event, data, kwargs) tuples that were emitted.
    """
    emitted = []

    async def capture_emit(event, payload=None, **kwargs):
        emitted.append((event, payload, kwargs))

    server.sio.emit = capture_emit

    handler = server.sio.handlers.get("/", {}).get(event_name)
    if handler is None:
        raise ValueError(f"No handler registered for event '{event_name}'")

    await handler("test-sid", data)
    return emitted


# =============================================================================
# Tests: request_block_config
# =============================================================================


class TestRequestBlockConfig:
    """Tests for the request_block_config handler."""

    @pytest.mark.asyncio
    async def test_returns_block_config(
        self, server_with_orchestrator: GUIServer, mock_orchestrator: MagicMock
    ) -> None:
        """request_block_config emits block_config_loaded with config data."""
        emitted = await _dispatch(server_with_orchestrator, "request_block_config", {})

        assert len(emitted) == 1
        event, payload, kwargs = emitted[0]
        assert event == "block_config_loaded"
        assert payload["order"] == SAMPLE_BLOCK_CONFIG["order"]
        assert payload["disabled"] == SAMPLE_BLOCK_CONFIG["disabled"]
        assert payload["blocks"] == SAMPLE_BLOCK_CONFIG["blocks"]
        assert kwargs["to"] == "test-sid"
        mock_orchestrator.get_block_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_orchestrator_pre_launch(self, server: GUIServer) -> None:
        """request_block_config returns defaults without orchestrator (NANO-064b)."""
        emitted = await _dispatch(server, "request_block_config", {})
        assert len(emitted) == 1
        event, payload, _ = emitted[0]
        assert event == "block_config_loaded"
        assert "order" in payload
        assert "blocks" in payload
        assert len(payload["blocks"]) > 0  # Has default blocks


# =============================================================================
# Tests: set_block_config
# =============================================================================


class TestSetBlockConfig:
    """Tests for the set_block_config handler."""

    @pytest.mark.asyncio
    async def test_updates_and_persists(
        self, server_with_orchestrator: GUIServer, mock_orchestrator: MagicMock
    ) -> None:
        """set_block_config calls update, persists, and emits response."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "set_block_config",
            {"disabled": ["voice_state", "rag_context"]},
        )

        # Should have called update_block_config with merged config
        mock_orchestrator.update_block_config.assert_called_once()
        call_args = mock_orchestrator.update_block_config.call_args[0][0]
        assert "disabled" in call_args
        assert "voice_state" in call_args["disabled"]
        assert "rag_context" in call_args["disabled"]

        # Should have persisted
        mock_orchestrator._config.save_block_config_to_yaml.assert_called_once_with(
            "/tmp/test_spindl.yaml"
        )

        # Should have emitted block_config_updated
        assert len(emitted) == 1
        event, payload, kwargs = emitted[0]
        assert event == "block_config_updated"
        assert payload["success"] is True
        assert payload["persisted"] is True
        assert kwargs["to"] == "test-sid"

    @pytest.mark.asyncio
    async def test_partial_update_merges(
        self, server_with_orchestrator: GUIServer, mock_orchestrator: MagicMock
    ) -> None:
        """set_block_config merges partial updates with existing config."""
        mock_orchestrator._config.prompt_blocks = {"disabled": ["voice_state"]}

        await _dispatch(
            server_with_orchestrator,
            "set_block_config",
            {"overrides": {"persona_name": "Custom name"}},
        )

        call_args = mock_orchestrator.update_block_config.call_args[0][0]
        # Should have both the existing disabled and new overrides
        assert call_args["disabled"] == ["voice_state"]
        assert call_args["overrides"] == {"persona_name": "Custom name"}

    @pytest.mark.asyncio
    async def test_persist_failure_still_emits(
        self, server_with_orchestrator: GUIServer, mock_orchestrator: MagicMock
    ) -> None:
        """set_block_config still emits response if persistence fails."""
        mock_orchestrator._config.save_block_config_to_yaml.side_effect = IOError(
            "disk full"
        )

        emitted = await _dispatch(
            server_with_orchestrator,
            "set_block_config",
            {"disabled": ["voice_state"]},
        )

        assert len(emitted) == 1
        event, payload, _ = emitted[0]
        assert event == "block_config_updated"
        assert payload["success"] is True
        assert payload["persisted"] is False

    @pytest.mark.asyncio
    async def test_no_orchestrator_pre_launch(self, server: GUIServer) -> None:
        """set_block_config saves to cache without orchestrator (NANO-064b)."""
        emitted = await _dispatch(
            server, "set_block_config", {"disabled": ["voice_state"]}
        )
        assert len(emitted) == 1
        event, payload, _ = emitted[0]
        assert event == "block_config_updated"
        assert payload["success"] is True
        assert "voice_state" in payload["disabled"]


# =============================================================================
# Tests: reset_block_config
# =============================================================================


class TestResetBlockConfig:
    """Tests for the reset_block_config handler."""

    @pytest.mark.asyncio
    async def test_resets_and_persists(
        self, server_with_orchestrator: GUIServer, mock_orchestrator: MagicMock
    ) -> None:
        """reset_block_config calls reset, persists, and emits response."""
        emitted = await _dispatch(
            server_with_orchestrator, "reset_block_config", {}
        )

        mock_orchestrator.reset_block_config.assert_called_once()

        # Should have persisted (removes section)
        mock_orchestrator._config.save_block_config_to_yaml.assert_called_once_with(
            "/tmp/test_spindl.yaml"
        )

        # Should emit block_config_updated with defaults
        assert len(emitted) == 1
        event, payload, kwargs = emitted[0]
        assert event == "block_config_updated"
        assert payload["success"] is True
        assert payload["persisted"] is True
        assert kwargs["to"] == "test-sid"

    @pytest.mark.asyncio
    async def test_no_orchestrator_pre_launch(self, server: GUIServer) -> None:
        """reset_block_config returns defaults without orchestrator (NANO-064b)."""
        emitted = await _dispatch(server, "reset_block_config", {})
        assert len(emitted) == 1
        event, payload, _ = emitted[0]
        assert event == "block_config_updated"
        assert payload["success"] is True
        assert payload["disabled"] == []  # All blocks enabled after reset
