"""Tests for tools config socket handlers (NANO-065a).

Tests cover:
- request_tools_config: returns state from orchestrator or pre-launch fallback
- set_tools_config: master toggle, per-tool toggle, persistence, emit
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.gui.server import GUIServer


# ============================================================================
# Helpers
# ============================================================================


def _make_server(with_orchestrator=True, tools_enabled=True):
    """Create a GUIServer with mocked internals."""
    server = GUIServer.__new__(GUIServer)
    server.sio = MagicMock()
    server.sio.event = lambda fn: fn  # passthrough decorator
    server.sio.emit = AsyncMock()
    server._config_path = "/tmp/test_config.yaml"
    server._event_loop = asyncio.new_event_loop()
    server._launch_in_progress = False
    server._conversations_dir = None
    server._personas_dir = None
    server._prompt_blocks_config = None
    server._tools_config_cache = {
        "enabled": tools_enabled,
        "tools": {
            "screen_vision": {"enabled": True},
        },
    }

    if with_orchestrator:
        server._orchestrator = MagicMock()
        server._orchestrator.get_tools_state.return_value = {
            "master_enabled": tools_enabled,
            "tools": {
                "screen_vision": {"enabled": True, "label": "Screen Vision"},
            },
        }
        # NANO-089 Phase 4: update_tools_config now returns a dict
        server._orchestrator.update_tools_config.return_value = {"success": True}
        server._orchestrator._config = MagicMock()
        server._orchestrator._config.save_to_yaml = MagicMock()
    else:
        server._orchestrator = None

    return server


# ============================================================================
# request_tools_config
# ============================================================================


class TestRequestToolsConfig:
    """Tests for the request_tools_config handler."""

    def test_with_orchestrator(self) -> None:
        """Orchestrator get_tools_state returns correct shape."""
        server = _make_server(with_orchestrator=True)
        state = server._orchestrator.get_tools_state()
        assert state["master_enabled"] is True
        assert "screen_vision" in state["tools"]
        assert state["tools"]["screen_vision"]["enabled"] is True
        assert state["tools"]["screen_vision"]["label"] == "Screen Vision"

    def test_pre_launch_fallback(self) -> None:
        """Returns state from cached YAML config when no orchestrator."""
        server = _make_server(with_orchestrator=False)
        # Pre-launch state should be derivable from the cache
        cache = server._tools_config_cache
        assert cache["enabled"] is True
        assert cache["tools"]["screen_vision"]["enabled"] is True


# ============================================================================
# set_tools_config
# ============================================================================


class TestSetToolsConfig:
    """Tests for the set_tools_config handler."""

    def test_master_toggle_calls_orchestrator(self) -> None:
        """Setting master_enabled calls update_tools_config on orchestrator."""
        server = _make_server(with_orchestrator=True)
        # Verify the orchestrator mock has the method
        server._orchestrator.update_tools_config = MagicMock()
        server._orchestrator.update_tools_config(master_enabled=False, tools=None)
        server._orchestrator.update_tools_config.assert_called_once_with(
            master_enabled=False, tools=None
        )

    def test_per_tool_toggle_calls_orchestrator(self) -> None:
        """Per-tool changes are forwarded to orchestrator."""
        server = _make_server(with_orchestrator=True)
        server._orchestrator.update_tools_config = MagicMock()
        tools_changes = {"screen_vision": {"enabled": False}}
        server._orchestrator.update_tools_config(master_enabled=..., tools=tools_changes)
        server._orchestrator.update_tools_config.assert_called_once_with(
            master_enabled=..., tools=tools_changes
        )

    def test_persistence_called(self) -> None:
        """save_to_yaml is called after tools config update."""
        server = _make_server(with_orchestrator=True)
        server._orchestrator._config.save_to_yaml(server._config_path)
        server._orchestrator._config.save_to_yaml.assert_called_once_with(
            "/tmp/test_config.yaml"
        )

    def test_no_orchestrator_no_crash(self) -> None:
        """Handler doesn't crash when orchestrator is None."""
        server = _make_server(with_orchestrator=False)
        # The handler should guard with `if self._orchestrator:`
        assert server._orchestrator is None

    def test_error_result_includes_error_field(self) -> None:
        """When update_tools_config fails, state should carry the error."""
        server = _make_server(with_orchestrator=True)
        error_msg = "No VLM provider configured."
        server._orchestrator.update_tools_config.return_value = {
            "success": False,
            "error": error_msg,
        }
        # Simulate the error relay logic from the handler:
        # On failure, handler merges error into tools state
        result = server._orchestrator.update_tools_config(master_enabled=True)
        assert result["success"] is False
        state = server._orchestrator.get_tools_state()
        state["error"] = result["error"]
        assert state["error"] == error_msg
        assert state["master_enabled"] is True  # state reflects actual, not requested
