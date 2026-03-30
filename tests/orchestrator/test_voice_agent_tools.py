"""Tests for runtime tools toggle methods (NANO-065a).

Tests cover:
- get_tools_state: returns correct state with/without tools
- update_tools_config: master toggle, per-tool toggle, config updates
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.voice_agent import VoiceAgentOrchestrator


def _make_orchestrator(tools_enabled=True) -> VoiceAgentOrchestrator:
    """Create a VoiceAgentOrchestrator with mocked internals."""
    orch = VoiceAgentOrchestrator.__new__(VoiceAgentOrchestrator)
    orch._config = MagicMock()
    orch._pipeline = MagicMock()
    orch._tool_registry = None
    orch._tool_executor = None

    if tools_enabled:
        # Mock registry
        registry = MagicMock()
        tool_mock = MagicMock()
        tool_mock.name = "screen_vision"
        registry._tools = {"screen_vision": tool_mock}
        registry.is_enabled.return_value = True
        orch._tool_registry = registry

        # Mock executor
        orch._tool_executor = MagicMock()

        # Wire executor into pipeline
        orch._pipeline._tool_executor = orch._tool_executor

        # Config
        orch._config.tools_config.enabled = True
        orch._config.tools_config.tools = {"screen_vision": {"enabled": True}}
    else:
        orch._config.tools_config.enabled = False
        orch._config.tools_config.tools = {}

    return orch


class TestGetToolsState:
    """Tests for get_tools_state()."""

    def test_no_tools(self) -> None:
        """Returns master_enabled=False when no tools configured."""
        orch = _make_orchestrator(tools_enabled=False)
        state = orch.get_tools_state()
        assert state["master_enabled"] is False
        assert state["tools"] == {}

    def test_with_tools(self) -> None:
        """Returns correct state when tools are active."""
        orch = _make_orchestrator(tools_enabled=True)
        state = orch.get_tools_state()
        assert state["master_enabled"] is True
        assert "screen_vision" in state["tools"]
        assert state["tools"]["screen_vision"]["enabled"] is True
        assert state["tools"]["screen_vision"]["label"] == "Screen Vision"

    def test_master_false_when_executor_disconnected(self) -> None:
        """master_enabled=False when pipeline has no executor."""
        orch = _make_orchestrator(tools_enabled=True)
        orch._pipeline._tool_executor = None
        state = orch.get_tools_state()
        assert state["master_enabled"] is False
        # Per-tool state still reflects registry
        assert state["tools"]["screen_vision"]["enabled"] is True


class TestUpdateToolsConfig:
    """Tests for update_tools_config()."""

    def test_master_disable(self) -> None:
        """Disabling master sets pipeline executor to None."""
        orch = _make_orchestrator(tools_enabled=True)
        orch.update_tools_config(master_enabled=False)
        orch._pipeline.set_tool_executor.assert_called_once_with(None)
        assert orch._config.tools_config.enabled is False

    def test_master_enable(self) -> None:
        """Enabling master restores pipeline executor."""
        orch = _make_orchestrator(tools_enabled=True)
        # First disable
        orch._pipeline._tool_executor = None
        orch._config.tools_config.enabled = False
        # Then enable
        orch.update_tools_config(master_enabled=True)
        orch._pipeline.set_tool_executor.assert_called_with(orch._tool_executor)
        assert orch._config.tools_config.enabled is True

    def test_per_tool_toggle(self) -> None:
        """Per-tool toggle calls registry.set_enabled and updates config."""
        orch = _make_orchestrator(tools_enabled=True)
        orch.update_tools_config(tools={"screen_vision": {"enabled": False}})
        orch._tool_registry.set_enabled.assert_called_once_with("screen_vision", False)
        assert orch._config.tools_config.tools["screen_vision"]["enabled"] is False

    def test_ellipsis_keeps_current(self) -> None:
        """Passing ... for master_enabled makes no changes."""
        orch = _make_orchestrator(tools_enabled=True)
        orch.update_tools_config(master_enabled=..., tools=None)
        orch._pipeline.set_tool_executor.assert_not_called()

    def test_combined_master_and_per_tool(self) -> None:
        """Both master and per-tool can be changed in one call."""
        orch = _make_orchestrator(tools_enabled=True)
        orch.update_tools_config(
            master_enabled=False,
            tools={"screen_vision": {"enabled": False}},
        )
        orch._pipeline.set_tool_executor.assert_called_once_with(None)
        orch._tool_registry.set_enabled.assert_called_once_with("screen_vision", False)
