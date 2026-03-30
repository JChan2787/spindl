"""NANO-089 Phase 4: State transition invariant tests.

Tests cover:
- Tools toggle precondition: VLM must be configured before enabling tools
- ToolsConfigResponse Pydantic model validation
- get_tools_state() response shape validation
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.gui.response_models import ToolsConfigResponse, ToolState
from spindl.orchestrator.voice_agent import VoiceAgentOrchestrator


# ============================================================================
# Helpers
# ============================================================================


def _make_orchestrator(
    tools_enabled=True,
    vlm_provider="llama",
    has_executor=True,
) -> VoiceAgentOrchestrator:
    """Create a VoiceAgentOrchestrator with mocked internals.

    Extends the pattern from test_voice_agent_tools.py with VLM config.
    """
    orch = VoiceAgentOrchestrator.__new__(VoiceAgentOrchestrator)
    orch._config = MagicMock()
    orch._pipeline = MagicMock()
    orch._tool_registry = None
    orch._tool_executor = None
    orch._event_bus = None

    # VLM config
    orch._config.vlm_config.provider = vlm_provider
    orch._config.vlm_config.providers = {"llama": {"url": "http://localhost:5558"}}

    if tools_enabled:
        # Mock registry
        registry = MagicMock()
        tool_mock = MagicMock()
        tool_mock.name = "screen_vision"
        registry._tools = {"screen_vision": tool_mock}
        registry.is_enabled.return_value = True
        orch._tool_registry = registry

        if has_executor:
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


# ============================================================================
# Class 1: Tools Toggle Precondition (Bugs #18, #19)
# ============================================================================


class TestToolsTogglePrecondition:
    """update_tools_config() must reject master ON when VLM is unconfigured."""

    def test_master_on_rejected_when_vlm_none(self) -> None:
        """VLM provider 'none' + no executor → precondition failure."""
        orch = _make_orchestrator(
            tools_enabled=False, vlm_provider="none", has_executor=False,
        )
        orch._tool_executor = None
        result = orch.update_tools_config(master_enabled=True)
        assert result["success"] is False
        assert "VLM" in result["error"]

    def test_master_on_rejected_when_vlm_empty(self) -> None:
        """VLM provider '' + no executor → precondition failure."""
        orch = _make_orchestrator(
            tools_enabled=False, vlm_provider="", has_executor=False,
        )
        orch._tool_executor = None
        result = orch.update_tools_config(master_enabled=True)
        assert result["success"] is False
        assert "VLM" in result["error"]

    def test_master_on_succeeds_when_vlm_configured(self) -> None:
        """VLM provider 'llama' + existing executor → success."""
        orch = _make_orchestrator(
            tools_enabled=True, vlm_provider="llama", has_executor=True,
        )
        result = orch.update_tools_config(master_enabled=True)
        assert result["success"] is True

    def test_master_off_always_succeeds(self) -> None:
        """Disabling tools always succeeds, even with VLM 'none'."""
        orch = _make_orchestrator(
            tools_enabled=True, vlm_provider="none", has_executor=True,
        )
        result = orch.update_tools_config(master_enabled=False)
        assert result["success"] is True

    def test_per_tool_toggle_returns_success(self) -> None:
        """Per-tool toggle returns success dict."""
        orch = _make_orchestrator(tools_enabled=True)
        result = orch.update_tools_config(
            tools={"screen_vision": {"enabled": False}},
        )
        assert result["success"] is True

    def test_update_tools_returns_dict(self) -> None:
        """update_tools_config always returns a dict, never None."""
        orch = _make_orchestrator(tools_enabled=True)
        result = orch.update_tools_config(master_enabled=..., tools=None)
        assert isinstance(result, dict)
        assert "success" in result


# ============================================================================
# Class 2: ToolsConfigResponse Model Validation
# ============================================================================


class TestToolsConfigResponseModel:
    """ToolsConfigResponse Pydantic model enforces tools state shape."""

    def test_valid_state_passes(self) -> None:
        """Full valid tools state passes validation."""
        state = {
            "master_enabled": True,
            "tools": {
                "screen_vision": {"enabled": True, "label": "Screen Vision"},
            },
        }
        resp = ToolsConfigResponse.model_validate(state)
        assert resp.master_enabled is True
        assert "screen_vision" in resp.tools
        assert resp.tools["screen_vision"].enabled is True

    def test_missing_master_enabled_rejected(self) -> None:
        """Missing master_enabled field raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolsConfigResponse.model_validate({"tools": {}})

    def test_malformed_tool_entry_rejected(self) -> None:
        """Tool entry missing 'label' raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolsConfigResponse.model_validate({
                "master_enabled": True,
                "tools": {"screen_vision": {"enabled": True}},
            })

    def test_error_field_allowed(self) -> None:
        """Response with error field passes validation."""
        resp = ToolsConfigResponse.model_validate({
            "master_enabled": False,
            "tools": {},
            "error": "No VLM provider configured.",
        })
        assert resp.error == "No VLM provider configured."


# ============================================================================
# Class 3: get_tools_state() Validation
# ============================================================================


class TestGetToolsStateValidation:
    """get_tools_state() output must validate against ToolsConfigResponse."""

    def test_get_tools_state_validates_against_response_model(self) -> None:
        """State from active orchestrator validates against Pydantic model."""
        orch = _make_orchestrator(tools_enabled=True)
        state = orch.get_tools_state()
        # Should not raise
        resp = ToolsConfigResponse.model_validate(state)
        assert resp.master_enabled is True
        assert "screen_vision" in resp.tools

    def test_get_tools_state_empty_tools_validates(self) -> None:
        """State with no tools registered still validates."""
        orch = _make_orchestrator(tools_enabled=False)
        state = orch.get_tools_state()
        resp = ToolsConfigResponse.model_validate(state)
        assert resp.master_enabled is False
        assert resp.tools == {}
