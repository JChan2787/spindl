"""Tests for runtime VLM provider swap methods (NANO-065c).

Tests cover:
- get_vlm_state: returns correct provider, available_providers, healthy
- swap_vlm_provider: state gating, config resolution, unified mode, cascade
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.voice_agent import VoiceAgentOrchestrator


def _make_vlm_orchestrator(
    vlm_provider="openai",
    llm_provider="openrouter",
    state="listening",
    vlm_providers=None,
    llm_provider_config=None,
    screen_vision_tool=None,
) -> VoiceAgentOrchestrator:
    """Create a VoiceAgentOrchestrator with mocked VLM internals."""
    orch = VoiceAgentOrchestrator.__new__(VoiceAgentOrchestrator)

    # VLM config
    if vlm_providers is None:
        vlm_providers = {
            "llama": {"host": "127.0.0.1", "port": 5558, "model_path": "/path/to/model"},
            "openai": {"api_key": "test-key", "base_url": "https://api.x.ai", "model": "grok-2-vision"},
            "llm": {},
        }
    orch._config = MagicMock()
    orch._config.vlm_config.provider = vlm_provider
    orch._config.vlm_config.providers = vlm_providers
    orch._config.vlm_config.plugin_paths = []

    # LLM config (for unified mode)
    if llm_provider_config is None:
        llm_provider_config = {
            "url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "model": "google/gemini-2.5-pro",
        }
    orch._config.llm_config.provider = llm_provider
    orch._config.llm_config.provider_config = llm_provider_config

    # Tool registry with screen_vision tool
    if screen_vision_tool is None:
        screen_vision_tool = MagicMock()
        screen_vision_tool.health_check.return_value = True
    orch._tool_registry = MagicMock()
    orch._tool_registry.get_tool.return_value = screen_vision_tool

    # State machine
    from spindl.core import AgentState

    orch._state_machine = MagicMock()
    orch._state_machine.state = AgentState(state)

    return orch


class TestGetVLMState:
    """Tests for get_vlm_state()."""

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_returns_provider_and_health(self, MockRegistry) -> None:
        orch = _make_vlm_orchestrator()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["llama", "openai", "llm"]
        MockRegistry.return_value = mock_registry

        state = orch.get_vlm_state()

        assert state["provider"] == "openai"
        assert state["healthy"] is True
        assert "llama" in state["available_providers"]

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_unhealthy_when_tool_returns_false(self, MockRegistry) -> None:
        tool = MagicMock()
        tool.health_check.return_value = False
        orch = _make_vlm_orchestrator(screen_vision_tool=tool)
        MockRegistry.return_value = MagicMock()
        MockRegistry.return_value.list_providers.return_value = []

        state = orch.get_vlm_state()

        assert state["healthy"] is False

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_cloud_config_omitted_when_not_openai(self, MockRegistry) -> None:
        """cloud_config key must be absent (not None or {}) for non-openai providers."""
        orch = _make_vlm_orchestrator(vlm_provider="llm")
        MockRegistry.return_value = MagicMock()
        MockRegistry.return_value.list_providers.return_value = ["llama", "openai", "llm"]

        state = orch.get_vlm_state()

        assert "cloud_config" not in state


class TestSwapVLMProvider:
    """Tests for swap_vlm_provider()."""

    def test_rejects_during_processing(self) -> None:
        """Swap rejected when state machine is PROCESSING."""
        orch = _make_vlm_orchestrator(state="processing")
        result = orch.swap_vlm_provider("llama", {})
        assert result["success"] is False
        assert "processing" in result["error"].lower()

    def test_swap_allowed_when_idle(self) -> None:
        """Swap allowed when state machine is IDLE."""
        orch = _make_vlm_orchestrator(state="idle")
        result = orch.swap_vlm_provider("llama", {})
        assert result["success"] is True

    def test_swap_allowed_when_listening(self) -> None:
        """Swap allowed when state machine is LISTENING."""
        orch = _make_vlm_orchestrator(state="listening")
        result = orch.swap_vlm_provider("openai", {})
        assert result["success"] is True

    def test_calls_tool_swap(self) -> None:
        """Calls swap_vlm_provider on the screen_vision tool."""
        tool = MagicMock()
        tool.health_check.return_value = True
        orch = _make_vlm_orchestrator(screen_vision_tool=tool)

        orch.swap_vlm_provider("llama", {})

        tool.swap_vlm_provider.assert_called_once()
        call_args = tool.swap_vlm_provider.call_args
        assert call_args[0][0] == "llama"  # provider_name

    def test_updates_config_provider(self) -> None:
        """In-memory config is updated for persistence."""
        orch = _make_vlm_orchestrator(vlm_provider="openai")
        orch.swap_vlm_provider("llama", {})
        assert orch._config.vlm_config.provider == "llama"

    def test_returns_error_when_no_tool_registry(self) -> None:
        """Returns error if tool registry not initialized."""
        orch = _make_vlm_orchestrator()
        orch._tool_registry = None
        result = orch.swap_vlm_provider("llama", {})
        assert result["success"] is False
        assert "not initialized" in result["error"].lower()

    def test_returns_error_when_no_screen_vision_tool(self) -> None:
        """Returns error if screen_vision tool not found."""
        orch = _make_vlm_orchestrator()
        orch._tool_registry.get_tool.return_value = None
        result = orch.swap_vlm_provider("llama", {})
        assert result["success"] is False
        assert "screen vision" in result["error"].lower()

    def test_empty_config_resolves_from_stored(self) -> None:
        """Empty override dict uses stored YAML config."""
        tool = MagicMock()
        tool.health_check.return_value = True
        orch = _make_vlm_orchestrator(screen_vision_tool=tool)

        orch.swap_vlm_provider("openai", {})

        call_args = tool.swap_vlm_provider.call_args
        resolved = call_args[0][1]  # vlm_config
        assert resolved.get("api_key") == "test-key"

    def test_config_overrides_merge_with_stored(self) -> None:
        """Caller overrides merge with stored config."""
        tool = MagicMock()
        tool.health_check.return_value = True
        orch = _make_vlm_orchestrator(screen_vision_tool=tool)

        orch.swap_vlm_provider("openai", {"model": "gpt-4o"})

        call_args = tool.swap_vlm_provider.call_args
        resolved = call_args[0][1]
        assert resolved["model"] == "gpt-4o"  # override
        assert resolved["api_key"] == "test-key"  # from stored

    def test_no_config_returns_error(self) -> None:
        """Returns error when no stored config and no overrides."""
        orch = _make_vlm_orchestrator(vlm_providers={"llama": {}})
        result = orch.swap_vlm_provider("unknown_provider", {})
        assert result["success"] is False
        assert "no config" in result["error"].lower()

    def test_unified_mode_derives_from_llm(self) -> None:
        """Unified mode ('llm') derives VLM config from current LLM provider."""
        tool = MagicMock()
        tool.health_check.return_value = True
        orch = _make_vlm_orchestrator(
            screen_vision_tool=tool,
            llm_provider_config={
                "url": "https://openrouter.ai/api/v1",
                "api_key": "or-key",
                "model": "google/gemini-2.5-pro",
            },
        )

        orch.swap_vlm_provider("llm", {})

        call_args = tool.swap_vlm_provider.call_args
        resolved = call_args[0][1]
        assert resolved["url"] == "https://openrouter.ai/api/v1"
        assert resolved["api_key"] == "or-key"
        assert resolved["model"] == "google/gemini-2.5-pro"

    def test_unified_mode_uses_host_port_fallback(self) -> None:
        """Unified mode falls back to host:port when no url in LLM config."""
        tool = MagicMock()
        tool.health_check.return_value = True
        orch = _make_vlm_orchestrator(
            screen_vision_tool=tool,
            llm_provider_config={
                "host": "127.0.0.1",
                "port": 5557,
                "model": "qwen3-14b",
            },
        )

        orch.swap_vlm_provider("llm", {})

        call_args = tool.swap_vlm_provider.call_args
        resolved = call_args[0][1]
        assert resolved["url"] == "http://127.0.0.1:5557"

    def test_tool_swap_failure_returns_error(self) -> None:
        """Returns error if tool's swap method raises."""
        tool = MagicMock()
        tool.health_check.return_value = True
        tool.swap_vlm_provider.side_effect = RuntimeError("provider init failed")
        orch = _make_vlm_orchestrator(screen_vision_tool=tool)

        result = orch.swap_vlm_provider("llama", {})

        assert result["success"] is False
        assert "init failed" in result["error"]


class TestVLMDisabledNone:
    """Tests for VLM provider='none' (disabled) behavior (Bug #12)."""

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_health_false_when_provider_none(self, MockRegistry) -> None:
        """VLM health check returns False when provider is 'none'."""
        orch = _make_vlm_orchestrator(vlm_provider="none")
        # Even if screen_vision tool exists and reports healthy,
        # the VLM state should reflect the provider is disabled
        MockRegistry.return_value = MagicMock()
        MockRegistry.return_value.list_providers.return_value = []
        state = orch.get_vlm_state()
        assert state["provider"] == "none"

    def test_swap_from_none_to_provider_works(self) -> None:
        """Swapping from 'none' to a real provider succeeds (Dashboard activation)."""
        tool = MagicMock()
        tool.health_check.return_value = True
        orch = _make_vlm_orchestrator(vlm_provider="none", screen_vision_tool=tool)
        result = orch.swap_vlm_provider("openai", {})
        assert result["success"] is True
        assert orch._config.vlm_config.provider == "openai"


class TestUnifiedModeCascade:
    """Tests for LLM swap cascading to unified VLM."""

    def test_llm_swap_cascades_when_vlm_unified(self) -> None:
        """When VLM is 'llm', swapping LLM triggers VLM re-derivation."""
        from spindl.llm.base import LLMProperties, LLMProvider
        from spindl.llm.provider_holder import ProviderHolder

        orch = _make_vlm_orchestrator(vlm_provider="llm")

        # Set up LLM swap infrastructure
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.get_properties.return_value = LLMProperties(
            model_name="deepseek-chat",
            supports_streaming=False,
            context_length=32768,
        )
        mock_llm_registry = MagicMock()
        mock_llm_registry.list_available.return_value = ["llama", "deepseek"]
        holder = ProviderHolder(mock_provider, mock_llm_registry)

        orch._provider_holder = holder
        orch._llm_provider = holder
        orch._llm_registry = mock_llm_registry

        # Set up new provider for swap
        new_llm_provider_class = MagicMock()
        new_llm_provider = MagicMock(spec=LLMProvider)
        new_llm_provider.get_properties.return_value = LLMProperties(
            model_name="deepseek-chat", supports_streaming=False, context_length=32768,
        )
        new_llm_provider_class.return_value = new_llm_provider
        new_llm_provider_class.validate_config.return_value = []
        mock_llm_registry.get_provider_class.return_value = new_llm_provider_class

        orch._config.llm_config.providers = {
            "deepseek": {"url": "https://api.deepseek.com/v1", "api_key": "dk-key", "model": "deepseek-chat"},
        }

        # Perform LLM swap — should cascade to VLM
        tool = orch._tool_registry.get_tool("screen_vision")
        result = orch.swap_llm_provider("deepseek", {})

        assert result["success"] is True
        # VLM tool's swap should have been called (cascade)
        tool.swap_vlm_provider.assert_called()

    def test_llm_swap_no_cascade_when_vlm_not_unified(self) -> None:
        """When VLM is not 'llm', swapping LLM does NOT cascade."""
        from spindl.llm.base import LLMProperties, LLMProvider
        from spindl.llm.provider_holder import ProviderHolder

        orch = _make_vlm_orchestrator(vlm_provider="openai")

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.get_properties.return_value = LLMProperties(
            model_name="deepseek-chat",
            supports_streaming=False,
            context_length=32768,
        )
        mock_llm_registry = MagicMock()
        mock_llm_registry.list_available.return_value = ["llama", "deepseek"]
        holder = ProviderHolder(mock_provider, mock_llm_registry)

        orch._provider_holder = holder
        orch._llm_provider = holder
        orch._llm_registry = mock_llm_registry

        new_llm_provider_class = MagicMock()
        new_llm_provider = MagicMock(spec=LLMProvider)
        new_llm_provider.get_properties.return_value = LLMProperties(
            model_name="deepseek-chat", supports_streaming=False, context_length=32768,
        )
        new_llm_provider_class.return_value = new_llm_provider
        new_llm_provider_class.validate_config.return_value = []
        mock_llm_registry.get_provider_class.return_value = new_llm_provider_class

        orch._config.llm_config.providers = {
            "deepseek": {"url": "https://api.deepseek.com/v1", "api_key": "dk-key", "model": "deepseek-chat"},
        }

        tool = orch._tool_registry.get_tool("screen_vision")
        result = orch.swap_llm_provider("deepseek", {})

        assert result["success"] is True
        # VLM tool's swap should NOT have been called
        tool.swap_vlm_provider.assert_not_called()
