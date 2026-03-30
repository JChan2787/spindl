"""Tests for runtime LLM provider swap methods (NANO-065b).

Tests cover:
- get_llm_state: returns correct provider/model/context info
- swap_llm_provider: state gating, validation, swap mechanics, config update
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.llm.base import LLMProperties, LLMProvider
from spindl.llm.provider_holder import ProviderHolder
from spindl.orchestrator.voice_agent import VoiceAgentOrchestrator


def _make_mock_provider(model_name="test-model", context_length=8192):
    """Create a mock LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_properties.return_value = LLMProperties(
        model_name=model_name,
        supports_streaming=False,
        context_length=context_length,
    )
    return provider


def _make_orchestrator(
    provider_name="openrouter",
    model_name="google/gemini-2.5-pro",
    context_length=128000,
    state="listening",
    providers=None,
) -> VoiceAgentOrchestrator:
    """Create a VoiceAgentOrchestrator with mocked internals for swap testing."""
    orch = VoiceAgentOrchestrator.__new__(VoiceAgentOrchestrator)

    # Mock provider + holder
    mock_provider = _make_mock_provider(model_name, context_length)
    mock_registry = MagicMock()
    mock_registry.list_available.return_value = ["llama", "deepseek", "openrouter"]
    holder = ProviderHolder(mock_provider, mock_registry)

    orch._provider_holder = holder
    orch._llm_provider = holder
    orch._llm_registry = mock_registry
    orch._pipeline = MagicMock()

    # Config — use real LLMConfig-like structure with providers dict
    if providers is None:
        providers = {
            "openrouter": {"url": "https://openrouter.ai/api/v1", "api_key": "test-key", "model": model_name},
            "deepseek": {"url": "https://api.deepseek.com/v1", "api_key": "test-key", "model": "deepseek-chat"},
            "llama": {"model_path": "/path/to/model", "host": "127.0.0.1", "port": 5557},
        }
    orch._config = MagicMock()
    orch._config.llm_config.provider = provider_name
    orch._config.llm_config.provider_config = {"model": model_name}
    orch._config.llm_config.providers = providers

    # State machine
    from spindl.core import AgentState

    orch._state_machine = MagicMock()
    orch._state_machine.state = AgentState(state)

    return orch


class TestGetLLMState:
    """Tests for get_llm_state()."""

    def test_returns_provider_info(self) -> None:
        orch = _make_orchestrator()
        state = orch.get_llm_state()
        assert state["provider"] == "openrouter"
        assert state["model"] == "google/gemini-2.5-pro"
        assert state["context_size"] == 128000
        assert "llama" in state["available_providers"]

    def test_returns_available_providers(self) -> None:
        orch = _make_orchestrator()
        state = orch.get_llm_state()
        assert state["available_providers"] == ["llama", "deepseek", "openrouter"]


class TestSwapLLMProvider:
    """Tests for swap_llm_provider()."""

    def test_rejects_during_processing(self) -> None:
        """Swap rejected when state machine is PROCESSING."""
        orch = _make_orchestrator(state="processing")
        result = orch.swap_llm_provider("deepseek", {"model": "deepseek-chat"})
        assert result["success"] is False
        assert "processing" in result["error"].lower()

    def test_rejects_unknown_provider(self) -> None:
        """Swap rejected when provider not found in registry."""
        from spindl.llm.registry import ProviderNotFoundError

        orch = _make_orchestrator()
        orch._llm_registry.get_provider_class.side_effect = ProviderNotFoundError("nonexistent", [])
        result = orch.swap_llm_provider("nonexistent", {})
        assert result["success"] is False
        assert "nonexistent" in result["error"]

    def test_rejects_invalid_config(self) -> None:
        """Swap rejected when provider config validation fails."""
        # Stored config has a URL but no API key — validation should catch it
        orch = _make_orchestrator(providers={
            "openrouter": {"model": "google/gemini-2.5-pro"},
            "bad_cloud": {"url": "https://example.com"},
        })
        mock_class = MagicMock()
        mock_class.validate_config.return_value = ["Missing API key"]
        orch._llm_registry.get_provider_class.return_value = mock_class
        result = orch.swap_llm_provider("bad_cloud", {})
        assert result["success"] is False
        assert "API key" in result["error"]

    def test_successful_swap(self) -> None:
        """Successful swap updates holder and config."""
        orch = _make_orchestrator()
        new_provider = _make_mock_provider("deepseek-chat", 32768)

        mock_class = MagicMock()
        mock_class.validate_config.return_value = []
        mock_class.return_value = new_provider
        orch._llm_registry.get_provider_class.return_value = mock_class

        result = orch.swap_llm_provider("deepseek", {"model": "deepseek-chat"})
        assert result["success"] is True
        assert result["provider"] == "deepseek"
        assert result["model"] == "deepseek-chat"
        assert result["context_size"] == 32768

        # Verify holder was swapped
        assert orch._provider_holder.provider is new_provider

        # Verify config updated — resolved config merges stored + overrides
        assert orch._config.llm_config.provider == "deepseek"
        resolved = orch._config.llm_config.provider_config
        assert resolved["model"] == "deepseek-chat"
        assert "api_key" in resolved  # From stored providers dict

    def test_swap_shuts_down_old_provider(self) -> None:
        """Old provider's shutdown() is called after swap."""
        orch = _make_orchestrator()
        old_provider = orch._provider_holder.provider

        new_provider = _make_mock_provider("new-model")
        mock_class = MagicMock()
        mock_class.validate_config.return_value = []
        mock_class.return_value = new_provider
        orch._llm_registry.get_provider_class.return_value = mock_class

        orch.swap_llm_provider("llama", {"model_path": "/path/to/model"})
        old_provider.shutdown.assert_called_once()

    def test_swap_allowed_when_idle(self) -> None:
        """Swap succeeds when state is IDLE."""
        orch = _make_orchestrator(state="idle")
        new_provider = _make_mock_provider("new-model")
        mock_class = MagicMock()
        mock_class.validate_config.return_value = []
        mock_class.return_value = new_provider
        orch._llm_registry.get_provider_class.return_value = mock_class

        result = orch.swap_llm_provider("llama", {"model_path": "/tmp/model"})
        assert result["success"] is True

    def test_swap_rejected_during_user_speaking(self) -> None:
        """Swap rejected when state is USER_SPEAKING."""
        orch = _make_orchestrator(state="user_speaking")
        result = orch.swap_llm_provider("deepseek", {})
        assert result["success"] is False

    def test_swap_rejected_during_system_speaking(self) -> None:
        """Swap rejected when state is SYSTEM_SPEAKING."""
        orch = _make_orchestrator(state="system_speaking")
        result = orch.swap_llm_provider("deepseek", {})
        assert result["success"] is False

    def test_init_failure_returns_error(self) -> None:
        """Provider initialization failure returns error dict."""
        orch = _make_orchestrator()
        mock_class = MagicMock()
        mock_class.validate_config.return_value = []
        bad_provider = MagicMock()
        bad_provider.initialize.side_effect = ConnectionError("unreachable")
        mock_class.return_value = bad_provider
        orch._llm_registry.get_provider_class.return_value = mock_class

        result = orch.swap_llm_provider("llama", {"host": "bad"})
        assert result["success"] is False
        assert "unreachable" in result["error"]

    def test_empty_config_resolves_from_stored_providers(self) -> None:
        """Empty config resolves from the stored providers dict in LLMConfig."""
        orch = _make_orchestrator()
        new_provider = _make_mock_provider("deepseek-chat", 32768)

        mock_class = MagicMock()
        mock_class.validate_config.return_value = []
        mock_class.return_value = new_provider
        orch._llm_registry.get_provider_class.return_value = mock_class

        # Send empty config — should resolve from stored providers
        result = orch.swap_llm_provider("deepseek", {})
        assert result["success"] is True

        # Verify the provider was initialized with stored config
        new_provider.initialize.assert_called_once()
        init_config = new_provider.initialize.call_args[0][0]
        assert init_config["url"] == "https://api.deepseek.com/v1"
        assert init_config["api_key"] == "test-key"
        assert init_config["model"] == "deepseek-chat"

    def test_config_overrides_merge_with_stored(self) -> None:
        """Explicit config overrides merge onto stored provider config."""
        orch = _make_orchestrator()
        new_provider = _make_mock_provider("custom-model", 65536)

        mock_class = MagicMock()
        mock_class.validate_config.return_value = []
        mock_class.return_value = new_provider
        orch._llm_registry.get_provider_class.return_value = mock_class

        # Send partial override — model changes but api_key comes from stored
        result = orch.swap_llm_provider("deepseek", {"model": "deepseek-v3"})
        assert result["success"] is True

        init_config = new_provider.initialize.call_args[0][0]
        assert init_config["model"] == "deepseek-v3"  # Override applied
        assert init_config["api_key"] == "test-key"  # Stored value preserved

    def test_no_stored_config_returns_error(self) -> None:
        """Swap with empty config and no stored config returns error."""
        orch = _make_orchestrator(providers={"openrouter": {"model": "test"}})
        result = orch.swap_llm_provider("unknown_provider", {})
        # Provider not found in registry (not in stored configs either)
        assert result["success"] is False
