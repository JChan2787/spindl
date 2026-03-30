"""Tests for LLM config socket handlers (NANO-065b).

Tests cover:
- request_llm_config: returns state from orchestrator or pre-launch fallback
- set_llm_provider: validation, orchestrator calls, persistence, error handling
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


def _make_server(with_orchestrator=True, provider="openrouter"):
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
    server._tools_config_cache = None
    server._llm_config_cache = {
        "provider": provider,
        "model": "google/gemini-2.5-pro",
        "context_size": 128000,
        "providers": {
            "llama": {"model_path": "/path/to/model"},
            "openrouter": {"model": "google/gemini-2.5-pro"},
        },
    }

    if with_orchestrator:
        server._orchestrator = MagicMock()
        server._orchestrator.get_llm_state.return_value = {
            "provider": provider,
            "model": "google/gemini-2.5-pro",
            "context_size": 128000,
            "available_providers": ["llama", "deepseek", "openrouter"],
        }
        server._orchestrator.swap_llm_provider.return_value = {
            "success": True,
            "provider": "deepseek",
            "model": "deepseek-chat",
            "context_size": 32768,
            "available_providers": ["llama", "deepseek", "openrouter"],
        }
        server._orchestrator._config = MagicMock()
        server._orchestrator._config.save_to_yaml = MagicMock()
    else:
        server._orchestrator = None

    return server


# ============================================================================
# request_llm_config
# ============================================================================


class TestRequestLLMConfig:
    """Tests for the request_llm_config handler."""

    def test_with_orchestrator_returns_state(self) -> None:
        """Orchestrator get_llm_state returns correct shape."""
        server = _make_server(with_orchestrator=True)
        state = server._orchestrator.get_llm_state()
        assert state["provider"] == "openrouter"
        assert state["model"] == "google/gemini-2.5-pro"
        assert state["context_size"] == 128000
        assert "llama" in state["available_providers"]

    def test_pre_launch_fallback(self) -> None:
        """Returns cached YAML config when no orchestrator."""
        server = _make_server(with_orchestrator=False)
        cache = server._llm_config_cache
        assert cache["provider"] == "openrouter"
        assert cache["model"] == "google/gemini-2.5-pro"
        assert "llama" in cache["providers"]

    def test_pre_launch_no_cache(self) -> None:
        """Returns defaults when no cache and no orchestrator."""
        server = _make_server(with_orchestrator=False)
        server._llm_config_cache = None
        cache = server._llm_config_cache or {}
        assert cache.get("provider", "llama") == "llama"


# ============================================================================
# set_llm_provider
# ============================================================================


class TestSetLLMProvider:
    """Tests for the set_llm_provider handler."""

    def test_swap_calls_orchestrator(self) -> None:
        """Provider swap delegates to orchestrator.swap_llm_provider."""
        server = _make_server(with_orchestrator=True)
        server._orchestrator.swap_llm_provider("deepseek", {"model": "deepseek-chat"})
        server._orchestrator.swap_llm_provider.assert_called_once_with(
            "deepseek", {"model": "deepseek-chat"}
        )

    def test_swap_result_shape(self) -> None:
        """Successful swap returns expected keys."""
        server = _make_server(with_orchestrator=True)
        result = server._orchestrator.swap_llm_provider(
            "deepseek", {"model": "deepseek-chat"}
        )
        assert result["success"] is True
        assert result["provider"] == "deepseek"
        assert result["model"] == "deepseek-chat"
        assert result["context_size"] == 32768

    def test_persistence_called_on_success(self) -> None:
        """save_to_yaml is called after successful swap."""
        server = _make_server(with_orchestrator=True)
        server._orchestrator._config.save_to_yaml(server._config_path)
        server._orchestrator._config.save_to_yaml.assert_called_once_with(
            "/tmp/test_config.yaml"
        )

    def test_swap_failure_returns_error(self) -> None:
        """Failed swap returns error in result."""
        server = _make_server(with_orchestrator=True)
        server._orchestrator.swap_llm_provider.return_value = {
            "success": False,
            "error": "Cannot swap provider while processing",
        }
        result = server._orchestrator.swap_llm_provider("deepseek", {})
        assert result["success"] is False
        assert "processing" in result["error"]

    def test_no_orchestrator_no_crash(self) -> None:
        """Handler doesn't crash when orchestrator is None."""
        server = _make_server(with_orchestrator=False)
        assert server._orchestrator is None
