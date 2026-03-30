"""Tests for ScreenVisionTool.swap_vlm_provider() (NANO-065c).

Tests cover:
- Successful provider swap (shutdown old, init new)
- Old provider shutdown called
- Init failure propagates
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.tools.builtin.screen_vision import ScreenVisionTool


def _make_tool_with_provider() -> tuple[ScreenVisionTool, MagicMock]:
    """Create a ScreenVisionTool with a mocked VLM provider."""
    tool = ScreenVisionTool()
    mock_provider = MagicMock()
    mock_provider.health_check.return_value = True
    tool._vlm_provider = mock_provider
    tool._initialized = True
    return tool, mock_provider


class TestSwapVLMProvider:
    """Tests for swap_vlm_provider()."""

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_successful_swap(self, MockRegistry) -> None:
        """New provider is initialized and assigned."""
        tool, old_provider = _make_tool_with_provider()

        new_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_provider_class.return_value = lambda: new_provider
        MockRegistry.return_value = mock_registry

        tool.swap_vlm_provider("openai", {"api_key": "test"}, [])

        assert tool._vlm_provider is new_provider
        new_provider.initialize.assert_called_once_with({"api_key": "test"})

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_shuts_down_old_provider(self, MockRegistry) -> None:
        """Old provider's shutdown() is called during swap."""
        tool, old_provider = _make_tool_with_provider()

        new_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_provider_class.return_value = lambda: new_provider
        MockRegistry.return_value = mock_registry

        tool.swap_vlm_provider("openai", {}, [])

        old_provider.shutdown.assert_called_once()

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_old_shutdown_error_does_not_block(self, MockRegistry) -> None:
        """Swap proceeds even if old provider shutdown raises."""
        tool, old_provider = _make_tool_with_provider()
        old_provider.shutdown.side_effect = RuntimeError("cleanup failed")

        new_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_provider_class.return_value = lambda: new_provider
        MockRegistry.return_value = mock_registry

        tool.swap_vlm_provider("openai", {}, [])

        assert tool._vlm_provider is new_provider

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_init_failure_propagates(self, MockRegistry) -> None:
        """If new provider init fails, exception propagates."""
        tool, old_provider = _make_tool_with_provider()

        new_provider = MagicMock()
        new_provider.initialize.side_effect = RuntimeError("init failed")
        mock_registry = MagicMock()
        mock_registry.get_provider_class.return_value = lambda: new_provider
        MockRegistry.return_value = mock_registry

        with pytest.raises(RuntimeError, match="init failed"):
            tool.swap_vlm_provider("openai", {}, [])

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_swap_from_none_provider(self, MockRegistry) -> None:
        """Swap works when starting with no provider (e.g., disabled state)."""
        tool = ScreenVisionTool()
        tool._initialized = True
        assert tool._vlm_provider is None

        new_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_provider_class.return_value = lambda: new_provider
        MockRegistry.return_value = mock_registry

        tool.swap_vlm_provider("llama", {"host": "127.0.0.1"}, [])

        assert tool._vlm_provider is new_provider

    @patch("spindl.vision.registry.VLMProviderRegistry")
    def test_plugin_paths_passed_to_registry(self, MockRegistry) -> None:
        """Plugin paths are forwarded to VLMProviderRegistry."""
        tool, _ = _make_tool_with_provider()

        new_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_provider_class.return_value = lambda: new_provider
        MockRegistry.return_value = mock_registry

        tool.swap_vlm_provider("openai", {}, ["./plugins/vlm"])

        MockRegistry.assert_called_once_with(plugin_paths=["./plugins/vlm"])
