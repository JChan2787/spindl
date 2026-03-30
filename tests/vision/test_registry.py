"""Tests for VLMProviderRegistry."""

import pytest

from spindl.vision.registry import VLMProviderRegistry
from spindl.vision.base import VLMProvider


class TestVLMProviderRegistry:
    """Tests for VLMProviderRegistry."""

    def test_list_providers_includes_builtins(self) -> None:
        """list_providers includes built-in providers."""
        registry = VLMProviderRegistry()
        providers = registry.list_providers()

        assert "llama" in providers
        assert "openai" in providers
        assert "llm" in providers

    def test_get_provider_class_llama(self) -> None:
        """get_provider_class returns LlamaVLMProvider for 'llama'."""
        registry = VLMProviderRegistry()
        provider_class = registry.get_provider_class("llama")

        assert issubclass(provider_class, VLMProvider)
        assert provider_class.__name__ == "LlamaVLMProvider"

    def test_get_provider_class_openai(self) -> None:
        """get_provider_class returns OpenAIVLMProvider for 'openai'."""
        registry = VLMProviderRegistry()
        provider_class = registry.get_provider_class("openai")

        assert issubclass(provider_class, VLMProvider)
        assert provider_class.__name__ == "OpenAIVLMProvider"

    def test_get_provider_class_unknown_raises(self) -> None:
        """get_provider_class raises ValueError for unknown provider."""
        registry = VLMProviderRegistry()

        with pytest.raises(ValueError) as exc_info:
            registry.get_provider_class("nonexistent")

        assert "Unknown VLM provider" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_validate_provider_config_valid(self) -> None:
        """validate_provider_config returns empty list for valid config."""
        registry = VLMProviderRegistry()

        errors = registry.validate_provider_config("llama", {
            "executable_path": "/opt/llama/llama-server",
            "model_type": "gemma3",
            "model_path": "/path/to/model.gguf",
            "mmproj_path": "/path/to/mmproj.gguf",
        })

        assert errors == []

    def test_validate_provider_config_invalid(self) -> None:
        """validate_provider_config returns errors for invalid config."""
        registry = VLMProviderRegistry()

        errors = registry.validate_provider_config("llama", {
            # Missing required fields
        })

        assert len(errors) > 0

    def test_validate_provider_config_unknown_provider(self) -> None:
        """validate_provider_config returns error for unknown provider."""
        registry = VLMProviderRegistry()

        errors = registry.validate_provider_config("nonexistent", {})

        assert len(errors) == 1
        assert "Unknown VLM provider" in errors[0]

    # --- LLM provider (NANO-030) ---

    def test_get_provider_class_llm(self) -> None:
        """get_provider_class returns LLMVisionProvider for 'llm'."""
        registry = VLMProviderRegistry()
        provider_class = registry.get_provider_class("llm")

        assert issubclass(provider_class, VLMProvider)
        assert provider_class.__name__ == "LLMVisionProvider"

    def test_validate_provider_config_llm_valid(self) -> None:
        """validate_provider_config returns empty list for valid LLM config."""
        registry = VLMProviderRegistry()

        errors = registry.validate_provider_config("llm", {
            "url": "http://127.0.0.1:5557",
        })

        assert errors == []

    def test_validate_provider_config_llm_invalid(self) -> None:
        """validate_provider_config returns errors for missing LLM url."""
        registry = VLMProviderRegistry()

        errors = registry.validate_provider_config("llm", {})

        assert len(errors) > 0
        assert "url" in errors[0].lower()
