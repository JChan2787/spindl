"""Tests for ProviderHolder indirection layer (NANO-065b).

Tests cover:
- Delegation of all LLMProvider methods
- swap() returns old provider and updates inner reference
- isinstance() compatibility with LLMProvider
- Registry storage
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.llm.base import LLMProvider, LLMProperties, LLMResponse
from spindl.llm.provider_holder import ProviderHolder


def _make_provider(**props) -> MagicMock:
    """Create a mock LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_properties.return_value = LLMProperties(
        model_name=props.get("model_name", "test-model"),
        supports_streaming=props.get("supports_streaming", False),
        context_length=props.get("context_length", 8192),
        supports_tools=props.get("supports_tools", False),
    )
    provider.generate.return_value = LLMResponse(
        content="hello", input_tokens=10, output_tokens=5
    )
    provider.count_tokens.return_value = 42
    provider.health_check.return_value = True
    return provider


def _make_holder(provider=None, registry=None) -> ProviderHolder:
    """Create a ProviderHolder with optional mocks."""
    provider = provider or _make_provider()
    registry = registry or MagicMock()
    return ProviderHolder(provider, registry)


class TestProviderHolderIsLLMProvider:
    """Verify ABC compliance."""

    def test_isinstance(self) -> None:
        holder = _make_holder()
        assert isinstance(holder, LLMProvider)

    def test_provider_property(self) -> None:
        provider = _make_provider()
        holder = _make_holder(provider=provider)
        assert holder.provider is provider


class TestDelegation:
    """Verify all methods delegate to inner provider."""

    def test_get_properties(self) -> None:
        provider = _make_provider(model_name="qwen3-14b")
        holder = _make_holder(provider=provider)
        props = holder.get_properties()
        assert props.model_name == "qwen3-14b"
        provider.get_properties.assert_called_once()

    def test_generate(self) -> None:
        provider = _make_provider()
        holder = _make_holder(provider=provider)
        result = holder.generate([{"role": "user", "content": "hi"}], temperature=0.5)
        assert result.content == "hello"
        provider.generate.assert_called_once_with(
            [{"role": "user", "content": "hi"}], 0.5, 256, tools=None
        )

    def test_generate_stream(self) -> None:
        provider = _make_provider()
        holder = _make_holder(provider=provider)
        holder.generate_stream([{"role": "user", "content": "hi"}])
        provider.generate_stream.assert_called_once()

    def test_count_tokens(self) -> None:
        provider = _make_provider()
        holder = _make_holder(provider=provider)
        result = holder.count_tokens("hello world")
        assert result == 42
        provider.count_tokens.assert_called_once_with("hello world")

    def test_health_check(self) -> None:
        provider = _make_provider()
        holder = _make_holder(provider=provider)
        assert holder.health_check() is True
        provider.health_check.assert_called_once()

    def test_initialize(self) -> None:
        provider = _make_provider()
        holder = _make_holder(provider=provider)
        holder.initialize({"api_key": "test"})
        provider.initialize.assert_called_once_with({"api_key": "test"})

    def test_shutdown(self) -> None:
        provider = _make_provider()
        holder = _make_holder(provider=provider)
        holder.shutdown()
        provider.shutdown.assert_called_once()


class TestSwap:
    """Verify swap() mechanics."""

    def test_swap_returns_old(self) -> None:
        old_provider = _make_provider(model_name="old-model")
        holder = _make_holder(provider=old_provider)
        new_provider = _make_provider(model_name="new-model")
        returned = holder.swap(new_provider)
        assert returned is old_provider

    def test_swap_updates_inner(self) -> None:
        old_provider = _make_provider(model_name="old-model")
        holder = _make_holder(provider=old_provider)
        new_provider = _make_provider(model_name="new-model")
        holder.swap(new_provider)
        assert holder.provider is new_provider
        assert holder.get_properties().model_name == "new-model"

    def test_swap_old_not_called_after_swap(self) -> None:
        old_provider = _make_provider()
        holder = _make_holder(provider=old_provider)
        new_provider = _make_provider()
        holder.swap(new_provider)
        old_provider.reset_mock()
        holder.generate([{"role": "user", "content": "test"}])
        old_provider.generate.assert_not_called()
        new_provider.generate.assert_called_once()


class TestClassMethodDelegation:
    """Verify class-method wrappers."""

    def test_validate_config_noop(self) -> None:
        """Class-level validate_config returns empty (never called on holder)."""
        assert ProviderHolder.validate_config({}) == []

    def test_validate_provider_config_delegates(self) -> None:
        """Instance-level validate_provider_config delegates to inner class."""
        provider = _make_provider()
        provider.__class__.validate_config = MagicMock(return_value=["error"])
        holder = _make_holder(provider=provider)
        errors = holder.validate_provider_config({"key": "val"})
        assert errors == ["error"]
