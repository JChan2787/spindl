"""
ProviderHolder — Thin indirection layer for runtime LLM provider swapping.

Wraps an LLMProvider instance and delegates all interface methods.
All consumers (pipeline, plugins, memory subsystems) receive the holder
instead of the raw provider. On swap, the inner reference updates and
consumers auto-see the new provider on their next call.

NANO-065b.
"""

import logging
from typing import Iterator, Optional

from .base import LLMProperties, LLMProvider, LLMResponse, StreamChunk
from .registry import LLMProviderRegistry

logger = logging.getLogger(__name__)


class ProviderHolder(LLMProvider):
    """Indirection layer enabling runtime provider hot-swap.

    Subclasses LLMProvider so it satisfies isinstance() checks and
    type hints throughout the codebase. All abstract methods delegate
    to the inner provider.
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: LLMProviderRegistry,
    ):
        self._provider = provider
        self._registry = registry

    @property
    def provider(self) -> LLMProvider:
        """The currently active inner provider."""
        return self._provider

    def swap(self, new_provider: LLMProvider) -> LLMProvider:
        """Swap the underlying provider. Returns the old one for cleanup."""
        old = self._provider
        self._provider = new_provider
        logger.info(
            "Provider swapped: %s -> %s",
            type(old).__name__,
            type(new_provider).__name__,
        )
        return old

    # ── LLMProvider abstract method delegation ──────────────────────

    def initialize(self, config: dict) -> None:
        return self._provider.initialize(config)

    def get_properties(self) -> LLMProperties:
        return self._provider.get_properties()

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        return self._provider.generate(
            messages, temperature, max_tokens, tools=tools, **kwargs
        )

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        return self._provider.generate_stream(
            messages, temperature, max_tokens, tools=tools, **kwargs
        )

    def count_tokens(self, text: str) -> int:
        return self._provider.count_tokens(text)

    def health_check(self) -> bool:
        return self._provider.health_check()

    def shutdown(self) -> None:
        return self._provider.shutdown()

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        # Class method — cannot delegate to instance. This is only called
        # on concrete provider classes before instantiation, never on the
        # holder itself. Provide a no-op implementation for ABC compliance.
        return []

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        return None

    @classmethod
    def is_cloud_provider(cls) -> bool:
        return False

    # ── Instance-level class method delegation ──────────────────────

    def validate_provider_config(self, config: dict) -> list[str]:
        """Delegate to the inner provider's class-level validate_config."""
        return self._provider.__class__.validate_config(config)

    def get_provider_server_command(self, config: dict) -> Optional[str]:
        """Delegate to the inner provider's class-level get_server_command."""
        return self._provider.__class__.get_server_command(config)

    def is_provider_cloud(self) -> bool:
        """Delegate to the inner provider's class-level is_cloud_provider."""
        return self._provider.__class__.is_cloud_provider()
