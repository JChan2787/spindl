"""
Provider registry - Central registration and management of context providers.

The registry holds all active providers and provides iteration for the
prompt builder. Providers are applied in registration order.
"""

import logging
from typing import Iterator, Optional

from ..context_provider import ContextProvider
from .history_provider import RecentHistoryProvider, SummaryProvider
from .input_provider import CurrentInputProvider
from .modality_provider import ModalityContextProvider, ModalityRulesProvider
from .persona_provider import (
    ExampleDialogueProvider,
    PersonaAppearanceProvider,
    PersonaNameProvider,
    PersonaPersonalityProvider,
    PersonaRulesProvider,
    ScenarioProvider,
)
from .voice_state_provider import VoiceStateProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Registry for context providers.

    Holds providers and provides iteration in registration order.
    The prompt builder iterates through providers and substitutes
    their content into the template.
    """

    def __init__(self) -> None:
        self._providers: list[ContextProvider] = []

    def register(self, provider: ContextProvider) -> None:
        """
        Register a provider.

        Providers are applied in registration order during prompt building.

        Args:
            provider: ContextProvider instance to register.
        """
        self._providers.append(provider)

    def register_all(self, providers: list[ContextProvider]) -> None:
        """
        Register multiple providers at once.

        Args:
            providers: List of ContextProvider instances to register.
        """
        self._providers.extend(providers)

    def __iter__(self) -> Iterator[ContextProvider]:
        """Iterate over registered providers in order."""
        return iter(self._providers)

    def __len__(self) -> int:
        """Return number of registered providers."""
        return len(self._providers)

    @property
    def providers(self) -> list[ContextProvider]:
        """Return list of registered providers (read-only copy)."""
        return list(self._providers)


def create_default_registry() -> ProviderRegistry:
    """
    Create a registry with all standard providers registered.

    Provider order:
    1. Persona providers (name, appearance, personality, scenario, example dialogue, rules)
    2. Modality providers (context, rules)
    3. Voice state provider
    4. History providers (summary, recent)
    5. Input provider

    Note: Vision is no longer injected into prompts. Use tools (NANO-024)
    for on-demand vision capabilities instead of always-on injection.

    Returns:
        ProviderRegistry with all default providers registered.
    """
    registry = ProviderRegistry()

    # Persona providers
    registry.register(PersonaNameProvider())
    registry.register(PersonaAppearanceProvider())
    registry.register(PersonaPersonalityProvider())
    registry.register(ScenarioProvider())
    registry.register(ExampleDialogueProvider())
    registry.register(PersonaRulesProvider())

    # Modality providers
    registry.register(ModalityContextProvider())
    registry.register(ModalityRulesProvider())

    # Voice state provider
    registry.register(VoiceStateProvider())

    # History providers
    # Note: SummaryProvider fills [CONVERSATION_SUMMARY] (collapses when empty).
    # RecentHistoryProvider is NOT registered — [RECENT_HISTORY] placeholder is
    # filled downstream by HistoryInjector (PreProcessor) which formats JSONL
    # history as inline text in the system prompt.
    registry.register(SummaryProvider())

    # Input provider
    registry.register(CurrentInputProvider())

    return registry


def create_prompt_provider_registry() -> ProviderRegistry:
    """
    Alias for create_default_registry.

    This is the function called by the orchestrator.

    Returns:
        ProviderRegistry with all providers registered.
    """
    return create_default_registry()
