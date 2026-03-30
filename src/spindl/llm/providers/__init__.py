"""
Providers package for modular prompt context sources.

Each provider implements the ContextProvider interface and fills
a specific placeholder in the prompt template.
"""

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
from .registry import ProviderRegistry, create_default_registry
from .voice_state_provider import VoiceStateProvider

__all__ = [
    # Persona providers
    "PersonaNameProvider",
    "PersonaAppearanceProvider",
    "PersonaPersonalityProvider",
    "ScenarioProvider",
    "ExampleDialogueProvider",
    "PersonaRulesProvider",
    # Modality providers
    "ModalityContextProvider",
    "ModalityRulesProvider",
    # Voice state provider
    "VoiceStateProvider",
    # History providers
    "SummaryProvider",
    "RecentHistoryProvider",
    # Input provider
    "CurrentInputProvider",
    # Registry
    "ProviderRegistry",
    "create_default_registry",
]
