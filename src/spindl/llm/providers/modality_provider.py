"""
Modality providers - Inject context and rules based on input modality.

These providers fill modality-related placeholders in the prompt template:
- [MODALITY_CONTEXT] - Description of the interaction mode (voice, text, etc.)
- [MODALITY_RULES] - Rules specific to the current modality
"""

from typing import Optional

from ..build_context import BuildContext, InputModality
from ..context_provider import ContextProvider
from ..prompt_library import MODALITY_CONTEXT, TEXT_MODALITY_RULES, VOICE_MODALITY_RULES


class ModalityContextProvider(ContextProvider):
    """
    Provides modality context for [MODALITY_CONTEXT] placeholder.

    Informs the model about the current interaction mode (voice conversation,
    text chat, etc.) so it can adjust response style appropriately.
    """

    @property
    def placeholder(self) -> str:
        return "[MODALITY_CONTEXT]"

    def provide(self, context: BuildContext) -> Optional[str]:
        # Stimulus uses text context; map to "text" key for lookup
        modality_key = "text" if context.input_modality == InputModality.STIMULUS else context.input_modality.value
        modality_text = MODALITY_CONTEXT.get(modality_key)

        if not modality_text:
            return None

        # NANO-110: Append addressing-others context when returning from addressing someone else
        if context.addressing_others_prompt:
            modality_text = f"{modality_text.strip()}\n\n{context.addressing_others_prompt.strip()}"

        return modality_text.strip()


class ModalityRulesProvider(ContextProvider):
    """
    Provides modality-specific rules for [MODALITY_RULES] placeholder.

    Voice mode gets TTS-friendly rules (no asterisks, concise responses).
    Text mode gets more relaxed rules (formatting allowed).
    """

    @property
    def placeholder(self) -> str:
        return "[MODALITY_RULES]"

    def provide(self, context: BuildContext) -> Optional[str]:
        if context.input_modality == InputModality.VOICE:
            return VOICE_MODALITY_RULES.strip()
        elif context.input_modality in (InputModality.TEXT, InputModality.STIMULUS):
            return TEXT_MODALITY_RULES.strip()

        # Unknown modality - collapse section
        return None
