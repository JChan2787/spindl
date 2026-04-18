"""
Voice state provider - Inject context based on voice state machine triggers.

This provider fills the [STATE_CONTEXT] placeholder based on the current
state_trigger from the AudioStateMachine. Used to inform the model about
special situations like barge-in (user interrupted) or empty transcription.
"""

from typing import Optional

from ..build_context import BuildContext
from ..context_provider import ContextProvider
from ..prompt_library import VOICE_STATE_INJECTIONS


class VoiceStateProvider(ContextProvider):
    """
    Provides voice state context for [STATE_CONTEXT] placeholder.

    Maps state_trigger values to human-readable context injections:
    - "barge_in" -> "The User interrupted you mid-sentence."
    - "empty_transcription" -> "The User made a sound but no words were detected."
    - "error" -> "An error occurred. Acknowledge briefly and continue."

    Returns None for normal state transitions (vad_speech_start, tts_complete),
    causing the section to collapse.
    """

    @property
    def placeholder(self) -> str:
        return "[STATE_CONTEXT]"

    def provide(self, context: BuildContext) -> Optional[str]:
        # No trigger = normal conversation flow, collapse section
        if not context.state_trigger:
            return None

        # Look up base injection text for this trigger
        injection = VOICE_STATE_INJECTIONS.get(context.state_trigger)

        if not injection:
            # Unknown trigger - collapse section (normal flow)
            return None

        return injection.strip()
