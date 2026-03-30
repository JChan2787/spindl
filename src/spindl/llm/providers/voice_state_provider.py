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

# Maximum characters to include from interrupted message
BARGE_IN_CONTEXT_MAX_CHARS = 200


class VoiceStateProvider(ContextProvider):
    """
    Provides voice state context for [STATE_CONTEXT] placeholder.

    Maps state_trigger values to human-readable context injections:
    - "barge_in" -> "The User interrupted you mid-sentence. You were saying: '...'"
    - "empty_transcription" -> "The User made a sound but no words were detected."
    - "error" -> "An error occurred. Acknowledge briefly and continue."

    For barge-in specifically, includes a truncated version of what the assistant
    was saying when interrupted, helping the model understand the context.

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

        # For barge-in, enhance with what was being said
        if context.state_trigger == "barge_in" and context.last_assistant_message:
            interrupted_text = self._truncate_message(
                context.last_assistant_message,
                BARGE_IN_CONTEXT_MAX_CHARS,
            )
            return f"{injection.strip()} You were saying: \"{interrupted_text}\""

        return injection.strip()

    def _truncate_message(self, message: str, max_chars: int) -> str:
        """
        Truncate message to max_chars, adding ellipsis if truncated.

        Tries to break at word boundary for cleaner truncation.
        """
        if len(message) <= max_chars:
            return message

        # Truncate and try to find last space for clean word break
        truncated = message[:max_chars]
        last_space = truncated.rfind(" ")

        if last_space > max_chars * 0.7:  # Only break at space if reasonably far in
            truncated = truncated[:last_space]

        return truncated.rstrip() + "..."
