"""
Event definitions for the spindl event system.

Provides typed events for publish/subscribe communication between components:
- Speech pipeline events (transcription, response, TTS lifecycle)
- State machine events
- Context events (for future multimodal integration)
- Error events
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import time


class EventType(Enum):
    """All event types in the spindl system."""

    # Speech pipeline events
    TRANSCRIPTION_READY = auto()  # STT produced text
    RESPONSE_READY = auto()  # LLM produced response
    TTS_STARTED = auto()  # Audio playback began
    TTS_COMPLETED = auto()  # Audio playback finished naturally
    TTS_INTERRUPTED = auto()  # Audio playback interrupted (barge-in)

    # State machine events
    STATE_CHANGED = auto()  # Agent state transition

    # Context events (future multimodal)
    CONTEXT_UPDATED = auto()  # ContextManager assembled new context

    # Token usage events (NANO-017)
    TOKEN_USAGE = auto()  # LLM token consumption report

    # Prompt inspection events (NANO-025 Phase 3)
    PROMPT_SNAPSHOT = auto()  # Full prompt sent to LLM with token breakdown

    # Tool events (NANO-025 Phase 7)
    TOOL_INVOKED = auto()  # Tool execution started
    TOOL_RESULT = auto()  # Tool execution completed

    # Audio level events (NANO-069)
    AUDIO_LEVEL = auto()  # Real-time audio output amplitude for portrait

    # Mic input level events (NANO-073b)
    MIC_LEVEL = auto()  # Real-time mic input amplitude for voice overlay

    # Stimuli events (NANO-056)
    STIMULUS_FIRED = auto()  # Stimulus module triggered autonomous response

    # Avatar events (NANO-093)
    AVATAR_MOOD = auto()       # Emotion classifier produced mood for avatar
    AVATAR_TOOL_MOOD = auto()  # Tool invocation mapped to avatar visual category

    # Error events
    PIPELINE_ERROR = auto()  # Processing error occurred


@dataclass
class Event:
    """
    Base event with common fields.

    All events carry a timestamp and can be consumed to stop propagation
    to lower-priority handlers.
    """

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    consumed: bool = field(default=False, init=False)

    def consume(self) -> None:
        """
        Mark event as consumed.

        When an event is consumed, lower-priority handlers will not receive it.
        Useful for command parsing that intercepts certain transcriptions.
        """
        self.consumed = True


@dataclass
class TranscriptionReadyEvent(Event):
    """
    Fired when STT produces transcription text.

    Attributes:
        text: The transcribed text from user speech.
        duration: Duration of the speech in seconds.
        input_modality: Origin of the input — "voice", "text", or "stimulus".
    """

    event_type: EventType = field(default=EventType.TRANSCRIPTION_READY, init=False)
    text: str = ""
    duration: float = 0.0
    input_modality: str = "voice"


@dataclass
class ResponseReadyEvent(Event):
    """
    Fired when LLM produces response text.

    Attributes:
        text: The generated response text.
        user_input: The transcription that triggered this response.
        activated_codex_entries: List of codex entries that fired for this response
                                (NANO-037 Phase 2). Each dict has: name, keys, activation_method.
    """

    event_type: EventType = field(default=EventType.RESPONSE_READY, init=False)
    text: str = ""
    user_input: str = ""
    activated_codex_entries: list = field(default_factory=list)
    retrieved_memories: list = field(default_factory=list)
    """List of retrieved memories for GUI display (NANO-044)."""
    reasoning: Optional[str] = None
    """Thinking/reasoning content from the LLM, if present (NANO-042)."""
    stimulus_source: Optional[str] = None
    """If set, identifies this as an autonomous stimulus response (NANO-056)."""
    emotion: Optional[str] = None
    """Classified emotion mood string for avatar/chat display (NANO-094)."""
    emotion_confidence: Optional[float] = None
    """Confidence score (0-100) of the emotion classification (NANO-094)."""
    tts_text: Optional[str] = None
    """TTS-safe version of the response with formatting stripped (NANO-109)."""


@dataclass
class TTSStartedEvent(Event):
    """
    Fired when TTS audio playback begins.

    Attributes:
        duration: Expected duration of the audio in seconds.
    """

    event_type: EventType = field(default=EventType.TTS_STARTED, init=False)
    duration: float = 0.0


@dataclass
class TTSCompletedEvent(Event):
    """Fired when TTS audio playback completes naturally."""

    event_type: EventType = field(default=EventType.TTS_COMPLETED, init=False)


@dataclass
class TTSInterruptedEvent(Event):
    """Fired when TTS audio playback is interrupted by barge-in."""

    event_type: EventType = field(default=EventType.TTS_INTERRUPTED, init=False)


@dataclass
class AudioLevelEvent(Event):
    """
    Real-time audio output amplitude for portrait visualization (NANO-069).

    Emitted at ~50ms intervals during TTS playback. Level is 0.0-1.0 RMS.
    """

    event_type: EventType = field(default=EventType.AUDIO_LEVEL, init=False)
    level: float = 0.0


@dataclass
class MicLevelEvent(Event):
    """
    Real-time mic input amplitude for voice overlay visualization (NANO-073b).

    Emitted at ~50ms intervals during user_speaking state. Level is 0.0-1.0 RMS.
    """

    event_type: EventType = field(default=EventType.MIC_LEVEL, init=False)
    level: float = 0.0


@dataclass
class StateChangedEvent(Event):
    """
    Fired on agent state transitions.

    Attributes:
        from_state: Previous state name.
        to_state: New state name.
        trigger: What caused the transition.
    """

    event_type: EventType = field(default=EventType.STATE_CHANGED, init=False)
    from_state: str = ""
    to_state: str = ""
    trigger: str = ""


@dataclass
class ContextUpdatedEvent(Event):
    """
    Fired when ContextManager assembles new context.

    Attributes:
        sources: Names of context sources that contributed.
    """

    event_type: EventType = field(default=EventType.CONTEXT_UPDATED, init=False)
    sources: list[str] = field(default_factory=list)


@dataclass
class PipelineErrorEvent(Event):
    """
    Fired when a pipeline stage encounters an error.

    Attributes:
        stage: Which stage failed ("stt", "llm", "tts").
        error_type: Exception class name.
        message: Error message.
    """

    event_type: EventType = field(default=EventType.PIPELINE_ERROR, init=False)
    stage: str = ""
    error_type: str = ""
    message: str = ""


@dataclass
class TokenUsageEvent(Event):
    """
    Fired after each LLM response with token consumption statistics.

    Attributes:
        prompt_tokens: Tokens used for the prompt (system + history + user input).
        completion_tokens: Tokens generated in the response.
        total_tokens: Sum of prompt + completion tokens.
        context_limit: Maximum context window size (n_ctx from server).

    Example console display:
        [Context] 142 + 45 = 187 tokens (2.3% of 8192)
    """

    event_type: EventType = field(default=EventType.TOKEN_USAGE, init=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_limit: int = 0

    @property
    def usage_percent(self) -> float:
        """Calculate percentage of context window used."""
        if self.context_limit <= 0:
            return 0.0
        return (self.total_tokens / self.context_limit) * 100


@dataclass
class PromptSnapshotEvent(Event):
    """
    Fired after each LLM call with the full prompt and token breakdown.

    Enables the Prompt Inspector GUI to show exactly what was sent to the LLM
    and how tokens are distributed across sections.

    Attributes:
        messages: The final message list sent to LLM ([{role, content}, ...]).
        token_breakdown: Token counts per section.
            {
                "total": int,
                "system": int,
                "user": int,
                "sections": {
                    "agent": int,
                    "context": int,
                    "rules": int,
                    "conversation": int,
                }
            }
        input_modality: "VOICE" or "TEXT".
        state_trigger: Optional trigger (e.g., "barge_in", "empty_transcription").
    """

    event_type: EventType = field(default=EventType.PROMPT_SNAPSHOT, init=False)
    messages: list[dict] = field(default_factory=list)
    token_breakdown: dict = field(default_factory=dict)
    input_modality: str = ""
    state_trigger: Optional[str] = None


@dataclass
class StimulusFiredEvent(Event):
    """
    Fired when the stimuli engine fires an autonomous stimulus (NANO-056).

    Attributes:
        source: Stimulus source identifier ("patience", "custom", etc.).
        prompt_text: Preview of the prompt text (truncated to 200 chars).
        elapsed_seconds: For PATIENCE, seconds since last activity.
    """

    event_type: EventType = field(default=EventType.STIMULUS_FIRED, init=False)
    source: str = ""
    prompt_text: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class ToolInvokedEvent(Event):
    """
    Fired when a tool execution begins (NANO-025 Phase 7).

    Attributes:
        tool_name: Name of the tool being invoked.
        arguments: Arguments passed to the tool.
        iteration: Current iteration in the tool loop (1-based).
        tool_call_id: Unique ID for this tool call.
    """

    event_type: EventType = field(default=EventType.TOOL_INVOKED, init=False)
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    iteration: int = 1
    tool_call_id: str = ""


@dataclass
class ToolResultEvent(Event):
    """
    Fired when a tool execution completes (NANO-025 Phase 7).

    Attributes:
        tool_name: Name of the tool that executed.
        success: Whether the tool executed successfully.
        result_summary: Truncated result output (first 200 chars).
        duration_ms: Execution time in milliseconds.
        iteration: Current iteration in the tool loop (1-based).
        tool_call_id: Unique ID for this tool call.
    """

    event_type: EventType = field(default=EventType.TOOL_RESULT, init=False)
    tool_name: str = ""
    success: bool = False
    result_summary: str = ""
    duration_ms: int = 0
    iteration: int = 1
    tool_call_id: str = ""


@dataclass
class AvatarMoodEvent(Event):
    """
    Emotion classifier produced a mood for the avatar (NANO-093).

    Emitted after LLM response when emotion classifier is active.
    The mood string corresponds to a key in the avatar's MOOD_SHAPES table.
    """

    event_type: EventType = field(default=EventType.AVATAR_MOOD, init=False)
    mood: str = ""
    confidence: float = 0.0


@dataclass
class AvatarToolMoodEvent(Event):
    """
    Tool invocation mapped to an avatar visual category (NANO-093).

    Emitted when a tool is invoked, mapping tool names to broad visual
    categories (search, execute, memory) for brief avatar expression flashes.
    """

    event_type: EventType = field(default=EventType.AVATAR_TOOL_MOOD, init=False)
    tool_mood: str = ""
