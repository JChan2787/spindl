"""Core state management and event system."""

from .state_machine import (
    AgentCallbacks,
    AgentState,
    AudioStateMachine,
    StateTransition,
)
from .events import (
    Event,
    EventType,
    TranscriptionReadyEvent,
    ResponseReadyEvent,
    TTSStartedEvent,
    TTSCompletedEvent,
    TTSInterruptedEvent,
    StateChangedEvent,
    ContextUpdatedEvent,
    PipelineErrorEvent,
    TokenUsageEvent,
    StimulusFiredEvent,
    AvatarMoodEvent,
    AvatarToolMoodEvent,
)
from .event_bus import EventBus
from .context_manager import ContextManager, AggregatedContext

__all__ = [
    # State machine
    "AgentCallbacks",
    "AgentState",
    "AudioStateMachine",
    "StateTransition",
    # Event system
    "Event",
    "EventType",
    "EventBus",
    "TranscriptionReadyEvent",
    "ResponseReadyEvent",
    "TTSStartedEvent",
    "TTSCompletedEvent",
    "TTSInterruptedEvent",
    "StateChangedEvent",
    "ContextUpdatedEvent",
    "PipelineErrorEvent",
    "TokenUsageEvent",
    "StimulusFiredEvent",
    "AvatarMoodEvent",
    "AvatarToolMoodEvent",
    # Context management
    "ContextManager",
    "AggregatedContext",
]
