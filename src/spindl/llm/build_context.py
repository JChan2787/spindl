"""
BuildContext - The linchpin of the modular prompt building system.

This module defines the BuildContext dataclass that carries all state needed
for prompt construction. Every ContextProvider receives a BuildContext and
extracts what it needs.

Design Principle: Per-call context composition. Nothing is "static" vs "dynamic" -
everything is assembled at prompt build time from available context sources.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..core.state_machine import AgentState
    from .prompt_block import PromptBlock


class InputModality(Enum):
    """
    Input modality for the current request.

    Determines which context providers are relevant and what rules apply.
    """

    VOICE = "voice"  # Transcribed STT string from live voice conversation
    TEXT = "text"  # Direct text input (DMs, typed messages)
    STIMULUS = "stimulus"  # Autonomous stimulus injection (PATIENCE, custom modules)
    # Future modalities (documented for scaling, not yet implemented):
    # CHAT_STREAM = "chat_stream"  # Aggregated messages from Twitch/YouTube
    # VISION = "vision"  # Image/video frame descriptions


@dataclass
class Message:
    """A single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None


@dataclass
class BuildContext:
    """
    Context object passed to all ContextProviders during prompt building.

    This is the linchpin of the entire system. Every provider receives it
    and extracts what it needs. Design this wrong and every provider suffers.

    Attributes:
        input_content: The current user input text.
        input_modality: How the input was received (VOICE, TEXT, etc).
        input_metadata: Modality-specific metadata (e.g., VAD trigger for voice).

        conversation_state: Current state from AudioStateMachine (if applicable).
        state_trigger: What triggered the current state (e.g., "barge_in", "vad_speech_start").

        persona: Loaded persona config dict from YAML.
        config: Orchestrator/pipeline configuration dict.

        recent_messages: Recent conversation messages (populated by history provider).
        summary: Conversation summary (populated by summary provider).
    """

    # Current input
    input_content: str
    input_modality: InputModality = InputModality.TEXT
    input_metadata: dict[str, Any] = field(default_factory=dict)

    # Conversation state (from state machine)
    conversation_state: Optional["AgentState"] = None
    state_trigger: Optional[str] = None

    # Session context
    persona: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    # History (populated by providers or pre-filled)
    recent_messages: list[Message] = field(default_factory=list)
    summary: Optional[str] = None

    # Last assistant message (for barge-in context)
    # When user interrupts, this contains what the assistant was saying
    last_assistant_message: Optional[str] = None

    # Addressing-others prompt (NANO-110)
    # When set, ModalityContextProvider appends this to the voice modality context.
    # One-shot: consumed on the first pipeline call after addressing-others ends.
    addressing_others_prompt: Optional[str] = None

    # Prompt block configuration (NANO-045a)
    # When set, PromptBuilder uses block-based assembly instead of template substitution.
    # None means legacy template path (byte-identical output).
    block_config: Optional[list["PromptBlock"]] = None

    # Per-block content tracking (NANO-045b: per-block token counting)
    # Populated by _build_with_blocks() during prompt assembly. Each dict has:
    # id, label, section (header name or None), chars (character count), deferred (bool).
    block_contents: Optional[list[dict]] = None

    def with_updates(self, **kwargs: Any) -> "BuildContext":
        """
        Create a new BuildContext with updated fields.

        BuildContext is effectively immutable during a build pass.
        Use this to create variants without mutating the original.

        Example:
            updated = context.with_updates(summary="New summary...")
        """
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return BuildContext(**current)
