"""
EventBridge: Connects the orchestrator's EventBus to the Socket.IO server.

Subscribes to internal events and forwards them to connected GUI clients.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from spindl.core.events import (
    EventType,
    Event,
    AudioLevelEvent,
    MicLevelEvent,
    TranscriptionReadyEvent,
    ResponseReadyEvent,
    TTSStartedEvent,
    TTSCompletedEvent,
    TTSInterruptedEvent,
    StateChangedEvent,
    ContextUpdatedEvent,
    PipelineErrorEvent,
    TokenUsageEvent,
    PromptSnapshotEvent,
    ToolInvokedEvent,
    ToolResultEvent,
    StimulusFiredEvent,
    AvatarMoodEvent,
    AvatarToolMoodEvent,
    LLMChunkEvent,
    LLMTokenEvent,
)

if TYPE_CHECKING:
    from spindl.core.event_bus import EventBus
    from spindl.gui.server import GUIServer

logger = logging.getLogger(__name__)


class EventBridge:
    """
    Bridges the orchestrator's EventBus to the GUI Socket.IO server.

    Subscribes to all relevant event types and forwards them to
    connected GUI clients via the GUIServer.
    """

    def __init__(self, event_bus: "EventBus", gui_server: "GUIServer"):
        self._event_bus = event_bus
        self._gui_server = gui_server
        self._subscription_ids: list[str] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Set the event loop to use for async emissions.

        Call this from the GUI server thread after the loop is running.
        """
        self._loop = loop
        logger.info("EventBridge: Event loop set from GUI server thread")

    def start(self) -> None:
        """
        Start bridging events.

        Subscribes to all event types and routes them to Socket.IO.
        Note: Call set_event_loop() from the GUI thread for emissions to work.
        """
        # Subscribe to all event types
        self._subscribe(EventType.TRANSCRIPTION_READY, self._on_transcription)
        self._subscribe(EventType.RESPONSE_READY, self._on_response)
        self._subscribe(EventType.TTS_STARTED, self._on_tts_started)
        self._subscribe(EventType.TTS_COMPLETED, self._on_tts_completed)
        self._subscribe(EventType.TTS_INTERRUPTED, self._on_tts_interrupted)
        self._subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        self._subscribe(EventType.CONTEXT_UPDATED, self._on_context_updated)
        self._subscribe(EventType.PIPELINE_ERROR, self._on_pipeline_error)
        self._subscribe(EventType.TOKEN_USAGE, self._on_token_usage)
        self._subscribe(EventType.PROMPT_SNAPSHOT, self._on_prompt_snapshot)
        self._subscribe(EventType.TOOL_INVOKED, self._on_tool_invoked)
        self._subscribe(EventType.TOOL_RESULT, self._on_tool_result)
        self._subscribe(EventType.STIMULUS_FIRED, self._on_stimulus_fired)
        self._subscribe(EventType.AUDIO_LEVEL, self._on_audio_level)
        self._subscribe(EventType.MIC_LEVEL, self._on_mic_level)
        self._subscribe(EventType.AVATAR_MOOD, self._on_avatar_mood)
        self._subscribe(EventType.AVATAR_TOOL_MOOD, self._on_avatar_tool_mood)
        self._subscribe(EventType.LLM_CHUNK, self._on_llm_chunk)
        self._subscribe(EventType.LLM_TOKEN, self._on_llm_token)

        logger.info(f"EventBridge started with {len(self._subscription_ids)} subscriptions")

    def stop(self) -> None:
        """Stop bridging events and clean up subscriptions."""
        for sub_id in self._subscription_ids:
            self._event_bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("EventBridge stopped")

    def _subscribe(self, event_type: EventType, handler) -> None:
        """Subscribe to an event type with the given handler."""
        sub_id = self._event_bus.subscribe(
            event_type,
            handler,
            priority=-100,  # Low priority so we don't interfere with core handlers
        )
        self._subscription_ids.append(sub_id)

    def _should_emit(self) -> bool:
        """Check if we should emit (has clients and valid event loop)."""
        return (
            self._gui_server.has_clients
            and self._loop is not None
            and self._loop.is_running()
        )

    def _emit_async(self, coro) -> None:
        """
        Schedule an async emit on the event loop.

        Since EventBus handlers are synchronous, we need to bridge
        to async Socket.IO emissions.

        IMPORTANT: Call _should_emit() BEFORE creating the coroutine to avoid
        'coroutine was never awaited' warnings when there are no clients.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            # This shouldn't happen if _should_emit() was checked first,
            # but handle it gracefully
            logger.warning("Could not schedule async emit - no event loop")

    def _timestamp_iso(self, event: Event) -> str:
        """Convert event timestamp to ISO8601 string."""
        return datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()

    # === Event Handlers ===

    def _on_transcription(self, event: TranscriptionReadyEvent) -> None:
        """Handle transcription ready event."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_transcription(
                text=event.text,
                duration=event.duration,
                is_final=True,
                input_modality=event.input_modality,
            )
        )

    def _on_response(self, event: ResponseReadyEvent) -> None:
        """Handle response ready event (NANO-037: codex, NANO-042: reasoning, NANO-044: memories, NANO-056: stimulus, NANO-094: emotion, NANO-109: tts_text)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_response(
                text=event.text,
                is_final=True,
                activated_codex_entries=event.activated_codex_entries,
                retrieved_memories=event.retrieved_memories,
                reasoning=event.reasoning,
                stimulus_source=event.stimulus_source,
                emotion=event.emotion,
                emotion_confidence=event.emotion_confidence,
                tts_text=event.tts_text,
                chunks=event.chunks,
            )
        )

    def _on_tts_started(self, event: TTSStartedEvent) -> None:
        """Handle TTS started event."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_tts_status(
                status="started",
                duration=event.duration,
            )
        )

    def _on_tts_completed(self, event: TTSCompletedEvent) -> None:
        """Handle TTS completed event."""
        if not self._should_emit():
            return
        self._emit_async(self._gui_server.emit_tts_status(status="completed"))

    def _on_tts_interrupted(self, event: TTSInterruptedEvent) -> None:
        """Handle TTS interrupted event."""
        if not self._should_emit():
            return
        self._emit_async(self._gui_server.emit_tts_status(status="interrupted"))

    def _on_state_changed(self, event: StateChangedEvent) -> None:
        """Handle state changed event."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_state_changed(
                from_state=event.from_state,
                to_state=event.to_state,
                trigger=event.trigger,
                timestamp=self._timestamp_iso(event),
            )
        )

    def _on_context_updated(self, event: ContextUpdatedEvent) -> None:
        """Handle context updated event."""
        if not self._should_emit():
            return
        self._emit_async(self._gui_server.emit_context_updated(sources=event.sources))

    def _on_pipeline_error(self, event: PipelineErrorEvent) -> None:
        """Handle pipeline error event."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_pipeline_error(
                stage=event.stage,
                error_type=event.error_type,
                message=event.message,
            )
        )

    def _on_token_usage(self, event: TokenUsageEvent) -> None:
        """Handle token usage event."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_token_usage(
                prompt=event.prompt_tokens,
                completion=event.completion_tokens,
                total=event.total_tokens,
                max_tokens=event.context_limit,
                percent=event.usage_percent,
            )
        )

    def _on_prompt_snapshot(self, event: PromptSnapshotEvent) -> None:
        """Handle prompt snapshot event (NANO-025 Phase 3)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_prompt_snapshot(
                messages=event.messages,
                token_breakdown=event.token_breakdown,
                input_modality=event.input_modality,
                state_trigger=event.state_trigger,
                timestamp=self._timestamp_iso(event),
            )
        )

    def _on_tool_invoked(self, event: ToolInvokedEvent) -> None:
        """Handle tool invoked event (NANO-025 Phase 7)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_tool_invoked(
                tool_name=event.tool_name,
                arguments=event.arguments,
                iteration=event.iteration,
                tool_call_id=event.tool_call_id,
                timestamp=self._timestamp_iso(event),
            )
        )

    def _on_tool_result(self, event: ToolResultEvent) -> None:
        """Handle tool result event (NANO-025 Phase 7)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_tool_result(
                tool_name=event.tool_name,
                success=event.success,
                result_summary=event.result_summary,
                duration_ms=event.duration_ms,
                iteration=event.iteration,
                tool_call_id=event.tool_call_id,
            )
        )

    def _on_stimulus_fired(self, event: StimulusFiredEvent) -> None:
        """Handle stimulus fired event (NANO-056)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_stimulus_fired(
                source=event.source,
                prompt_text=event.prompt_text,
                elapsed_seconds=event.elapsed_seconds,
            )
        )

    def _on_audio_level(self, event: AudioLevelEvent) -> None:
        """Handle audio level event (NANO-069)."""
        if not self._should_emit():
            return
        self._emit_async(self._gui_server.emit_audio_level(level=event.level))

    def _on_mic_level(self, event: MicLevelEvent) -> None:
        """Handle mic input level event (NANO-073b)."""
        if not self._should_emit():
            return
        self._emit_async(self._gui_server.emit_mic_level(level=event.level))

    def _on_avatar_mood(self, event: AvatarMoodEvent) -> None:
        """Handle avatar mood event (NANO-093)."""
        if not self._should_emit():
            return
        self._emit_async(self._gui_server.emit_avatar_mood(mood=event.mood, confidence=event.confidence))

    def _on_avatar_tool_mood(self, event: AvatarToolMoodEvent) -> None:
        """Handle avatar tool mood event (NANO-093)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_avatar_tool_mood(tool_mood=event.tool_mood)
        )

    def _on_llm_chunk(self, event: LLMChunkEvent) -> None:
        """Handle streaming LLM sentence chunk (NANO-111)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_llm_chunk(
                text=event.text,
                is_final=event.is_final,
                emotion=event.emotion,
                emotion_confidence=event.emotion_confidence,
            )
        )

    def _on_llm_token(self, event: LLMTokenEvent) -> None:
        """Handle token-level LLM text for real-time dashboard display (NANO-111)."""
        if not self._should_emit():
            return
        self._emit_async(
            self._gui_server.emit_llm_token(token=event.token, is_final=event.is_final)
        )
