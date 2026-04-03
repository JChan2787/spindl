"""
Callback implementations for the Voice Agent Orchestrator.

Bridges state machine events to the processing pipeline:
- on_user_speech_end: STT -> LLM -> TTS
- on_barge_in: Stop playback
- on_state_change: Logging

Emits events via EventBus for external subscribers.

Session 5 (NANO-014): Added state trigger tracking for voice state injection.
"""

import threading
import time
from typing import Callable, Optional

import numpy as np

from ..core import StateTransition
from ..core.event_bus import EventBus
from ..core.context_manager import ContextManager
from ..core.events import (
    TranscriptionReadyEvent,
    ResponseReadyEvent,
    PipelineErrorEvent,
    TokenUsageEvent,
    PromptSnapshotEvent,
    StateChangedEvent,
    AvatarMoodEvent,
)
from ..history.snapshot_store import append_snapshot
from ..llm import LLMPipeline
from ..llm.build_context import InputModality
from ..stt import STTProvider
from ..tts import TTSProvider

import logging

logger = logging.getLogger(__name__)


class OrchestratorCallbacks:
    """
    Callback implementations for AudioStateMachine.

    Handles the processing pipeline when state machine events fire:
    - User finishes speaking: transcribe, generate response, synthesize
    - User interrupts: stop playback
    - State changes: log for debugging

    Processing runs in a background thread to avoid blocking audio callbacks.
    """

    def __init__(
        self,
        stt_client: STTProvider,
        tts_provider: TTSProvider,
        llm_pipeline: LLMPipeline,
        persona: dict,
        on_response_ready: Optional[Callable[[np.ndarray], None]] = None,
        on_barge_in_triggered: Optional[Callable[[], None]] = None,
        on_empty_transcription: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        on_pause_listening: Optional[Callable[[], None]] = None,
        on_resume_listening: Optional[Callable[[], None]] = None,
        event_bus: Optional[EventBus] = None,
        context_manager: Optional[ContextManager] = None,
        context_limit_getter: Optional[Callable[[], int]] = None,
    ):
        """
        Initialize orchestrator callbacks.

        Args:
            stt_client: Speech-to-text provider (NANO-061a: provider-based).
            tts_provider: TTS provider instance (NANO-015: provider-based architecture).
            llm_pipeline: LLM pipeline with registered plugins.
            persona: Persona configuration dict (id, name, system_prompt, voice, etc).
            on_response_ready: Called with TTS audio when response is ready to play.
            on_barge_in_triggered: Called when barge-in is detected.
            on_empty_transcription: Called when STT returns empty (noise/silence).
            on_error: Called on processing errors with (stage, exception).
            on_pause_listening: Called to suppress VAD during text input processing.
            on_resume_listening: Called to re-enable VAD after text input processing.
            event_bus: Optional EventBus for emitting events to subscribers.
            context_manager: Optional ContextManager for multimodal context assembly.
            context_limit_getter: Callable returning current context window size in tokens.
                Queries the active LLM provider's context_length (NANO-096).
        """
        self._stt = stt_client
        self._tts_provider = tts_provider
        self._pipeline = llm_pipeline
        self._persona = persona

        self._on_response_ready = on_response_ready
        self._on_barge_in_triggered = on_barge_in_triggered
        self._on_empty_transcription = on_empty_transcription
        self._on_error = on_error
        self._on_pause_listening = on_pause_listening
        self._on_resume_listening = on_resume_listening

        # NANO-111 Phase 2: Streaming playback callbacks (set by voice agent)
        self._on_response_ready_streaming: Optional[Callable[[np.ndarray], None]] = None
        self._append_playback_audio: Optional[Callable[[np.ndarray], None]] = None
        self._finalize_playback_streaming: Optional[Callable[[], None]] = None
        self._event_bus = event_bus
        self._context_manager = context_manager

        # Runtime generation parameter overrides (NANO-053)
        self._generation_params: Optional[dict] = None

        # NANO-076: Session file getter for snapshot sidecar persistence.
        # Set by VoiceAgentOrchestrator after construction.
        self._session_file_getter: Optional[Callable] = None

        # NANO-094: Emotion classifier (set post-construction via setter)
        self._emotion_classifier = None

        # NANO-110: Addressing-others state getters (set by orchestrator)
        self._is_addressing_others: Optional[Callable[[], bool]] = None
        self._consume_addressing_others_prompt: Optional[Callable[[], Optional[str]]] = None

        # Processing state
        self._processing_thread: Optional[threading.Thread] = None
        self._last_transcription: Optional[str] = None
        self._last_response: Optional[str] = None
        self._processing_start_time: Optional[float] = None

        # Voice state tracking (NANO-014 Session 5)
        # Captures significant triggers (barge_in, etc.) for prompt injection
        self._pending_state_trigger: Optional[str] = None

        # Token usage tracking (NANO-017)
        # Context limit getter — queries active provider (NANO-096)
        self._context_limit_getter = context_limit_getter or (lambda: 8192)
        # Last recorded context usage (prompt_tokens = current context size)
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0

        # Statistics
        self._total_turns = 0
        self._total_errors = 0
        self._total_empty_transcriptions = 0

    def on_user_speech_end(
        self,
        audio: np.ndarray,
        duration: float,
    ) -> None:
        """
        Handle end of user speech.

        Runs the full pipeline: STT -> LLM -> TTS

        Executes in a background thread to avoid blocking the audio callback.

        Args:
            audio: Audio buffer containing user speech (float32, 16kHz).
            duration: Duration of speech in seconds.
        """
        def process():
            self._processing_start_time = time.time()

            try:
                # NANO-110: Suppress voice pipeline while addressing others
                if self._is_addressing_others and self._is_addressing_others():
                    logger.info("[NANO-110] Voice input suppressed — addressing others")
                    if self._on_empty_transcription is not None:
                        self._on_empty_transcription()
                    return

                # 1. Transcribe speech to text
                transcription = self._stt.transcribe(audio)
                self._last_transcription = transcription

                # Emit transcription event
                if self._event_bus is not None:
                    self._event_bus.emit(
                        TranscriptionReadyEvent(
                            text=transcription or "",
                            duration=duration,
                        )
                    )

                # Check for empty transcription (noise, breathing, etc.)
                if not transcription or transcription.strip() == "":
                    self._total_empty_transcriptions += 1
                    if self._on_empty_transcription is not None:
                        self._on_empty_transcription()
                    return

                # 2. Assemble context from all sources (for future multimodal)
                if self._context_manager is not None:
                    context = self._context_manager.assemble(transcription)
                    # Future: use context.sources for lorebook/vision data
                    # For now, just pass through the primary input

                # 3. Capture and clear pending state trigger (NANO-014 Session 5)
                # This allows VoiceStateProvider to inject context like "User interrupted"
                state_trigger = self._pending_state_trigger
                self._pending_state_trigger = None

                # NANO-111: Check if streaming playback callbacks are wired
                # The pipeline's run_stream() handles provider/tool fallback internally —
                # it yields a single chunk from blocking run() if streaming isn't available.
                # We only need to check if the voice agent wired the streaming callbacks.
                from ..llm.provider_holder import ProviderHolder
                _inner = self._pipeline.provider.provider if isinstance(self._pipeline.provider, ProviderHolder) else self._pipeline.provider
                can_stream = _inner.get_properties().supports_streaming

                tts_config = self._persona.get("tts_voice_config", {})

                logger.debug(f"[NANO-111] can_stream={can_stream}, streaming_cb={self._on_response_ready_streaming is not None}")
                if can_stream and self._on_response_ready_streaming is not None:
                    # --- NANO-111 Phase 2: Streaming path ---
                    # Playback starts inline during _synthesize_streaming via
                    # play_streaming/append_audio/finalize_streaming callbacks.
                    result = self._synthesize_streaming(transcription, tts_config)

                    if result is None:
                        return

                    self._total_turns += 1
                else:
                    # --- Blocking path (existing behavior) ---

                    # 4. Capture last assistant message for barge-in context
                    last_msg = self._last_response if state_trigger == "barge_in" else None

                    # NANO-110: Consume one-shot addressing-others prompt
                    addressing_prompt = None
                    if self._consume_addressing_others_prompt:
                        addressing_prompt = self._consume_addressing_others_prompt()

                    # 5. Generate response via LLM pipeline (returns PipelineResult)
                    result = self._pipeline.run(
                        transcription,
                        self._persona,
                        generation_params=self._generation_params,
                        state_trigger=state_trigger,
                        input_modality=InputModality.VOICE,
                        last_assistant_message=last_msg,
                        addressing_others_prompt=addressing_prompt,
                    )
                    response = result.content
                    tts_response = result.tts_text or response  # NANO-109
                    self._last_response = response

                    # Classify emotion for avatar + chat display (NANO-094)
                    emotion, emotion_confidence = self._classify_emotion(response or "")

                    # Emit response event
                    if self._event_bus is not None:
                        self._event_bus.emit(
                            ResponseReadyEvent(
                                text=response or "",
                                user_input=transcription,
                                activated_codex_entries=result.activated_codex_entries,
                                retrieved_memories=result.retrieved_memories,
                                reasoning=result.reasoning,
                                emotion=emotion,
                                emotion_confidence=emotion_confidence,
                                tts_text=result.tts_text,
                            )
                        )

                    # Track token usage (NANO-017)
                    self._last_prompt_tokens = result.usage.prompt_tokens
                    self._last_completion_tokens = result.usage.completion_tokens

                    # Emit token usage event
                    if self._event_bus is not None:
                        self._event_bus.emit(
                            TokenUsageEvent(
                                prompt_tokens=result.usage.prompt_tokens,
                                completion_tokens=result.usage.completion_tokens,
                                total_tokens=result.usage.total_tokens,
                                context_limit=self._get_context_limit(),
                            )
                        )

                    # Emit prompt snapshot for GUI inspection (NANO-025 Phase 3)
                    if self._event_bus is not None:
                        token_breakdown = self._build_token_breakdown(
                            result.messages,
                            result.usage.prompt_tokens,
                            result.usage.completion_tokens,
                            block_contents=result.block_contents,
                        )
                        self._event_bus.emit(
                            PromptSnapshotEvent(
                                messages=result.messages,
                                token_breakdown=token_breakdown,
                                input_modality=result.input_modality,
                                state_trigger=result.state_trigger,
                            )
                        )

                        # NANO-076: Persist snapshot to sidecar JSONL
                        self._persist_snapshot(
                            result.messages, token_breakdown,
                            result.input_modality, result.state_trigger,
                            result.block_contents,
                        )

                    # Check for empty response
                    if not response or response.strip() == "":
                        return

                    # Synthesize speech via provider (NANO-015, NANO-054a, NANO-109)
                    audio_result = self._tts_provider.synthesize(
                        tts_response,
                        voice=tts_config.get("voice"),
                        **{k: v for k, v in tts_config.items() if k != "voice"},
                    )
                    audio_response = np.frombuffer(audio_result.data, dtype=np.float32)

                    # Signal response is ready
                    self._total_turns += 1
                    if self._on_response_ready is not None:
                        self._on_response_ready(audio_response)

            except Exception as e:
                self._total_errors += 1
                stage = self._determine_error_stage(e)
                print(f"[OrchestratorCallbacks] {stage} error: {e}")

                # Emit error event
                if self._event_bus is not None:
                    self._event_bus.emit(
                        PipelineErrorEvent(
                            stage=stage,
                            error_type=type(e).__name__,
                            message=str(e),
                        )
                    )

                if self._on_error is not None:
                    self._on_error(stage, e)

        # Run processing in background thread
        self._processing_thread = threading.Thread(target=process, daemon=True)
        self._processing_thread.start()

    def _determine_error_stage(self, error: Exception) -> str:
        """Determine which stage the error occurred in based on state."""
        if self._last_transcription is None:
            return "stt"
        elif self._last_response is None:
            return "llm"
        else:
            return "tts"

    def _synthesize_streaming(
        self,
        transcription: str,
        tts_config: dict,
    ) -> Optional[np.ndarray]:
        """
        Stream LLM response, synthesize TTS per-sentence in parallel,
        start playback on first chunk and append subsequent chunks (NANO-111 Phase 2).

        Uses pipeline.run_stream() to get sentence-level chunks, fires TTS
        synthesis for each sentence in a separate thread, delivers audio to
        the playback device in sentence order as chunks complete.

        Per-sentence emotion classification drives avatar expression transitions.

        Args:
            transcription: User's transcribed speech text.
            tts_config: TTS voice configuration from persona.

        Returns:
            None — playback is started inline via play_streaming/append_audio.
            Returns the full audio array for callers that need it (voice agent
            duration calculation still uses the concatenated result).
        """
        from ..llm.pipeline import StreamingPipelineChunk
        from ..core.events import LLMChunkEvent

        audio_results: dict[int, np.ndarray] = {}  # index → audio
        audio_ready = threading.Event()  # signaled when any new audio arrives
        tts_threads: list[threading.Thread] = []
        lock = threading.Lock()
        total_chunks_dispatched = 0  # how many TTS threads were fired

        def synthesize_chunk(tts_text: str, index: int):
            """Synthesize one sentence in a thread."""
            try:
                result = self._tts_provider.synthesize(
                    tts_text,
                    voice=tts_config.get("voice"),
                    **{k: v for k, v in tts_config.items() if k != "voice"},
                )
                audio = np.frombuffer(result.data, dtype=np.float32)
                with lock:
                    audio_results[index] = audio
                audio_ready.set()
            except Exception as e:
                logger.warning(f"[NANO-111] TTS failed for chunk {index}: {e}")
                with lock:
                    audio_results[index] = np.array([], dtype=np.float32)
                audio_ready.set()

        # Capture pending state trigger and last assistant message
        state_trigger = self._pending_state_trigger
        self._pending_state_trigger = None
        last_msg = self._last_response if state_trigger == "barge_in" else None

        # NANO-110: Consume one-shot addressing-others prompt
        addressing_prompt = None
        if self._consume_addressing_others_prompt:
            addressing_prompt = self._consume_addressing_others_prompt()

        full_display_text = []
        full_tts_text = []
        playback_started = False
        next_to_deliver = 0  # next sentence index to deliver to playback

        # NANO-111: Token-level callback for real-time dashboard display
        def _on_token(token_text: str, is_final: bool):
            if self._event_bus is not None:
                from ..core.events import LLMTokenEvent
                self._event_bus.emit(LLMTokenEvent(token=token_text, is_final=is_final))

        for chunk in self._pipeline.run_stream(
            transcription,
            self._persona,
            generation_params=self._generation_params,
            state_trigger=state_trigger,
            input_modality=InputModality.VOICE,
            last_assistant_message=last_msg,
            addressing_others_prompt=addressing_prompt,
            on_token=_on_token,
        ):
            full_display_text.append(chunk.display_text)
            full_tts_text.append(chunk.tts_text)

            # Emit LLM chunk event for real-time dashboard text
            if self._event_bus is not None:
                self._event_bus.emit(LLMChunkEvent(
                    text=chunk.display_text,
                    is_final=chunk.is_final,
                ))
                # NANO-111: Yield control so the asyncio event loop can flush
                # the Socket.IO emit to the client between sentences.
                # Without this, all emits queue up and flush at once after
                # the streaming loop completes.
                time.sleep(0.01)

            # Per-sentence emotion classification for avatar (NANO-111 Phase 2)
            if self._event_bus is not None and chunk.display_text.strip():
                emotion, confidence = self._classify_emotion(chunk.display_text)
                if emotion:
                    self._event_bus.emit(AvatarMoodEvent(
                        mood=emotion,
                        confidence=confidence,
                    ))

            # Fire TTS in parallel thread if there's text to speak
            if chunk.tts_text.strip():
                t = threading.Thread(
                    target=synthesize_chunk,
                    args=(chunk.tts_text, chunk.index),
                    daemon=True,
                )
                t.start()
                tts_threads.append(t)
                total_chunks_dispatched += 1

            # Try to deliver any ready audio chunks in order
            while True:
                with lock:
                    if next_to_deliver not in audio_results:
                        break
                    audio_chunk = audio_results[next_to_deliver]

                if len(audio_chunk) > 0:
                    if not playback_started:
                        # Start playback with first chunk — audio begins now
                        self._on_response_ready_streaming(audio_chunk)
                        playback_started = True
                    else:
                        self._append_playback_audio(audio_chunk)

                next_to_deliver += 1

        # LLM stream complete — wait for remaining TTS threads
        for t in tts_threads:
            t.join()

        # Deliver any remaining audio chunks that completed after the LLM stream ended
        while next_to_deliver < total_chunks_dispatched:
            with lock:
                if next_to_deliver in audio_results:
                    audio_chunk = audio_results[next_to_deliver]
                else:
                    break

            if len(audio_chunk) > 0:
                if not playback_started:
                    self._on_response_ready_streaming(audio_chunk)
                    playback_started = True
                else:
                    self._append_playback_audio(audio_chunk)

            next_to_deliver += 1

        # Signal no more audio coming
        self._finalize_playback_streaming()

        # Assemble full response for events and history
        response = " ".join(full_display_text)
        tts_text_combined = " ".join(full_tts_text)
        self._last_response = response

        # Emit response event with full text (final summary after all chunks)
        emotion, emotion_confidence = self._classify_emotion(response or "")
        if self._event_bus is not None:
            self._event_bus.emit(
                ResponseReadyEvent(
                    text=response or "",
                    user_input=transcription,
                    activated_codex_entries=[],
                    retrieved_memories=[],
                    reasoning=None,
                    emotion=emotion,
                    emotion_confidence=emotion_confidence,
                    tts_text=tts_text_combined,
                )
            )

        if not response or response.strip() == "":
            return None

        # Return concatenated audio for duration calculation
        ordered = [audio_results[i] for i in sorted(audio_results.keys()) if len(audio_results.get(i, [])) > 0]
        return np.concatenate(ordered) if ordered else None

    def on_barge_in(self) -> None:
        """
        Handle user interruption during system speech.

        Called by the state machine when speech is detected during playback.
        The actual playback stop is handled by the orchestrator.

        NANO-014 Session 5: Stores "barge_in" trigger for next pipeline call,
        enabling VoiceStateProvider to inject "User interrupted you" context.
        """
        # Store trigger for the upcoming pipeline.run() call
        self._pending_state_trigger = "barge_in"

        if self._on_barge_in_triggered is not None:
            self._on_barge_in_triggered()

    def _emit_state_change(self, from_state: str, to_state: str, trigger: str) -> None:
        """
        Emit a synthetic state change event.

        NANO-031: Used by text input to emit state changes without going through
        the AudioStateMachine.
        """
        if self._event_bus is not None:
            self._event_bus.emit(
                StateChangedEvent(
                    from_state=from_state,
                    to_state=to_state,
                    trigger=trigger,
                )
            )

    def process_text_input(
        self,
        text: str,
        skip_tts: bool = False,
        stimulus_source: Optional[str] = None,
        stimulus_metadata: Optional[dict] = None,
    ) -> None:
        """
        Process text input directly, bypassing STT.

        NANO-029/NANO-031: Text injection for E2E testing and user-facing text input.
        NANO-056: Also used by StimuliEngine for autonomous stimulus injection.
        Runs the LLM pipeline (and optionally TTS) with text as if it came from STT.

        Args:
            text: Transcription text to process (simulates STT output).
            skip_tts: If True, skip TTS synthesis (text-only response mode).
            stimulus_source: If set, identifies this as an autonomous stimulus
                (e.g., "patience", "custom"). Used for GUI display and logging.
            stimulus_metadata: Optional metadata from stimulus module (e.g.,
                Twitch chat content for prompt block injection). NANO-056b.
        """
        def process():
            # Suppress VAD during text input to prevent phantom triggers
            if self._on_pause_listening is not None:
                self._on_pause_listening()

            self._processing_start_time = time.time()
            current_state = "idle"  # Track state for synthetic transitions

            try:
                # Store as transcription (matches normal flow)
                transcription = text.strip()
                self._last_transcription = transcription

                # NANO-031/056: Emit state change to processing
                trigger = f"stimulus_{stimulus_source}" if stimulus_source else "text_input"
                self._emit_state_change(current_state, "processing", trigger)
                current_state = "processing"

                # Emit transcription event (for GUI synchronization)
                # NANO-073d: Tag origin so frontend can route user message rendering
                modality_origin = "stimulus" if stimulus_source else "text"
                if self._event_bus is not None:
                    self._event_bus.emit(
                        TranscriptionReadyEvent(
                            text=transcription,
                            duration=0.0,  # No audio duration for injected text
                            input_modality=modality_origin,
                        )
                    )

                # Check for empty transcription
                if not transcription:
                    self._total_empty_transcriptions += 1
                    if self._on_empty_transcription is not None:
                        self._on_empty_transcription()
                    # Return to idle on empty
                    self._emit_state_change(current_state, "idle", "empty_input")
                    return

                # Assemble context from all sources
                if self._context_manager is not None:
                    self._context_manager.assemble(transcription)

                # Capture and clear pending state trigger
                state_trigger = self._pending_state_trigger
                self._pending_state_trigger = None

                # Generate response via LLM pipeline
                # NANO-073d: Text input uses TEXT modality (was hardcoded VOICE)
                # NANO-075: Thread stimulus_source for JSONL metadata persistence
                # NANO-111: Use run_stream() for incremental dashboard display
                modality = InputModality.STIMULUS if stimulus_source else InputModality.TEXT

                from ..llm.provider_holder import ProviderHolder
                from ..core.events import LLMChunkEvent, LLMTokenEvent
                _inner = self._pipeline.provider.provider if isinstance(self._pipeline.provider, ProviderHolder) else self._pipeline.provider
                can_stream_text = _inner.get_properties().supports_streaming

                if can_stream_text:
                    # NANO-111: Token-level callback for real-time display
                    def _on_token_text(token_text: str, is_final: bool):
                        if self._event_bus is not None:
                            self._event_bus.emit(LLMTokenEvent(token=token_text, is_final=is_final))

                    full_display = []
                    full_tts = []
                    for chunk in self._pipeline.run_stream(
                        transcription,
                        self._persona,
                        generation_params=self._generation_params,
                        state_trigger=state_trigger,
                        input_modality=modality,
                        last_assistant_message=None,
                        stimulus_source=stimulus_source,
                        stimulus_metadata=stimulus_metadata,
                        on_token=_on_token_text,
                    ):
                        full_display.append(chunk.display_text)
                        full_tts.append(chunk.tts_text)
                        if self._event_bus is not None:
                            self._event_bus.emit(LLMChunkEvent(
                                text=chunk.display_text,
                                is_final=chunk.is_final,
                            ))
                            time.sleep(0.01)  # Yield for Socket.IO flush

                    response = " ".join(full_display)
                    tts_response = " ".join(full_tts)
                    # Build a minimal result-like object for downstream code
                    result = type("StreamResult", (), {
                        "content": response,
                        "tts_text": tts_response,
                        "activated_codex_entries": [],
                        "retrieved_memories": [],
                        "reasoning": None,
                        "usage": type("Usage", (), {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        })(),
                        "messages": [],
                        "input_modality": modality.value,
                        "state_trigger": state_trigger,
                        "block_contents": None,
                    })()
                else:
                    result = self._pipeline.run(
                        transcription,
                        self._persona,
                        generation_params=self._generation_params,
                        state_trigger=state_trigger,
                        input_modality=modality,
                        last_assistant_message=None,
                        stimulus_source=stimulus_source,
                        stimulus_metadata=stimulus_metadata,
                    )
                    response = result.content
                    tts_response = result.tts_text or response  # NANO-109

                self._last_response = response

                # Classify emotion for avatar + chat display (NANO-094)
                emotion, emotion_confidence = self._classify_emotion(response or "")

                # Emit response event (NANO-037: codex, NANO-042: reasoning, NANO-044: memories, NANO-056: stimulus, NANO-094: emotion, NANO-109: tts_text)
                if self._event_bus is not None:
                    self._event_bus.emit(
                        ResponseReadyEvent(
                            text=response or "",
                            user_input=transcription,
                            activated_codex_entries=result.activated_codex_entries,
                            retrieved_memories=result.retrieved_memories,
                            reasoning=result.reasoning,
                            stimulus_source=stimulus_source,
                            emotion=emotion,
                            emotion_confidence=emotion_confidence,
                            tts_text=result.tts_text,
                        )
                    )

                # Track token usage
                self._last_prompt_tokens = result.usage.prompt_tokens
                self._last_completion_tokens = result.usage.completion_tokens

                # Emit token usage event
                if self._event_bus is not None:
                    self._event_bus.emit(
                        TokenUsageEvent(
                            prompt_tokens=result.usage.prompt_tokens,
                            completion_tokens=result.usage.completion_tokens,
                            total_tokens=result.usage.total_tokens,
                            context_limit=self._get_context_limit(),
                        )
                    )

                # Emit prompt snapshot for GUI inspection
                if self._event_bus is not None:
                    token_breakdown = self._build_token_breakdown(
                        result.messages,
                        result.usage.prompt_tokens,
                        result.usage.completion_tokens,
                        block_contents=result.block_contents,
                    )
                    self._event_bus.emit(
                        PromptSnapshotEvent(
                            messages=result.messages,
                            token_breakdown=token_breakdown,
                            input_modality=result.input_modality,
                            state_trigger=result.state_trigger,
                        )
                    )

                    # NANO-076: Persist snapshot to sidecar JSONL
                    self._persist_snapshot(
                        result.messages, token_breakdown,
                        result.input_modality, result.state_trigger,
                        result.block_contents,
                    )

                # Log stimulus source for tracing (NANO-056)
                if stimulus_source:
                    logger.info(
                        "[Stimuli] Response generated for %s stimulus",
                        stimulus_source,
                    )

                # Check for empty response
                if not response or response.strip() == "":
                    # Return to idle on empty response
                    self._emit_state_change(current_state, "idle", "empty_response")
                    return

                # Synthesize speech via provider (unless skip_tts requested)
                if not skip_tts:
                    # NANO-031: Emit state change to system_speaking
                    self._emit_state_change(current_state, "system_speaking", "tts_start")
                    current_state = "system_speaking"

                    # NANO-054a: provider-agnostic TTS config, NANO-109: use TTS-cleaned text
                    tts_config = self._persona.get("tts_voice_config", {})
                    audio_result = self._tts_provider.synthesize(
                        tts_response,
                        voice=tts_config.get("voice"),
                        **{k: v for k, v in tts_config.items() if k != "voice"},
                    )
                    audio_response = np.frombuffer(audio_result.data, dtype=np.float32)

                    # Signal response is ready
                    self._total_turns += 1
                    if self._on_response_ready is not None:
                        self._on_response_ready(audio_response)

                    # NANO-031: Emit state change back to idle after TTS
                    self._emit_state_change(current_state, "idle", "tts_complete")
                else:
                    # Text-only mode: still count the turn, return to idle
                    self._total_turns += 1
                    self._emit_state_change(current_state, "idle", "response_complete")

            except Exception as e:
                self._total_errors += 1
                stage = self._determine_error_stage(e)
                print(f"[OrchestratorCallbacks] {stage} error: {e}")

                # Emit error event
                if self._event_bus is not None:
                    self._event_bus.emit(
                        PipelineErrorEvent(
                            stage=stage,
                            error_type=type(e).__name__,
                            message=str(e),
                        )
                    )

                if self._on_error is not None:
                    self._on_error(stage, e)

                # NANO-031: Return to idle on error
                self._emit_state_change(current_state, "idle", "error")

            finally:
                # Re-enable VAD after text input processing completes
                if self._on_resume_listening is not None:
                    self._on_resume_listening()

        # Run processing in background thread
        self._processing_thread = threading.Thread(target=process, daemon=True)
        self._processing_thread.start()

    def on_state_change(self, transition: StateTransition) -> None:
        """
        Log state transitions for debugging.

        Args:
            transition: State transition record.
        """
        print(f"[State] {transition.from_state.value} -> {transition.to_state.value} ({transition.trigger})")

    def update_persona(self, persona: dict) -> None:
        """
        Update the persona reference used by the pipeline.

        Called by VoiceAgentOrchestrator.reload_persona() to propagate
        hot-reloaded persona data to the callbacks/pipeline layer.

        Args:
            persona: New persona dict from CharacterLoader.

        NANO-036: Character hot-reload support.
        """
        self._persona = persona

    def update_generation_params(self, params: Optional[dict]) -> None:
        """
        Update runtime generation parameter overrides (NANO-053).

        Called by VoiceAgentOrchestrator.update_generation_params() to propagate
        dashboard slider changes to the pipeline layer.

        Args:
            params: Dict with temperature/max_tokens/top_p overrides, or None.
        """
        self._generation_params = params

    @property
    def last_transcription(self) -> Optional[str]:
        """Most recent transcription result."""
        return self._last_transcription

    @property
    def last_response(self) -> Optional[str]:
        """Most recent LLM response."""
        return self._last_response

    @property
    def processing_latency(self) -> Optional[float]:
        """Time since processing started (seconds), or None if not processing."""
        if self._processing_start_time is None:
            return None
        return time.time() - self._processing_start_time

    @property
    def is_processing(self) -> bool:
        """Whether a processing thread is currently active."""
        return (
            self._processing_thread is not None
            and self._processing_thread.is_alive()
        )

    @property
    def stats(self) -> dict:
        """Processing statistics."""
        return {
            "total_turns": self._total_turns,
            "total_errors": self._total_errors,
            "total_empty_transcriptions": self._total_empty_transcriptions,
        }

    @property
    def context_usage(self) -> dict:
        """
        Current context window usage.

        Returns:
            Dict with:
                - tokens: Current tokens in context (prompt_tokens from last turn)
                - max_context: Context window size from active provider (NANO-096)
                - usage_percent: Percentage of budget used
        """
        limit = self._get_context_limit()
        pct = (self._last_prompt_tokens / limit * 100) if limit > 0 else 0.0
        return {
            "tokens": self._last_prompt_tokens,
            "max_context": limit,
            "usage_percent": round(pct, 1),
        }

    def wait_for_processing(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for current processing to complete.

        Args:
            timeout: Maximum time to wait (seconds). None = wait forever.

        Returns:
            True if processing completed, False if timed out.
        """
        if self._processing_thread is None:
            return True

        self._processing_thread.join(timeout=timeout)
        return not self._processing_thread.is_alive()

    def _get_context_limit(self) -> int:
        """
        Get the context window size from the active LLM provider (NANO-096).

        Returns:
            Context window size in tokens.
        """
        return self._context_limit_getter()

    def _build_token_breakdown(
        self,
        messages: list[dict],
        prompt_tokens: int,
        completion_tokens: int,
        block_contents: list[dict] | None = None,
    ) -> dict:
        """
        Build token breakdown from messages for prompt inspection.

        When block_contents is provided (NANO-045b block mode), distributes
        system tokens proportionally across individual blocks and computes
        legacy section rollups. Otherwise falls back to header-parsing.

        Args:
            messages: Message list sent to LLM ([{role, content}, ...]).
            prompt_tokens: Actual prompt tokens from LLM response.
            completion_tokens: Actual completion tokens from LLM response.
            block_contents: Per-block content data from prompt assembly
                           (NANO-045b). None for legacy template mode.

        Returns:
            Token breakdown dict with structure:
            {
                "total": int,
                "prompt": int,
                "completion": int,
                "system": int,
                "user": int,
                "sections": {
                    "agent": int,
                    "context": int,
                    "rules": int,
                    "conversation": int,
                },
                "blocks": [  # Only present in block mode
                    {"id": str, "label": str, "section": str|None, "tokens": int},
                    ...
                ]
            }
        """
        breakdown = {
            "total": prompt_tokens + completion_tokens,
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "system": 0,
            "user": 0,
            "sections": {
                "agent": 0,
                "context": 0,
                "rules": 0,
                "conversation": 0,
            },
        }

        # Extract system and user content
        system_content = ""
        user_content = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content

        # Estimate system vs user token split based on character ratio
        total_chars = len(system_content) + len(user_content)
        if total_chars > 0 and prompt_tokens > 0:
            system_ratio = len(system_content) / total_chars
            breakdown["system"] = int(prompt_tokens * system_ratio)
            breakdown["user"] = prompt_tokens - breakdown["system"]

        # NANO-045b: Per-block token distribution
        if block_contents:
            self._distribute_tokens_by_blocks(breakdown, block_contents)
        else:
            # Legacy fallback: parse section headers from assembled prompt
            self._distribute_tokens_by_headers(breakdown, system_content)

        return breakdown

    def _distribute_tokens_by_blocks(
        self,
        breakdown: dict,
        block_contents: list[dict],
    ) -> None:
        """
        Distribute system tokens across blocks proportionally by char count.

        Also computes legacy section rollups by accumulating block tokens
        into their parent section (section header starts a new section;
        subsequent blocks without headers inherit the current section).

        Args:
            breakdown: Token breakdown dict to mutate (adds "blocks" key,
                      populates "sections").
            block_contents: Per-block data from prompt assembly.
        """
        system_tokens = breakdown["system"]
        total_block_chars = sum(b["chars"] for b in block_contents)

        # Map section header names to legacy section keys
        _SECTION_KEY_MAP = {
            "Agent": "agent",
            "Context": "context",
            "Rules": "rules",
            "Conversation": "conversation",
        }

        blocks_data = []
        current_section_key = None

        for block in block_contents:
            # Calculate proportional token count
            if total_block_chars > 0 and system_tokens > 0:
                ratio = block["chars"] / total_block_chars
                tokens = int(system_tokens * ratio)
            else:
                tokens = 0

            blocks_data.append({
                "id": block["id"],
                "label": block["label"],
                "section": block.get("section"),
                "tokens": tokens,
                "content": block.get("content", ""),
            })

            # Track current section for legacy rollup
            if block.get("section"):
                current_section_key = _SECTION_KEY_MAP.get(block["section"])

            # Accumulate into legacy section
            if current_section_key and current_section_key in breakdown["sections"]:
                breakdown["sections"][current_section_key] += tokens

        breakdown["blocks"] = blocks_data

    def _distribute_tokens_by_headers(
        self,
        breakdown: dict,
        system_content: str,
    ) -> None:
        """
        Legacy fallback: distribute system tokens by parsing ### headers.

        Used when block_config is not set (legacy template mode).

        Args:
            breakdown: Token breakdown dict to mutate (populates "sections").
            system_content: The assembled system prompt string.
        """
        section_map = {
            "### Agent": "agent",
            "### Context": "context",
            "### Rules": "rules",
            "### Conversation": "conversation",
        }

        if system_content and breakdown["system"] > 0:
            # Find section boundaries
            sections_found = []
            for header, key in section_map.items():
                pos = system_content.find(header)
                if pos != -1:
                    sections_found.append((pos, header, key))

            # Sort by position
            sections_found.sort(key=lambda x: x[0])

            # Calculate character counts per section
            section_chars = {}
            for i, (pos, header, key) in enumerate(sections_found):
                start = pos
                end = (
                    sections_found[i + 1][0]
                    if i + 1 < len(sections_found)
                    else len(system_content)
                )
                section_chars[key] = end - start

            # Distribute system tokens proportionally
            total_section_chars = sum(section_chars.values())
            if total_section_chars > 0:
                for key, chars in section_chars.items():
                    ratio = chars / total_section_chars
                    breakdown["sections"][key] = int(breakdown["system"] * ratio)

    def set_session_file_getter(self, getter: Callable) -> None:
        """
        Set the session file getter for snapshot sidecar persistence (NANO-076).

        Args:
            getter: Callable that returns the current session file Path or None.
        """
        self._session_file_getter = getter

    def set_emotion_classifier(self, classifier) -> None:
        """Set the emotion classifier for response mood detection (NANO-094)."""
        self._emotion_classifier = classifier

    def _classify_emotion(self, response_text: str):
        """Classify response text and emit avatar mood event. Returns (emotion, confidence)."""
        if not self._emotion_classifier:
            return None, None
        emotion, confidence = self._emotion_classifier.classify(response_text)
        if emotion and self._event_bus is not None:
            self._event_bus.emit(AvatarMoodEvent(mood=emotion, confidence=confidence or 0.0))
        # Persist emotion to JSONL so it survives hydration on reload
        if emotion is not None and self._session_file_getter:
            session_file = self._session_file_getter()
            if session_file:
                from ..history.jsonl_store import patch_last_turn
                try:
                    patch_last_turn(session_file, {
                        "emotion": emotion,
                        "emotion_confidence": confidence,
                    })
                except Exception as e:
                    logger.warning("Failed to persist emotion to JSONL: %s", e)
        return emotion, confidence

    def _persist_snapshot(
        self,
        messages: list[dict],
        token_breakdown: dict,
        input_modality: str,
        state_trigger: Optional[str],
        block_contents: Optional[list[dict]],
    ) -> None:
        """
        Persist a prompt snapshot to the sidecar JSONL (NANO-076).

        Called after live snapshot emission. Wrapped in try/except so a
        sidecar write failure never blocks the LLM response.
        """
        if not self._session_file_getter:
            return
        session_file = self._session_file_getter()
        if not session_file:
            return

        snapshot = {
            "turn_id": self._total_turns + 1,
            "messages": messages,
            "token_breakdown": token_breakdown,
            "block_contents": block_contents,
            "input_modality": input_modality,
            "state_trigger": state_trigger,
            "timestamp": time.time(),
            "estimated": False,
        }
        append_snapshot(session_file, snapshot)

    def build_prompt_snapshot(self) -> Optional[dict]:
        """
        Build a cold-reconstructed prompt snapshot via pipeline (NANO-076).

        Used when no sidecar exists (legacy sessions, new sessions).
        Returns None if pipeline or persona is not available.
        """
        if not self._pipeline or not self._persona:
            return None
        try:
            return self._pipeline.build_snapshot(self._persona)
        except Exception:
            logger.warning("Failed to build cold prompt snapshot", exc_info=True)
            return None
