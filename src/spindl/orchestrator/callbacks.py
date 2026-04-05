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
        stt_client: Optional[STTProvider],
        tts_provider: Optional[TTSProvider],
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

        # NANO-111 Phase 2.5: Delivery tracking for barge-in history truncation.
        # Populated by delivery threads in _parallel_tts_delivery / _synthesize_streaming.
        # on_barge_in() reads this to truncate _last_response and amend history.
        self._delivered_sentences: list[str] = []
        self._delivery_lock = threading.Lock()
        # Set post-construction by voice agent (same pattern as streaming callbacks)
        self._history_manager = None

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
                # NANO-112: Guard against disabled STT
                if self._stt is None:
                    logger.debug("STT disabled — ignoring speech input")
                    if self._on_empty_transcription is not None:
                        self._on_empty_transcription()
                    return

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
                # Tools require blocking _run_with_tools() — can't stream
                has_tools = self._pipeline._tool_executor is not None
                use_streaming = can_stream and not has_tools

                tts_config = self._persona.get("tts_voice_config", {})

                logger.debug(f"[NANO-111] can_stream={can_stream}, has_tools={has_tools}, streaming_cb={self._on_response_ready_streaming is not None}")
                # 4. Capture last assistant message for barge-in context
                last_msg = self._last_response if state_trigger == "barge_in" else None

                # NANO-110: Consume one-shot addressing-others prompt
                addressing_prompt = None
                if self._consume_addressing_others_prompt:
                    addressing_prompt = self._consume_addressing_others_prompt()

                if use_streaming and self._on_response_ready_streaming is not None and self._tts_provider is not None:
                    # NANO-111: Streaming voice path with parallel TTS.
                    # run_stream() yields sentence chunks for parallel TTS,
                    # then runs deferred post-processors on the accumulated
                    # response (Session 606 refactor).
                    audio_response = self._synthesize_streaming(
                        transcription, tts_config,
                        state_trigger=state_trigger,
                        last_assistant_message=last_msg,
                        addressing_others_prompt=addressing_prompt,
                    )
                    if audio_response is not None:
                        self._total_turns += 1
                else:

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

                    # NANO-112: Skip TTS synthesis when provider is disabled
                    if self._tts_provider is not None:
                        # NANO-111: Parallel TTS with streaming playback on the
                        # completed response. Split into sentences, fire TTS threads,
                        # delivery thread feeds playback in order.
                        if self._on_response_ready_streaming is not None:
                            audio_response, _ = self._parallel_tts_delivery(
                                tts_response, tts_config, display_text=response,
                            )
                            if audio_response is not None:
                                self._total_turns += 1
                        else:
                            # Fallback: single blocking TTS call (no streaming callbacks)
                            audio_result = self._tts_provider.synthesize(
                                tts_response,
                                voice=tts_config.get("voice"),
                                **{k: v for k, v in tts_config.items() if k != "voice"},
                            )
                            audio_response = np.frombuffer(audio_result.data, dtype=np.float32)
                            self._total_turns += 1
                    else:
                        # TTS disabled — text-only response
                        self._total_turns += 1

                    # Emit response event AFTER TTS so llm_chunk events
                    # from _parallel_tts_delivery land first.
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
                        # Only fire _on_response_ready for the fallback (non-streaming) path.
                        # When _parallel_tts_delivery handled playback, audio is already
                        # playing via play_streaming — calling play() here would kill the
                        # stream and replay from scratch (Session 607: first-chunk stutter).
                        if self._tts_provider is not None and self._on_response_ready_streaming is None and self._on_response_ready is not None:
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

    def _parallel_tts_delivery(
        self,
        tts_text: str,
        tts_config: dict,
        display_text: Optional[str] = None,
    ) -> tuple[Optional[np.ndarray], Optional[list]]:
        """
        Split completed response into sentences, synthesize TTS in parallel,
        deliver audio to streaming playback in order (NANO-111).

        Called after pipeline.run() completes — history, metadata, snapshots
        are all intact. This only handles the TTS synthesis and delivery.

        Args:
            tts_text: TTS-cleaned response text.
            tts_config: TTS voice config from persona.
            display_text: Original display text for sentence splitting + emotion.
                         Falls back to tts_text if not provided.

        Returns:
            Tuple of (audio array or None, chunks list or None).
            Chunks: [{text, emotion, emotion_confidence}, ...] for sub-bubble display.
        """
        import re
        from ..llm.sentence_segmenter import merge_punctuation_fragments

        # Split TTS text into sentences
        sentences = re.split(r'(?<=[.!?。！？])\s+', tts_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = merge_punctuation_fragments(sentences)

        # Split display text into sentences for sub-bubble display
        raw = display_text or tts_text
        display_sentences = re.split(r'(?<=[.!?。！？])\s+', raw.strip())
        display_sentences = [s.strip() for s in display_sentences if s.strip()]
        display_sentences = merge_punctuation_fragments(display_sentences)

        if not sentences:
            return None, None

        # Build sentence chunks for sub-bubble display (text only, no per-sentence emotion)
        sentence_chunks = []
        for ds in display_sentences:
            sentence_chunks.append({
                "text": ds,
            })

        # If only one sentence, just synthesize directly — no need for threading
        if len(sentences) == 1:
            audio_result = self._tts_provider.synthesize(
                sentences[0],
                voice=tts_config.get("voice"),
                **{k: v for k, v in tts_config.items() if k != "voice"},
            )
            audio = np.frombuffer(audio_result.data, dtype=np.float32)
            # Track delivered sentence (Phase 2.5)
            with self._delivery_lock:
                self._delivered_sentences = [display_sentences[0] if display_sentences else sentences[0]]
            self._on_response_ready_streaming(audio)
            self._finalize_playback_streaming()
            return audio, sentence_chunks

        audio_results: dict[int, np.ndarray] = {}
        lock = threading.Lock()
        tts_threads: list[threading.Thread] = []
        tts_semaphore = threading.Semaphore(3)  # NANO-111: cap concurrent Kokoro calls

        def synthesize_chunk(text: str, index: int):
            with tts_semaphore:
                try:
                    result = self._tts_provider.synthesize(
                        text,
                        voice=tts_config.get("voice"),
                        **{k: v for k, v in tts_config.items() if k != "voice"},
                    )
                    audio = np.frombuffer(result.data, dtype=np.float32)
                    with lock:
                        audio_results[index] = audio
                except Exception as e:
                    logger.warning(f"[NANO-111] TTS failed for chunk {index}: {e}")
                    with lock:
                        audio_results[index] = np.array([], dtype=np.float32)

        # Build closure factory for chunk start callbacks (same as _synthesize_streaming)
        def _make_chunk_start_cb(idx: int):
            """Emit sentence text when playback reaches this chunk."""
            def cb():
                # Phase 2.5: Track delivery at playback time, not queue time.
                # This fires when the audio device actually starts playing this chunk.
                if idx < len(display_sentences):
                    with self._delivery_lock:
                        self._delivered_sentences.append(display_sentences[idx])
                if idx < len(sentence_chunks) and self._event_bus is not None:
                    from ..core.events import LLMChunkEvent
                    sc = sentence_chunks[idx]
                    self._event_bus.emit(LLMChunkEvent(
                        text=sc["text"],
                        is_final=(idx >= len(sentence_chunks) - 1),
                    ))
            return cb

        # Delivery thread: feeds playback in order as chunks complete
        playback_started = False
        total = len(sentences)

        # NANO-111 Phase 2.5: Reset delivery tracking for this response
        with self._delivery_lock:
            self._delivered_sentences = []

        def delivery_loop():
            nonlocal playback_started
            next_idx = 0
            while next_idx < total:
                with lock:
                    ready = next_idx in audio_results
                if ready:
                    with lock:
                        chunk = audio_results[next_idx]
                    start_cb = _make_chunk_start_cb(next_idx)
                    if len(chunk) > 0:
                        if not playback_started:
                            start_cb()
                            self._on_response_ready_streaming(chunk)
                            playback_started = True
                        else:
                            self._append_playback_audio(
                                chunk,
                                on_chunk_start=start_cb,
                            )
                    next_idx += 1
                else:
                    time.sleep(0.02)
            self._finalize_playback_streaming()

        delivery_thread = threading.Thread(target=delivery_loop, daemon=True)
        delivery_thread.start()

        # Fire all TTS threads
        for i, sentence in enumerate(sentences):
            t = threading.Thread(target=synthesize_chunk, args=(sentence, i), daemon=True)
            t.start()
            tts_threads.append(t)

        # Wait for all TTS + delivery to complete
        for t in tts_threads:
            t.join()
        delivery_thread.join()

        # Return concatenated audio + chunks
        ordered = [audio_results[i] for i in range(total) if len(audio_results.get(i, [])) > 0]
        audio = np.concatenate(ordered) if ordered else None
        return audio, sentence_chunks

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
        state_trigger: Optional[str] = None,
        last_assistant_message: Optional[str] = None,
        addressing_others_prompt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Stream LLM response, synthesize TTS per-sentence in parallel,
        start playback on first chunk and append subsequent chunks (NANO-111 Phase 2).

        Uses pipeline.run_stream() to get sentence-level chunks, fires TTS
        synthesis for each sentence in a separate thread, delivers audio to
        the playback device in sentence order as chunks complete.

        Per-sentence emotion classification drives avatar expression transitions.
        Post-processors run after stream ends (Session 606 refactor).

        Args:
            transcription: User's transcribed speech text.
            tts_config: TTS voice configuration from persona.
            state_trigger: Pending state trigger (e.g., "barge_in").
            last_assistant_message: Previous response for barge-in context.
            addressing_others_prompt: One-shot addressing-others prompt (NANO-110).

        Returns:
            The full concatenated audio array, or None if empty.
        """
        from ..llm.pipeline import StreamingPipelineChunk
        from ..core.events import LLMChunkEvent

        audio_results: dict[int, np.ndarray] = {}  # index → audio
        sentence_texts: dict[int, str] = {}  # index → display text for delivery-synced rendering
        # sentence_emotions removed — emotion is classified once on full response
        tts_threads: list[threading.Thread] = []
        lock = threading.Lock()
        tts_semaphore = threading.Semaphore(3)  # NANO-111: cap concurrent Kokoro calls
        total_chunks_dispatched = 0
        all_chunks_dispatched = threading.Event()  # set when LLM stream ends

        def synthesize_chunk(tts_text: str, index: int):
            """Synthesize one sentence in a thread."""
            with tts_semaphore:
                try:
                    result = self._tts_provider.synthesize(
                        tts_text,
                        voice=tts_config.get("voice"),
                        **{k: v for k, v in tts_config.items() if k != "voice"},
                    )
                    audio = np.frombuffer(result.data, dtype=np.float32)
                    with lock:
                        audio_results[index] = audio
                except Exception as e:
                    logger.warning(f"[NANO-111] TTS failed for chunk {index}: {e}")
                    with lock:
                        audio_results[index] = np.array([], dtype=np.float32)

        # Separate delivery thread: continuously delivers TTS audio to playback
        # in order, independent of the LLM generation loop.
        playback_started = False

        def _make_chunk_start_cb(
            display_text: str,
            is_final: bool,
        ):
            """Create a closure that emits sentence text when playback reaches this chunk."""
            def cb():
                # Phase 2.5: Track delivery at playback time, not queue time.
                if display_text:
                    with self._delivery_lock:
                        self._delivered_sentences.append(display_text)
                if display_text and self._event_bus is not None:
                    from ..core.events import LLMChunkEvent
                    self._event_bus.emit(LLMChunkEvent(
                        text=display_text,
                        is_final=is_final,
                    ))
            return cb

        # NANO-111 Phase 2.5: Reset delivery tracking for this response
        with self._delivery_lock:
            self._delivered_sentences = []

        def delivery_loop():
            nonlocal playback_started
            next_to_deliver = 0
            while True:
                # Check if the next chunk is ready
                with lock:
                    ready = next_to_deliver in audio_results
                    total = total_chunks_dispatched
                    done = all_chunks_dispatched.is_set()

                if ready:
                    with lock:
                        audio_chunk = audio_results[next_to_deliver]
                        display_text = sentence_texts.get(next_to_deliver, "")

                    # NANO-111 Session 606: Text renders when playback
                    # reaches this chunk — [text][tts][text][tts] via playback callbacks.
                    is_final_chunk = done and next_to_deliver >= total - 1
                    start_cb = _make_chunk_start_cb(
                        display_text, is_final_chunk,
                    )

                    if len(audio_chunk) > 0:
                        if not playback_started:
                            # First chunk: fire text callback immediately from
                            # delivery thread — don't wait for monitor's 10ms poll.
                            # The monitor would check pos >= 0 which is always true,
                            # but by then response event may have already arrived.
                            start_cb()
                            self._on_response_ready_streaming(audio_chunk)
                            playback_started = True
                        else:
                            self._append_playback_audio(
                                audio_chunk,
                                on_chunk_start=start_cb,
                            )
                    next_to_deliver += 1
                elif done and next_to_deliver >= total:
                    # All chunks dispatched and delivered
                    break
                else:
                    # Not ready yet — poll briefly
                    time.sleep(0.02)

            self._finalize_playback_streaming()

        delivery_thread = threading.Thread(target=delivery_loop, daemon=True)
        delivery_thread.start()

        full_display_text = []
        full_tts_text = []

        # NANO-111 Session 606: No per-token callback on voice path.
        # Text renders in sync with TTS delivery, not LLM generation.
        # The delivery thread emits LLMChunkEvent per sentence right
        # before audio plays — [text][tts][text][tts] ordering.

        for chunk in self._pipeline.run_stream(
            transcription,
            self._persona,
            generation_params=self._generation_params,
            state_trigger=state_trigger,
            input_modality=InputModality.VOICE,
            last_assistant_message=last_assistant_message,
            addressing_others_prompt=addressing_others_prompt,
        ):
            full_display_text.append(chunk.display_text)
            full_tts_text.append(chunk.tts_text)

            # Fire TTS in parallel thread if there's text to speak
            if chunk.tts_text.strip():
                # Store display text for delivery-synced rendering
                with lock:
                    sentence_texts[chunk.index] = chunk.display_text
                t = threading.Thread(
                    target=synthesize_chunk,
                    args=(chunk.tts_text, chunk.index),
                    daemon=True,
                )
                t.start()
                tts_threads.append(t)
                with lock:
                    total_chunks_dispatched += 1

        # LLM stream complete — signal delivery thread
        for t in tts_threads:
            t.join()
        all_chunks_dispatched.set()

        # Wait for delivery to finish
        delivery_thread.join()

        # --- Deferred post-processor results (NANO-111 Session 606) ---
        # run_stream() now runs post-processors after the stream ends and
        # stores the result on _last_stream_result. Use it for full metadata.
        result = self._pipeline._last_stream_result

        if result is not None:
            response = result.content
            self._last_response = response

            # Classify emotion on full response for final event
            emotion, emotion_confidence = self._classify_emotion(response or "")

            # Build per-sentence chunks list for sub-bubble display (text only)
            response_chunks = None
            with lock:
                if sentence_texts:
                    response_chunks = []
                    for idx in sorted(sentence_texts.keys()):
                        response_chunks.append({
                            "text": sentence_texts[idx],
                        })

            # Emit response event with full metadata + chunks
            print(f"[NANO-111] response_chunks={response_chunks}", flush=True)
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
                        chunks=response_chunks,
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
        else:
            # Fallback: no stream result (shouldn't happen, but defensive)
            response = " ".join(full_display_text)
            self._last_response = response

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

        NANO-111 Phase 2.5: Truncates _last_response and amends history to
        reflect only the sentences that were actually delivered before barge-in.
        """
        # Store trigger for the upcoming pipeline.run() call
        self._pending_state_trigger = "barge_in"

        # Phase 2.5: Truncate to delivered sentences
        with self._delivery_lock:
            delivered = list(self._delivered_sentences)

        if delivered and self._last_response:
            truncated = " ".join(delivered)
            # Only truncate if we actually delivered less than the full response
            if len(truncated) < len(self._last_response):
                logger.info(
                    "[Phase 2.5] Barge-in truncation: %d/%d chars delivered (%d sentences)",
                    len(truncated), len(self._last_response), len(delivered),
                )
                self._last_response = truncated

                # Amend the history entry to reflect what was actually spoken
                if self._history_manager is not None:
                    self._history_manager.amend_last_assistant_content(truncated)

                # Emit truncation event for frontend bubble update
                if self._event_bus is not None:
                    from ..core.events import BargeInTruncatedEvent
                    self._event_bus.emit(BargeInTruncatedEvent(
                        truncated_text=truncated,
                        delivered_sentences=len(delivered),
                    ))

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

                # Text input uses blocking pipeline for full metadata
                # (token usage, codex, memories, prompt snapshot).
                # Voice input uses run_stream() for latency; text doesn't need it.
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

                # NANO-111: Chunks will be populated by _parallel_tts_delivery if TTS runs
                _text_input_chunks = None

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
                # NANO-112: Also skip TTS when provider is disabled
                if not skip_tts and self._tts_provider is not None:
                    # NANO-031: Emit state change to system_speaking
                    self._emit_state_change(current_state, "system_speaking", "tts_start")
                    current_state = "system_speaking"

                    # NANO-054a: provider-agnostic TTS config, NANO-109: use TTS-cleaned text
                    tts_config = self._persona.get("tts_voice_config", {})

                    # NANO-111: Parallel TTS delivery for text input path
                    if self._on_response_ready_streaming is not None:
                        audio_response, _text_input_chunks = self._parallel_tts_delivery(
                            tts_response, tts_config, display_text=response,
                        )
                        if audio_response is not None:
                            self._total_turns += 1
                    else:
                        # Fallback: single blocking TTS call (no streaming callbacks)
                        audio_result = self._tts_provider.synthesize(
                            tts_response,
                            voice=tts_config.get("voice"),
                            **{k: v for k, v in tts_config.items() if k != "voice"},
                        )
                        audio_response = np.frombuffer(audio_result.data, dtype=np.float32)
                        self._total_turns += 1
                        if self._on_response_ready is not None:
                            self._on_response_ready(audio_response)

                    # NANO-031: Emit state change back to idle after TTS
                    self._emit_state_change(current_state, "idle", "tts_complete")
                else:
                    # Text-only mode: still count the turn, return to idle
                    self._total_turns += 1
                    self._emit_state_change(current_state, "idle", "response_complete")

                # NANO-111 Session 606: Emit response event AFTER TTS.
                # When streaming callbacks are wired, llm_chunk events already
                # built sub-bubbles incrementally — don't pass chunks here
                # (would re-render all at once). Pass chunks only for fallback
                # path where no llm_chunk events fired.
                fallback_chunks = _text_input_chunks if self._on_response_ready_streaming is None else None
                print(f"[NANO-111] text_input_chunks={_text_input_chunks}", flush=True)
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
                            chunks=fallback_chunks,
                        )
                    )

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

    def _classify_emotion(self, response_text: str, *, emit_avatar_mood: bool = True):
        """Classify response text and optionally emit avatar mood event. Returns (emotion, confidence)."""
        if not self._emotion_classifier:
            return None, None
        emotion, confidence = self._emotion_classifier.classify(response_text)
        if emit_avatar_mood and emotion and self._event_bus is not None:
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
