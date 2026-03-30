"""Agent state machine for voice agent orchestration."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Callable, Optional

import numpy as np

from ..audio.vad import SpeechEvent, SpeechState, VADTracker

logger = logging.getLogger(__name__)

# Pre-roll buffer constants
# VAD requires min_speech_ms of consecutive speech frames before triggering.
# Without pre-roll, we lose the first ~250ms of speech (word clipping).
# The pre-roll buffer captures audio BEFORE VAD triggers, so we can include
# the beginning of the utterance.
DEFAULT_PRE_ROLL_MS = 300
CHUNK_SIZE = 512
SAMPLE_RATE = 16000

# Audio input timeout: if no audio chunk arrives for this long while in
# USER_SPEAKING, force transition to LISTENING. Prevents state machine from
# getting stuck when audio stream dies mid-utterance. (Pipecat pattern)
AUDIO_INPUT_TIMEOUT = 0.5  # seconds


class AgentState(Enum):
    """
    High-level agent states.

    IDLE: Microphone off, waiting for activation.
    LISTENING: Mic on, monitoring VAD, waiting for user to speak.
    USER_SPEAKING: User is talking, accumulating audio.
    PROCESSING: User finished, audio sent to STT/LLM pipeline.
    SYSTEM_SPEAKING: Playing TTS response, monitoring for barge-in.
    """

    IDLE = "idle"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    PROCESSING = "processing"
    SYSTEM_SPEAKING = "system_speaking"


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: AgentState
    to_state: AgentState
    timestamp: float
    trigger: str  # What caused the transition


@dataclass
class AgentCallbacks:
    """
    Callbacks for agent state machine events.

    All callbacks are optional. They run synchronously in the audio thread,
    so keep them fast or dispatch to another thread.
    """

    on_state_change: Optional[Callable[[StateTransition], None]] = None
    on_user_speech_start: Optional[Callable[[], None]] = None
    on_user_speech_end: Optional[Callable[[np.ndarray, float], None]] = None  # (audio, duration)
    on_barge_in: Optional[Callable[[], None]] = None
    on_processing_complete: Optional[Callable[[], None]] = None
    on_system_speech_end: Optional[Callable[[], None]] = None


class AudioStateMachine:
    """
    State machine that orchestrates the voice agent flow.

    Integrates with VADTracker to handle:
    - Detecting when user starts/stops speaking
    - Accumulating audio during user speech
    - Detecting barge-in during system speech
    - Managing state transitions with callbacks

    Thread-safe: state transitions are protected by a lock.
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 300,
        pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
        callbacks: Optional[AgentCallbacks] = None,
    ):
        """
        Initialize the state machine.

        Args:
            vad_threshold: Speech probability threshold (0.0-1.0).
            min_speech_ms: Minimum speech duration to trigger (ms).
            min_silence_ms: Minimum silence to end speech (ms).
            speech_pad_ms: Padding added to speech segments (ms).
            pre_roll_ms: Audio to capture before VAD triggers (ms). Prevents word clipping.
            callbacks: Optional callbacks for state events.
        """
        self._lock = Lock()
        self._state = AgentState.IDLE
        self._callbacks = callbacks or AgentCallbacks()
        self._state_entered_at = time.time()

        # Pre-roll buffer: circular buffer capturing audio BEFORE speech_start
        # This prevents word clipping when VAD takes time to trigger
        # Uses speech_pad_ms from config (exposed via Settings slider)
        effective_pre_roll = speech_pad_ms if speech_pad_ms > 0 else pre_roll_ms
        pre_roll_chunks = max(1, int((effective_pre_roll / 1000) * SAMPLE_RATE / CHUNK_SIZE))
        self._pre_roll_buffer: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)

        # Audio buffer for user speech
        self._audio_buffer: list[np.ndarray] = []
        self._speech_start_time: Optional[float] = None

        # Transition history (for debugging)
        self._transitions: list[StateTransition] = []
        self._max_history = 100

        # Audio input timeout tracking (NANO-108 Layer 2)
        self._last_audio_time: Optional[float] = None
        self._audio_received = False  # First-frame guard

        # Initialize VAD tracker with our callbacks
        self._vad = VADTracker(
            threshold=vad_threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            on_speech_start=self._handle_speech_start,
            on_speech_end=self._handle_speech_end,
        )

    def update_vad_params(self, **kwargs: object) -> None:
        """Update VAD parameters at runtime. Delegates to VADTracker.update_params()."""
        # Resize pre-roll buffer if speech_pad_ms changed
        speech_pad = kwargs.get("speech_pad_ms")
        if speech_pad is not None and isinstance(speech_pad, (int, float)):
            pad_ms = int(speech_pad)
            if pad_ms > 0:
                new_maxlen = max(1, int((pad_ms / 1000) * SAMPLE_RATE / CHUNK_SIZE))
                old_data = list(self._pre_roll_buffer)
                self._pre_roll_buffer = deque(old_data[-new_maxlen:], maxlen=new_maxlen)

        self._vad.update_params(**kwargs)  # type: ignore[arg-type]

    def _transition(self, new_state: AgentState, trigger: str) -> None:
        """
        Perform a state transition.

        Args:
            new_state: Target state.
            trigger: What caused this transition.
        """
        old_state = self._state
        if old_state == new_state:
            return  # No-op

        now = time.time()

        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=now,
            trigger=trigger,
        )

        self._state = new_state
        self._state_entered_at = now

        # Maintain history
        self._transitions.append(transition)
        if len(self._transitions) > self._max_history:
            self._transitions.pop(0)

        # Callback
        if self._callbacks.on_state_change is not None:
            self._callbacks.on_state_change(transition)

    def _handle_speech_start(self, event: SpeechEvent) -> None:
        """Handle VAD speech_start event."""
        with self._lock:
            if self._state == AgentState.LISTENING:
                # User started speaking - transition to USER_SPEAKING
                # Seed audio buffer with pre-roll (audio from BEFORE VAD triggered)
                # This prevents word clipping at the start of utterances
                self._audio_buffer = list(self._pre_roll_buffer)
                self._speech_start_time = event.timestamp
                self._transition(AgentState.USER_SPEAKING, "vad_speech_start")

                if self._callbacks.on_user_speech_start is not None:
                    self._callbacks.on_user_speech_start()

            elif self._state == AgentState.SYSTEM_SPEAKING:
                # User interrupted system - BARGE-IN!
                # Transition to USER_SPEAKING (not LISTENING) so we capture the
                # user's speech that triggered the barge-in. Seed audio buffer
                # with pre-roll to capture the beginning of their utterance.
                self._audio_buffer = list(self._pre_roll_buffer)
                self._speech_start_time = event.timestamp
                self._transition(AgentState.USER_SPEAKING, "barge_in")

                if self._callbacks.on_barge_in is not None:
                    self._callbacks.on_barge_in()

                if self._callbacks.on_user_speech_start is not None:
                    self._callbacks.on_user_speech_start()

    def _handle_speech_end(self, event: SpeechEvent) -> None:
        """Handle VAD speech_end event."""
        with self._lock:
            if self._state == AgentState.USER_SPEAKING:
                # User finished speaking - transition to PROCESSING
                self._transition(AgentState.PROCESSING, "vad_speech_end")

                # Concatenate buffered audio
                if self._audio_buffer:
                    audio = np.concatenate(self._audio_buffer)
                else:
                    audio = np.array([], dtype=np.float32)

                duration = event.duration or 0.0

                # Clear buffer
                self._audio_buffer = []
                self._speech_start_time = None

                if self._callbacks.on_user_speech_end is not None:
                    self._callbacks.on_user_speech_end(audio, duration)

    def process_audio(self, chunk: np.ndarray) -> tuple[AgentState, float]:
        """
        Process an audio chunk through VAD and update state.

        This should be called for every audio chunk from the microphone.

        Args:
            chunk: Audio chunk as float32 numpy array (512 samples at 16kHz).

        Returns:
            Tuple of (current state, speech probability).
        """
        self._last_audio_time = time.monotonic()
        self._audio_received = True

        with self._lock:
            # Only process audio when listening or user speaking
            # Also process during system speaking (for barge-in detection)
            if self._state in (
                AgentState.LISTENING,
                AgentState.USER_SPEAKING,
                AgentState.SYSTEM_SPEAKING,
            ):
                # Fill pre-roll buffer during LISTENING and SYSTEM_SPEAKING
                # LISTENING: captures audio before normal speech start
                # SYSTEM_SPEAKING: captures audio before barge-in so the
                # first word of an interruption isn't clipped
                if self._state in (AgentState.LISTENING, AgentState.SYSTEM_SPEAKING):
                    self._pre_roll_buffer.append(chunk.copy())

                # Buffer audio if user is speaking
                if self._state == AgentState.USER_SPEAKING:
                    self._audio_buffer.append(chunk.copy())

        # Process VAD outside the lock to avoid deadlock
        # (VAD callbacks will acquire the lock themselves)
        vad_state, prob = self._vad.process_chunk(chunk)

        return self._state, prob

    def check_audio_timeout(self) -> bool:
        """
        Check if audio input has timed out while in USER_SPEAKING.

        Should be called periodically (e.g., from the mic level monitor thread).
        Forces transition to LISTENING if no audio chunk has arrived for
        AUDIO_INPUT_TIMEOUT seconds while the user was speaking.

        Returns:
            True if a timeout was detected and state was forced to LISTENING.
        """
        if not self._audio_received:
            return False  # First-frame guard: no timeout before first chunk

        if self._last_audio_time is None:
            return False

        with self._lock:
            if self._state != AgentState.USER_SPEAKING:
                return False

            gap = time.monotonic() - self._last_audio_time
            if gap <= AUDIO_INPUT_TIMEOUT:
                return False

            logger.warning(
                "Audio input timeout: no audio for %.2fs while in USER_SPEAKING, "
                "forcing transition to LISTENING",
                gap,
            )
            self._audio_buffer = []
            self._pre_roll_buffer.clear()
            self._speech_start_time = None
            self._transition(AgentState.LISTENING, "audio_timeout")

        # Reset VAD outside the lock (same pattern as process_audio)
        self._vad.reset()
        return True

    def activate(self) -> None:
        """
        Activate the agent (transition from IDLE to LISTENING).

        Call this when user presses a button, says a wake word, etc.
        """
        with self._lock:
            if self._state == AgentState.IDLE:
                self._vad.reset()
                self._transition(AgentState.LISTENING, "activation")

    def deactivate(self) -> None:
        """
        Deactivate the agent (transition to IDLE from any state).

        Call this to put the agent to sleep.
        """
        with self._lock:
            if self._state != AgentState.IDLE:
                self._audio_buffer = []
                self._pre_roll_buffer.clear()
                self._speech_start_time = None
                self._transition(AgentState.IDLE, "deactivation")

    def start_system_speaking(self) -> None:
        """
        Signal that system is about to speak (play TTS).

        Call this when TTS playback begins.
        """
        with self._lock:
            if self._state == AgentState.PROCESSING:
                self._transition(AgentState.SYSTEM_SPEAKING, "tts_start")

    def finish_system_speaking(self) -> None:
        """
        Signal that system finished speaking (TTS complete).

        Call this when TTS playback ends normally (not interrupted).
        """
        with self._lock:
            if self._state == AgentState.SYSTEM_SPEAKING:
                self._transition(AgentState.LISTENING, "tts_complete")

                if self._callbacks.on_system_speech_end is not None:
                    self._callbacks.on_system_speech_end()

    def finish_processing(self) -> None:
        """
        Signal that processing is complete (ready for TTS or back to listening).

        Call this when LLM response is ready.
        """
        with self._lock:
            if self._state == AgentState.PROCESSING:
                if self._callbacks.on_processing_complete is not None:
                    self._callbacks.on_processing_complete()

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        with self._lock:
            return self._state

    @property
    def state_duration(self) -> float:
        """Time spent in current state (seconds)."""
        with self._lock:
            return time.time() - self._state_entered_at

    @property
    def is_active(self) -> bool:
        """Whether agent is active (not IDLE)."""
        with self._lock:
            return self._state != AgentState.IDLE

    @property
    def is_listening(self) -> bool:
        """Whether agent is listening for user speech."""
        with self._lock:
            return self._state in (AgentState.LISTENING, AgentState.SYSTEM_SPEAKING)

    @property
    def transitions(self) -> list[StateTransition]:
        """Recent state transitions (for debugging)."""
        with self._lock:
            return self._transitions.copy()

    @property
    def vad_speech_ratio(self) -> float:
        """Ratio of frames detected as speech by VAD."""
        return self._vad.speech_ratio

    def reset(self) -> None:
        """Reset state machine to initial state."""
        with self._lock:
            self._state = AgentState.IDLE
            self._state_entered_at = time.time()
            self._audio_buffer = []
            self._pre_roll_buffer.clear()
            self._speech_start_time = None
            self._transitions = []
            self._last_audio_time = None
            self._audio_received = False
            self._vad.reset()
