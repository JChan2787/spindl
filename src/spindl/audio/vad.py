"""Voice Activity Detection using Silero VAD."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Periodic model state reset to prevent RNN hidden state drift (Pipecat pattern)
MODEL_RESET_INTERVAL = 5.0  # seconds


class SpeechState(Enum):
    """Current state of speech detection."""

    SILENCE = "silence"
    SPEAKING = "speaking"


@dataclass
class SpeechEvent:
    """Event emitted when speech state changes."""

    event_type: str  # "speech_start" or "speech_end"
    timestamp: float  # time.time() when event occurred
    duration: Optional[float] = None  # For speech_end: duration of speech segment


class SileroVAD:
    """
    Wrapper around Silero VAD model.

    Silero expects 512 samples at 16kHz (32ms) - matches our AudioCapture chunk size.
    Returns probability that chunk contains speech.
    """

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512  # Silero's expected input size

    def __init__(self):
        """Load Silero VAD model."""
        # Load model from torch hub (cached after first download)
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model.eval()

        # Reset model state
        self._model.reset_states()
        self._last_reset_time = time.monotonic()

    def process_chunk(self, audio: np.ndarray) -> float:
        """
        Process a single audio chunk and return speech probability.

        Args:
            audio: Audio chunk as float32 numpy array (512 samples at 16kHz).

        Returns:
            Speech probability between 0.0 and 1.0.
        """
        try:
            # Periodic state reset to prevent RNN hidden state drift
            now = time.monotonic()
            if (now - self._last_reset_time) >= MODEL_RESET_INTERVAL:
                self._model.reset_states()
                self._last_reset_time = now

            # Convert to torch tensor
            tensor = torch.from_numpy(audio).float()

            # Run inference
            with torch.no_grad():
                prob = self._model(tensor, self.SAMPLE_RATE).item()

            return prob
        except Exception as e:
            logger.error("SileroVAD inference error: %s", e)
            return 0.0

    def reset(self) -> None:
        """Reset model state (call between sessions)."""
        self._model.reset_states()
        self._last_reset_time = time.monotonic()


class VADTracker:
    """
    Tracks speech state with debouncing and hangover.

    Debouncing: Requires N consecutive frames above threshold to trigger speech_start.
    Hangover: Waits N consecutive frames below threshold before triggering speech_end.

    This prevents choppy detection from brief pauses or noise spikes.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        speech_pad_ms: int = 300,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        on_speech_start: Optional[Callable[[SpeechEvent], None]] = None,
        on_speech_end: Optional[Callable[[SpeechEvent], None]] = None,
    ):
        """
        Initialize VAD tracker.

        Args:
            threshold: Speech probability threshold (0.0-1.0). Default 0.5.
            speech_pad_ms: Padding added to speech segments (ms). Default 300ms.
            min_speech_ms: Minimum speech duration to trigger start event (ms). Default 250ms.
            min_silence_ms: Minimum silence duration to trigger end event (ms). Default 500ms.
            on_speech_start: Callback for speech start events.
            on_speech_end: Callback for speech end events.
        """
        self.threshold = threshold
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        # Initialize Silero VAD
        self._vad = SileroVAD()

        # State tracking
        self._state = SpeechState.SILENCE
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None

        # Counters for debouncing (in frames)
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0

        # Frame duration at 16kHz with 512-sample chunks
        self._frame_duration_ms = (512 / 16000) * 1000  # 32ms

        # Convert ms thresholds to frame counts
        self._min_speech_frames = max(1, int(min_speech_ms / self._frame_duration_ms))
        self._min_silence_frames = max(1, int(min_silence_ms / self._frame_duration_ms))

        # Statistics
        self._total_frames = 0
        self._speech_frames = 0

    def update_params(
        self,
        threshold: Optional[float] = None,
        min_speech_ms: Optional[int] = None,
        min_silence_ms: Optional[int] = None,
        speech_pad_ms: Optional[int] = None,
    ) -> None:
        """Update VAD parameters at runtime without resetting state."""
        if threshold is not None:
            self.threshold = threshold
        if speech_pad_ms is not None:
            self.speech_pad_ms = speech_pad_ms
        if min_speech_ms is not None:
            self.min_speech_ms = min_speech_ms
            self._min_speech_frames = max(1, int(min_speech_ms / self._frame_duration_ms))
        if min_silence_ms is not None:
            self.min_silence_ms = min_silence_ms
            self._min_silence_frames = max(1, int(min_silence_ms / self._frame_duration_ms))

    def process_chunk(self, audio: np.ndarray) -> tuple[SpeechState, float]:
        """
        Process an audio chunk and update speech state.

        Args:
            audio: Audio chunk as float32 numpy array (512 samples).

        Returns:
            Tuple of (current state, speech probability).
        """
        # Get speech probability from Silero
        prob = self._vad.process_chunk(audio)
        self._total_frames += 1

        is_speech = prob >= self.threshold
        current_time = time.time()

        if is_speech:
            self._speech_frames += 1
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0

            # Check for speech start
            if self._state == SpeechState.SILENCE:
                if self._consecutive_speech_frames >= self._min_speech_frames:
                    self._state = SpeechState.SPEAKING
                    self._speech_start_time = current_time
                    self._silence_start_time = None

                    # Emit event
                    if self.on_speech_start is not None:
                        event = SpeechEvent(
                            event_type="speech_start",
                            timestamp=current_time,
                        )
                        self.on_speech_start(event)

        else:
            self._consecutive_silence_frames += 1
            self._consecutive_speech_frames = 0

            # Check for speech end (hangover)
            if self._state == SpeechState.SPEAKING:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time

                if self._consecutive_silence_frames >= self._min_silence_frames:
                    self._state = SpeechState.SILENCE
                    speech_duration = None

                    if self._speech_start_time is not None:
                        speech_duration = current_time - self._speech_start_time

                    # Emit event
                    if self.on_speech_end is not None:
                        event = SpeechEvent(
                            event_type="speech_end",
                            timestamp=current_time,
                            duration=speech_duration,
                        )
                        self.on_speech_end(event)

                    self._speech_start_time = None
                    self._silence_start_time = None

        return self._state, prob

    @property
    def state(self) -> SpeechState:
        """Current speech state."""
        return self._state

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently detected."""
        return self._state == SpeechState.SPEAKING

    @property
    def speech_duration(self) -> Optional[float]:
        """Duration of current speech segment (if speaking), or None."""
        if self._state == SpeechState.SPEAKING and self._speech_start_time is not None:
            return time.time() - self._speech_start_time
        return None

    @property
    def total_frames(self) -> int:
        """Total frames processed."""
        return self._total_frames

    @property
    def speech_ratio(self) -> float:
        """Ratio of frames detected as speech."""
        if self._total_frames == 0:
            return 0.0
        return self._speech_frames / self._total_frames

    def reset(self) -> None:
        """Reset tracker state."""
        self._vad.reset()
        self._state = SpeechState.SILENCE
        self._speech_start_time = None
        self._silence_start_time = None
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._total_frames = 0
        self._speech_frames = 0
