"""Audio playback with interrupt support for voice agent output."""

import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioPlayback:
    """
    Interruptible audio playback for TTS output.

    Plays audio through the system output device with the ability to stop
    mid-playback (for barge-in support). Uses callback-based streaming to
    allow real-time interruption.

    Sample rate is configurable via configure() to support different TTS providers:
    - Kokoro TTS: 24000 Hz
    - Qwen3 TTS: 12000 Hz
    - Default: 16000 Hz (backward compatible)
    """

    # Default values for backward compatibility
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DTYPE = "float32"

    # NANO-069: Audio level emission interval (seconds)
    AUDIO_LEVEL_INTERVAL = 0.05  # 50ms

    def __init__(
        self,
        device: Optional[int] = None,
        on_complete: Optional[Callable[[], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
    ):
        """
        Initialize audio playback.

        Args:
            device: Audio output device ID (None = default).
            on_complete: Callback when playback finishes normally.
            on_interrupt: Callback when playback is interrupted.
            on_audio_level: Callback with RMS level (0.0-1.0) during playback.
        """
        self.device = device
        self.on_complete = on_complete
        self.on_interrupt = on_interrupt
        self.on_audio_level = on_audio_level

        # Dynamic sample rate configuration (set via configure())
        self._sample_rate: Optional[int] = None
        self._channels: Optional[int] = None

        self._stream: Optional[sd.OutputStream] = None
        self._audio_data: Optional[np.ndarray] = None
        self._playback_position = 0
        self._playing = False
        self._interrupted = False
        self._finished = False  # Set when audio exhausted (distinct from _playing)
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._streaming_finalized: bool = True  # NANO-111: True = not streaming or done

        # NANO-069: Audio level tracking (written in audio thread, read in monitor)
        self._current_rms: float = 0.0

    def configure(self, sample_rate: int, channels: int = 1) -> None:
        """
        Configure playback for provider's output format.

        Call this after getting TTSProvider.get_properties() to match
        the provider's native sample rate. If not called, defaults to
        16kHz mono for backward compatibility.

        Args:
            sample_rate: Output sample rate (e.g., 24000 for Kokoro, 12000 for Qwen3)
            channels: Number of channels (1 = mono, 2 = stereo)

        Raises:
            ValueError: If sample_rate <= 0 or channels not in [1, 2]
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {channels}")

        self._sample_rate = sample_rate
        self._channels = channels

    @property
    def sample_rate(self) -> int:
        """Current sample rate (default 16000 if not configured)."""
        return self._sample_rate if self._sample_rate is not None else self.DEFAULT_SAMPLE_RATE

    @property
    def channels(self) -> int:
        """Current channel count (default 1 if not configured)."""
        return self._channels if self._channels is not None else self.DEFAULT_CHANNELS

    @property
    def is_configured(self) -> bool:
        """Whether configure() has been called."""
        return self._sample_rate is not None

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called by sounddevice when output buffer needs data.

        Runs in high-priority audio thread.
        """
        with self._lock:
            if self._audio_data is None or self._interrupted:
                # Fill with silence
                outdata.fill(0)
                return

            # Calculate how much audio we can provide
            remaining = len(self._audio_data) - self._playback_position
            to_copy = min(frames, remaining)

            if to_copy > 0:
                # Copy audio data
                chunk = self._audio_data[
                    self._playback_position : self._playback_position + to_copy
                ]
                outdata[:to_copy, 0] = chunk
                self._playback_position += to_copy

                # NANO-069: Compute RMS for portrait visualization
                self._current_rms = float(np.sqrt(np.mean(chunk ** 2)))

            # Fill remainder with silence
            if to_copy < frames:
                outdata[to_copy:, :] = 0

            # Check if playback complete
            # NANO-111: During streaming, don't finish until finalized
            if self._playback_position >= len(self._audio_data) and self._streaming_finalized:
                self._playing = False
                self._finished = True

    def _monitor_completion(self) -> None:
        """Background thread that monitors for playback completion."""
        last_level_time = 0.0

        while True:
            with self._lock:
                if self._finished and not self._interrupted:
                    # Playback finished normally
                    break
                if self._interrupted or not self._playing and not self._finished:
                    # Interrupted or stopped externally
                    # NANO-069: Emit zero level on stop
                    if self.on_audio_level is not None:
                        try:
                            self.on_audio_level(0.0)
                        except Exception:
                            pass
                    return

            # NANO-069: Emit audio level at ~50ms intervals
            now = time.monotonic()
            if self.on_audio_level is not None and (now - last_level_time) >= self.AUDIO_LEVEL_INTERVAL:
                last_level_time = now
                # Clamp to 0.0-1.0 range
                level = min(1.0, self._current_rms)
                try:
                    self.on_audio_level(level)
                except Exception:
                    pass

            time.sleep(0.01)

        # NANO-069: Emit zero level when playback ends
        if self.on_audio_level is not None:
            try:
                self.on_audio_level(0.0)
            except Exception:
                pass

        # Clean up stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Fire completion callback
        if self.on_complete is not None:
            self.on_complete()

    def play(self, audio: np.ndarray, blocking: bool = False) -> None:
        """
        Start playing audio.

        Args:
            audio: Audio data as float32 numpy array at configured sample rate.
                   If configure() was not called, assumes 16kHz mono.
            blocking: If True, wait for playback to complete.
        """
        self.stop()

        with self._lock:
            # Ensure audio is float32 and 1D
            self._audio_data = audio.astype(np.float32).flatten()
            self._playback_position = 0
            self._playing = True
            self._interrupted = False
            self._finished = False

        # Create and start output stream with configured (or default) sample rate
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.DTYPE,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

        if blocking:
            self.wait()
        else:
            # Start background monitor for non-blocking playback
            self._monitor_thread = threading.Thread(
                target=self._monitor_completion, daemon=True
            )
            self._monitor_thread.start()

    def stop(self) -> None:
        """Stop playback immediately (for barge-in)."""
        was_playing = False

        with self._lock:
            if self._playing or self._finished:
                was_playing = self._playing
                self._interrupted = True
                self._playing = False
                self._finished = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if was_playing and self.on_interrupt is not None:
            self.on_interrupt()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for playback to complete.

        Args:
            timeout: Maximum time to wait (seconds). None = wait forever.

        Returns:
            True if playback completed normally, False if timed out or interrupted.
        """
        start = time.time()

        while True:
            with self._lock:
                if not self._playing:
                    break
                if self._interrupted:
                    return False

            if timeout is not None and (time.time() - start) > timeout:
                return False

            time.sleep(0.01)

        # Clean up stream
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Fire completion callback if not interrupted
        with self._lock:
            was_interrupted = self._interrupted

        if not was_interrupted and self.on_complete is not None:
            self.on_complete()

        return not was_interrupted

    def play_streaming(self, first_chunk: np.ndarray) -> None:
        """
        Start playing audio with the first chunk, expecting more via append_audio() (NANO-111).

        Begins playback immediately on the first chunk. Subsequent chunks are
        appended via append_audio(). Call finalize_streaming() when the last
        chunk has been appended so the monitor thread knows to fire on_complete.

        Args:
            first_chunk: First audio chunk as float32 numpy array.
        """
        self.stop()

        with self._lock:
            self._audio_data = first_chunk.astype(np.float32).flatten()
            self._playback_position = 0
            self._playing = True
            self._interrupted = False
            self._finished = False
            self._streaming_finalized = False

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.DTYPE,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

        # Start background monitor (same as non-blocking play)
        self._monitor_thread = threading.Thread(
            target=self._monitor_completion, daemon=True
        )
        self._monitor_thread.start()

    def append_audio(self, chunk: np.ndarray) -> None:
        """
        Append an audio chunk to the currently playing stream (NANO-111).

        Thread-safe. The audio callback will seamlessly continue into
        the appended data. If playback has already passed the end of
        the previous data, there may be a brief silence gap (natural
        inter-sentence pause).

        Args:
            chunk: Audio data as float32 numpy array at the same sample rate.
        """
        with self._lock:
            if self._audio_data is None:
                return
            new_data = chunk.astype(np.float32).flatten()
            self._audio_data = np.concatenate([self._audio_data, new_data])

    def finalize_streaming(self) -> None:
        """
        Signal that no more audio chunks will be appended (NANO-111).

        After this call, the monitor thread will fire on_complete when
        playback reaches the end of the audio data.
        """
        with self._lock:
            self._streaming_finalized = True

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        with self._lock:
            return self._playing

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        with self._lock:
            return self._playback_position / self.sample_rate

    @property
    def duration(self) -> float:
        """Total duration of loaded audio in seconds."""
        with self._lock:
            if self._audio_data is None:
                return 0.0
            return len(self._audio_data) / self.sample_rate


class FullDuplexStream:
    """
    Simultaneous audio input and output.

    Manages both AudioCapture and AudioPlayback to enable full-duplex
    operation where the agent can listen while speaking (for barge-in).

    This is the coordination layer - it doesn't do echo cancellation,
    just manages the streams.

    Note: Input (capture) is fixed at 16kHz due to Silero VAD model constraint.
    Output (playback) sample rate is configurable via configure_playback().
    """

    # Input sample rate is fixed at 16kHz (Silero VAD requirement)
    INPUT_SAMPLE_RATE = 16000

    def __init__(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        chunk_size: int = 512,
        on_input_chunk: Optional[Callable[[np.ndarray], None]] = None,
        on_playback_complete: Optional[Callable[[], None]] = None,
        on_playback_interrupt: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize full-duplex audio stream.

        Args:
            input_device: Audio input device ID (None = default).
            output_device: Audio output device ID (None = default).
            chunk_size: Samples per input callback (512 = 32ms at 16kHz).
            on_input_chunk: Callback for each input audio chunk.
            on_playback_complete: Callback when playback finishes normally.
            on_playback_interrupt: Callback when playback is interrupted.
        """
        # Import here to avoid circular dependency
        from .capture import AudioCapture

        self._capture = AudioCapture(
            chunk_size=chunk_size,
            buffer_chunks=100,
            device=input_device,
            on_chunk=on_input_chunk,
        )

        self._playback = AudioPlayback(
            device=output_device,
            on_complete=on_playback_complete,
            on_interrupt=on_playback_interrupt,
        )

        self._running = False

    def configure_playback(self, sample_rate: int, channels: int = 1) -> None:
        """
        Configure playback sample rate for TTS provider output.

        Args:
            sample_rate: Output sample rate (e.g., 24000 for Kokoro)
            channels: Number of channels (1 = mono, 2 = stereo)
        """
        self._playback.configure(sample_rate, channels)

    def start(self) -> None:
        """Start capturing audio from microphone."""
        if self._running:
            return
        self._capture.start()
        self._running = True

    def stop(self) -> None:
        """Stop all audio streams."""
        self._capture.stop()
        self._playback.stop()
        self._running = False

    def play(self, audio: np.ndarray) -> None:
        """
        Start playing audio through output device.

        Capture continues during playback for barge-in detection.

        Args:
            audio: Audio data as float32 numpy array (16kHz mono).
        """
        self._playback.play(audio, blocking=False)

    def stop_playback(self) -> None:
        """Stop audio playback immediately (barge-in)."""
        self._playback.stop()

    def get_captured_audio(self) -> np.ndarray:
        """Get all buffered input audio without clearing."""
        return self._capture.get_audio()

    def consume_captured_audio(self) -> np.ndarray:
        """Get all buffered input audio and clear the buffer."""
        return self._capture.consume_audio()

    @property
    def is_running(self) -> bool:
        """Whether capture is active."""
        return self._running

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently being played."""
        return self._playback.is_playing

    @property
    def capture(self):
        """Direct access to AudioCapture instance."""
        return self._capture

    @property
    def playback(self):
        """Direct access to AudioPlayback instance."""
        return self._playback

    def __enter__(self) -> "FullDuplexStream":
        """Context manager entry - start capture."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop all streams."""
        self.stop()
