"""Continuous microphone capture with callback-based streaming."""

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from ..utils.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

# Stream watchdog constants (NANO-108 Layer 1)
WATCHDOG_TIMEOUT = 3.0  # seconds without a chunk before restart
WATCHDOG_CHECK_INTERVAL = 1.0  # how often the watchdog checks
MAX_RESTART_ATTEMPTS = 3  # cap retries before giving up
RESTART_BACKOFF = 1.0  # wait between restart attempts


class AudioCapture:
    """
    Callback-based microphone capture at 16kHz mono.

    Uses sounddevice's InputStream with a high-priority audio thread.
    Audio chunks are accumulated in a thread-safe ring buffer.

    Includes a stream health watchdog that detects when the audio stream
    stops delivering chunks (e.g., after an OS freeze) and automatically
    restarts it.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "float32"
    DEFAULT_CHUNK_SIZE = 512  # 32ms at 16kHz

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        buffer_chunks: int = 100,
        device: Optional[int] = None,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None,
        on_health_change: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize audio capture.

        Args:
            chunk_size: Samples per callback (512 = 32ms at 16kHz).
            buffer_chunks: Max chunks in ring buffer (100 = ~3.2s).
            device: Audio input device ID (None = default).
            on_chunk: Optional callback for each audio chunk.
            on_health_change: Optional callback when stream health changes.
                Called with "ok", "restarting", or "down".
        """
        self.chunk_size = chunk_size
        self.device = device
        self.on_chunk = on_chunk
        self._on_health_change = on_health_change

        self._buffer = RingBuffer(max_chunks=buffer_chunks)
        self._stream: Optional[sd.InputStream] = None
        self._stream_lock = threading.Lock()
        self._overflow_events = 0
        self._running = False

        # Watchdog state
        self._last_chunk_time: Optional[float] = None
        self._stream_health = "ok"
        self._restart_count = 0
        self._watchdog_thread: Optional[threading.Thread] = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called by sounddevice when audio is available.

        Runs in high-priority audio thread - keep it fast!
        """
        try:
            self._last_chunk_time = time.monotonic()

            if status:
                if status.input_overflow:
                    self._overflow_events += 1
                logger.debug("Audio callback status: %s", status)

            # Non-blocking lock: if restart is in progress, skip this chunk
            if not self._stream_lock.acquire(blocking=False):
                return
            try:
                # Flatten to 1D (indata shape is (frames, channels))
                chunk = indata[:, 0].astype(np.float32)

                # Store in buffer
                self._buffer.append(chunk)

                # User callback (if provided)
                if self.on_chunk is not None:
                    self.on_chunk(chunk)
            finally:
                self._stream_lock.release()
        except Exception as e:
            logger.error("Audio callback exception: %s", e)

    def start(self) -> None:
        """Start capturing audio from microphone."""
        if self._running:
            return

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True
        self._last_chunk_time = time.monotonic()
        self._stream_health = "ok"
        self._restart_count = 0

        # Start watchdog thread
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._running:
            return

        self._running = False

        # Stop watchdog first so it doesn't trigger during shutdown
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=WATCHDOG_CHECK_INTERVAL + 0.5)
            self._watchdog_thread = None

        with self._stream_lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def _watchdog_loop(self) -> None:
        """Background thread that monitors stream health."""
        while self._running:
            if self._last_chunk_time is not None and self._stream_health != "down":
                gap = time.monotonic() - self._last_chunk_time
                if gap > WATCHDOG_TIMEOUT:
                    self._restart_stream()
            time.sleep(WATCHDOG_CHECK_INTERVAL)

    def _restart_stream(self) -> None:
        """Tear down and rebuild the audio stream."""
        logger.warning(
            "Audio stream watchdog: no chunks for %.1fs, restarting stream "
            "(attempt %d/%d)",
            WATCHDOG_TIMEOUT,
            self._restart_count + 1,
            MAX_RESTART_ATTEMPTS,
        )
        self._restart_count += 1

        if self._restart_count > MAX_RESTART_ATTEMPTS:
            logger.error(
                "Audio stream watchdog: max restart attempts (%d) exceeded",
                MAX_RESTART_ATTEMPTS,
            )
            self._stream_health = "down"
            self._notify_health_change()
            return

        self._stream_health = "restarting"
        self._notify_health_change()

        try:
            with self._stream_lock:
                if self._stream is not None:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception as e:
                        logger.warning("Error closing dead stream: %s", e)
                    self._stream = None

                time.sleep(RESTART_BACKOFF)

                self._stream = sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS,
                    dtype=self.DTYPE,
                    blocksize=self.chunk_size,
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._stream.start()

            self._last_chunk_time = time.monotonic()
            self._stream_health = "ok"
            self._restart_count = 0
            self._notify_health_change()
            logger.info("Audio stream watchdog: stream restarted successfully")
        except Exception as e:
            logger.error("Audio stream watchdog: restart failed: %s", e)
            self._stream_health = "down"
            self._notify_health_change()

    def _notify_health_change(self) -> None:
        """Fire health change callback."""
        if self._on_health_change is not None:
            try:
                self._on_health_change(self._stream_health)
            except Exception as e:
                logger.error("Health change callback error: %s", e)

    @property
    def stream_health(self) -> str:
        """Stream health: 'ok', 'restarting', or 'down'."""
        return self._stream_health

    @property
    def restart_count(self) -> int:
        """Number of stream restarts since last successful start."""
        return self._restart_count

    def get_audio(self) -> np.ndarray:
        """Get all buffered audio without clearing."""
        return self._buffer.get_all()

    def consume_audio(self) -> np.ndarray:
        """Get all buffered audio and clear the buffer."""
        return self._buffer.clear()

    @property
    def is_running(self) -> bool:
        """Whether capture is currently active."""
        return self._running

    @property
    def total_samples(self) -> int:
        """Total samples captured since start."""
        return self._buffer.total_samples

    @property
    def overflow_count(self) -> int:
        """Number of audio overflow events (chunks lost)."""
        return self._overflow_events

    @property
    def buffer_duration(self) -> float:
        """Duration of currently buffered audio in seconds."""
        return self._buffer.duration_seconds

    def __enter__(self) -> "AudioCapture":
        """Context manager entry - start capture."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop capture."""
        self.stop()
