"""Thread-safe ring buffer for audio accumulation."""

import threading
from collections import deque
from typing import Optional

import numpy as np


class RingBuffer:
    """
    Thread-safe circular buffer for accumulating audio chunks.

    Uses a deque internally for O(1) append/pop operations.
    Overwrites oldest data when capacity is exceeded.
    """

    # Default sample rate for backward compatibility
    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self, max_chunks: int = 100, sample_rate: int = 16000):
        """
        Initialize ring buffer.

        Args:
            max_chunks: Maximum number of audio chunks to store.
                        At 512 samples/chunk @ 16kHz, 100 chunks = 3.2 seconds.
            sample_rate: Sample rate for duration calculations (default 16kHz).
                         Use 24000 for Kokoro TTS, 12000 for Qwen3 TTS, etc.
        """
        self._buffer: deque[np.ndarray] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._total_samples = 0
        self._overflow_count = 0
        self._sample_rate = sample_rate

    def append(self, chunk: np.ndarray) -> None:
        """
        Add an audio chunk to the buffer.

        Args:
            chunk: Audio data as numpy array (will be copied).
        """
        with self._lock:
            was_full = len(self._buffer) == self._buffer.maxlen
            self._buffer.append(chunk.copy())
            self._total_samples += len(chunk)
            if was_full:
                self._overflow_count += 1

    def get_all(self) -> np.ndarray:
        """
        Get all buffered audio as a single array.

        Returns:
            Concatenated audio data, or empty array if buffer is empty.
        """
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buffer))

    def clear(self) -> np.ndarray:
        """
        Clear buffer and return all accumulated audio.

        Returns:
            All buffered audio before clearing.
        """
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            result = np.concatenate(list(self._buffer))
            self._buffer.clear()
            return result

    def __len__(self) -> int:
        """Return number of chunks currently buffered."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_samples(self) -> int:
        """Total samples ever added (even if overwritten)."""
        with self._lock:
            return self._total_samples

    @property
    def overflow_count(self) -> int:
        """Number of times oldest chunk was overwritten."""
        with self._lock:
            return self._overflow_count

    @property
    def sample_rate(self) -> int:
        """Sample rate used for duration calculations."""
        return self._sample_rate

    @property
    def duration_seconds(self) -> float:
        """
        Approximate duration of buffered audio at configured sample rate.

        Returns:
            Duration in seconds.
        """
        with self._lock:
            if not self._buffer:
                return 0.0
            total = sum(len(chunk) for chunk in self._buffer)
            return total / self._sample_rate
