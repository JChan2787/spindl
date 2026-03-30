"""Tests for RingBuffer with parameterized sample rate."""

import pytest
import numpy as np

from spindl.utils.ring_buffer import RingBuffer


class TestRingBufferSampleRate:
    """Tests for RingBuffer sample rate parameterization."""

    def test_default_sample_rate_is_16khz(self) -> None:
        """RingBuffer defaults to 16kHz sample rate."""
        buffer = RingBuffer()

        assert buffer.sample_rate == 16000

    def test_custom_sample_rate_24khz(self) -> None:
        """RingBuffer accepts 24kHz for Kokoro TTS."""
        buffer = RingBuffer(sample_rate=24000)

        assert buffer.sample_rate == 24000

    def test_custom_sample_rate_12khz(self) -> None:
        """RingBuffer accepts 12kHz for Qwen3 TTS."""
        buffer = RingBuffer(sample_rate=12000)

        assert buffer.sample_rate == 12000

    def test_max_chunks_still_configurable(self) -> None:
        """max_chunks parameter still works with sample_rate."""
        buffer = RingBuffer(max_chunks=50, sample_rate=24000)

        # Fill buffer to verify max_chunks
        chunk = np.zeros(512, dtype=np.float32)
        for _ in range(60):
            buffer.append(chunk)

        # Should only have 50 chunks
        assert len(buffer) == 50
        assert buffer.sample_rate == 24000


class TestRingBufferDuration:
    """Tests for duration_seconds with parameterized sample rate."""

    def test_duration_at_16khz_default(self) -> None:
        """duration_seconds uses default 16kHz rate."""
        buffer = RingBuffer()

        # Add 16000 samples (1 second at 16kHz)
        chunk = np.zeros(16000, dtype=np.float32)
        buffer.append(chunk)

        assert buffer.duration_seconds == 1.0

    def test_duration_at_24khz(self) -> None:
        """duration_seconds uses configured 24kHz rate."""
        buffer = RingBuffer(sample_rate=24000)

        # Add 24000 samples (1 second at 24kHz)
        chunk = np.zeros(24000, dtype=np.float32)
        buffer.append(chunk)

        assert buffer.duration_seconds == 1.0

    def test_duration_at_12khz(self) -> None:
        """duration_seconds uses configured 12kHz rate."""
        buffer = RingBuffer(sample_rate=12000)

        # Add 12000 samples (1 second at 12kHz)
        chunk = np.zeros(12000, dtype=np.float32)
        buffer.append(chunk)

        assert buffer.duration_seconds == 1.0

    def test_duration_multiple_chunks(self) -> None:
        """duration_seconds sums across multiple chunks."""
        buffer = RingBuffer(sample_rate=24000)

        # Add 3 chunks of 8000 samples each = 24000 total = 1 second
        chunk = np.zeros(8000, dtype=np.float32)
        buffer.append(chunk)
        buffer.append(chunk)
        buffer.append(chunk)

        assert buffer.duration_seconds == 1.0

    def test_duration_empty_buffer_is_zero(self) -> None:
        """Empty buffer has zero duration regardless of sample rate."""
        buffer_16k = RingBuffer(sample_rate=16000)
        buffer_24k = RingBuffer(sample_rate=24000)

        assert buffer_16k.duration_seconds == 0.0
        assert buffer_24k.duration_seconds == 0.0

    def test_duration_fractional_seconds(self) -> None:
        """duration_seconds handles fractional durations."""
        buffer = RingBuffer(sample_rate=24000)

        # Add 12000 samples = 0.5 seconds at 24kHz
        chunk = np.zeros(12000, dtype=np.float32)
        buffer.append(chunk)

        assert buffer.duration_seconds == 0.5

    def test_duration_wrong_rate_gives_wrong_answer(self) -> None:
        """Demonstrates importance of matching sample rate to audio."""
        # If you have 24kHz audio but use 16kHz buffer...
        buffer_wrong = RingBuffer(sample_rate=16000)
        buffer_correct = RingBuffer(sample_rate=24000)

        # 24000 samples of "24kHz audio"
        audio_24k = np.zeros(24000, dtype=np.float32)
        buffer_wrong.append(audio_24k)
        buffer_correct.append(audio_24k)

        # Wrong rate: 24000 / 16000 = 1.5 seconds (INCORRECT)
        assert buffer_wrong.duration_seconds == 1.5

        # Correct rate: 24000 / 24000 = 1.0 second (CORRECT)
        assert buffer_correct.duration_seconds == 1.0


class TestRingBufferExistingBehavior:
    """Verify existing RingBuffer behavior is preserved."""

    def test_append_and_get_all(self) -> None:
        """Basic append and retrieval still works."""
        buffer = RingBuffer(sample_rate=24000)

        chunk1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        chunk2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        buffer.append(chunk1)
        buffer.append(chunk2)

        result = buffer.get_all()
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_clear_returns_data_and_empties(self) -> None:
        """clear() returns accumulated audio and empties buffer."""
        buffer = RingBuffer(sample_rate=24000)

        chunk = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buffer.append(chunk)

        result = buffer.clear()
        np.testing.assert_array_equal(result, chunk)

        assert len(buffer) == 0
        assert buffer.duration_seconds == 0.0

    def test_overflow_tracking(self) -> None:
        """Overflow count tracks overwritten chunks."""
        buffer = RingBuffer(max_chunks=2, sample_rate=24000)

        chunk = np.zeros(100, dtype=np.float32)
        buffer.append(chunk)  # 1
        buffer.append(chunk)  # 2 (full)
        buffer.append(chunk)  # 3 (overflow)

        assert buffer.overflow_count == 1
        assert len(buffer) == 2

    def test_total_samples_tracking(self) -> None:
        """total_samples counts all samples ever added."""
        buffer = RingBuffer(max_chunks=2, sample_rate=24000)

        chunk = np.zeros(100, dtype=np.float32)
        buffer.append(chunk)  # 100
        buffer.append(chunk)  # 200
        buffer.append(chunk)  # 300 (even though first was overwritten)

        assert buffer.total_samples == 300
