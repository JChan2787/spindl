"""Tests for AudioCapture stream watchdog — dead stream detection and restart."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spindl.audio.capture import (
    WATCHDOG_TIMEOUT,
    MAX_RESTART_ATTEMPTS,
    AudioCapture,
)


@pytest.fixture
def capture():
    """Create an AudioCapture with mocked sounddevice stream."""
    with patch("spindl.audio.capture.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd.InputStream.return_value = mock_stream

        cap = AudioCapture(on_chunk=MagicMock())
        yield cap, mock_sd, mock_stream


class TestWatchdogDetection:
    """Tests for dead stream detection."""

    def test_stream_health_starts_ok(self, capture):
        """Stream health is 'ok' after start."""
        cap, mock_sd, _ = capture
        cap.start()
        assert cap.stream_health == "ok"
        cap.stop()

    def test_restart_triggered_after_timeout(self, capture):
        """Watchdog triggers restart when no chunks arrive for WATCHDOG_TIMEOUT."""
        cap, mock_sd, _ = capture
        cap.start()

        # Simulate time passing beyond timeout
        cap._last_chunk_time = time.monotonic() - (WATCHDOG_TIMEOUT + 1.0)
        cap._restart_stream()

        # Should have tried to create a new stream
        assert mock_sd.InputStream.call_count >= 2  # initial + restart
        assert cap.stream_health == "ok"
        cap.stop()

    def test_no_restart_within_timeout(self, capture):
        """No restart when chunks arrived recently."""
        cap, mock_sd, _ = capture
        cap.start()

        initial_call_count = mock_sd.InputStream.call_count

        # Last chunk just arrived
        cap._last_chunk_time = time.monotonic()

        # Manually run one watchdog check cycle
        # (gap < WATCHDOG_TIMEOUT, so no restart)
        gap = time.monotonic() - cap._last_chunk_time
        assert gap <= WATCHDOG_TIMEOUT
        assert cap.stream_health == "ok"
        assert mock_sd.InputStream.call_count == initial_call_count
        cap.stop()


class TestStreamRestart:
    """Tests for stream restart logic."""

    def test_restart_rebuilds_stream(self, capture):
        """Restart tears down and rebuilds sd.InputStream."""
        cap, mock_sd, mock_stream = capture
        cap.start()

        cap._last_chunk_time = time.monotonic() - (WATCHDOG_TIMEOUT + 1.0)
        cap._restart_stream()

        # Old stream should have been stopped and closed
        assert mock_stream.stop.called
        assert mock_stream.close.called

        # New stream should have been created
        assert mock_sd.InputStream.call_count == 2
        assert cap.stream_health == "ok"
        cap.stop()

    def test_restart_resets_count_on_success(self, capture):
        """Successful restart resets the attempt counter."""
        cap, mock_sd, _ = capture
        cap.start()

        cap._last_chunk_time = time.monotonic() - (WATCHDOG_TIMEOUT + 1.0)
        cap._restart_stream()

        assert cap._restart_count == 0
        assert cap.restart_count == 0
        cap.stop()

    def test_max_retries_sets_health_down(self, capture):
        """Exceeding MAX_RESTART_ATTEMPTS sets health to 'down'."""
        cap, mock_sd, _ = capture
        cap.start()

        # Exhaust all attempts
        cap._restart_count = MAX_RESTART_ATTEMPTS
        cap._restart_stream()

        assert cap.stream_health == "down"
        cap.stop()

    def test_restart_failure_sets_health_down(self, capture):
        """If sd.InputStream raises on restart, health goes to 'down'."""
        cap, mock_sd, _ = capture
        cap.start()

        # Make InputStream constructor fail on second call
        mock_sd.InputStream.side_effect = [
            mock_sd.InputStream.return_value,  # first call (start)
            Exception("Device disconnected"),  # restart attempt
        ]

        cap._last_chunk_time = time.monotonic() - (WATCHDOG_TIMEOUT + 1.0)

        # Reset side_effect to only fail on new calls
        mock_sd.InputStream.side_effect = Exception("Device disconnected")
        cap._restart_stream()

        assert cap.stream_health == "down"
        cap.stop()


class TestCallbackHardening:
    """Tests for audio callback exception handling."""

    def test_callback_exception_does_not_crash(self, capture):
        """Bad chunk in callback doesn't kill the audio thread."""
        cap, _, _ = capture
        cap.start()

        # Set on_chunk to something that throws
        cap.on_chunk = MagicMock(side_effect=RuntimeError("bad callback"))

        # Simulate a callback — should not raise
        indata = np.zeros((512, 1), dtype=np.float32)
        status = MagicMock()
        status.input_overflow = False
        status.__bool__ = lambda self: False

        cap._audio_callback(indata, 512, {}, status)

        # last_chunk_time should still have been set (timestamp comes first)
        assert cap._last_chunk_time is not None
        cap.stop()

    def test_callback_timestamps_chunk(self, capture):
        """Every callback updates _last_chunk_time."""
        cap, _, _ = capture
        cap.start()
        cap.on_chunk = None

        # Set to a known past time
        cap._last_chunk_time = time.monotonic() - 1.0
        old_time = cap._last_chunk_time

        indata = np.zeros((512, 1), dtype=np.float32)
        status = MagicMock()
        status.input_overflow = False
        status.__bool__ = lambda self: False

        cap._audio_callback(indata, 512, {}, status)

        assert cap._last_chunk_time > old_time
        cap.stop()


class TestHealthCallback:
    """Tests for health change notification."""

    def test_health_callback_fires_on_restart(self, capture):
        """on_health_change fires with 'restarting' then 'ok' on successful restart."""
        health_states = []
        cap, mock_sd, _ = capture
        cap._on_health_change = lambda state: health_states.append(state)
        cap.start()

        cap._last_chunk_time = time.monotonic() - (WATCHDOG_TIMEOUT + 1.0)
        cap._restart_stream()

        assert "restarting" in health_states
        assert "ok" in health_states
        assert health_states.index("restarting") < health_states.index("ok")
        cap.stop()

    def test_health_callback_fires_on_down(self, capture):
        """on_health_change fires with 'down' when max retries exceeded."""
        health_states = []
        cap, _, _ = capture
        cap._on_health_change = lambda state: health_states.append(state)
        cap.start()

        cap._restart_count = MAX_RESTART_ATTEMPTS
        cap._restart_stream()

        assert "down" in health_states
        cap.stop()

    def test_no_crash_if_health_callback_throws(self, capture):
        """Health callback exception doesn't crash the watchdog."""
        cap, _, _ = capture
        cap._on_health_change = MagicMock(side_effect=RuntimeError("callback failed"))
        cap.start()

        # Should not raise
        cap._restart_count = MAX_RESTART_ATTEMPTS
        cap._restart_stream()

        assert cap.stream_health == "down"
        cap.stop()


class TestWatchdogLifecycle:
    """Tests for watchdog thread lifecycle."""

    def test_watchdog_thread_starts_with_capture(self, capture):
        """Watchdog thread is created and alive after start()."""
        cap, _, _ = capture
        cap.start()

        assert cap._watchdog_thread is not None
        assert cap._watchdog_thread.is_alive()
        cap.stop()

    def test_watchdog_thread_stops_with_capture(self, capture):
        """Watchdog thread stops after stop()."""
        cap, _, _ = capture
        cap.start()

        thread = cap._watchdog_thread
        cap.stop()

        assert not thread.is_alive()

    def test_watchdog_is_daemon_thread(self, capture):
        """Watchdog thread is a daemon so it dies with the process."""
        cap, _, _ = capture
        cap.start()

        assert cap._watchdog_thread.daemon is True
        cap.stop()

    def test_stream_health_property(self, capture):
        """stream_health property reflects current state."""
        cap, _, _ = capture

        assert cap.stream_health == "ok"  # default before start

        cap.start()
        assert cap.stream_health == "ok"

        cap._stream_health = "restarting"
        assert cap.stream_health == "restarting"

        cap._stream_health = "down"
        assert cap.stream_health == "down"
        cap.stop()
