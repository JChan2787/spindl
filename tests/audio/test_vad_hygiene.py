"""Tests for SileroVAD model hygiene — periodic reset and exception handling."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spindl.audio.vad import MODEL_RESET_INTERVAL, SileroVAD


@pytest.fixture
def vad():
    """Create a SileroVAD with mocked torch model."""
    with patch("spindl.audio.vad.torch.hub.load") as mock_load:
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.3
        mock_load.return_value = (mock_model, None)
        v = SileroVAD()
        yield v


def test_periodic_model_reset(vad):
    """Model state is reset after MODEL_RESET_INTERVAL seconds."""
    chunk = np.zeros(512, dtype=np.float32)

    # Process once — no reset yet (just initialized)
    vad._last_reset_time = time.monotonic()
    vad.process_chunk(chunk)
    vad._model.reset_states.reset_mock()

    # Simulate time passing beyond the reset interval
    vad._last_reset_time = time.monotonic() - (MODEL_RESET_INTERVAL + 1.0)
    vad.process_chunk(chunk)

    vad._model.reset_states.assert_called_once()


def test_no_reset_before_interval(vad):
    """Model state is NOT reset before MODEL_RESET_INTERVAL elapses."""
    chunk = np.zeros(512, dtype=np.float32)

    # Reset the mock after init
    vad._last_reset_time = time.monotonic()
    vad._model.reset_states.reset_mock()

    # Process immediately — well within interval
    vad.process_chunk(chunk)

    vad._model.reset_states.assert_not_called()


def test_inference_exception_returns_zero(vad):
    """Bad tensor or model failure returns 0.0 instead of crashing."""
    vad._model.side_effect = RuntimeError("ONNX inference failed")

    chunk = np.zeros(512, dtype=np.float32)
    prob = vad.process_chunk(chunk)

    assert prob == 0.0


def test_inference_exception_does_not_crash(vad):
    """Multiple calls with failing model don't accumulate errors."""
    vad._model.side_effect = RuntimeError("bad tensor")

    chunk = np.zeros(512, dtype=np.float32)
    for _ in range(10):
        prob = vad.process_chunk(chunk)
        assert prob == 0.0


def test_reset_updates_last_reset_time(vad):
    """Calling reset() updates _last_reset_time."""
    old_time = vad._last_reset_time
    time.sleep(0.01)
    vad.reset()
    assert vad._last_reset_time > old_time


def test_reset_after_periodic_updates_timer(vad):
    """After periodic reset triggers, the timer is updated so it doesn't fire every call."""
    chunk = np.zeros(512, dtype=np.float32)

    # Force past interval
    vad._last_reset_time = time.monotonic() - (MODEL_RESET_INTERVAL + 1.0)
    vad._model.reset_states.reset_mock()

    # First call — should trigger reset
    vad.process_chunk(chunk)
    assert vad._model.reset_states.call_count == 1

    # Second call immediately after — should NOT trigger reset
    vad.process_chunk(chunk)
    assert vad._model.reset_states.call_count == 1
