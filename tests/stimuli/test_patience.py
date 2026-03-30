"""Tests for PatienceModule."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.patience import PatienceModule
from spindl.stimuli.models import StimulusSource


class TestPatienceModule:
    """Tests for PatienceModule basic properties."""

    def test_defaults(self):
        module = PatienceModule()
        assert module.name == "patience"
        assert module.priority == 0
        assert module.enabled is True
        assert module.timeout_seconds == 60.0
        assert module.health_check() is True

    def test_custom_config(self):
        module = PatienceModule(
            timeout_seconds=30.0,
            prompt="Say something!",
            enabled=False,
        )
        assert module.timeout_seconds == 30.0
        assert module.prompt == "Say something!"
        assert module.enabled is False

    def test_enable_disable(self):
        module = PatienceModule()
        module.enabled = False
        assert module.enabled is False
        module.enabled = True
        assert module.enabled is True

    def test_timeout_setter(self):
        module = PatienceModule()
        module.timeout_seconds = 120.0
        assert module.timeout_seconds == 120.0

    def test_timeout_setter_negative_clamped(self):
        module = PatienceModule()
        module.timeout_seconds = -10.0
        assert module.timeout_seconds == 0.0

    def test_prompt_setter(self):
        module = PatienceModule()
        module.prompt = "New prompt"
        assert module.prompt == "New prompt"


class TestPatienceHasStimulus:
    """Tests for PatienceModule.has_stimulus() logic."""

    def test_not_running_returns_false(self):
        module = PatienceModule(timeout_seconds=0.01)
        # Module not started
        assert module.has_stimulus() is False

    def test_disabled_returns_false(self):
        module = PatienceModule(timeout_seconds=0.01, enabled=False)
        module.start()
        time.sleep(0.02)
        assert module.has_stimulus() is False
        module.stop()

    def test_zero_timeout_returns_false(self):
        module = PatienceModule(timeout_seconds=0.0)
        module.start()
        assert module.has_stimulus() is False
        module.stop()

    def test_not_expired_returns_false(self):
        module = PatienceModule(timeout_seconds=999.0)
        module.start()
        assert module.has_stimulus() is False
        module.stop()

    def test_expired_returns_true(self):
        module = PatienceModule(timeout_seconds=0.01)
        module.start()
        time.sleep(0.03)
        assert module.has_stimulus() is True
        module.stop()


class TestPatienceGetStimulus:
    """Tests for PatienceModule.get_stimulus()."""

    def test_returns_none_when_not_expired(self):
        module = PatienceModule(timeout_seconds=999.0)
        module.start()
        assert module.get_stimulus() is None
        module.stop()

    def test_returns_data_when_expired(self):
        module = PatienceModule(timeout_seconds=0.01, prompt="Talk now!")
        module.start()
        time.sleep(0.03)

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.source == StimulusSource.PATIENCE
        assert stimulus.user_input == "Talk now!"
        assert "elapsed_seconds" in stimulus.metadata
        assert "timeout_seconds" in stimulus.metadata
        assert stimulus.metadata["timeout_seconds"] == 0.01
        module.stop()

    def test_get_stimulus_resets_timer(self):
        """After firing, PATIENCE should not fire again immediately."""
        module = PatienceModule(timeout_seconds=0.01)
        module.start()
        time.sleep(0.03)

        # First call should succeed
        stimulus = module.get_stimulus()
        assert stimulus is not None

        # Immediately after, should not be expired (timer reset)
        assert module.has_stimulus() is False
        module.stop()


class TestPatienceResetActivity:
    """Tests for PatienceModule.reset_activity()."""

    def test_reset_prevents_firing(self):
        module = PatienceModule(timeout_seconds=0.05)
        module.start()
        time.sleep(0.03)

        # Reset before expiry
        module.reset_activity()

        # Should not be expired now
        assert module.has_stimulus() is False
        module.stop()


class TestPatienceProgress:
    """Tests for PatienceModule.get_progress()."""

    def test_progress_at_start(self):
        module = PatienceModule(timeout_seconds=60.0)
        module.start()

        progress = module.get_progress()
        assert progress["elapsed"] < 1.0
        assert progress["total"] == 60.0
        assert progress["progress"] < 0.1
        module.stop()

    def test_progress_capped_at_one(self):
        module = PatienceModule(timeout_seconds=0.01)
        module.start()
        time.sleep(0.03)

        progress = module.get_progress()
        assert progress["progress"] == 1.0
        module.stop()

    def test_progress_zero_timeout(self):
        module = PatienceModule(timeout_seconds=0.0)
        module.start()

        progress = module.get_progress()
        assert progress["progress"] == 0.0
        module.stop()


class TestPatienceLifecycle:
    """Tests for PatienceModule start/stop."""

    def test_start_stop(self):
        module = PatienceModule()
        module.start()
        assert module._running is True
        module.stop()
        assert module._running is False

    def test_double_start(self):
        """Starting twice should be safe."""
        module = PatienceModule()
        module.start()
        module.start()  # No error
        module.stop()

    def test_double_stop(self):
        """Stopping twice should be safe."""
        module = PatienceModule()
        module.start()
        module.stop()
        module.stop()  # No error
