"""Tests for StimuliEngine."""

import sys
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.core.event_bus import EventBus
from spindl.core.events import EventType, StateChangedEvent, StimulusFiredEvent
from spindl.core.state_machine import AgentState
from spindl.stimuli.engine import StimuliEngine
from spindl.stimuli.base import StimulusModule
from spindl.stimuli.models import StimulusData, StimulusSource
from spindl.stimuli.patience import PatienceModule


# -- Test fixtures --


class MockStateMachine:
    """Minimal mock of AudioStateMachine for engine tests."""

    def __init__(self, state: AgentState = AgentState.LISTENING):
        self._state = state

    @property
    def state(self) -> AgentState:
        return self._state

    @state.setter
    def state(self, value: AgentState):
        self._state = value


class MockCallbacks:
    """Minimal mock of OrchestratorCallbacks for engine tests."""

    def __init__(self):
        self._is_processing = False
        self.calls: list[dict] = []

    @property
    def is_processing(self) -> bool:
        return self._is_processing

    def process_text_input(
        self,
        text: str,
        skip_tts: bool = False,
        stimulus_source: Optional[str] = None,
        stimulus_metadata: Optional[dict] = None,
    ) -> None:
        self.calls.append(
            {
                "text": text,
                "skip_tts": skip_tts,
                "stimulus_source": stimulus_source,
                "stimulus_metadata": stimulus_metadata,
            }
        )


class DummyModule(StimulusModule):
    """Test module with controllable stimulus state."""

    def __init__(
        self,
        name: str = "dummy",
        priority: int = 50,
        enabled: bool = True,
        has: bool = False,
        user_input: str = "dummy stimulus",
    ):
        self._name = name
        self._priority = priority
        self._enabled = enabled
        self._has = has
        self._user_input = user_input
        self._started = False
        self._stopped = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._stopped = True

    def has_stimulus(self) -> bool:
        return self._has

    def get_stimulus(self) -> Optional[StimulusData]:
        if not self._has:
            return None
        self._has = False  # One-shot
        return StimulusData(
            source=StimulusSource.MODULE,
            user_input=self._user_input,
        )

    def health_check(self) -> bool:
        return True


@pytest.fixture
def engine_components():
    """Create engine with mocked dependencies."""
    sm = MockStateMachine(AgentState.LISTENING)
    cb = MockCallbacks()
    bus = EventBus()
    return sm, cb, bus


@pytest.fixture
def engine(engine_components):
    """Create and return a StimuliEngine (not started)."""
    sm, cb, bus = engine_components
    eng = StimuliEngine(
        state_machine=sm,
        callbacks=cb,
        event_bus=bus,
        enabled=True,
    )
    yield eng
    # Cleanup
    if eng._running:
        eng.stop()


# -- Tests --


class TestEngineRegistration:
    """Tests for module registration."""

    def test_register_module(self, engine):
        mod = DummyModule(name="test")
        engine.register_module(mod)
        assert len(engine.modules) == 1
        assert engine.modules[0].name == "test"

    def test_register_duplicate_rejected(self, engine):
        mod1 = DummyModule(name="test")
        mod2 = DummyModule(name="test")
        engine.register_module(mod1)
        engine.register_module(mod2)
        assert len(engine.modules) == 1

    def test_unregister_module(self, engine):
        mod = DummyModule(name="test")
        engine.register_module(mod)
        result = engine.unregister_module("test")
        assert result is True
        assert len(engine.modules) == 0
        assert mod._stopped is True

    def test_unregister_nonexistent(self, engine):
        result = engine.unregister_module("nonexistent")
        assert result is False

    def test_modules_sorted_by_priority(self, engine):
        engine.register_module(DummyModule(name="low", priority=0))
        engine.register_module(DummyModule(name="high", priority=100))
        engine.register_module(DummyModule(name="mid", priority=50))

        modules = engine.modules
        assert modules[0].name == "high"
        assert modules[1].name == "mid"
        assert modules[2].name == "low"


class TestEngineShouldFire:
    """Tests for _should_fire() decision logic."""

    def test_fires_in_listening_state(self, engine, engine_components):
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        assert engine._should_fire() is True

    def test_fires_in_idle_state(self, engine, engine_components):
        """IDLE state allows firing — supports text-only / paused-listening mode."""
        sm, cb, bus = engine_components
        sm.state = AgentState.IDLE
        assert engine._should_fire() is True

    def test_does_not_fire_in_processing(self, engine, engine_components):
        sm, cb, bus = engine_components
        sm.state = AgentState.PROCESSING
        assert engine._should_fire() is False

    def test_does_not_fire_in_user_speaking(self, engine, engine_components):
        sm, cb, bus = engine_components
        sm.state = AgentState.USER_SPEAKING
        assert engine._should_fire() is False

    def test_does_not_fire_in_system_speaking(self, engine, engine_components):
        sm, cb, bus = engine_components
        sm.state = AgentState.SYSTEM_SPEAKING
        assert engine._should_fire() is False

    def test_does_not_fire_when_callbacks_processing(self, engine, engine_components):
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        cb._is_processing = True
        assert engine._should_fire() is False

    def test_does_not_fire_during_cooldown(self, engine, engine_components):
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        engine._last_fire_time = time.monotonic()  # Just fired
        assert engine._should_fire() is False

    def test_does_not_fire_while_audio_playing(self, engine_components):
        """Audio playback blocks firing even when state is LISTENING (text-path TTS)."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: True,  # Audio is playing
        )
        assert engine._should_fire() is False

    def test_fires_when_audio_not_playing(self, engine_components):
        """is_speaking=False should not block firing."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: False,
        )
        assert engine._should_fire() is True

    def test_fires_when_no_is_speaking_callback(self, engine_components):
        """Without is_speaking callback (backward compat), firing is not blocked."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=None,
        )
        assert engine._should_fire() is True

    def test_does_not_fire_while_user_typing(self, engine, engine_components):
        """User typing in text input blocks firing."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        engine.user_typing = True
        assert engine._should_fire() is False

    def test_fires_when_user_not_typing(self, engine, engine_components):
        """Clearing user_typing allows firing again."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING
        engine.user_typing = True
        assert engine._should_fire() is False
        engine.user_typing = False
        assert engine._should_fire() is True

    def test_user_typing_pauses_patience(self, engine):
        """Setting user_typing=True pauses PATIENCE, progress reports 0."""
        patience = PatienceModule(timeout_seconds=5.0)
        engine.register_module(patience)
        patience.start()
        # Expire patience
        patience._last_activity_time = time.monotonic() - 10.0
        assert patience.has_stimulus() is True
        # Typing starts — should pause (has_stimulus=False, progress=0)
        engine.user_typing = True
        assert patience.has_stimulus() is False
        assert patience.paused is True
        progress = patience.get_progress()
        assert progress["elapsed"] == 0.0
        assert progress["progress"] == 0.0

    def test_user_typing_resume_restarts_from_zero(self, engine):
        """Clearing user_typing resumes PATIENCE from zero."""
        patience = PatienceModule(timeout_seconds=5.0)
        engine.register_module(patience)
        patience.start()
        # Expire patience, then pause
        patience._last_activity_time = time.monotonic() - 10.0
        engine.user_typing = True
        assert patience.paused is True
        # Resume — timer restarts from zero
        engine.user_typing = False
        assert patience.paused is False
        assert patience.has_stimulus() is False  # Just resumed, 0 elapsed
        progress = patience.get_progress()
        assert progress["elapsed"] < 1.0  # Near zero


    def test_playback_pauses_patience_progress(self, engine_components):
        """Audio playback pauses PATIENCE so progress reports 0."""
        sm, cb, bus = engine_components
        is_playing = {"value": False}
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: is_playing["value"],
        )
        patience = PatienceModule(timeout_seconds=5.0)
        engine.register_module(patience)
        patience.start()

        # Expire patience
        patience._last_activity_time = time.monotonic() - 10.0
        assert patience.has_stimulus() is True

        # Start playback — trigger transition detection
        is_playing["value"] = True
        engine._check_playback_transitions()

        # PATIENCE should be paused (progress = 0, has_stimulus = False)
        assert patience.paused is True
        assert patience.has_stimulus() is False
        progress = patience.get_progress()
        assert progress["elapsed"] == 0.0
        assert progress["progress"] == 0.0

    def test_playback_resume_restarts_from_zero(self, engine_components):
        """Ending playback resumes PATIENCE from zero."""
        sm, cb, bus = engine_components
        is_playing = {"value": True}
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: is_playing["value"],
        )
        patience = PatienceModule(timeout_seconds=5.0)
        engine.register_module(patience)
        patience.start()

        # Start playback → pause
        engine._check_playback_transitions()
        assert patience.paused is True

        # End playback → resume
        is_playing["value"] = False
        engine._check_playback_transitions()
        assert patience.paused is False
        assert patience.has_stimulus() is False  # Just resumed, near zero elapsed
        progress = patience.get_progress()
        assert progress["elapsed"] < 1.0

    def test_playback_end_does_not_resume_if_typing(self, engine_components):
        """If user is typing when playback ends, PATIENCE stays paused."""
        sm, cb, bus = engine_components
        is_playing = {"value": True}
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: is_playing["value"],
        )
        patience = PatienceModule(timeout_seconds=5.0)
        engine.register_module(patience)
        patience.start()

        # Start playback → pause
        engine._check_playback_transitions()
        assert patience.paused is True

        # User starts typing during playback
        engine.user_typing = True

        # End playback — should NOT resume because typing is active
        is_playing["value"] = False
        engine._check_playback_transitions()
        assert patience.paused is True  # Still paused by typing

    def test_typing_end_does_not_resume_if_playback_active(self, engine_components):
        """If playback is active when typing ends, PATIENCE stays paused."""
        sm, cb, bus = engine_components
        is_playing = {"value": True}
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: is_playing["value"],
        )
        patience = PatienceModule(timeout_seconds=5.0)
        engine.register_module(patience)
        patience.start()

        # Playback starts → pause
        engine._check_playback_transitions()
        assert patience.paused is True

        # User starts typing (pause called again — idempotent)
        engine.user_typing = True
        assert patience.paused is True

        # User stops typing — should NOT resume because playback still active
        engine.user_typing = False
        assert patience.paused is True


class TestEngineSelectStimulus:
    """Tests for _select_stimulus() priority logic."""

    def test_selects_highest_priority(self, engine):
        low = DummyModule(name="low", priority=0, has=True, user_input="low msg")
        high = DummyModule(name="high", priority=100, has=True, user_input="high msg")
        engine.register_module(low)
        engine.register_module(high)

        stimulus = engine._select_stimulus()
        assert stimulus is not None
        assert stimulus.user_input == "high msg"

    def test_skips_disabled_modules(self, engine):
        disabled = DummyModule(
            name="disabled", priority=100, enabled=False, has=True
        )
        active = DummyModule(
            name="active", priority=50, has=True, user_input="active"
        )
        engine.register_module(disabled)
        engine.register_module(active)

        stimulus = engine._select_stimulus()
        assert stimulus is not None
        assert stimulus.user_input == "active"

    def test_returns_none_when_no_stimulus(self, engine):
        engine.register_module(DummyModule(name="empty", has=False))
        assert engine._select_stimulus() is None

    def test_returns_none_when_no_modules(self, engine):
        assert engine._select_stimulus() is None


class TestEngineFire:
    """Tests for _fire() execution."""

    def test_fire_calls_process_text_input(self, engine, engine_components):
        sm, cb, bus = engine_components
        stimulus = StimulusData(
            source=StimulusSource.PATIENCE,
            user_input="Hello from PATIENCE",
            metadata={"elapsed_seconds": 60.0},
        )

        engine._fire(stimulus)

        assert len(cb.calls) == 1
        assert cb.calls[0]["text"] == "Hello from PATIENCE"
        assert cb.calls[0]["stimulus_source"] == "patience"
        assert cb.calls[0]["skip_tts"] is False

    def test_fire_emits_event(self, engine, engine_components):
        sm, cb, bus = engine_components
        received = []
        bus.subscribe(EventType.STIMULUS_FIRED, lambda e: received.append(e))

        stimulus = StimulusData(
            source=StimulusSource.CUSTOM,
            user_input="Custom prompt text",
            metadata={"elapsed_seconds": 0.0},
        )
        engine._fire(stimulus)

        assert len(received) == 1
        assert received[0].source == "custom"
        assert received[0].prompt_text == "Custom prompt text"

    def test_fire_updates_cooldown(self, engine, engine_components):
        sm, cb, bus = engine_components
        before = engine._last_fire_time

        stimulus = StimulusData(
            source=StimulusSource.PATIENCE,
            user_input="test",
        )
        engine._fire(stimulus)

        assert engine._last_fire_time > before


class TestEngineLifecycle:
    """Tests for engine start/stop."""

    def test_start_creates_thread(self, engine):
        engine.start()
        assert engine._running is True
        assert engine._thread is not None
        assert engine._thread.is_alive()
        engine.stop()

    def test_stop_joins_thread(self, engine):
        engine.start()
        engine.stop()
        assert engine._running is False
        assert engine._thread is None or not engine._thread.is_alive()

    def test_start_starts_enabled_modules(self, engine):
        mod = DummyModule(name="test", enabled=True)
        engine.register_module(mod)
        engine.start()
        assert mod._started is True
        engine.stop()

    def test_stop_stops_modules(self, engine):
        mod = DummyModule(name="test")
        engine.register_module(mod)
        engine.start()
        engine.stop()
        assert mod._stopped is True

    def test_double_start_is_safe(self, engine):
        engine.start()
        engine.start()  # No error
        engine.stop()

    def test_double_stop_is_safe(self, engine):
        engine.start()
        engine.stop()
        engine.stop()  # No error


class TestEngineEventSubscriptions:
    """Tests for event-driven activity tracking."""

    def test_activity_event_resets_patience(self, engine, engine_components):
        sm, cb, bus = engine_components
        patience = PatienceModule(timeout_seconds=5.0)  # Long timeout so it doesn't re-expire
        engine.register_module(patience)
        engine.start()

        # Manually expire patience for testing
        patience._last_activity_time = time.monotonic() - 10.0
        assert patience.has_stimulus() is True

        # Simulate activity event — engine's _on_activity_event calls reset_activity()
        bus.emit(StateChangedEvent(from_state="processing", to_state="listening", trigger="tts_complete"))

        # Give engine time to process event callback
        time.sleep(0.05)

        # PATIENCE should have been reset (5s timeout, just reset)
        assert patience.has_stimulus() is False
        engine.stop()

    def test_state_changed_to_listening_wakes_engine(self, engine, engine_components):
        sm, cb, bus = engine_components
        engine.start()

        # The wake event should be set when LISTENING is entered
        bus.emit(StateChangedEvent(from_state="processing", to_state="listening", trigger="tts_complete"))

        # Brief sleep to let the event propagate
        time.sleep(0.1)

        # Engine should have woken up (no way to directly assert, but no error = good)
        engine.stop()


class TestEngineEnabled:
    """Tests for enable/disable toggling."""

    def test_disabled_engine_does_not_fire(self, engine, engine_components):
        sm, cb, bus = engine_components
        engine.enabled = False
        patience = PatienceModule(timeout_seconds=0.01)
        engine.register_module(patience)
        engine.start()
        time.sleep(0.1)

        # Should not have fired
        assert len(cb.calls) == 0
        engine.stop()

    def test_enable_toggle(self, engine):
        engine.enabled = False
        assert engine.enabled is False
        engine.enabled = True
        assert engine.enabled is True


class TestEngineModuleStatus:
    """Tests for get_module_status()."""

    def test_status_includes_all_modules(self, engine):
        engine.register_module(DummyModule(name="a", priority=10))
        engine.register_module(DummyModule(name="b", priority=20))

        status = engine.get_module_status()
        assert len(status) == 2

    def test_status_sorted_by_priority(self, engine):
        engine.register_module(DummyModule(name="low", priority=0))
        engine.register_module(DummyModule(name="high", priority=100))

        status = engine.get_module_status()
        assert status[0]["name"] == "high"
        assert status[1]["name"] == "low"

    def test_status_fields(self, engine):
        engine.register_module(
            DummyModule(name="test", priority=50, enabled=True, has=True)
        )

        status = engine.get_module_status()
        entry = status[0]
        assert entry["name"] == "test"
        assert entry["priority"] == 50
        assert entry["enabled"] is True
        assert entry["has_stimulus"] is True
        assert entry["healthy"] is True


class TestEngineIntegration:
    """Integration tests: engine + PATIENCE + mocked pipeline."""

    def test_patience_fires_through_engine(self, engine_components):
        """End-to-end: PATIENCE expires → engine detects → fires stimulus."""
        sm, cb, bus = engine_components
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
        )
        # Very short timeout + fast loop for testing
        engine._cooldown_seconds = 0.0
        engine._loop_interval = 0.05
        patience = PatienceModule(timeout_seconds=0.05)
        engine.register_module(patience)
        engine.start()

        # Wait for PATIENCE to expire and engine to fire
        time.sleep(0.5)

        engine.stop()

        # Should have fired at least once
        assert len(cb.calls) >= 1
        assert cb.calls[0]["stimulus_source"] == "patience"

    def test_patience_fires_in_idle_state(self, engine_components):
        """PATIENCE fires when state machine is IDLE (text-only / paused listening)."""
        sm, cb, bus = engine_components
        sm.state = AgentState.IDLE
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
        )
        engine._cooldown_seconds = 0.0
        engine._loop_interval = 0.05
        patience = PatienceModule(timeout_seconds=0.05)
        engine.register_module(patience)
        engine.start()

        time.sleep(0.5)
        engine.stop()

        assert len(cb.calls) >= 1
        assert cb.calls[0]["stimulus_source"] == "patience"

    def test_higher_priority_module_preempts_patience(self, engine_components):
        """A higher-priority module should fire instead of PATIENCE."""
        sm, cb, bus = engine_components
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
        )
        engine._cooldown_seconds = 0.0
        engine._loop_interval = 0.05

        # PATIENCE: low priority, expired
        patience = PatienceModule(timeout_seconds=0.01)
        engine.register_module(patience)

        # High-priority module with stimulus ready
        high = DummyModule(
            name="urgent", priority=100, has=True, user_input="urgent msg"
        )
        engine.register_module(high)

        engine.start()
        time.sleep(0.3)
        engine.stop()

        # Should have fired the high-priority module first
        assert len(cb.calls) >= 1
        assert cb.calls[0]["stimulus_source"] == "module"
        assert cb.calls[0]["text"] == "urgent msg"

    def test_patience_blocked_during_audio_playback(self, engine_components):
        """PATIENCE should NOT fire while audio is playing (text-path TTS bug fix)."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING

        is_playing = {"value": True}

        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: is_playing["value"],
        )
        engine._cooldown_seconds = 0.0
        engine._loop_interval = 0.05

        patience = PatienceModule(timeout_seconds=0.05)
        engine.register_module(patience)
        engine.start()

        # Audio is "playing" — PATIENCE should NOT fire even though it's expired
        time.sleep(0.3)
        assert len(cb.calls) == 0

        # Stop "playing" — PATIENCE should fire now
        is_playing["value"] = False
        time.sleep(0.3)
        engine.stop()

        assert len(cb.calls) >= 1
        assert cb.calls[0]["stimulus_source"] == "patience"

    def test_patience_progress_frozen_during_playback(self, engine_components):
        """PATIENCE progress should report 0 during audio playback (live loop)."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING

        is_playing = {"value": False}
        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
            is_speaking=lambda: is_playing["value"],
        )
        engine._cooldown_seconds = 0.0
        engine._loop_interval = 0.05

        patience = PatienceModule(timeout_seconds=60.0)
        engine.register_module(patience)
        engine.start()

        # Let some time elapse
        time.sleep(0.15)
        progress_before = patience.get_progress()
        assert progress_before["elapsed"] > 0.0  # Timer is ticking

        # Start playback — loop should detect transition and pause
        is_playing["value"] = True
        time.sleep(0.15)  # Let loop run a few cycles

        # Progress should now be frozen at 0
        progress_during = patience.get_progress()
        assert progress_during["elapsed"] == 0.0
        assert progress_during["progress"] == 0.0
        assert patience.paused is True

        # Stop playback — loop should detect and resume
        is_playing["value"] = False
        time.sleep(0.15)

        # Timer should be running again, near zero (just resumed)
        progress_after = patience.get_progress()
        assert patience.paused is False
        assert progress_after["elapsed"] < 1.0  # Near zero, freshly resumed

        engine.stop()

    def test_patience_blocked_while_user_typing(self, engine_components):
        """PATIENCE should NOT fire while user is typing, then resume after."""
        sm, cb, bus = engine_components
        sm.state = AgentState.LISTENING

        engine = StimuliEngine(
            state_machine=sm,
            callbacks=cb,
            event_bus=bus,
            enabled=True,
        )
        engine._cooldown_seconds = 0.0
        engine._loop_interval = 0.05

        patience = PatienceModule(timeout_seconds=0.05)
        engine.register_module(patience)

        # Start typing BEFORE engine starts
        engine.user_typing = True
        engine.start()

        # PATIENCE should NOT fire while typing
        time.sleep(0.3)
        assert len(cb.calls) == 0

        # Stop typing — PATIENCE timer was reset when typing started,
        # so it needs to re-expire before firing
        engine.user_typing = False
        time.sleep(0.3)
        engine.stop()

        assert len(cb.calls) >= 1
        assert cb.calls[0]["stimulus_source"] == "patience"
