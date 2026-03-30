"""Tests for VTSDriver EventBus expression/animation wiring (NANO-060c)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.config import VTubeStudioConfig
from spindl.vts.driver import VTSDriver, _EMOTION_PATTERN


def _make_config(**overrides) -> VTubeStudioConfig:
    """Create a VTubeStudioConfig with test defaults."""
    defaults = {
        "enabled": True,
        "host": "localhost",
        "port": 8001,
        "token_path": "./test_token.txt",
        "plugin_name": "test-plugin",
        "developer": "test-dev",
        "expressions": {},
        "positions": {},
        "thinking_hotkey": "",
        "idle_hotkey": "",
    }
    defaults.update(overrides)
    return VTubeStudioConfig(**defaults)


def _make_response_event(text: str, **kwargs) -> SimpleNamespace:
    """Create a fake ResponseReadyEvent."""
    return SimpleNamespace(text=text, **kwargs)


def _make_state_event(from_state: str, to_state: str, trigger: str) -> SimpleNamespace:
    """Create a fake StateChangedEvent."""
    return SimpleNamespace(from_state=from_state, to_state=to_state, trigger=trigger)


# ------------------------------------------------------------------ #
# Emotion regex unit tests
# ------------------------------------------------------------------ #


class TestEmotionPattern:
    """Tests for the emotion bracket regex pattern."""

    def test_simple_match(self):
        m = _EMOTION_PATTERN.search("[happy] I'm so glad!")
        assert m is not None
        assert m.group(1) == "happy"

    def test_match_at_end(self):
        m = _EMOTION_PATTERN.search("I feel great! [excited]")
        assert m is not None
        assert m.group(1) == "excited"

    def test_no_match(self):
        m = _EMOTION_PATTERN.search("Just a normal sentence.")
        assert m is None

    def test_first_match_wins(self):
        m = _EMOTION_PATTERN.search("[happy] blah blah [sad]")
        assert m.group(1) == "happy"

    def test_mixed_case(self):
        m = _EMOTION_PATTERN.search("[Happy]")
        assert m.group(1) == "Happy"

    def test_empty_brackets_no_match(self):
        m = _EMOTION_PATTERN.search("[]")
        assert m is None


# ------------------------------------------------------------------ #
# _on_response_ready tests
# ------------------------------------------------------------------ #


class TestOnResponseReady:
    """Tests for emotion extraction from LLM responses."""

    def test_happy_triggers_expression(self):
        """Response with [happy] triggers the configured expression file."""
        config = _make_config(expressions={"happy": "happy.exp3.json"})
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event("[happy] I'm so glad!"))

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert cmd == "trigger_expression"
        assert args == ("happy.exp3.json", True)

    def test_sad_triggers_expression(self):
        """Response with [sad] triggers the configured expression file."""
        config = _make_config(expressions={"sad": "sad.exp3.json"})
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event("[sad] That's unfortunate."))

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert cmd == "trigger_expression"
        assert args == ("sad.exp3.json", True)

    def test_no_brackets_no_trigger(self):
        """Response without emotion brackets triggers nothing."""
        config = _make_config(expressions={"happy": "happy.exp3.json"})
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event("Just a normal response."))

        assert driver._queue.qsize() == 0

    def test_unknown_emotion_no_trigger(self):
        """Response with [confused] not in mapping triggers nothing."""
        config = _make_config(expressions={"happy": "happy.exp3.json"})
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event("[confused] Huh?"))

        assert driver._queue.qsize() == 0

    def test_multiple_brackets_first_wins(self):
        """Response with multiple brackets uses the first match."""
        config = _make_config(expressions={
            "happy": "happy.exp3.json",
            "sad": "sad.exp3.json",
        })
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event("[happy] then later [sad]"))

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert args == ("happy.exp3.json", True)

    def test_empty_expression_map_no_trigger(self):
        """Empty expression mapping means all emotions are no-op."""
        config = _make_config(expressions={})
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event("[happy] I'm glad!"))

        assert driver._queue.qsize() == 0

    def test_empty_text_no_trigger(self):
        """Empty response text triggers nothing."""
        config = _make_config(expressions={"happy": "happy.exp3.json"})
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event(""))

        assert driver._queue.qsize() == 0

    def test_emotion_case_insensitive(self):
        """Emotion lookup is case-insensitive (lowered before lookup)."""
        config = _make_config(expressions={"happy": "happy.exp3.json"})
        driver = VTSDriver(config=config)

        driver._on_response_ready(_make_response_event("[Happy] Yay!"))

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert args == ("happy.exp3.json", True)


# ------------------------------------------------------------------ #
# _on_state_changed tests
# ------------------------------------------------------------------ #


class TestOnStateChanged:
    """Tests for thinking/idle hotkey wiring on state transitions."""

    def test_processing_triggers_thinking_hotkey(self):
        """Transition to processing triggers thinking hotkey."""
        config = _make_config(thinking_hotkey="ThinkingAnim")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("listening", "processing", "vad_speech_end")
        )

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert cmd == "trigger_hotkey"
        assert args == ("ThinkingAnim",)

    def test_processing_no_hotkey_configured_noop(self):
        """Transition to processing with no thinking hotkey is a no-op."""
        config = _make_config(thinking_hotkey="")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("listening", "processing", "vad_speech_end")
        )

        assert driver._queue.qsize() == 0

    def test_listening_triggers_idle_hotkey(self):
        """Voice path: transition to listening triggers idle hotkey."""
        config = _make_config(idle_hotkey="IdleBreathing")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("system_speaking", "listening", "tts_complete")
        )

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert cmd == "trigger_hotkey"
        assert args == ("IdleBreathing",)

    def test_idle_tts_complete_triggers_idle_hotkey(self):
        """Text path: transition to idle on tts_complete triggers idle hotkey."""
        config = _make_config(idle_hotkey="IdleBreathing")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("system_speaking", "idle", "tts_complete")
        )

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert cmd == "trigger_hotkey"
        assert args == ("IdleBreathing",)

    def test_idle_response_complete_triggers_idle_hotkey(self):
        """Text path (no TTS): transition to idle on response_complete triggers idle hotkey."""
        config = _make_config(idle_hotkey="IdleBreathing")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("processing", "idle", "response_complete")
        )

        assert driver._queue.qsize() == 1
        cmd, args, _ = driver._queue.get()
        assert cmd == "trigger_hotkey"
        assert args == ("IdleBreathing",)

    def test_idle_error_does_not_trigger(self):
        """Transition to idle on error does NOT trigger idle hotkey."""
        config = _make_config(idle_hotkey="IdleBreathing")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("processing", "idle", "error")
        )

        assert driver._queue.qsize() == 0

    def test_idle_empty_input_does_not_trigger(self):
        """Transition to idle on empty_input does NOT trigger idle hotkey."""
        config = _make_config(idle_hotkey="IdleBreathing")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("processing", "idle", "empty_input")
        )

        assert driver._queue.qsize() == 0

    def test_system_speaking_no_trigger(self):
        """Transition to system_speaking triggers nothing."""
        config = _make_config(
            thinking_hotkey="ThinkingAnim",
            idle_hotkey="IdleBreathing",
        )
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("processing", "system_speaking", "tts_start")
        )

        assert driver._queue.qsize() == 0

    def test_idle_no_hotkey_configured_noop(self):
        """Transition to listening with no idle hotkey is a no-op."""
        config = _make_config(idle_hotkey="")
        driver = VTSDriver(config=config)

        driver._on_state_changed(
            _make_state_event("system_speaking", "listening", "tts_complete")
        )

        assert driver._queue.qsize() == 0


# ------------------------------------------------------------------ #
# EventBus subscription lifecycle tests
# ------------------------------------------------------------------ #


class TestEventBusSubscriptionLifecycle:
    """Tests for subscribe/unsubscribe on start/stop."""

    def test_subscribe_on_start(self):
        """start() subscribes to RESPONSE_READY and STATE_CHANGED."""
        config = _make_config()
        event_bus = MagicMock()
        event_bus.subscribe.return_value = 42
        driver = VTSDriver(config=config, event_bus=event_bus)

        with patch.object(driver, "_run_loop"):
            driver.start()
            assert event_bus.subscribe.call_count == 2
            assert len(driver._sub_ids) == 2
            driver.stop()

    def test_unsubscribe_on_stop(self):
        """stop() unsubscribes all stored subscription IDs."""
        config = _make_config()
        event_bus = MagicMock()
        event_bus.subscribe.side_effect = [10, 20]
        driver = VTSDriver(config=config, event_bus=event_bus)

        with patch.object(driver, "_run_loop"):
            driver.start()
            driver.stop()

        event_bus.unsubscribe.assert_any_call(10)
        event_bus.unsubscribe.assert_any_call(20)
        assert event_bus.unsubscribe.call_count == 2
        assert driver._sub_ids == []

    def test_no_event_bus_no_crash(self):
        """Driver without event_bus starts and stops without error."""
        config = _make_config()
        driver = VTSDriver(config=config, event_bus=None)

        with patch.object(driver, "_run_loop"):
            driver.start()
            assert driver._sub_ids == []
            driver.stop()
