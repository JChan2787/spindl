"""Tests for AudioStateMachine: barge-in behavior and audio input timeout."""

import time

import numpy as np
import pytest

from spindl.audio.vad import SpeechEvent
from spindl.core.state_machine import (
    AUDIO_INPUT_TIMEOUT,
    AgentCallbacks,
    AgentState,
    AudioStateMachine,
)


class TestBargeInSpeechCapture:
    """
    Tests for NANO-012: Barge-in should capture speech, not discard it.

    These tests directly invoke the internal _handle_speech_start/end methods
    to test state machine logic without relying on Silero VAD to detect
    synthetic audio as speech.
    """

    def test_barge_in_transitions_to_user_speaking(self):
        """Barge-in should transition to USER_SPEAKING, not LISTENING."""
        transitions = []

        def on_state_change(t):
            transitions.append((t.from_state, t.to_state, t.trigger))

        sm = AudioStateMachine(
            callbacks=AgentCallbacks(on_state_change=on_state_change),
        )

        # Get to SYSTEM_SPEAKING state
        sm.activate()  # IDLE -> LISTENING
        sm._transition(AgentState.SYSTEM_SPEAKING, "test_tts_start")

        assert sm.state == AgentState.SYSTEM_SPEAKING

        # Simulate VAD detecting speech (barge-in) by calling handler directly
        speech_event = SpeechEvent(event_type="speech_start", timestamp=1.0, duration=None)
        sm._handle_speech_start(speech_event)

        # Should have transitioned to USER_SPEAKING, not LISTENING
        assert sm.state == AgentState.USER_SPEAKING

        # Find the barge_in transition
        barge_in_transition = [t for t in transitions if t[2] == "barge_in"]
        assert len(barge_in_transition) == 1
        assert barge_in_transition[0][0] == AgentState.SYSTEM_SPEAKING
        assert barge_in_transition[0][1] == AgentState.USER_SPEAKING

    def test_barge_in_seeds_audio_buffer_with_preroll(self):
        """Barge-in should seed audio buffer with pre-roll data."""
        sm = AudioStateMachine(pre_roll_ms=100)

        sm.activate()
        sm._transition(AgentState.SYSTEM_SPEAKING, "test")

        # Manually populate pre-roll buffer with identifiable chunks
        test_chunk = np.ones(512, dtype=np.float32) * 0.123
        sm._pre_roll_buffer.append(test_chunk.copy())
        sm._pre_roll_buffer.append(test_chunk.copy())

        pre_roll_count = len(sm._pre_roll_buffer)
        assert pre_roll_count == 2

        # Trigger barge-in by calling handler directly
        speech_event = SpeechEvent(event_type="speech_start", timestamp=1.0, duration=None)
        sm._handle_speech_start(speech_event)

        # Audio buffer should have been seeded with pre-roll
        assert len(sm._audio_buffer) == pre_roll_count
        # Verify it's actually the pre-roll data
        assert np.allclose(sm._audio_buffer[0], test_chunk)

    def test_barge_in_fires_both_callbacks(self):
        """Barge-in should fire on_barge_in AND on_user_speech_start."""
        barge_in_fired = []
        speech_start_fired = []

        def on_barge_in():
            barge_in_fired.append(True)

        def on_speech_start():
            speech_start_fired.append(True)

        sm = AudioStateMachine(
            callbacks=AgentCallbacks(
                on_barge_in=on_barge_in,
                on_user_speech_start=on_speech_start,
            ),
        )

        sm.activate()
        sm._transition(AgentState.SYSTEM_SPEAKING, "test")

        # Trigger barge-in
        speech_event = SpeechEvent(event_type="speech_start", timestamp=1.0, duration=None)
        sm._handle_speech_start(speech_event)

        assert len(barge_in_fired) == 1, "on_barge_in should fire"
        assert len(speech_start_fired) == 1, "on_user_speech_start should fire"

    def test_barge_in_speech_captured_on_speech_end(self):
        """After barge-in, speech_end should deliver captured audio."""
        captured_audio = []
        captured_duration = []

        def on_speech_end(audio, duration):
            captured_audio.append(audio)
            captured_duration.append(duration)

        sm = AudioStateMachine(
            callbacks=AgentCallbacks(on_user_speech_end=on_speech_end),
        )

        sm.activate()
        sm._transition(AgentState.SYSTEM_SPEAKING, "test")

        # Pre-populate pre-roll so we have something to capture
        test_chunk = np.ones(512, dtype=np.float32) * 0.5
        sm._pre_roll_buffer.append(test_chunk.copy())

        # Trigger barge-in
        speech_event = SpeechEvent(event_type="speech_start", timestamp=1.0, duration=None)
        sm._handle_speech_start(speech_event)

        assert sm.state == AgentState.USER_SPEAKING

        # Simulate more audio being captured during USER_SPEAKING
        sm._audio_buffer.append(test_chunk.copy())
        sm._audio_buffer.append(test_chunk.copy())

        # Trigger speech end
        end_event = SpeechEvent(event_type="speech_end", timestamp=2.0, duration=1.0)
        sm._handle_speech_end(end_event)

        # Should have transitioned to PROCESSING and delivered audio
        assert sm.state == AgentState.PROCESSING
        assert len(captured_audio) == 1
        assert len(captured_audio[0]) > 0, "Should have captured audio data"
        # Should have 3 chunks worth: 1 pre-roll + 2 during speaking
        assert len(captured_audio[0]) == 512 * 3

    def test_normal_speech_still_works(self):
        """Ensure normal LISTENING -> USER_SPEAKING flow still works."""
        transitions = []

        def on_state_change(t):
            transitions.append((t.from_state, t.to_state, t.trigger))

        sm = AudioStateMachine(
            callbacks=AgentCallbacks(on_state_change=on_state_change),
        )

        sm.activate()
        assert sm.state == AgentState.LISTENING

        # Simulate VAD detecting speech start
        speech_event = SpeechEvent(event_type="speech_start", timestamp=1.0, duration=None)
        sm._handle_speech_start(speech_event)

        assert sm.state == AgentState.USER_SPEAKING

        # Verify transition
        speech_start = [t for t in transitions if t[2] == "vad_speech_start"]
        assert len(speech_start) == 1
        assert speech_start[0][0] == AgentState.LISTENING
        assert speech_start[0][1] == AgentState.USER_SPEAKING

    def test_barge_in_sets_speech_start_time(self):
        """Barge-in should set _speech_start_time from the event."""
        sm = AudioStateMachine()

        sm.activate()
        sm._transition(AgentState.SYSTEM_SPEAKING, "test")

        assert sm._speech_start_time is None

        # Trigger barge-in with specific timestamp
        speech_event = SpeechEvent(event_type="speech_start", timestamp=12345.0, duration=None)
        sm._handle_speech_start(speech_event)

        assert sm._speech_start_time == 12345.0


class TestAudioInputTimeout:
    """
    Tests for NANO-108 Layer 2: Audio input timeout.

    When stuck in USER_SPEAKING with no audio chunks arriving (stream died),
    the state machine should force a transition back to LISTENING.
    """

    def _make_sm(self, **kwargs):
        """Create a state machine and advance to USER_SPEAKING."""
        sm = AudioStateMachine(**kwargs)
        sm.activate()  # IDLE -> LISTENING
        # Feed one chunk so _audio_received is True
        chunk = np.zeros(512, dtype=np.float32)
        sm.process_audio(chunk)
        # Force to USER_SPEAKING
        event = SpeechEvent(event_type="speech_start", timestamp=1.0)
        sm._handle_speech_start(event)
        assert sm.state == AgentState.USER_SPEAKING
        return sm

    def test_timeout_forces_listening(self):
        """USER_SPEAKING with no audio for > AUDIO_INPUT_TIMEOUT forces LISTENING."""
        sm = self._make_sm()

        # Simulate time passing beyond timeout
        sm._last_audio_time = time.monotonic() - (AUDIO_INPUT_TIMEOUT + 0.1)

        result = sm.check_audio_timeout()
        assert result is True
        assert sm.state == AgentState.LISTENING

    def test_no_timeout_within_interval(self):
        """No timeout triggered when audio arrived recently."""
        sm = self._make_sm()

        # Last audio just arrived
        sm._last_audio_time = time.monotonic()

        result = sm.check_audio_timeout()
        assert result is False
        assert sm.state == AgentState.USER_SPEAKING

    def test_first_frame_guard(self):
        """No timeout before first audio chunk is received."""
        sm = AudioStateMachine()
        sm.activate()

        # Force to USER_SPEAKING without processing audio
        event = SpeechEvent(event_type="speech_start", timestamp=1.0)
        sm._handle_speech_start(event)

        # _audio_received is False — should not trigger
        assert sm._audio_received is False
        result = sm.check_audio_timeout()
        assert result is False

    def test_timeout_clears_buffers(self):
        """Timeout clears audio buffer and pre-roll buffer."""
        sm = self._make_sm()

        # Add some data to buffers
        chunk = np.ones(512, dtype=np.float32)
        sm._audio_buffer = [chunk, chunk, chunk]
        sm._pre_roll_buffer.append(chunk)

        # Force timeout
        sm._last_audio_time = time.monotonic() - (AUDIO_INPUT_TIMEOUT + 0.1)
        sm.check_audio_timeout()

        assert sm._audio_buffer == []
        assert len(sm._pre_roll_buffer) == 0
        assert sm._speech_start_time is None

    def test_timeout_does_not_fire_in_listening(self):
        """Timeout only applies to USER_SPEAKING, not LISTENING."""
        sm = AudioStateMachine()
        sm.activate()

        chunk = np.zeros(512, dtype=np.float32)
        sm.process_audio(chunk)

        # In LISTENING with stale audio time
        sm._last_audio_time = time.monotonic() - 10.0
        assert sm.state == AgentState.LISTENING

        result = sm.check_audio_timeout()
        assert result is False
        assert sm.state == AgentState.LISTENING

    def test_timeout_does_not_fire_in_processing(self):
        """Timeout only applies to USER_SPEAKING, not PROCESSING."""
        sm = self._make_sm()

        # Transition to PROCESSING
        end_event = SpeechEvent(event_type="speech_end", timestamp=2.0, duration=1.0)
        sm._handle_speech_end(end_event)
        assert sm.state == AgentState.PROCESSING

        sm._last_audio_time = time.monotonic() - 10.0
        result = sm.check_audio_timeout()
        assert result is False

    def test_timeout_transition_recorded(self):
        """Timeout transition appears in history with 'audio_timeout' trigger."""
        transitions = []

        def on_state_change(t):
            transitions.append(t)

        sm = self._make_sm(
            callbacks=AgentCallbacks(on_state_change=on_state_change)
        )

        sm._last_audio_time = time.monotonic() - (AUDIO_INPUT_TIMEOUT + 0.1)
        sm.check_audio_timeout()

        timeout_transitions = [t for t in transitions if t.trigger == "audio_timeout"]
        assert len(timeout_transitions) == 1
        assert timeout_transitions[0].from_state == AgentState.USER_SPEAKING
        assert timeout_transitions[0].to_state == AgentState.LISTENING

    def test_reset_clears_audio_tracking(self):
        """reset() clears audio timeout tracking fields."""
        sm = self._make_sm()
        assert sm._audio_received is True
        assert sm._last_audio_time is not None

        sm.reset()
        assert sm._audio_received is False
        assert sm._last_audio_time is None
