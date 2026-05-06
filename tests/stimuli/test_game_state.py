"""Tests for GameStateModule (NANO-116 Phase B.1, NANO-123 restructure)."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.game_state.module import GameStateModule
from spindl.stimuli.game_state.models import GameEvent
from spindl.stimuli.game_state.validator import (
    check_protocol_version,
    validate_envelope,
)
from spindl.stimuli.models import StimulusSource


# -- Validator tests -------------------------------------------------------


class TestValidateEnvelope:
    """Tests for envelope validation."""

    def _make_valid_event(self, **overrides) -> dict:
        base = {
            "protocol_version": "0.1.2",
            "event_type": "dialogue_line",
            "event_source": "direct_hook",
            "timestamp": "2026-04-25T04:00:00.000Z",
            "sequence": 42,
            "payload": {"speaker": "Diana", "text": "Watch out!", "source": "chatter"},
        }
        base.update(overrides)
        return base

    def test_valid_event_passes(self):
        ok, reason = validate_envelope(self._make_valid_event())
        assert ok is True
        assert reason == ""

    def test_missing_required_field(self):
        event = self._make_valid_event()
        del event["event_type"]
        ok, reason = validate_envelope(event)
        assert ok is False
        assert "event_type" in reason

    def test_missing_multiple_fields(self):
        event = {"protocol_version": "0.1.2", "payload": {}}
        ok, reason = validate_envelope(event)
        assert ok is False

    def test_sequence_must_be_int(self):
        ok, reason = validate_envelope(self._make_valid_event(sequence="not_int"))
        assert ok is False
        assert "sequence" in reason

    def test_payload_must_be_dict(self):
        ok, reason = validate_envelope(self._make_valid_event(payload="not_dict"))
        assert ok is False
        assert "payload" in reason

    def test_extra_fields_are_tolerated(self):
        event = self._make_valid_event(game_id="pragmata", save_slot_hint=0)
        ok, reason = validate_envelope(event)
        assert ok is True


class TestCheckProtocolVersion:
    """Tests for protocol version comparison."""

    def test_exact_match(self):
        ok, level = check_protocol_version("0.1.2", "0.1.2")
        assert ok is True
        assert level == "match"

    def test_patch_drift(self):
        ok, level = check_protocol_version("0.1.3", "0.1.2")
        assert ok is True
        assert level == "patch"

    def test_minor_drift(self):
        ok, level = check_protocol_version("0.2.0", "0.1.2")
        assert ok is True
        assert level == "minor"

    def test_major_mismatch(self):
        ok, level = check_protocol_version("1.0.0", "0.1.2")
        assert ok is False
        assert level == "major"

    def test_parse_error_non_numeric(self):
        ok, level = check_protocol_version("abc", "0.1.2")
        assert ok is False
        assert level == "parse_error"

    def test_parse_error_too_few_parts(self):
        ok, level = check_protocol_version("0.1", "0.1.2")
        assert ok is False
        assert level == "parse_error"

    def test_parse_error_none_input(self):
        ok, level = check_protocol_version(None, "0.1.2")
        assert ok is False
        assert level == "parse_error"


# -- Module property tests -------------------------------------------------


class TestGameStateModuleProperties:
    """Tests for GameStateModule basic properties."""

    def test_defaults(self):
        module = GameStateModule()
        assert module.name == "game_state"
        assert module.priority == 60
        assert module.enabled is False
        assert module.host == "127.0.0.1"
        assert module.port == 53817
        assert module.buffer_size == 20
        assert module.connected is False
        assert module.version_mismatch is False
        assert module.bridge_protocol_version is None
        assert module.buffer_count == 0

    def test_custom_config(self):
        module = GameStateModule(
            host="192.168.1.100",
            port=9999,
            buffer_size=50,
            prompt_template="Custom: {events}",
            enabled=True,
        )
        assert module.host == "192.168.1.100"
        assert module.port == 9999
        assert module.buffer_size == 50
        assert module.prompt_template == "Custom: {events}"
        assert module.enabled is True

    def test_enable_disable(self):
        module = GameStateModule()
        module.enabled = True
        assert module.enabled is True
        module.enabled = False
        assert module.enabled is False

    def test_disable_clears_buffer(self):
        module = GameStateModule(enabled=True)
        module._running = True
        module._buffer.append(
            GameEvent(
                event_type="dialogue_line",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z",
                sequence=1,
                payload={"speaker": "Diana", "text": "Hello", "source": "chatter"},
            )
        )
        assert module.buffer_count == 1
        module.enabled = False
        assert module.buffer_count == 0

    def test_buffer_size_setter_preserves_events(self):
        module = GameStateModule(buffer_size=5)
        for i in range(3):
            module._buffer.append(
                GameEvent(
                    event_type="snapshot",
                    event_source="snapshot_aggregate",
                    timestamp=f"2026-04-25T04:00:0{i}Z",
                    sequence=i,
                )
            )
        assert module.buffer_count == 3
        module.buffer_size = 10
        assert module.buffer_size == 10
        assert module.buffer_count == 3

    def test_buffer_size_minimum_is_1(self):
        module = GameStateModule(buffer_size=0)
        assert module.buffer_size == 1


# -- Buffer and drain tests ------------------------------------------------


class TestGameStateModuleBufferDrain:
    """Tests for buffer/drain — B.3 deferred, general buffer does not produce stimuli."""

    def _make_event(self, event_type: str = "dialogue_line", seq: int = 0) -> GameEvent:
        return GameEvent(
            event_type=event_type,
            event_source="direct_hook",
            timestamp="2026-04-25T04:00:00Z",
            sequence=seq,
            payload={"speaker": "Diana", "text": "Watch out!", "source": "chatter"},
            game_id="pragmata",
        )

    def test_has_stimulus_from_dialogue_buffer(self):
        module = GameStateModule(enabled=True, buffer_size=10)
        module._running = True
        assert module.has_stimulus() is False
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine
        module._dialogue_buffer._buffer.append(
            DialogueLine(speaker="Diana", text="Watch out!", source="chatter",
                         event_source="direct_hook",
                         timestamp="2026-04-25T04:00:00Z", sequence=0)
        )
        assert module.has_stimulus() is True

    def test_get_stimulus_drains_dialogue_buffer(self):
        module = GameStateModule(enabled=True, buffer_size=10)
        module._running = True
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine
        module._dialogue_buffer._buffer.append(
            DialogueLine(speaker="Diana", text="Watch out!", source="chatter",
                         event_source="direct_hook",
                         timestamp="2026-04-25T04:00:00Z", sequence=0)
        )
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "[Diana]: Watch out!" in stimulus.user_input
        assert module._dialogue_buffer.count == 0

    def test_buffer_fifo_eviction(self):
        module = GameStateModule(enabled=True, buffer_size=3)
        module._running = True
        for i in range(5):
            module._buffer.append(self._make_event(seq=i))

        assert module.buffer_count == 3
        events = list(module._buffer)
        assert events[0].sequence == 2
        assert events[2].sequence == 4


# -- Process line tests ----------------------------------------------------


class TestGameStateModuleProcessLine:
    """Tests for _process_line (event parsing and validation)."""

    def _make_module(self) -> GameStateModule:
        module = GameStateModule(enabled=True, buffer_size=20)
        module._running = True
        module._schema_version = "0.1.2"
        return module

    def _make_raw_event(self, **overrides) -> bytes:
        event = {
            "protocol_version": "0.1.2",
            "event_type": "dialogue_line",
            "event_source": "direct_hook",
            "timestamp": "2026-04-25T04:00:00.000Z",
            "sequence": 1,
            "payload": {"speaker": "Ken", "text": "Stay close.", "source": "cinematic"},
            "game_id": "pragmata",
        }
        event.update(overrides)
        return (json.dumps(event) + "\n").encode("utf-8")

    def test_valid_event_buffered(self):
        module = self._make_module()
        module._process_line(self._make_raw_event())
        assert module.buffer_count == 1

    def test_invalid_json_dropped(self):
        module = self._make_module()
        module._process_line(b"not json at all\n")
        assert module.buffer_count == 0

    def test_empty_line_ignored(self):
        module = self._make_module()
        module._process_line(b"\n")
        assert module.buffer_count == 0
        module._process_line(b"   \n")
        assert module.buffer_count == 0

    def test_missing_required_field_dropped(self):
        module = self._make_module()
        bad_event = json.dumps({"protocol_version": "0.1.2", "payload": {}}).encode()
        module._process_line(bad_event + b"\n")
        assert module.buffer_count == 0

    def test_disabled_module_does_not_buffer(self):
        module = self._make_module()
        module._enabled = False
        module._process_line(self._make_raw_event())
        assert module.buffer_count == 0

    def test_sequence_monotonicity_warning(self):
        module = self._make_module()
        module._process_line(self._make_raw_event(sequence=5))
        module._process_line(self._make_raw_event(sequence=3))
        # Both buffered (warning logged but not rejected)
        assert module.buffer_count == 2

    def test_game_id_and_save_slot_preserved(self):
        module = self._make_module()
        module._process_line(
            self._make_raw_event(game_id="pragmata", save_slot_hint=2)
        )
        event = module._buffer[0]
        assert event.game_id == "pragmata"
        assert event.save_slot_hint == 2


# -- Start/stop tests -----------------------------------------------------


class TestGameStateModuleLifecycle:
    """Tests for start/stop lifecycle."""

    @patch(
        "spindl.stimuli.game_state.module.load_vendored_version",
        return_value="0.1.2",
    )
    def test_start_sets_running(self, _mock):
        module = GameStateModule(enabled=True)
        module.start()
        assert module._running is True
        module.stop()

    @patch(
        "spindl.stimuli.game_state.module.load_vendored_version",
        return_value=None,
    )
    def test_start_fails_without_schema(self, _mock):
        module = GameStateModule(enabled=True)
        module.start()
        assert module._running is False

    def test_stop_clears_state(self):
        module = GameStateModule(enabled=True)
        module._running = True
        module._connected = True
        module._version_mismatch = True
        module._bridge_protocol_version = "0.1.2"
        module._buffer.append(
            GameEvent(
                event_type="snapshot",
                event_source="snapshot_aggregate",
                timestamp="2026-04-25T04:00:00Z",
                sequence=1,
            )
        )
        module.stop()
        assert module._connected is False
        assert module._version_mismatch is False
        assert module._bridge_protocol_version is None
        assert module.buffer_count == 0
        assert module._last_sequence == -1

    def test_double_stop_is_safe(self):
        module = GameStateModule()
        module.stop()
        module.stop()

    def test_double_start_is_safe(self):
        with patch(
            "spindl.stimuli.game_state.module.load_vendored_version",
            return_value="0.1.2",
        ):
            module = GameStateModule(enabled=True)
            module.start()
            thread1 = module._thread
            module.start()
            assert module._thread is thread1
            module.stop()

    def test_health_check_requires_connected_and_no_mismatch(self):
        module = GameStateModule()
        assert module.health_check() is False

        module._connected = True
        assert module.health_check() is True

        module._version_mismatch = True
        assert module.health_check() is False


# -- NANO-123: Dialogue responsiveness & stimulus restructure tests --------


class TestNANO123LatestLineOnly:
    """NANO-123: Only the latest dialogue line appears in the user message."""

    def _make_module(self) -> GameStateModule:
        module = GameStateModule(enabled=True, gameplay_enabled=True)
        module._running = True
        return module

    def test_single_line_in_user_message(self):
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine
        module = self._make_module()
        module._dialogue_buffer._buffer.append(
            DialogueLine(speaker="Diana", text="Watch out!", source="chatter",
                         event_source="direct_hook",
                         timestamp="2026-04-25T04:00:00Z", sequence=0)
        )
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "[Diana]: Watch out!" in stimulus.user_input

    def test_multi_line_drain_only_latest_in_user_message(self):
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine
        module = self._make_module()
        for i, (speaker, text) in enumerate([
            ("Hugh", "Take this!"),
            ("Diana", "Just like that!"),
            ("Hugh", "Time to dance."),
            ("Diana", "Hacking online!"),
        ]):
            module._dialogue_buffer._buffer.append(
                DialogueLine(speaker=speaker, text=text, source="chatter",
                             event_source="direct_hook",
                             timestamp=f"2026-04-25T04:00:0{i}Z", sequence=i)
            )
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "[Diana]: Hacking online!" in stimulus.user_input
        assert "[Hugh]: Take this!" not in stimulus.user_input
        assert "[Diana]: Just like that!" not in stimulus.user_input
        assert "[Hugh]: Time to dance." not in stimulus.user_input


class TestNANO123CombatSnapshotBundling:
    """NANO-123: Combat chatter bundles current gameplay snapshot."""

    def _make_module(self) -> GameStateModule:
        module = GameStateModule(enabled=True, gameplay_enabled=True)
        module._running = True
        return module

    def test_combat_chatter_includes_snapshot(self):
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine, GameplaySnapshot
        module = self._make_module()
        module._current_snapshot = {
            "hp_ratio": 0.73,
            "weapon_name": "Grip Gun",
            "in_combat": True,
            "is_dead": False,
            "enemies": [
                {"display_name": "Walker", "hp_ratio": 1.0, "is_dead": False},
            ],
        }
        module._dialogue_buffer._buffer.append(
            DialogueLine(
                speaker="Diana", text="Hacking online!", source="chatter",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z", sequence=0,
                gameplay_context=GameplaySnapshot(combat_active=True, enemy_count=1),
            )
        )
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "[Diana]: Hacking online!" in stimulus.user_input
        assert "**Current gameplay state:**" in stimulus.user_input
        assert "Hugh: 73% HP" in stimulus.user_input
        assert "Walker" in stimulus.user_input
        assert stimulus.metadata["combat_snapshot_bundled"] is True

    def test_exploration_chatter_no_snapshot(self):
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine, GameplaySnapshot
        module = self._make_module()
        module._current_snapshot = {
            "hp_ratio": 1.0, "weapon_name": "Grip Gun",
            "in_combat": False, "is_dead": False,
        }
        module._dialogue_buffer._buffer.append(
            DialogueLine(
                speaker="Diana", text="Look at that.", source="chatter",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z", sequence=0,
                gameplay_context=GameplaySnapshot(combat_active=False),
            )
        )
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "**Current gameplay state:**" not in stimulus.user_input
        assert stimulus.metadata["combat_snapshot_bundled"] is False


class TestNANO123PriorityEvents:
    """NANO-123: Boss/chapter events fire independently above dialogue."""

    def _make_module(self) -> GameStateModule:
        module = GameStateModule(enabled=True, gameplay_enabled=True)
        module._running = True
        return module

    def test_boss_event_fires_independently(self):
        module = self._make_module()
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="boss_battle_started",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z",
                sequence=1,
                payload={},
                game_id="pragmata",
            )
        )
        assert module.has_stimulus() is True
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "Boss battle started" in stimulus.user_input
        assert stimulus.metadata["stimulus_type"] == "priority_event"

    def test_chapter_event_fires_independently(self):
        module = self._make_module()
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="chapter_status_changed",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z",
                sequence=1,
                payload={"chapter_name": "The Audit"},
                game_id="pragmata",
            )
        )
        assert module.has_stimulus() is True
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "The Audit" in stimulus.user_input

    def test_priority_event_preempts_dialogue(self):
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine
        module = self._make_module()
        module._dialogue_buffer._buffer.append(
            DialogueLine(speaker="Diana", text="Watch out!", source="chatter",
                         event_source="direct_hook",
                         timestamp="2026-04-25T04:00:00Z", sequence=0)
        )
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="boss_battle_started",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:01Z",
                sequence=1,
                payload={},
                game_id="pragmata",
            )
        )
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "Boss battle started" in stimulus.user_input
        assert module._dialogue_buffer.count == 1

    def test_non_priority_events_do_not_fire(self):
        module = self._make_module()
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="enemy_engaged_player",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z",
                sequence=1,
                payload={"enemy_name": "Walker"},
                game_id="pragmata",
            )
        )
        assert module.has_stimulus() is False

    def test_priority_extraction_leaves_non_priority_in_buffer(self):
        module = self._make_module()
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="enemy_engaged_player",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z",
                sequence=1,
                payload={"enemy_name": "Walker"},
                game_id="pragmata",
            )
        )
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="boss_battle_started",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:01Z",
                sequence=2,
                payload={},
                game_id="pragmata",
            )
        )
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "Boss battle started" in stimulus.user_input
        assert len(module._gameplay_event_buffer) == 1
        assert module._gameplay_event_buffer[0].event_type == "enemy_engaged_player"


class TestNANO123KilledPaths:
    """NANO-123: Independent gameplay event/snapshot paths no longer fire."""

    def _make_module(self) -> GameStateModule:
        module = GameStateModule(enabled=True, gameplay_enabled=True)
        module._running = True
        return module

    def test_gameplay_events_alone_no_stimulus(self):
        module = self._make_module()
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="enemy_engaged_player",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:00Z",
                sequence=1,
                payload={"enemy_name": "Walker"},
                game_id="pragmata",
            )
        )
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="enemy_died",
                event_source="direct_hook",
                timestamp="2026-04-25T04:00:01Z",
                sequence=2,
                payload={"enemy_name": "Walker"},
                game_id="pragmata",
            )
        )
        module._gameplay_first_event_time = 0.0
        assert module.has_stimulus() is False

    def test_snapshot_alone_no_stimulus(self):
        module = self._make_module()
        module._current_snapshot = {
            "hp_ratio": 0.5, "weapon_name": "Grip Gun",
            "in_combat": True, "is_dead": False,
        }
        assert module.has_stimulus() is False

    def test_no_probability_gates(self):
        module = self._make_module()
        assert not hasattr(module, "_snapshot_roll_passed")
        assert not hasattr(module, "_last_evaluated_snapshot")


class TestNANO123Priority:
    """NANO-123: GameStateModule priority above Twitch."""

    def test_priority_above_50(self):
        module = GameStateModule()
        assert module.priority > 50


# -- NANO-123 Phase 2: Stale-line dropping -----------------------------------


class TestNANO123StaleLineDropping:
    """NANO-123 Phase 2: Lines arriving while engine is busy don't trigger."""

    def _make_module(self, idle: bool = True) -> GameStateModule:
        module = GameStateModule(
            enabled=True,
            gameplay_enabled=True,
            is_engine_idle=lambda: idle,
        )
        module._running = True
        return module

    def _make_dialogue_event(self, speaker="Diana", text="Watch out!", seq=1):
        return {
            "protocol_version": "0.1.2",
            "event_type": "dialogue_line",
            "event_source": "direct_hook",
            "timestamp": "2026-05-06T10:00:00.000Z",
            "sequence": seq,
            "payload": {"speaker": speaker, "text": text, "source": "chatter"},
        }

    def test_line_while_idle_triggers(self):
        module = self._make_module(idle=True)
        module._buffer_event(self._make_dialogue_event())
        assert module._has_fresh_trigger is True
        assert module.has_stimulus() is True

    def test_line_while_busy_no_trigger(self):
        module = self._make_module(idle=False)
        module._buffer_event(self._make_dialogue_event())
        assert module._has_fresh_trigger is False
        assert module.has_stimulus() is False

    def test_line_while_busy_still_buffered(self):
        module = self._make_module(idle=False)
        module._buffer_event(self._make_dialogue_event())
        assert module._dialogue_buffer.count == 1

    def test_stale_lines_reach_store_on_next_drain(self):
        from unittest.mock import MagicMock
        store = MagicMock()
        idle_state = [False]
        module = GameStateModule(
            enabled=True,
            gameplay_enabled=True,
            is_engine_idle=lambda: idle_state[0],
        )
        module._running = True
        module._dialogue_store = store

        module._buffer_event(self._make_dialogue_event(
            speaker="Hugh", text="Stale line", seq=1))
        assert module._has_fresh_trigger is False
        assert store.record_dialogue_line.call_count == 0

        idle_state[0] = True
        module._buffer_event(self._make_dialogue_event(
            speaker="Diana", text="Fresh line", seq=2))
        assert module._has_fresh_trigger is True

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "[Diana]: Fresh line" in stimulus.user_input
        assert "[Hugh]: Stale line" not in stimulus.user_input
        assert store.record_dialogue_line.call_count == 2

    def test_drain_resets_fresh_trigger(self):
        module = self._make_module(idle=True)
        module._buffer_event(self._make_dialogue_event())
        assert module._has_fresh_trigger is True
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert module._has_fresh_trigger is False
        assert module.has_stimulus() is False

    def test_silence_after_busy_period(self):
        """No fresh line after becoming idle = silence (correct behavior)."""
        idle_state = [False]
        module = GameStateModule(
            enabled=True,
            gameplay_enabled=True,
            is_engine_idle=lambda: idle_state[0],
        )
        module._running = True

        for i in range(5):
            module._buffer_event(self._make_dialogue_event(
                text=f"Line {i}", seq=i + 1))
        assert module._dialogue_buffer.count == 5
        assert module._has_fresh_trigger is False

        idle_state[0] = True
        assert module.has_stimulus() is False

    def test_no_callback_falls_back_to_phase1(self):
        """Without is_engine_idle callback, any buffered line triggers (legacy)."""
        module = GameStateModule(enabled=True, gameplay_enabled=True)
        module._running = True
        module._buffer_event(self._make_dialogue_event())
        assert module.has_stimulus() is True


class TestNANO123StaleLinePriorityEvents:
    """NANO-123 Phase 2: Boss/chapter events fire regardless of idle state."""

    def _make_module(self, idle: bool = False) -> GameStateModule:
        module = GameStateModule(
            enabled=True,
            gameplay_enabled=True,
            is_engine_idle=lambda: idle,
        )
        module._running = True
        return module

    def test_boss_event_fires_while_busy(self):
        module = self._make_module(idle=False)
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="boss_battle_started",
                event_source="direct_hook",
                timestamp="2026-05-06T10:00:00Z",
                sequence=1,
                payload={"boss_name": "Clavis"},
            )
        )
        assert module.has_stimulus() is True
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.metadata["stimulus_type"] == "priority_event"

    def test_chapter_event_fires_while_busy(self):
        module = self._make_module(idle=False)
        module._gameplay_event_buffer.append(
            GameEvent(
                event_type="chapter_status_changed",
                event_source="direct_hook",
                timestamp="2026-05-06T10:00:00Z",
                sequence=1,
                payload={"chapter_hash": "0xABC", "chapter_name": "Chapter 2"},
            )
        )
        assert module.has_stimulus() is True


class TestNANO123StaleLineCombatBundling:
    """NANO-123 Phase 2: Combat snapshot still bundles with fresh combat line."""

    def test_fresh_combat_line_bundles_snapshot(self):
        from spindl.stimuli.game_state.dialogue_buffer import DialogueLine, GameplaySnapshot
        module = GameStateModule(
            enabled=True,
            gameplay_enabled=True,
            is_engine_idle=lambda: True,
        )
        module._running = True
        module._current_snapshot = {
            "hp_ratio": 0.5,
            "weapon_name": "Grip Gun",
            "in_combat": True,
            "is_dead": False,
        }
        module._dialogue_buffer._buffer.append(
            DialogueLine(
                speaker="Diana", text="Hacking!", source="chatter",
                event_source="direct_hook",
                timestamp="2026-05-06T10:00:00Z", sequence=1,
                gameplay_context=GameplaySnapshot(combat_active=True),
            )
        )
        module._has_fresh_trigger = True
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "Current gameplay state" in stimulus.user_input
        assert "Grip Gun" in stimulus.user_input
