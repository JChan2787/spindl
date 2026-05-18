"""Tests for DialogueBuffer (NANO-116 Phase B.2)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.game_state.dialogue_buffer import (
    DialogueBuffer,
    DialogueLine,
    GameplaySnapshot,
)


def _make_dialogue_event(
    speaker: str = "Diana",
    text: str = "Watch out!",
    source: str = "chatter",
    seq: int = 1,
    event_source: str = "direct_hook",
    game_id: str = "pragmata",
) -> dict:
    return {
        "protocol_version": "0.1.2",
        "event_type": "dialogue_line",
        "event_source": event_source,
        "timestamp": f"2026-04-25T04:00:{seq:02d}.000Z",
        "sequence": seq,
        "payload": {"speaker": speaker, "text": text, "source": source},
        "game_id": game_id,
    }


def _make_boundary_event(event_type: str = "dialogue_started", seq: int = 0) -> dict:
    return {
        "protocol_version": "0.1.2",
        "event_type": event_type,
        "event_source": "direct_hook",
        "timestamp": "2026-04-25T04:00:00.000Z",
        "sequence": seq,
        "payload": {"speaker_name": "Diana", "conversation_id": "abc123"},
    }


class TestDialogueBufferBasics:
    """Basic buffer properties and operations."""

    def test_empty_buffer(self):
        buf = DialogueBuffer(max_size=10)
        assert buf.count == 0
        assert buf.drain() == []
        assert buf.peek() == []

    def test_accept_dialogue_line(self):
        buf = DialogueBuffer()
        result = buf.accept_event(_make_dialogue_event())
        assert result is not None
        assert result.speaker == "Diana"
        assert result.text == "Watch out!"
        assert buf.count == 1

    def test_boundary_events_not_buffered(self):
        buf = DialogueBuffer()
        result = buf.accept_event(_make_boundary_event("dialogue_started"))
        assert result is None
        assert buf.count == 0

        result = buf.accept_event(_make_boundary_event("dialogue_ended"))
        assert result is None
        assert buf.count == 0

    def test_non_dialogue_events_ignored(self):
        buf = DialogueBuffer()
        event = _make_dialogue_event()
        event["event_type"] = "snapshot"
        result = buf.accept_event(event)
        assert result is None
        assert buf.count == 0

    def test_empty_speaker_dropped(self):
        buf = DialogueBuffer()
        event = _make_dialogue_event(speaker="")
        result = buf.accept_event(event)
        assert result is None
        assert buf.count == 0

    def test_empty_text_dropped(self):
        buf = DialogueBuffer()
        event = _make_dialogue_event(text="")
        result = buf.accept_event(event)
        assert result is None
        assert buf.count == 0


class TestDialogueBufferDedup:
    """Per-line dedup: same speaker + same text collapses."""

    def test_same_speaker_text_collapses(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=1))
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=2))
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=3))

        assert buf.count == 1
        lines = buf.peek()
        assert lines[0].repeat_count == 3
        assert lines[0].sequence == 3  # Updated to latest

    def test_different_text_no_collapse(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=1))
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Stay close!", seq=2))

        assert buf.count == 2

    def test_different_speaker_no_collapse(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=1))
        buf.accept_event(_make_dialogue_event(speaker="Ken", text="Watch out!", seq=2))

        assert buf.count == 2

    def test_dedup_only_collapses_consecutive(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=1))
        buf.accept_event(_make_dialogue_event(speaker="Ken", text="Roger!", seq=2))
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=3))

        assert buf.count == 3  # Not collapsed — Ken's line in between


class TestDialogueBufferFIFO:
    """FIFO eviction when buffer is full."""

    def test_fifo_eviction(self):
        buf = DialogueBuffer(max_size=3)
        for i in range(5):
            buf.accept_event(_make_dialogue_event(
                speaker=f"NPC_{i}", text=f"Line {i}", seq=i
            ))

        assert buf.count == 3
        lines = buf.peek()
        assert lines[0].speaker == "NPC_2"
        assert lines[2].speaker == "NPC_4"

    def test_resize_preserves_events(self):
        buf = DialogueBuffer(max_size=5)
        for i in range(3):
            buf.accept_event(_make_dialogue_event(speaker=f"NPC_{i}", text=f"L{i}", seq=i))

        assert buf.count == 3
        buf.max_size = 10
        assert buf.count == 3
        assert buf.max_size == 10


class TestDialogueBufferGameplayContext:
    """Gameplay snapshot stapling at capture time."""

    def test_snapshot_stapled_at_capture_time(self):
        buf = DialogueBuffer()
        buf.update_gameplay_snapshot(
            combat_active=True, enemy_count=3, hp_ratio=0.75, chapter_hash="ch01"
        )

        buf.accept_event(_make_dialogue_event(seq=1))
        lines = buf.peek()
        ctx = lines[0].gameplay_context

        assert ctx.combat_active is True
        assert ctx.enemy_count == 3
        assert ctx.hp_ratio == 0.75
        assert ctx.chapter_hash == "ch01"

    def test_snapshot_is_point_in_time(self):
        buf = DialogueBuffer()
        buf.update_gameplay_snapshot(combat_active=True, enemy_count=3)

        buf.accept_event(_make_dialogue_event(seq=1))

        # Update snapshot AFTER first line
        buf.update_gameplay_snapshot(combat_active=False, enemy_count=0)

        buf.accept_event(_make_dialogue_event(speaker="Ken", text="Clear!", seq=2))

        lines = buf.peek()
        assert lines[0].gameplay_context.combat_active is True
        assert lines[0].gameplay_context.enemy_count == 3
        assert lines[1].gameplay_context.combat_active is False
        assert lines[1].gameplay_context.enemy_count == 0


class TestDialogueBufferDrain:
    """Drain returns all lines and clears buffer."""

    def test_drain_returns_and_clears(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Line 1", seq=1))
        buf.accept_event(_make_dialogue_event(speaker="Ken", text="Line 2", seq=2))

        lines = buf.drain()
        assert len(lines) == 2
        assert buf.count == 0

    def test_drain_empty_buffer(self):
        buf = DialogueBuffer()
        assert buf.drain() == []


class TestDialogueBufferFormat:
    """Formatting drained lines for stimulus template."""

    def test_basic_format(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=1))
        lines = buf.drain()
        formatted = buf.format_for_stimulus(lines)
        assert "Diana: Watch out!" in formatted

    def test_repeat_count_format(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=1))
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=2))
        buf.accept_event(_make_dialogue_event(speaker="Diana", text="Watch out!", seq=3))
        lines = buf.drain()
        formatted = buf.format_for_stimulus(lines)
        assert "(x3)" in formatted

    def test_cinematic_tag(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(
            speaker="Ken", text="Stay close.", source="cinematic", seq=1
        ))
        lines = buf.drain()
        formatted = buf.format_for_stimulus(lines)
        assert "[cinematic]" in formatted

    def test_chatter_no_tag(self):
        buf = DialogueBuffer()
        buf.accept_event(_make_dialogue_event(
            speaker="Diana", text="Watch out!", source="chatter", seq=1
        ))
        lines = buf.drain()
        formatted = buf.format_for_stimulus(lines)
        assert "[cinematic]" not in formatted


class TestIsDialogueEvent:
    """Static method for event type routing."""

    def test_dialogue_events(self):
        assert DialogueBuffer.is_dialogue_event("dialogue_line") is True
        assert DialogueBuffer.is_dialogue_event("dialogue_started") is True
        assert DialogueBuffer.is_dialogue_event("dialogue_ended") is True

    def test_non_dialogue_events(self):
        assert DialogueBuffer.is_dialogue_event("snapshot") is False
        assert DialogueBuffer.is_dialogue_event("enemy_engaged") is False
        assert DialogueBuffer.is_dialogue_event("bridge_ready") is False


class TestGameplaySnapshot:
    """GameplaySnapshot serialization."""

    def test_to_dict_minimal(self):
        snap = GameplaySnapshot()
        d = snap.to_dict()
        assert "combat_active" in d
        assert d["combat_active"] is False
        assert "enemy_count" not in d  # Only included when combat_active

    def test_to_dict_combat(self):
        snap = GameplaySnapshot(
            combat_active=True, enemy_count=3, hp_ratio=0.756, chapter_hash="ch01"
        )
        d = snap.to_dict()
        assert d["combat_active"] is True
        assert d["enemy_count"] == 3
        assert d["hp_ratio"] == 0.76  # Rounded
        assert d["chapter_hash"] == "ch01"
