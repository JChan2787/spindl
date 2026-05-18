"""Tests for DialogueStore (NANO-116 Phase B.2)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.game_state.dialogue_buffer import DialogueLine, GameplaySnapshot
from spindl.stimuli.game_state.dialogue_store import DialogueStore


def _make_line(
    speaker: str = "Diana",
    text: str = "Watch out!",
    source: str = "chatter",
    seq: int = 1,
) -> DialogueLine:
    return DialogueLine(
        speaker=speaker,
        text=text,
        source=source,
        event_source="direct_hook",
        timestamp="2026-04-25T04:00:00.000Z",
        sequence=seq,
        gameplay_context=GameplaySnapshot(combat_active=True, enemy_count=2),
        game_id="pragmata",
    )


class TestDialogueStoreBasics:
    """Basic store operations."""

    def test_unbound_store_returns_negative(self):
        store = DialogueStore()
        assert store.record_dialogue_line(_make_line()) == -1
        assert store.has_content is False

    def test_ensure_store_creates_file(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()

        store.ensure_store(voice_file)
        assert store.store_file is not None
        assert store.store_file.name == "spindle_20260425_043900.dialogue.jsonl"

    def test_record_line_persists(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        turn_id = store.record_dialogue_line(_make_line())
        assert turn_id == 1
        assert store.has_content is True
        assert store.turn_count == 1

        # Verify on disk
        content = store.store_file.read_text(encoding="utf-8").strip()
        entry = json.loads(content)
        assert entry["role"] == "dialogue"
        assert entry["speaker"] == "Diana"
        assert entry["text"] == "Watch out!"
        assert entry["gameplay_context"]["combat_active"] is True


class TestDialogueStoreAssistantReply:
    """Dual-write assistant replies."""

    def test_record_assistant_reply(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        store.record_dialogue_line(_make_line())
        turn_id = store.record_assistant_reply("Diana seems worried!", [1])

        assert turn_id == 2
        assert store.turn_count == 2

        # Verify on disk
        lines = store.store_file.read_text(encoding="utf-8").strip().split("\n")
        assistant_entry = json.loads(lines[1])
        assert assistant_entry["role"] == "assistant"
        assert assistant_entry["responding_to_lines"] == [1]


class TestDialogueStoreSummary:
    """Summary blob persistence and reconstruction."""

    def test_record_summary(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        store.record_dialogue_line(_make_line(speaker="Diana", text="L1", seq=1))
        store.record_dialogue_line(_make_line(speaker="Ken", text="L2", seq=2))
        store.record_summary("Diana is Hugh's partner. Ken is cautious.")

        assert store.summary_blob == "Diana is Hugh's partner. Ken is cautious."
        assert store.summary_version == 1
        assert store.summarized_through_turn_id == 2

    def test_rolling_summary_supersedes(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        store.record_dialogue_line(_make_line(speaker="Diana", text="L1", seq=1))
        store.record_summary("Summary v1")

        store.record_dialogue_line(_make_line(speaker="Ken", text="L2", seq=2))
        store.record_summary("Summary v2 — more context")

        assert store.summary_version == 2
        assert store.summary_blob == "Summary v2 — more context"

    def test_unsummarized_lines(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        store.record_dialogue_line(_make_line(speaker="Diana", text="L1", seq=1))
        store.record_dialogue_line(_make_line(speaker="Ken", text="L2", seq=2))
        store.record_summary("Summary covering L1 and L2")

        store.record_dialogue_line(_make_line(speaker="Diana", text="L3", seq=3))

        unsummarized = store.get_unsummarized_lines()
        assert len(unsummarized) == 1
        assert unsummarized[0]["speaker"] == "Diana"
        assert unsummarized[0]["text"] == "L3"


class TestDialogueStoreRestart:
    """Reconstruction from disk on restart."""

    def test_restart_loads_summary_and_tail(self, tmp_path):
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()

        # First session: write data
        store1 = DialogueStore(conversations_dir=str(tmp_path))
        store1.ensure_store(voice_file)
        store1.record_dialogue_line(_make_line(speaker="Diana", text="L1", seq=1))
        store1.record_dialogue_line(_make_line(speaker="Ken", text="L2", seq=2))
        store1.record_summary("Diana and Ken are allies.")
        store1.record_dialogue_line(_make_line(speaker="Diana", text="L3", seq=3))

        # Second session: reconstruct
        store2 = DialogueStore(conversations_dir=str(tmp_path))
        store2.ensure_store(voice_file)

        assert store2.summary_blob == "Diana and Ken are allies."
        assert store2.summary_version == 1
        assert store2.summarized_through_turn_id == 2  # L1=1, L2=2, summary covers through 2
        assert store2.turn_count == 3  # 3 dialogue turns in LRU (summary not in LRU)
        assert len(store2.get_unsummarized_lines()) == 1  # L3 (turn_id=4) after summary

    def test_restart_no_summary(self, tmp_path):
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()

        store1 = DialogueStore(conversations_dir=str(tmp_path))
        store1.ensure_store(voice_file)
        store1.record_dialogue_line(_make_line(speaker="Diana", text="L1", seq=1))
        store1.record_dialogue_line(_make_line(speaker="Ken", text="L2", seq=2))

        store2 = DialogueStore(conversations_dir=str(tmp_path))
        store2.ensure_store(voice_file)

        assert store2.summary_blob == ""
        assert store2.summary_version == 0
        assert store2.turn_count == 2


class TestDialogueStoreInjection:
    """Injection content generation."""

    def test_under_budget_returns_raw(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        store.record_dialogue_line(_make_line(speaker="Diana", text="Watch out!", seq=1))
        store.record_dialogue_line(_make_line(speaker="Ken", text="Stay close.", seq=2))

        content = store.get_injection_content(token_budget=5000)
        assert "Diana: Watch out!" in content
        assert "Ken: Stay close." in content

    def test_over_budget_with_summary(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        # Fill with enough lines to exceed a small budget
        for i in range(20):
            store.record_dialogue_line(
                _make_line(speaker=f"NPC_{i}", text=f"Long dialogue line {i} " * 5, seq=i)
            )
        store.record_summary("Summary of 20 NPC conversations.")
        store.record_dialogue_line(_make_line(speaker="Diana", text="New line after summary", seq=20))

        content = store.get_injection_content(token_budget=200)
        assert "Accumulated character knowledge from dialogue:" in content
        assert "Summary of 20 NPC conversations." in content
        assert "New line after summary" in content

    def test_empty_store_returns_empty(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        assert store.get_injection_content() == ""

    def test_needs_summarization(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        assert store.needs_summarization(token_budget=100) is False

        for i in range(10):
            store.record_dialogue_line(
                _make_line(speaker=f"NPC_{i}", text=f"Long line {i} " * 10, seq=i)
            )

        assert store.needs_summarization(token_budget=100) is True
        assert store.needs_summarization(token_budget=100000) is False


class TestDialogueStoreGlobSafety:
    """Ensure .dialogue.jsonl naming doesn't match session listing globs."""

    def test_filename_contains_dialogue_marker(self, tmp_path):
        store = DialogueStore(conversations_dir=str(tmp_path))
        voice_file = tmp_path / "spindle_20260425_043900.jsonl"
        voice_file.touch()
        store.ensure_store(voice_file)

        assert ".dialogue." in store.store_file.name
        # This is the Lesson 14 assertion — the sidecar marker is present
        # so glob exclusion filters in jsonl_store.py and server_sessions.py
        # will skip it.
