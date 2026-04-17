"""Tests for ConversationHistoryManager metadata persistence (NANO-075).

Tests cover:
- store_turn persists base fields (user + assistant)
- store_turn persists reasoning (NANO-042, pre-existing)
- store_turn persists input_modality on user turn
- store_turn persists stimulus_source on assistant turn
- store_turn persists activated_codex_entries on assistant turn
- store_turn persists retrieved_memories on assistant turn
- Omitted metadata fields produce no extra keys (backward compat)
- HistoryRecorder.process extracts metadata from PipelineContext
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from spindl.llm.plugins.conversation_history import (
    ConversationHistoryManager,
    HistoryRecorder,
)
from spindl.llm.plugins.base import PipelineContext
from spindl.codex.models import ActivationResult


# ============================================================================
# Helpers
# ============================================================================


def _read_jsonl(filepath: Path) -> list[dict]:
    """Read all lines from a JSONL file."""
    turns = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def _make_manager(tmp_path: Path, persona_id: str = "test") -> ConversationHistoryManager:
    """Create a manager with a fresh session."""
    mgr = ConversationHistoryManager(conversations_dir=str(tmp_path))
    mgr.ensure_session(persona_id)
    return mgr


# ============================================================================
# Tests: store_turn metadata persistence
# ============================================================================


class TestStoreTurnMetadata:
    """Tests for NANO-075 metadata fields in store_turn."""

    def test_base_fields_persisted(self, tmp_path) -> None:
        """User and assistant turns have base fields."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Hello")
        mgr.store_turn("Hi there!")

        turns = _read_jsonl(mgr.session_file)
        assert len(turns) == 2

        user = turns[0]
        assert user["role"] == "user"
        assert user["content"] == "Hello"
        assert "turn_id" in user
        assert "uuid" in user
        assert "timestamp" in user
        assert user["hidden"] is False

        assistant = turns[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "Hi there!"

    def test_reasoning_persisted(self, tmp_path) -> None:
        """Reasoning is stored on assistant turn (NANO-042)."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Think about this")
        mgr.store_turn("My answer", reasoning="I thought carefully...")

        turns = _read_jsonl(mgr.session_file)
        assert turns[1]["reasoning"] == "I thought carefully..."

    def test_input_modality_persisted_on_user_turn(self, tmp_path) -> None:
        """input_modality is stored on the user turn."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Hello")
        mgr.store_turn("Hi", input_modality="VOICE")

        turns = _read_jsonl(mgr.session_file)
        assert turns[0]["input_modality"] == "VOICE"
        # Assistant turn should NOT have input_modality
        assert "input_modality" not in turns[1]

    def test_stimulus_source_persisted_on_assistant_turn(self, tmp_path) -> None:
        """stimulus_source is stored on the assistant turn."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("patience prompt")
        mgr.store_turn("I noticed you're quiet", stimulus_source="patience")

        turns = _read_jsonl(mgr.session_file)
        assert turns[1]["stimulus_source"] == "patience"
        # User turn should NOT have stimulus_source
        assert "stimulus_source" not in turns[0]

    def test_codex_entries_persisted(self, tmp_path) -> None:
        """activated_codex_entries stored on assistant turn."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Tell me about cats")
        entries = [{"name": "cats", "keys": ["cat", "feline"], "activation_method": "keyword"}]
        mgr.store_turn("Cats are great!", activated_codex_entries=entries)

        turns = _read_jsonl(mgr.session_file)
        assert turns[1]["activated_codex_entries"] == entries

    def test_retrieved_memories_persisted(self, tmp_path) -> None:
        """retrieved_memories stored on assistant turn."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Remember when...")
        memories = [{"content_preview": "Last Tuesday...", "collection": "general", "distance": 0.42}]
        mgr.store_turn("Yes, I remember!", retrieved_memories=memories)

        turns = _read_jsonl(mgr.session_file)
        assert turns[1]["retrieved_memories"] == memories

    def test_omitted_metadata_produces_no_extra_keys(self, tmp_path) -> None:
        """When metadata is None/empty, no extra keys appear in JSONL (backward compat)."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Plain message")
        mgr.store_turn("Plain reply")

        turns = _read_jsonl(mgr.session_file)
        user_keys = set(turns[0].keys())
        assistant_keys = set(turns[1].keys())

        # Base keys only
        assert user_keys == {"turn_id", "uuid", "role", "content", "timestamp", "hidden"}
        assert assistant_keys == {"turn_id", "uuid", "role", "content", "timestamp", "hidden"}

    def test_all_metadata_together(self, tmp_path) -> None:
        """All metadata fields persisted on a single exchange."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("stimulus text")
        codex = [{"name": "lore", "keys": ["dragon"], "activation_method": "keyword"}]
        memories = [{"content_preview": "Dragons exist", "collection": "general", "distance": 0.1}]
        mgr.store_turn(
            "Response with everything",
            reasoning="Deep thoughts",
            input_modality="stimulus",
            stimulus_source="patience",
            activated_codex_entries=codex,
            retrieved_memories=memories,
        )

        turns = _read_jsonl(mgr.session_file)
        user = turns[0]
        assistant = turns[1]

        assert user["input_modality"] == "stimulus"
        assert assistant["reasoning"] == "Deep thoughts"
        assert assistant["stimulus_source"] == "patience"
        assert assistant["activated_codex_entries"] == codex
        assert assistant["retrieved_memories"] == memories


# ============================================================================
# Tests: HistoryRecorder extracts metadata from PipelineContext
# ============================================================================


class TestHistoryRecorderMetadataExtraction:
    """Tests that HistoryRecorder.process extracts NANO-075 metadata from context."""

    def test_extracts_input_modality(self, tmp_path) -> None:
        """HistoryRecorder reads input_modality from context.metadata."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Hello")
        recorder = HistoryRecorder(mgr)

        context = PipelineContext(user_input="Hello", persona={"id": "test"})
        context.metadata["input_modality"] = "VOICE"

        recorder.process(context, "Hi!")

        turns = _read_jsonl(mgr.session_file)
        assert turns[0]["input_modality"] == "VOICE"

    def test_extracts_stimulus_source(self, tmp_path) -> None:
        """HistoryRecorder reads stimulus_source from context.metadata."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Are you there?")
        recorder = HistoryRecorder(mgr)

        context = PipelineContext(user_input="Are you there?", persona={"id": "test"})
        context.metadata["stimulus_source"] = "patience"

        recorder.process(context, "Yes!")

        turns = _read_jsonl(mgr.session_file)
        assert turns[1]["stimulus_source"] == "patience"

    def test_extracts_codex_results(self, tmp_path) -> None:
        """HistoryRecorder transforms codex_results into activated_codex_entries."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Tell me about dragons")
        recorder = HistoryRecorder(mgr)

        context = PipelineContext(user_input="Tell me about dragons", persona={"id": "test"})
        context.metadata["codex_results"] = [
            ActivationResult(
                entry_id=1, entry_name="dragon_lore", activated=True,
                content="...", reason="keyword_match", matched_keyword="dragon",
            ),
            ActivationResult(
                entry_id=2, entry_name="inactive_entry", activated=False,
                content="...", reason="no_match",
            ),
        ]

        recorder.process(context, "Dragons are...")

        turns = _read_jsonl(mgr.session_file)
        codex = turns[1]["activated_codex_entries"]
        assert len(codex) == 1  # only activated entries persisted
        assert codex[0]["name"] == "dragon_lore"
        assert codex[0]["keys"] == ["dragon"]
        assert codex[0]["activation_method"] == "keyword_match"
        # content should NOT be persisted (only display fields)
        assert "content" not in codex[0]

    def test_extracts_rag_results(self, tmp_path) -> None:
        """HistoryRecorder transforms rag_results into retrieved_memories."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Remember?")
        recorder = HistoryRecorder(mgr)

        context = PipelineContext(user_input="Remember?", persona={"id": "test"})
        context.metadata["rag_results"] = [
            {"content": "A very long memory content that should be truncated", "collection": "general", "distance": 0.25}
        ]

        recorder.process(context, "I remember!")

        turns = _read_jsonl(mgr.session_file)
        memories = turns[1]["retrieved_memories"]
        assert len(memories) == 1
        assert memories[0]["collection"] == "general"
        assert memories[0]["distance"] == 0.25
        assert len(memories[0]["content_preview"]) <= 100

    def test_no_metadata_produces_clean_turns(self, tmp_path) -> None:
        """When context has no metadata, turns have no extra fields."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Plain")
        recorder = HistoryRecorder(mgr)

        context = PipelineContext(user_input="Plain", persona={"id": "test"})
        recorder.process(context, "Plain reply")

        turns = _read_jsonl(mgr.session_file)
        assert "input_modality" not in turns[0]
        assert "stimulus_source" not in turns[1]
        assert "activated_codex_entries" not in turns[1]
        assert "retrieved_memories" not in turns[1]


# ============================================================================
# Tests: NANO-115 item #4 — empty assistant response guard
# ============================================================================


class TestEmptyAssistantGuard:
    """
    R1 timeouts and similar silent failures produce a blank assistant response.
    Committing that response to history leaves a dangling `[assistant]:` turn
    that pollutes the next LLM turn's context. The guard writes the user turn
    (they did speak) and skips the assistant turn.
    """

    def test_empty_string_writes_user_only(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Did you hear me?")
        mgr.store_turn("")

        turns = _read_jsonl(mgr.session_file)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Did you hear me?"

    def test_whitespace_only_writes_user_only(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Hello?")
        mgr.store_turn("   \n\t  ")

        turns = _read_jsonl(mgr.session_file)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"

    def test_turn_id_advances_by_one_on_empty(self, tmp_path) -> None:
        """Next turn after an empty-response incident uses the next sequential id."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("first")
        mgr.store_turn("")  # timeout — user turn 1 written, assistant skipped

        mgr.stash_user_input("second")
        mgr.store_turn("real reply")  # turn_ids 2 and 3

        turns = _read_jsonl(mgr.session_file)
        assert [t["turn_id"] for t in turns] == [1, 2, 3]
        assert [t["role"] for t in turns] == ["user", "user", "assistant"]

    def test_in_memory_history_matches_jsonl_on_empty(self, tmp_path) -> None:
        """No phantom assistant turn in _history when response is empty."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Anyone there?")
        mgr.store_turn("")

        history = mgr.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_tts_text_empty_but_response_whitespace_skips(self, tmp_path) -> None:
        """When history_content resolves to whitespace (tts_text empty, response blank), skip."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Say something")
        mgr.store_turn("", tts_text="")

        turns = _read_jsonl(mgr.session_file)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"

    def test_non_empty_response_unaffected(self, tmp_path) -> None:
        """Sanity: the guard does not trigger on normal responses."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("Hi")
        mgr.store_turn("Hi back")

        turns = _read_jsonl(mgr.session_file)
        assert len(turns) == 2
        assert turns[1]["content"] == "Hi back"

    def test_pending_user_input_cleared_on_empty(self, tmp_path) -> None:
        """The stash is cleared even on the skip path — prevents leaking into next turn."""
        mgr = _make_manager(tmp_path)
        mgr.stash_user_input("lost in the void")
        mgr.store_turn("")

        assert mgr._pending_user_input is None
