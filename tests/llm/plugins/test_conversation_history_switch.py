"""Tests for ConversationHistoryManager.switch_to_persona (NANO-077).

Tests cover:
- switch_to_persona creates new session file with correct persona prefix
- switch_to_persona to a new persona (no sessions) starts empty
- switch_to_persona preserves old session file on disk
- Switching back to a previous persona resumes the latest session
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from spindl.llm.plugins.conversation_history import ConversationHistoryManager


def _make_manager(tmp_path: Path, persona_id: str = "spindle") -> ConversationHistoryManager:
    """Create a manager with a fresh session."""
    mgr = ConversationHistoryManager(conversations_dir=str(tmp_path))
    mgr.ensure_session(persona_id)
    return mgr


class TestSwitchToPersona:

    def test_new_session_file_has_correct_persona_prefix(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        mgr.switch_to_persona("mryummers")
        assert mgr.session_file is not None
        assert mgr.session_file.name.startswith("mryummers_")

    def test_history_empty_for_new_persona(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        # Add a turn to spindle's session
        mgr.stash_user_input("Hello Spindle")
        mgr.store_turn("Hi there!")
        assert mgr.turn_count == 2

        # mryummers has no prior sessions — should start empty
        mgr.switch_to_persona("mryummers")
        assert mgr.turn_count == 0
        assert mgr.get_history() == []

    def test_old_session_file_preserved(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        mgr.stash_user_input("Hello Spindle")
        mgr.store_turn("Hi there!")
        old_file = mgr.session_file

        mgr.switch_to_persona("mryummers")

        assert old_file.exists()
        # Old file should still contain the turn
        turns = []
        with open(old_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))
        assert len(turns) == 2
        assert turns[0]["role"] == "user"

    def test_switch_back_resumes_latest_session(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        mgr.stash_user_input("Hello Spindle")
        mgr.store_turn("Hi there!")
        first_spindle_file = mgr.session_file

        mgr.switch_to_persona("mryummers")
        mryummers_file = mgr.session_file

        mgr.switch_to_persona("spindle")
        resumed_spindle_file = mgr.session_file

        # Spindle and mryummers should be different
        assert first_spindle_file != mryummers_file
        # Switching back resumes the original spindle session
        assert first_spindle_file == resumed_spindle_file
        # History should be restored
        assert mgr.turn_count == 2
        history = mgr.get_history()
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_pending_input_cleared(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        mgr.stash_user_input("unfinished thought")
        mgr.switch_to_persona("mryummers")
        assert mgr._pending_user_input is None

    def test_next_turn_id_reset_for_new_persona(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        mgr.stash_user_input("Hello")
        mgr.store_turn("Hi!")
        assert mgr._next_turn_id == 3

        # New persona with no sessions — turn_id starts at 1
        mgr.switch_to_persona("mryummers")
        assert mgr._next_turn_id == 1

    def test_next_turn_id_restored_on_resume(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        mgr.stash_user_input("Hello")
        mgr.store_turn("Hi!")
        assert mgr._next_turn_id == 3

        mgr.switch_to_persona("mryummers")
        mgr.switch_to_persona("spindle")
        # Should resume from where we left off
        assert mgr._next_turn_id == 3

    def test_persona_id_updated(self, tmp_path) -> None:
        mgr = _make_manager(tmp_path, "spindle")
        assert mgr._persona_id == "spindle"
        mgr.switch_to_persona("mryummers")
        assert mgr._persona_id == "mryummers"
