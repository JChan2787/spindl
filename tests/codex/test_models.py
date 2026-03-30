"""Tests for codex state models."""

import pytest

from spindl.codex.models import (
    ActivationResult,
    CodexState,
    EntryState,
    SelectiveLogic,
)


class TestEntryState:
    """Tests for EntryState."""

    def test_initial_state(self):
        """New entry state has no activations or effects."""
        state = EntryState(entry_id=1)
        assert state.last_activated_turn is None
        assert state.sticky_until_turn is None
        assert state.cooldown_until_turn is None
        assert not state.is_on_cooldown(0)
        assert not state.is_sticky_active(0)

    def test_cooldown_tracking(self):
        """Cooldown is tracked correctly."""
        state = EntryState(entry_id=1)

        # Activate with 2-turn cooldown at turn 0
        state.activate(current_turn=0, sticky=None, cooldown=2)

        # Cooldown until turn = 0 + 2 + 1 = 3
        assert state.is_on_cooldown(0)  # Same turn
        assert state.is_on_cooldown(1)
        assert state.is_on_cooldown(2)
        assert not state.is_on_cooldown(3)  # Cooldown expired

    def test_sticky_tracking(self):
        """Sticky effect is tracked correctly."""
        state = EntryState(entry_id=1)

        # Activate with 2-turn sticky at turn 0
        state.activate(current_turn=0, sticky=2, cooldown=None)

        # Sticky until turn = 0 + 2 = 2
        assert state.is_sticky_active(0)
        assert state.is_sticky_active(1)
        assert not state.is_sticky_active(2)  # Sticky expired

    def test_activation_records_turn(self):
        """Activation records the turn number."""
        state = EntryState(entry_id=1)
        state.activate(current_turn=5, sticky=None, cooldown=None)
        assert state.last_activated_turn == 5


class TestCodexState:
    """Tests for CodexState."""

    def test_initial_state(self):
        """New codex state starts at turn 0."""
        state = CodexState()
        assert state.current_turn == 0
        assert len(state.entry_states) == 0

    def test_get_entry_state_creates(self):
        """Getting entry state creates it if missing."""
        state = CodexState()
        entry_state = state.get_entry_state(42)
        assert entry_state.entry_id == 42
        assert 42 in state.entry_states

    def test_advance_turn(self):
        """Advancing turn increments counter."""
        state = CodexState()
        assert state.current_turn == 0
        state.advance_turn()
        assert state.current_turn == 1
        state.advance_turn()
        assert state.current_turn == 2

    def test_reset(self):
        """Reset clears all state."""
        state = CodexState()
        state.advance_turn()
        state.advance_turn()
        state.get_entry_state(1).activate(1, sticky=3, cooldown=2)

        state.reset()

        assert state.current_turn == 0
        assert len(state.entry_states) == 0

    def test_get_active_sticky_entries(self):
        """Get list of entries with active sticky effect."""
        state = CodexState()
        state.current_turn = 0

        # Entry 1: sticky until turn 2
        state.get_entry_state(1).activate(0, sticky=2, cooldown=None)
        # Entry 2: sticky until turn 1
        state.get_entry_state(2).activate(0, sticky=1, cooldown=None)
        # Entry 3: no sticky
        state.get_entry_state(3).activate(0, sticky=None, cooldown=None)

        active = state.get_active_sticky_entries()
        assert 1 in active
        assert 2 in active
        assert 3 not in active

        # Advance to turn 1, entry 2's sticky expires
        state.advance_turn()
        active = state.get_active_sticky_entries()
        assert 1 in active
        assert 2 not in active

    def test_get_entries_on_cooldown(self):
        """Get list of entries on cooldown."""
        state = CodexState()
        state.current_turn = 0

        # Entry 1: cooldown until turn 3
        state.get_entry_state(1).activate(0, sticky=None, cooldown=2)
        # Entry 2: no cooldown
        state.get_entry_state(2).activate(0, sticky=None, cooldown=None)

        on_cooldown = state.get_entries_on_cooldown()
        assert 1 in on_cooldown
        assert 2 not in on_cooldown

    def test_to_dict(self):
        """Serialize state to dict."""
        state = CodexState()
        state.current_turn = 5
        state.get_entry_state(1).activate(4, sticky=2, cooldown=3)

        data = state.to_dict()

        assert data["current_turn"] == 5
        assert 1 in data["active_sticky"]
        assert 1 in data["on_cooldown"]
        assert data["entry_count"] == 1


class TestActivationResult:
    """Tests for ActivationResult."""

    def test_basic_result(self):
        """Create basic activation result."""
        result = ActivationResult(
            entry_id=1,
            entry_name="Test Entry",
            activated=True,
            content="Entry content",
            reason="keyword_match",
            matched_keyword="coffee",
        )

        assert result.entry_id == 1
        assert result.entry_name == "Test Entry"
        assert result.activated
        assert result.content == "Entry content"
        assert result.reason == "keyword_match"
        assert result.matched_keyword == "coffee"
        assert result.position == "after_char"  # Default
        assert result.priority == 0  # Default

    def test_result_with_position(self):
        """Result with custom position."""
        result = ActivationResult(
            entry_id=1,
            entry_name=None,
            activated=True,
            content="test",
            reason="constant",
            position="before_char",
        )

        assert result.position == "before_char"


class TestSelectiveLogic:
    """Tests for SelectiveLogic enum."""

    def test_enum_values(self):
        """Enum has expected values."""
        assert SelectiveLogic.AND_ANY.value == "AND_ANY"
        assert SelectiveLogic.AND_ALL.value == "AND_ALL"
        assert SelectiveLogic.NOT_ANY.value == "NOT_ANY"
        assert SelectiveLogic.NOT_ALL.value == "NOT_ALL"

    def test_enum_from_string(self):
        """Can create enum from string."""
        assert SelectiveLogic("AND_ANY") == SelectiveLogic.AND_ANY
        assert SelectiveLogic("NOT_ALL") == SelectiveLogic.NOT_ALL
