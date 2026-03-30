"""Tests for codex activation engine."""

import pytest

from spindl.characters.models import CharacterBookEntry
from spindl.codex.activation import (
    activate_entries,
    activate_entry,
    check_primary_keys,
    check_secondary_keys,
    match_keyword,
    parse_regex,
)
from spindl.codex.models import CodexState, SelectiveLogic


class TestParseRegex:
    """Tests for regex parsing."""

    def test_simple_regex(self):
        """Parse simple /pattern/ format."""
        result = parse_regex("/hello/")
        assert result is not None
        pattern, _ = result
        assert pattern.search("hello world")
        assert not pattern.search("goodbye")

    def test_regex_with_case_flag(self):
        """Parse /pattern/i for case insensitive."""
        result = parse_regex("/hello/i")
        assert result is not None
        pattern, _ = result
        assert pattern.search("HELLO world")
        assert pattern.search("Hello")

    def test_regex_with_multiline_flag(self):
        """Parse /pattern/m for multiline."""
        result = parse_regex("/^test/m")
        assert result is not None
        pattern, _ = result
        assert pattern.search("first\ntest line")

    def test_invalid_regex(self):
        """Invalid regex returns None."""
        result = parse_regex("/[invalid/")
        assert result is None

    def test_not_regex(self):
        """Non-regex string returns None."""
        assert parse_regex("hello") is None
        assert parse_regex("/partial") is None
        assert parse_regex("partial/") is None


class TestMatchKeyword:
    """Tests for keyword matching."""

    def test_simple_substring(self):
        """Simple substring matching."""
        assert match_keyword("hello world", "world")
        assert match_keyword("hello world", "llo wor")
        assert not match_keyword("hello world", "foo")

    def test_case_insensitive_default(self):
        """Matching is case insensitive by default."""
        assert match_keyword("Hello World", "hello")
        assert match_keyword("Hello World", "WORLD")

    def test_case_sensitive(self):
        """Case sensitive matching when enabled."""
        assert match_keyword("Hello World", "Hello", case_sensitive=True)
        assert not match_keyword("Hello World", "hello", case_sensitive=True)

    def test_whole_word_matching(self):
        """Whole word matching with word boundaries."""
        # Should match whole word
        assert match_keyword("hello world", "hello", match_whole_words=True)
        assert match_keyword("hello world", "world", match_whole_words=True)

        # Should not match partial word
        assert not match_keyword("helloworld", "hello", match_whole_words=True)
        assert not match_keyword("worldly", "world", match_whole_words=True)

        # Word at start/end
        assert match_keyword("world", "world", match_whole_words=True)
        assert match_keyword("the world!", "world", match_whole_words=True)

    def test_regex_keyword(self):
        """Regex pattern in keyword."""
        assert match_keyword("hello world", "/wor.d/")
        assert match_keyword("HELLO", "/hello/i")
        assert not match_keyword("hello", "/^world/")

    def test_empty_keyword(self):
        """Empty keyword never matches."""
        assert not match_keyword("hello world", "")


class TestCheckPrimaryKeys:
    """Tests for primary key checking."""

    def test_single_key_match(self):
        """Match with single key."""
        entry = CharacterBookEntry(keys=["coffee"], content="test")
        matched, keyword = check_primary_keys("I love coffee", entry)
        assert matched
        assert keyword == "coffee"

    def test_multiple_keys_first_match(self):
        """First matching key is returned."""
        entry = CharacterBookEntry(keys=["tea", "coffee", "water"], content="test")
        matched, keyword = check_primary_keys("I love coffee", entry)
        assert matched
        assert keyword == "coffee"

    def test_no_match(self):
        """No keys match."""
        entry = CharacterBookEntry(keys=["tea", "juice"], content="test")
        matched, keyword = check_primary_keys("I love coffee", entry)
        assert not matched
        assert keyword is None

    def test_case_sensitive_keys(self):
        """Case sensitive key matching."""
        entry = CharacterBookEntry(
            keys=["Coffee"],
            content="test",
            case_sensitive=True,
        )
        matched, _ = check_primary_keys("Coffee time", entry)
        assert matched

        matched, _ = check_primary_keys("coffee time", entry)
        assert not matched


class TestCheckSecondaryKeys:
    """Tests for secondary key logic."""

    def test_no_selective_returns_primary(self):
        """Without selective flag, just returns primary result."""
        entry = CharacterBookEntry(
            keys=["coffee"],
            secondary_keys=["morning"],
            content="test",
            selective=False,
        )
        assert check_secondary_keys("coffee", entry, primary_matched=True)
        assert not check_secondary_keys("coffee", entry, primary_matched=False)

    def test_and_any_logic(self):
        """AND ANY: primary AND (any secondary)."""
        entry = CharacterBookEntry(
            keys=["drink"],
            secondary_keys=["coffee", "tea"],
            content="test",
            selective=True,
            extensions={"selective_logic": "AND_ANY"},
        )
        # Primary + any secondary
        assert check_secondary_keys("drink coffee", entry, primary_matched=True)
        assert check_secondary_keys("drink tea", entry, primary_matched=True)
        # Primary without secondary
        assert not check_secondary_keys("drink water", entry, primary_matched=True)

    def test_and_all_logic(self):
        """AND ALL: primary AND (all secondaries)."""
        entry = CharacterBookEntry(
            keys=["drink"],
            secondary_keys=["hot", "caffeinated"],
            content="test",
            selective=True,
            extensions={"selective_logic": "AND_ALL"},
        )
        # Primary + all secondaries
        assert check_secondary_keys(
            "drink hot caffeinated beverage", entry, primary_matched=True
        )
        # Missing one secondary
        assert not check_secondary_keys("drink hot water", entry, primary_matched=True)

    def test_not_any_logic(self):
        """NOT ANY: primary AND NOT (any secondary)."""
        entry = CharacterBookEntry(
            keys=["drink"],
            secondary_keys=["alcohol", "beer"],
            content="test",
            selective=True,
            extensions={"selective_logic": "NOT_ANY"},
        )
        # Primary without any secondary
        assert check_secondary_keys("drink coffee", entry, primary_matched=True)
        # Primary with a secondary (should NOT match)
        assert not check_secondary_keys("drink beer", entry, primary_matched=True)

    def test_not_all_logic(self):
        """NOT ALL: primary AND NOT (all secondaries)."""
        entry = CharacterBookEntry(
            keys=["drink"],
            secondary_keys=["alcohol", "beer"],
            content="test",
            selective=True,
            extensions={"selective_logic": "NOT_ALL"},
        )
        # Only one secondary present (not all)
        assert check_secondary_keys("drink beer only", entry, primary_matched=True)
        # All secondaries present (should NOT match)
        assert not check_secondary_keys(
            "drink alcohol and beer", entry, primary_matched=True
        )


class TestActivateEntry:
    """Tests for single entry activation."""

    def test_basic_activation(self):
        """Entry activates on keyword match."""
        entry = CharacterBookEntry(
            keys=["coffee"],
            content="User likes coffee",
            enabled=True,
            insertion_order=100,
        )
        state = CodexState()
        result = activate_entry("I want coffee", entry, entry_id=1, state=state)

        assert result.activated
        assert result.reason == "keyword_match"
        assert result.matched_keyword == "coffee"
        assert result.content == "User likes coffee"

    def test_disabled_entry(self):
        """Disabled entries don't activate."""
        entry = CharacterBookEntry(
            keys=["coffee"],
            content="test",
            enabled=False,
        )
        state = CodexState()
        result = activate_entry("I want coffee", entry, entry_id=1, state=state)

        assert not result.activated
        assert result.reason == "disabled"

    def test_constant_entry(self):
        """Constant entries always activate."""
        entry = CharacterBookEntry(
            keys=["never_matches"],
            content="Always here",
            enabled=True,
            constant=True,
        )
        state = CodexState()
        result = activate_entry("random text", entry, entry_id=1, state=state)

        assert result.activated
        assert result.reason == "constant"

    def test_delay_effect(self):
        """Entry doesn't activate until delay turns pass."""
        entry = CharacterBookEntry(
            keys=["coffee"],
            content="test",
            enabled=True,
            delay=3,
        )
        state = CodexState()

        # Turn 0, 1, 2 - blocked
        for turn in range(3):
            state.current_turn = turn
            result = activate_entry("coffee", entry, entry_id=1, state=state)
            assert not result.activated
            assert result.reason == "blocked_delay"

        # Turn 3 - allowed
        state.current_turn = 3
        result = activate_entry("coffee", entry, entry_id=1, state=state)
        assert result.activated

    def test_cooldown_effect(self):
        """Entry can't re-activate during cooldown."""
        entry = CharacterBookEntry(
            keys=["coffee"],
            content="test",
            enabled=True,
            cooldown=2,
        )
        state = CodexState()

        # First activation
        result = activate_entry("coffee", entry, entry_id=1, state=state)
        assert result.activated

        # Try again same turn - on cooldown
        result = activate_entry("coffee", entry, entry_id=1, state=state)
        assert not result.activated
        assert result.reason == "blocked_cooldown"

        # Advance turns through cooldown
        state.advance_turn()  # Turn 1
        result = activate_entry("coffee", entry, entry_id=1, state=state)
        assert not result.activated

        state.advance_turn()  # Turn 2
        result = activate_entry("coffee", entry, entry_id=1, state=state)
        assert not result.activated

        state.advance_turn()  # Turn 3 - cooldown expired
        result = activate_entry("coffee", entry, entry_id=1, state=state)
        assert result.activated

    def test_sticky_effect(self):
        """Sticky entries remain active for N turns after trigger."""
        entry = CharacterBookEntry(
            keys=["coffee"],
            content="Sticky content",
            enabled=True,
            sticky=2,  # Active for 2 turns (turns 0 and 1)
        )
        state = CodexState()

        # Turn 0 - Activate via keyword
        result = activate_entry("coffee", entry, entry_id=1, state=state)
        assert result.activated
        assert result.reason == "keyword_match"

        # Turn 1 - sticky still active (no keyword needed)
        state.advance_turn()
        result = activate_entry("random text", entry, entry_id=1, state=state)
        assert result.activated
        assert result.reason == "sticky_active"

        # Turn 2 - sticky expired (was active for 2 turns: 0 and 1)
        state.advance_turn()
        result = activate_entry("other text", entry, entry_id=1, state=state)
        assert not result.activated
        assert result.reason == "no_match"


class TestActivateEntries:
    """Tests for batch entry activation."""

    def test_multiple_entries(self):
        """Activate multiple entries."""
        entries = [
            CharacterBookEntry(
                id=1, keys=["coffee"], content="Coffee info", insertion_order=100
            ),
            CharacterBookEntry(
                id=2, keys=["tea"], content="Tea info", insertion_order=200
            ),
            CharacterBookEntry(
                id=3, keys=["water"], content="Water info", insertion_order=50
            ),
        ]
        state = CodexState()

        results = activate_entries("I like coffee and tea", entries, state)

        assert len(results) == 2
        # Should be sorted by insertion_order
        assert results[0].entry_id == 1  # coffee (100)
        assert results[1].entry_id == 2  # tea (200)

    def test_max_entries_limit(self):
        """Respect max_entries limit, keeping highest priority."""
        entries = [
            CharacterBookEntry(
                id=1, keys=["a"], content="A", priority=1, insertion_order=100
            ),
            CharacterBookEntry(
                id=2, keys=["a"], content="B", priority=3, insertion_order=200
            ),
            CharacterBookEntry(
                id=3, keys=["a"], content="C", priority=2, insertion_order=300
            ),
        ]
        state = CodexState()

        results = activate_entries("a", entries, state, max_entries=2)

        assert len(results) == 2
        # Should keep highest priority (3 and 2), sorted by insertion_order
        entry_ids = [r.entry_id for r in results]
        assert 2 in entry_ids  # priority 3
        assert 3 in entry_ids  # priority 2
        assert 1 not in entry_ids  # priority 1 (dropped)

    def test_position_preserved(self):
        """Entry position is preserved in results."""
        entries = [
            CharacterBookEntry(
                id=1, keys=["before"], content="Before", position="before_char"
            ),
            CharacterBookEntry(
                id=2, keys=["after"], content="After", position="after_char"
            ),
        ]
        state = CodexState()

        results = activate_entries("before and after", entries, state)

        assert len(results) == 2
        before_result = next(r for r in results if r.entry_id == 1)
        after_result = next(r for r in results if r.entry_id == 2)
        assert before_result.position == "before_char"
        assert after_result.position == "after_char"
