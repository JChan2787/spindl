"""
Codex activation engine - keyword matching and entry activation logic.

Implements SillyTavern-compatible lorebook activation with:
- Substring matching (default)
- Regex support (/pattern/ or /pattern/i format)
- Whole-word matching
- Case sensitivity toggle
- Secondary key logic (AND ANY, AND ALL, NOT ANY, NOT ALL)
- Timed effects (sticky, cooldown, delay)
"""

import re
from typing import TYPE_CHECKING

from spindl.codex.models import (
    ActivationResult,
    CodexState,
    SelectiveLogic,
)

if TYPE_CHECKING:
    from spindl.characters.models import CharacterBookEntry


def parse_regex(keyword: str) -> tuple[re.Pattern[str], bool] | None:
    """
    Parse a keyword as regex if it matches /pattern/ or /pattern/flags format.

    Args:
        keyword: The keyword to check

    Returns:
        Tuple of (compiled pattern, success) or None if not a regex
    """
    if not keyword.startswith("/"):
        return None

    # Check for /pattern/flags format
    # Find the last / that isn't escaped
    last_slash = keyword.rfind("/")
    if last_slash <= 0:
        return None

    pattern = keyword[1:last_slash]
    flags_str = keyword[last_slash + 1 :]

    # Parse flags
    flags = 0
    for char in flags_str:
        if char == "i":
            flags |= re.IGNORECASE
        elif char == "m":
            flags |= re.MULTILINE
        elif char == "s":
            flags |= re.DOTALL
        # Ignore unknown flags

    try:
        return (re.compile(pattern, flags), True)
    except re.error:
        return None


def match_keyword(
    text: str,
    keyword: str,
    case_sensitive: bool = False,
    match_whole_words: bool = False,
) -> bool:
    """
    Check if a keyword matches in the given text.

    Matching modes (in precedence order):
    1. Regex: If keyword is /pattern/ or /pattern/flags format
    2. Whole word: If match_whole_words is True
    3. Substring: Default simple contains check

    Args:
        text: The text to search in
        keyword: The keyword to search for
        case_sensitive: Whether matching is case-sensitive
        match_whole_words: Whether to match whole words only

    Returns:
        True if keyword matches in text
    """
    # Empty keyword never matches
    if not keyword:
        return False

    # Try regex first
    regex_result = parse_regex(keyword)
    if regex_result is not None:
        pattern, _ = regex_result
        return bool(pattern.search(text))

    # Apply case sensitivity
    search_text = text if case_sensitive else text.lower()
    search_keyword = keyword if case_sensitive else keyword.lower()

    # Whole word matching
    if match_whole_words:
        # Word boundary pattern
        escaped = re.escape(search_keyword)
        pattern = rf"(?:^|\W)({escaped})(?:$|\W)"
        return bool(re.search(pattern, search_text))

    # Simple substring
    return search_keyword in search_text


def check_primary_keys(
    text: str,
    entry: "CharacterBookEntry",
    match_whole_words: bool = False,
) -> tuple[bool, str | None]:
    """
    Check if any primary key matches.

    Args:
        text: Text to search in
        entry: The codex entry
        match_whole_words: Whether to match whole words only

    Returns:
        Tuple of (matched, matched_keyword)
    """
    case_sensitive = entry.case_sensitive or False

    for key in entry.keys:
        if match_keyword(text, key, case_sensitive, match_whole_words):
            return (True, key)

    return (False, None)


def check_secondary_keys(
    text: str,
    entry: "CharacterBookEntry",
    primary_matched: bool,
    match_whole_words: bool = False,
) -> bool:
    """
    Apply secondary key logic to determine final activation.

    Secondary key logic (when entry.selective is True):
    - AND_ANY: Primary AND (any secondary matches)
    - AND_ALL: Primary AND (all secondaries match)
    - NOT_ANY: Primary AND NOT (any secondary matches)
    - NOT_ALL: Primary AND NOT (all secondaries match)

    Args:
        text: Text to search in
        entry: The codex entry
        primary_matched: Whether primary keys matched
        match_whole_words: Whether to match whole words only

    Returns:
        Whether entry should activate
    """
    # If not selective or no secondary keys, just use primary result
    if not entry.selective or not entry.secondary_keys:
        return primary_matched

    # Primary must match for any secondary logic to apply
    if not primary_matched:
        return False

    case_sensitive = entry.case_sensitive or False

    # Check which secondary keys match
    secondary_matches = [
        match_keyword(text, key, case_sensitive, match_whole_words)
        for key in entry.secondary_keys
    ]

    # Determine logic operator from extensions or default to AND_ANY
    logic = SelectiveLogic.AND_ANY
    if entry.extensions and "selective_logic" in entry.extensions:
        try:
            logic = SelectiveLogic(entry.extensions["selective_logic"])
        except ValueError:
            pass  # Use default

    # Apply logic
    if logic == SelectiveLogic.AND_ANY:
        return any(secondary_matches)
    elif logic == SelectiveLogic.AND_ALL:
        return all(secondary_matches)
    elif logic == SelectiveLogic.NOT_ANY:
        return not any(secondary_matches)
    elif logic == SelectiveLogic.NOT_ALL:
        return not all(secondary_matches)

    return primary_matched


def can_activate_entry(
    entry: "CharacterBookEntry",
    state: CodexState,
    entry_id: int,
) -> tuple[bool, str]:
    """
    Check if an entry can activate based on timed effects.

    Args:
        entry: The codex entry
        state: Current codex state
        entry_id: ID of the entry

    Returns:
        Tuple of (can_activate, reason)
    """
    current_turn = state.current_turn
    entry_state = state.get_entry_state(entry_id)

    # Delay: can't activate until N turns into conversation
    if entry.delay is not None and entry.delay > 0:
        if current_turn < entry.delay:
            return (False, "blocked_delay")

    # Cooldown: can't activate for N turns after last activation
    if entry_state.is_on_cooldown(current_turn):
        return (False, "blocked_cooldown")

    return (True, "allowed")


def activate_entry(
    text: str,
    entry: "CharacterBookEntry",
    entry_id: int,
    state: CodexState,
    match_whole_words: bool = False,
) -> ActivationResult:
    """
    Attempt to activate a single codex entry.

    Args:
        text: Text to search for keywords
        entry: The codex entry to check
        entry_id: ID of the entry (for state tracking)
        state: Current codex state
        match_whole_words: Whether to match whole words only

    Returns:
        ActivationResult with activation details
    """
    # Base result for non-activation
    base_result = ActivationResult(
        entry_id=entry_id,
        entry_name=entry.name,
        activated=False,
        content=entry.content,
        reason="no_match",
        position=entry.position or "after_char",
        priority=entry.priority or 0,
        insertion_order=entry.insertion_order,
    )

    # Skip disabled entries
    if not entry.enabled:
        base_result.reason = "disabled"
        return base_result

    # Check timed effects first
    can_activate, reason = can_activate_entry(entry, state, entry_id)

    # Constant entries always activate (within budget), ignoring keywords
    if entry.constant:
        if not can_activate:
            base_result.reason = reason
            return base_result

        # Activate constant entry
        entry_state = state.get_entry_state(entry_id)
        entry_state.activate(
            state.current_turn,
            sticky=entry.sticky,
            cooldown=entry.cooldown,
        )
        return ActivationResult(
            entry_id=entry_id,
            entry_name=entry.name,
            activated=True,
            content=entry.content,
            reason="constant",
            position=entry.position or "after_char",
            priority=entry.priority or 0,
            insertion_order=entry.insertion_order,
        )

    # Check if sticky is still active (even without new keyword match)
    entry_state = state.get_entry_state(entry_id)
    if entry_state.is_sticky_active(state.current_turn):
        return ActivationResult(
            entry_id=entry_id,
            entry_name=entry.name,
            activated=True,
            content=entry.content,
            reason="sticky_active",
            position=entry.position or "after_char",
            priority=entry.priority or 0,
            insertion_order=entry.insertion_order,
        )

    # Can't activate due to timed effects
    if not can_activate:
        base_result.reason = reason
        return base_result

    # Check primary keys
    primary_matched, matched_keyword = check_primary_keys(
        text, entry, match_whole_words
    )

    # Check secondary keys
    final_match = check_secondary_keys(
        text, entry, primary_matched, match_whole_words
    )

    if not final_match:
        return base_result

    # Activate!
    entry_state.activate(
        state.current_turn,
        sticky=entry.sticky,
        cooldown=entry.cooldown,
    )

    return ActivationResult(
        entry_id=entry_id,
        entry_name=entry.name,
        activated=True,
        content=entry.content,
        reason="keyword_match",
        matched_keyword=matched_keyword,
        position=entry.position or "after_char",
        priority=entry.priority or 0,
        insertion_order=entry.insertion_order,
    )


def activate_entries(
    text: str,
    entries: list["CharacterBookEntry"],
    state: CodexState,
    match_whole_words: bool = False,
    max_entries: int | None = None,
) -> list[ActivationResult]:
    """
    Process all entries and return activated ones.

    Results are sorted by:
    1. insertion_order (ascending - lower = earlier in prompt)
    2. priority (descending - higher = kept if over budget)

    Args:
        text: Text to search for keywords
        entries: List of codex entries to check
        state: Current codex state
        match_whole_words: Whether to match whole words only
        max_entries: Maximum number of entries to activate (None = unlimited)

    Returns:
        List of ActivationResults for activated entries only
    """
    results: list[ActivationResult] = []

    for i, entry in enumerate(entries):
        # Use entry.id if available, otherwise use list index
        entry_id = entry.id if entry.id is not None else i

        result = activate_entry(text, entry, entry_id, state, match_whole_words)

        if result.activated:
            results.append(result)

    # Sort by insertion_order (ascending), then priority (descending)
    results.sort(key=lambda r: (r.insertion_order, -r.priority))

    # Apply max entries limit
    if max_entries is not None and len(results) > max_entries:
        # Keep highest priority entries
        results.sort(key=lambda r: -r.priority)
        results = results[:max_entries]
        # Re-sort by insertion order for final output
        results.sort(key=lambda r: (r.insertion_order, -r.priority))

    return results
