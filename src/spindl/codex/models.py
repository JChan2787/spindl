"""
Codex activation models for tracking entry state.

These models track the runtime state of codex entries during a conversation,
including sticky effects, cooldowns, and activation history.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SelectiveLogic(str, Enum):
    """Logic operators for secondary key matching."""

    AND_ANY = "AND_ANY"  # Primary AND (any secondary)
    AND_ALL = "AND_ALL"  # Primary AND (all secondaries)
    NOT_ANY = "NOT_ANY"  # Primary AND NOT (any secondary)
    NOT_ALL = "NOT_ALL"  # Primary AND NOT (all secondaries)


@dataclass
class EntryState:
    """
    Runtime state for a single codex entry.

    Tracks activation history and timed effects (sticky, cooldown).
    This state persists across the conversation but resets on session restart.
    """

    entry_id: int
    last_activated_turn: int | None = None  # Turn index when last activated
    sticky_until_turn: int | None = None  # Entry remains active until this turn
    cooldown_until_turn: int | None = None  # Entry cannot activate until this turn

    def is_on_cooldown(self, current_turn: int) -> bool:
        """Check if entry is currently on cooldown."""
        if self.cooldown_until_turn is None:
            return False
        return current_turn < self.cooldown_until_turn

    def is_sticky_active(self, current_turn: int) -> bool:
        """Check if entry is still active from a previous sticky trigger."""
        if self.sticky_until_turn is None:
            return False
        return current_turn < self.sticky_until_turn

    def activate(self, current_turn: int, sticky: int | None, cooldown: int | None) -> None:
        """
        Mark entry as activated and set timed effects.

        Args:
            current_turn: Current conversation turn index
            sticky: Number of turns to remain active (None = no sticky)
            cooldown: Number of turns before re-activation allowed (None = no cooldown)
        """
        self.last_activated_turn = current_turn

        if sticky is not None and sticky > 0:
            self.sticky_until_turn = current_turn + sticky

        if cooldown is not None and cooldown > 0:
            self.cooldown_until_turn = current_turn + cooldown + 1  # +1 so cooldown starts next turn


@dataclass
class ActivationResult:
    """Result of attempting to activate a codex entry."""

    entry_id: int
    entry_name: str | None
    activated: bool
    content: str
    reason: str  # "keyword_match", "sticky_active", "constant", "blocked_cooldown", "blocked_delay", "no_match"
    matched_keyword: str | None = None
    position: str = "after_char"  # "before_char" or "after_char"
    priority: int = 0
    insertion_order: int = 0


@dataclass
class CodexState:
    """
    Aggregate state for all codex entries in a conversation.

    Manages the runtime state of all entries, tracking activations,
    cooldowns, and sticky effects across conversation turns.
    """

    current_turn: int = 0
    entry_states: dict[int, EntryState] = field(default_factory=dict)

    def get_entry_state(self, entry_id: int) -> EntryState:
        """Get or create state for an entry."""
        if entry_id not in self.entry_states:
            self.entry_states[entry_id] = EntryState(entry_id=entry_id)
        return self.entry_states[entry_id]

    def advance_turn(self) -> None:
        """Advance to the next conversation turn."""
        self.current_turn += 1

    def reset(self) -> None:
        """Reset all state (new conversation)."""
        self.current_turn = 0
        self.entry_states.clear()

    def get_active_sticky_entries(self) -> list[int]:
        """Get IDs of entries currently active via sticky effect."""
        return [
            entry_id
            for entry_id, state in self.entry_states.items()
            if state.is_sticky_active(self.current_turn)
        ]

    def get_entries_on_cooldown(self) -> list[int]:
        """Get IDs of entries currently on cooldown."""
        return [
            entry_id
            for entry_id, state in self.entry_states.items()
            if state.is_on_cooldown(self.current_turn)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for debugging/logging."""
        return {
            "current_turn": self.current_turn,
            "active_sticky": self.get_active_sticky_entries(),
            "on_cooldown": self.get_entries_on_cooldown(),
            "entry_count": len(self.entry_states),
        }
