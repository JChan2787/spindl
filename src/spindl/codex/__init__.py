"""
Codex (Lorebook) System for spindl.

Keyword-activated knowledge injection following SillyTavern conventions.
Supports substring matching, regex, whole-word matching, secondary key logic,
and timed effects (sticky, cooldown, delay).

Main Components:
    - CodexManager: Orchestrates entry loading and activation
    - CodexState: Tracks activation state per conversation
    - activate_entries: Core activation algorithm

Usage:
    from spindl.codex import CodexManager

    manager = CodexManager(characters_dir="./characters")
    manager.load_character("spindle")

    results = manager.activate("Tell me about the creator")
    for result in results:
        print(f"Activated: {result.entry_name} via {result.reason}")

    manager.advance_turn()
"""

from spindl.codex.activation import (
    activate_entries,
    activate_entry,
    match_keyword,
)
from spindl.codex.manager import CodexManager
from spindl.codex.models import (
    ActivationResult,
    CodexState,
    EntryState,
    SelectiveLogic,
)

__all__ = [
    # Manager
    "CodexManager",
    # Activation
    "activate_entries",
    "activate_entry",
    "match_keyword",
    # Models
    "ActivationResult",
    "CodexState",
    "EntryState",
    "SelectiveLogic",
]
