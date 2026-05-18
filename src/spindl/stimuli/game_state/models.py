"""
Data models for the game-state stimulus module (NANO-116).

Lightweight dataclasses for buffered bridge events. These are internal
to the module — the wire format is defined by the vendored
stimulus_event.schema.json from SPNDL-001.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameEvent:
    """A single buffered game-state event from the bridge."""

    event_type: str
    event_source: str
    timestamp: str
    sequence: int
    payload: dict[str, Any] = field(default_factory=dict)
    game_id: str | None = None
    save_slot_hint: int | None = None
