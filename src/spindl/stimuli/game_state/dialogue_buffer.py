"""
Dialogue-specific buffer for the game-state bridge (NANO-116 Phase B.2).

Filters dialogue events from the bridge stream, deduplicates per-line
(same speaker + same text collapses with repeat count), and staples
a gameplay context snapshot to each buffered line at capture time.

Follows the Twitch bounded-buffer pattern — FIFO eviction when full.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Event types routed to the dialogue buffer
_DIALOGUE_EVENT_TYPES = frozenset({
    "dialogue_line",
    "dialogue_started",
    "dialogue_ended",
})


@dataclass
class GameplaySnapshot:
    """Lightweight snapshot of gameplay state at dialogue capture time.

    Stapled to each dialogue line so the summarizer and conversational
    model have situational context the User can't provide live on stream.
    """

    chapter_hash: str | None = None
    combat_active: bool = False
    enemy_count: int = 0
    hp_ratio: float | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"timestamp": self.timestamp}
        if self.chapter_hash is not None:
            d["chapter_hash"] = self.chapter_hash
        d["combat_active"] = self.combat_active
        if self.combat_active:
            d["enemy_count"] = self.enemy_count
        if self.hp_ratio is not None:
            d["hp_ratio"] = round(self.hp_ratio, 2)
        return d


@dataclass
class DialogueLine:
    """A single buffered dialogue line with dedup tracking.

    Each line captures the speaker, text, capture path (cinematic/chatter),
    event source reliability tag, and a gameplay snapshot from the moment
    of capture. repeat_count tracks dedup collapses.
    """

    speaker: str
    text: str
    source: str  # "cinematic" | "chatter"
    event_source: str  # "direct_hook" | "inferred_from_hud" | "snapshot_aggregate"
    timestamp: str
    sequence: int
    gameplay_context: GameplaySnapshot = field(default_factory=GameplaySnapshot)
    repeat_count: int = 1
    game_id: str | None = None

    @property
    def dedup_key(self) -> tuple[str, str]:
        """Key for per-line dedup: same speaker + same text collapses."""
        return (self.speaker, self.text)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "speaker": self.speaker,
            "text": self.text,
            "source": self.source,
            "event_source": self.event_source,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "gameplay_context": self.gameplay_context.to_dict(),
        }
        if self.repeat_count > 1:
            d["repeat_count"] = self.repeat_count
        if self.game_id:
            d["game_id"] = self.game_id
        return d


class DialogueBuffer:
    """
    Bounded buffer for dialogue events with per-line dedup.

    Receives raw GameEvent dicts from the GameStateModule's _buffer_event
    path, extracts dialogue payload fields, staples the current gameplay
    snapshot, and applies per-line dedup (same speaker + same text within
    the buffer collapses with a repeat count increment).

    FIFO eviction when full — oldest unique lines drop first.
    """

    def __init__(self, max_size: int = 30):
        self._max_size = max(1, max_size)
        self._buffer: deque[DialogueLine] = deque(maxlen=self._max_size)
        self._gameplay_snapshot = GameplaySnapshot()

    @property
    def max_size(self) -> int:
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        self._max_size = max(1, value)
        old = list(self._buffer)
        self._buffer = deque(old, maxlen=self._max_size)

    @property
    def count(self) -> int:
        return len(self._buffer)

    @property
    def gameplay_snapshot(self) -> GameplaySnapshot:
        return self._gameplay_snapshot

    def update_gameplay_snapshot(
        self,
        *,
        chapter_hash: str | None = None,
        combat_active: bool | None = None,
        enemy_count: int | None = None,
        hp_ratio: float | None = None,
        timestamp: str = "",
    ) -> None:
        """Update the current gameplay snapshot from recent bridge events.

        Called by the GameStateModule when non-dialogue events arrive
        (snapshots, combat state changes, etc.) so dialogue lines capture
        the gameplay context at the moment they arrive.
        """
        if chapter_hash is not None:
            self._gameplay_snapshot.chapter_hash = chapter_hash
        if combat_active is not None:
            self._gameplay_snapshot.combat_active = combat_active
        if enemy_count is not None:
            self._gameplay_snapshot.enemy_count = enemy_count
        if hp_ratio is not None:
            self._gameplay_snapshot.hp_ratio = hp_ratio
        if timestamp:
            self._gameplay_snapshot.timestamp = timestamp

    @staticmethod
    def is_dialogue_event(event_type: str) -> bool:
        """Check if an event type should be routed to the dialogue buffer."""
        return event_type in _DIALOGUE_EVENT_TYPES

    def accept_event(self, event: dict) -> Optional[DialogueLine]:
        """Accept a validated bridge event dict and buffer if it's a dialogue_line.

        dialogue_started / dialogue_ended are noted but not buffered as
        lines — they don't carry speaker text. Only dialogue_line events
        produce buffered DialogueLine entries.

        Returns the DialogueLine if one was buffered (or dedup-collapsed),
        None otherwise.
        """
        event_type = event.get("event_type", "")

        if event_type in ("dialogue_started", "dialogue_ended"):
            logger.debug(
                "Dialogue boundary event: %s (seq=%s)",
                event_type,
                event.get("sequence"),
            )
            return None

        if event_type != "dialogue_line":
            return None

        payload = event.get("payload", {})
        speaker = payload.get("speaker", "")
        text = payload.get("text", "")
        source = payload.get("source", "chatter")

        if not speaker or not text:
            logger.warning(
                "Dropping dialogue_line with empty speaker/text: seq=%s",
                event.get("sequence"),
            )
            return None

        line = DialogueLine(
            speaker=speaker,
            text=text,
            source=source,
            event_source=event.get("event_source", "direct_hook"),
            timestamp=event.get("timestamp", ""),
            sequence=event.get("sequence", 0),
            gameplay_context=GameplaySnapshot(
                chapter_hash=self._gameplay_snapshot.chapter_hash,
                combat_active=self._gameplay_snapshot.combat_active,
                enemy_count=self._gameplay_snapshot.enemy_count,
                hp_ratio=self._gameplay_snapshot.hp_ratio,
                timestamp=self._gameplay_snapshot.timestamp,
            ),
            game_id=event.get("game_id"),
        )

        # Per-line dedup: if the last entry in the buffer has the same
        # speaker+text, increment repeat_count instead of adding a new entry.
        if self._buffer and self._buffer[-1].dedup_key == line.dedup_key:
            self._buffer[-1].repeat_count += 1
            self._buffer[-1].timestamp = line.timestamp
            self._buffer[-1].sequence = line.sequence
            logger.debug(
                "Dedup collapse: %s says '%s' (x%d)",
                speaker,
                text[:50],
                self._buffer[-1].repeat_count,
            )
            return self._buffer[-1]

        self._buffer.append(line)
        logger.debug(
            "Buffered dialogue: %s says '%s' (buffer=%d/%d)",
            speaker,
            text[:50],
            len(self._buffer),
            self._max_size,
        )
        return line

    def drain(self) -> list[DialogueLine]:
        """Drain all buffered lines. Returns a list and clears the buffer."""
        lines = list(self._buffer)
        self._buffer.clear()
        return lines

    def peek(self) -> list[DialogueLine]:
        """Peek at buffered lines without draining."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def format_for_stimulus(self, lines: list[DialogueLine]) -> str:
        """Format drained dialogue lines for stimulus template injection.

        Produces a compact human-readable block:
            Diana: Watch out!
            Ken: Stay close. [cinematic]
            Diana: Watch out! (x3)
        """
        formatted: list[str] = []
        for line in lines:
            parts = [f"{line.speaker}: {line.text}"]
            if line.repeat_count > 1:
                parts.append(f"(x{line.repeat_count})")
            if line.source == "cinematic":
                parts.append("[cinematic]")
            formatted.append(" ".join(parts))
        return "\n".join(formatted)
