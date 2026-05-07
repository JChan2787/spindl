"""
Dialogue persistence for the game-state bridge (NANO-116 Phase B.2).

JSONL sibling file alongside the voice conversation session:
    spindle_20260425_043900.jsonl          (voice)
    spindle_20260425_043900.twitch.jsonl   (audience transcript)
    spindle_20260425_043900.dialogue.jsonl (game dialogue — this file)

Raw dialogue lines are written to disk and never deleted. The summary
blob is written as a special entry (role: "summary") and updated in
place by appending a new summary entry — the latest summary entry is
authoritative. On restart, load_tail reconstructs the LRU window from
the latest summary + unsummarized raw lines.

Follows the TwitchTranscriptManager pattern from NANO-115.
"""

import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .dialogue_buffer import DialogueLine
from ...utils.tokens import count_tokens as _count_tokens

logger = logging.getLogger(__name__)


class DialogueStore:
    """
    Persistent JSONL store for game dialogue lines and summary blobs.

    Storage format:
        {"turn_id": 1, "role": "dialogue", "speaker": "Diana", "text": "Watch out!",
         "source": "chatter", "event_source": "direct_hook", "timestamp": "ISO8601",
         "gameplay_context": {...}, "repeat_count": 1}

        {"turn_id": 5, "role": "assistant", "text": "Diana seems worried...",
         "timestamp": "ISO8601", "responding_to_lines": [1, 2, 3]}

        {"turn_id": 10, "role": "summary", "version": 1,
         "summary_text": "Diana is Hugh's partner...",
         "summarized_through_turn_id": 9, "timestamp": "ISO8601"}
    """

    def __init__(
        self,
        conversations_dir: str = "./conversations",
        lru_size: int = 200,
        debug: bool = False,
    ):
        self._conversations_dir = Path(conversations_dir)
        self._lru_size = lru_size
        self._debug = debug
        self._lock = threading.Lock()

        self._store_file: Optional[Path] = None
        self._lru: deque[dict] = deque(maxlen=lru_size)
        self._next_turn_id: int = 1

        # Summary state
        self._summary_blob: str = ""
        self._summary_version: int = 0
        self._summarized_through_turn_id: int = 0

    @property
    def store_file(self) -> Optional[Path]:
        return self._store_file

    @property
    def summary_blob(self) -> str:
        return self._summary_blob

    @property
    def summary_version(self) -> int:
        return self._summary_version

    @property
    def summarized_through_turn_id(self) -> int:
        return self._summarized_through_turn_id

    @property
    def has_content(self) -> bool:
        return len(self._lru) > 0

    @property
    def turn_count(self) -> int:
        return len(self._lru)

    def ensure_store(self, voice_session_file: Optional[Path]) -> None:
        """Ensure a dialogue store file exists as a sibling of the voice session.

        Derives filename from voice session:
            spindle_20260425_043900.jsonl -> spindle_20260425_043900.dialogue.jsonl
        """
        if voice_session_file is None:
            return

        store_name = voice_session_file.stem + ".dialogue.jsonl"
        store_path = voice_session_file.parent / store_name

        if self._store_file == store_path:
            return

        self._store_file = store_path
        self._lru.clear()
        self._next_turn_id = 1
        self._summary_blob = ""
        self._summary_version = 0
        self._summarized_through_turn_id = 0

        if store_path.exists():
            self._load_tail(store_path)

        if self._debug:
            logger.info(
                "DialogueStore bound to %s (loaded %d turns, summary_v%d)",
                store_path,
                len(self._lru),
                self._summary_version,
            )

    def _load_tail(self, filepath: Path) -> None:
        """Load state from disk: latest summary + all turns into LRU."""
        turns: list[dict] = []
        latest_summary: Optional[dict] = None
        max_turn_id = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_text in f:
                line_text = line_text.strip()
                if not line_text:
                    continue
                try:
                    entry = json.loads(line_text)
                except json.JSONDecodeError:
                    continue

                tid = entry.get("turn_id", 0)
                if tid > max_turn_id:
                    max_turn_id = tid

                if entry.get("role") == "summary":
                    latest_summary = entry
                else:
                    turns.append(entry)

        self._next_turn_id = max_turn_id + 1

        # Restore summary state
        if latest_summary:
            self._summary_blob = latest_summary.get("summary_text", "")
            self._summary_version = latest_summary.get("version", 0)
            self._summarized_through_turn_id = latest_summary.get(
                "summarized_through_turn_id", 0
            )

        # Load tail into LRU
        tail = turns[-self._lru_size :]
        for t in tail:
            self._lru.append(t)

    def record_dialogue_line(self, line: DialogueLine) -> int:
        """Persist a dialogue line to the store. Returns the turn_id."""
        if self._store_file is None:
            return -1

        with self._lock:
            turn = {
                "turn_id": self._next_turn_id,
                "role": "dialogue",
                "speaker": line.speaker,
                "text": line.text,
                "source": line.source,
                "event_source": line.event_source,
                "timestamp": line.timestamp or datetime.now(timezone.utc).isoformat(),
                "gameplay_context": line.gameplay_context.to_dict(),
            }
            if line.repeat_count > 1:
                turn["repeat_count"] = line.repeat_count
            if line.game_id:
                turn["game_id"] = line.game_id

            self._append_to_disk(turn)
            self._lru.append(turn)
            turn_id = self._next_turn_id
            self._next_turn_id += 1
            return turn_id

    def record_assistant_reply(
        self, text: str, responding_to_turn_ids: list[int]
    ) -> int:
        """Persist SpindL's response to dialogue stimulus. Returns turn_id."""
        if self._store_file is None:
            return -1

        with self._lock:
            turn = {
                "turn_id": self._next_turn_id,
                "role": "assistant",
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "responding_to_lines": responding_to_turn_ids,
            }

            self._append_to_disk(turn)
            self._lru.append(turn)
            turn_id = self._next_turn_id
            self._next_turn_id += 1
            return turn_id

    def record_summary(self, summary_text: str) -> int:
        """Persist a new summary blob. Returns the turn_id.

        The summary blob is rolling — each new summary supersedes the last.
        On restart, the latest summary entry in the JSONL is authoritative.
        """
        if self._store_file is None:
            return -1

        with self._lock:
            self._summary_version += 1
            self._summary_blob = summary_text
            self._summarized_through_turn_id = self._next_turn_id - 1

            turn = {
                "turn_id": self._next_turn_id,
                "role": "summary",
                "version": self._summary_version,
                "summary_text": summary_text,
                "summarized_through_turn_id": self._summarized_through_turn_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._append_to_disk(turn)
            turn_id = self._next_turn_id
            self._next_turn_id += 1
            return turn_id

    def get_unsummarized_lines(self) -> list[dict]:
        """Get dialogue lines that haven't been covered by a summary yet.

        Returns lines from the LRU whose turn_id > summarized_through_turn_id
        and whose role is 'dialogue'.
        """
        with self._lock:
            return [
                t
                for t in self._lru
                if t.get("role") == "dialogue"
                and t.get("turn_id", 0) > self._summarized_through_turn_id
            ]

    def get_all_dialogue_lines(self) -> list[dict]:
        """Get all dialogue lines from the LRU (regardless of summary state)."""
        with self._lock:
            return [t for t in self._lru if t.get("role") == "dialogue"]

    def get_injection_content(self, token_budget: int = 500) -> str:
        """Build the character knowledge injection content.

        If a summary exists: summary blob + unsummarized raw tail.
        If no summary and under budget: all raw dialogue lines.
        If no summary and over budget: truncated recent lines.
        Token counting via tiktoken cl100k_base.

        Returns empty string if no content.
        """
        if not self._lru:
            return ""

        all_dialogue = self.get_all_dialogue_lines()
        if not all_dialogue:
            return ""

        # Once a summary exists, always use summary + unsummarized tail
        if self._summary_blob:
            unsummarized = self.get_unsummarized_lines()
            tail_block = self._format_lines(unsummarized) if unsummarized else ""

            parts = [
                "Accumulated character knowledge from dialogue:",
                self._summary_blob,
            ]
            if tail_block:
                parts.append("")
                parts.append("Recent unsummarized dialogue:")
                parts.append(tail_block)
            return "\n".join(parts)

        # No summary yet — inject raw if under budget
        raw_block = self._format_lines(all_dialogue)
        if _count_tokens(raw_block) <= token_budget:
            return raw_block

        # Over budget, no summary — truncate from front, keep recent
        lines_reversed = list(reversed(all_dialogue))
        kept: list[dict] = []
        running_tokens = 0
        for line_dict in lines_reversed:
            line_str = self._format_single_line(line_dict)
            line_tokens = _count_tokens(line_str)
            if running_tokens + line_tokens > token_budget:
                break
            kept.append(line_dict)
            running_tokens += line_tokens

        kept.reverse()
        return self._format_lines(kept) if kept else ""

    def needs_summarization(self, token_budget: int = 500) -> bool:
        """Check if unsummarized dialogue + existing summary exceeds the token budget.

        Measures unsummarized raw lines + summary blob (if any) against budget.
        Token counting via tiktoken cl100k_base.
        """
        unsummarized = self.get_unsummarized_lines()
        if not unsummarized:
            return False
        raw_tokens = _count_tokens(self._format_lines(unsummarized))
        summary_tokens = _count_tokens(self._summary_blob) if self._summary_blob else 0
        return (raw_tokens + summary_tokens) > token_budget

    def get_summarizer_input(self) -> tuple[str, list[dict]]:
        """Get inputs for the summarizer: previous summary + unsummarized lines.

        Returns (previous_summary, unsummarized_lines).
        """
        return self._summary_blob, self.get_unsummarized_lines()

    @staticmethod
    def _format_lines(lines: list[dict]) -> str:
        """Format dialogue line dicts for prompt injection."""
        formatted: list[str] = []
        for line_dict in lines:
            formatted.append(DialogueStore._format_single_line(line_dict))
        return "\n".join(formatted)

    @staticmethod
    def _format_single_line(line_dict: dict) -> str:
        """Format a single dialogue line dict."""
        speaker = line_dict.get("speaker", "???")
        text = line_dict.get("text", "")
        repeat = line_dict.get("repeat_count", 1)
        source = line_dict.get("source", "")

        parts = [f"{speaker}: {text}"]
        if repeat and repeat > 1:
            parts.append(f"(x{repeat})")
        if source == "cinematic":
            parts.append("[cinematic]")
        return " ".join(parts)

    def _append_to_disk(self, turn: dict) -> None:
        if self._store_file is None:
            return
        self._store_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._store_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
