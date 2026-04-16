"""Twitch audience transcript persistence and injection plugin (NANO-115)."""

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import PipelineContext, PreProcessor


class TwitchTranscriptManager:
    """
    Manages persistent Twitch audience chat transcripts.

    Stores audience messages and the character's broadcast replies in a
    per-stream JSONL sibling of the voice conversation file. Provides
    an in-memory LRU window for fast injection into the system prompt.

    Storage format: One JSONL file per stream session.
    File location: {conversations_dir}/{persona}_{timestamp}.twitch.jsonl

    Turn format:
        {"turn_id": 1,
         "role": "audience",
         "username": "apubloo_tw",
         "text": "hey spindle",
         "timestamp": "ISO8601",
         "channel": "jubbydubby"}

        {"turn_id": 2,
         "role": "assistant",
         "text": "Hey Apubloo!",
         "timestamp": "ISO8601",
         "responding_to": ["apubloo_tw"]}
    """

    def __init__(
        self,
        conversations_dir: str = "./conversations",
        lru_size: int = 100,
        debug: bool = False,
    ):
        self._conversations_dir = Path(conversations_dir)
        self._lru_size = lru_size
        self._debug = debug

        self._transcript_file: Optional[Path] = None
        self._lru: deque[dict] = deque(maxlen=lru_size)
        self._next_turn_id: int = 1
        self._replied_usernames: dict[str, datetime] = {}

    def ensure_transcript(self, voice_session_file: Optional[Path]) -> None:
        """
        Ensure a transcript file exists as a sibling of the voice session.

        Derives the transcript filename from the voice session filename:
            spindle_20260415_112456.jsonl -> spindle_20260415_112456.twitch.jsonl
        """
        if voice_session_file is None:
            return

        transcript_name = voice_session_file.stem + ".twitch.jsonl"
        transcript_path = voice_session_file.parent / transcript_name

        if self._transcript_file == transcript_path:
            return

        self._transcript_file = transcript_path
        self._lru.clear()
        self._next_turn_id = 1
        self._replied_usernames.clear()

        if transcript_path.exists():
            self._load_tail(transcript_path)

        if self._debug:
            print(
                f"[DEBUG:TwitchTranscript] Bound to {transcript_path} "
                f"(loaded {len(self._lru)} turns into LRU)"
            )

    def _load_tail(self, filepath: Path) -> None:
        """Load the last N turns from disk into the LRU."""
        turns = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))

        max_turn_id = 0
        for t in turns:
            tid = t.get("turn_id", 0)
            if tid > max_turn_id:
                max_turn_id = tid
            if t.get("role") == "assistant" and t.get("responding_to"):
                for uname in t["responding_to"]:
                    ts_str = t.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except (ValueError, TypeError):
                        ts = datetime.now(timezone.utc)
                    self._replied_usernames[uname] = ts

        self._next_turn_id = max_turn_id + 1

        tail = turns[-self._lru_size:]
        for t in tail:
            self._lru.append(t)

    def record_audience_message(
        self, username: str, text: str, channel: str = ""
    ) -> None:
        """Persist an incoming audience message to the transcript."""
        if self._transcript_file is None:
            return

        turn = {
            "turn_id": self._next_turn_id,
            "role": "audience",
            "username": username,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "channel": channel,
        }

        self._append_to_disk(turn)
        self._lru.append(turn)
        self._next_turn_id += 1

    def record_assistant_reply(
        self, text: str, responding_to: list[str]
    ) -> None:
        """
        Persist the character's broadcast reply to the transcript (dual-write).

        Called when stimulus_source == "twitch" after the LLM responds.
        """
        if self._transcript_file is None:
            return

        now = datetime.now(timezone.utc)
        turn = {
            "turn_id": self._next_turn_id,
            "role": "assistant",
            "text": text,
            "timestamp": now.isoformat(),
            "responding_to": responding_to,
        }

        self._append_to_disk(turn)
        self._lru.append(turn)
        self._next_turn_id += 1

        for uname in responding_to:
            self._replied_usernames[uname] = now

    def get_injection_window(
        self,
        max_messages: int = 25,
        char_cap: int = 150,
        pin_window_minutes: int = 10,
    ) -> list[dict]:
        """
        Build the audience-chat injection window for the system prompt.

        Returns a list of turns (audience + assistant) ordered chronologically,
        respecting the message limit and per-message character cap.

        Pinning: usernames the character replied to within pin_window_minutes
        are guaranteed inclusion (budget permitting).
        """
        if not self._lru:
            return []

        now = datetime.now(timezone.utc)

        pinned_usernames: set[str] = set()
        for uname, replied_at in self._replied_usernames.items():
            if replied_at.tzinfo is None:
                replied_at = replied_at.replace(tzinfo=timezone.utc)
            elapsed = (now - replied_at).total_seconds()
            if elapsed <= pin_window_minutes * 60:
                pinned_usernames.add(uname)

        all_turns = list(self._lru)

        pinned: list[dict] = []
        unpinned: list[dict] = []

        for turn in reversed(all_turns):
            is_pinned = False
            if turn.get("role") == "audience" and turn.get("username") in pinned_usernames:
                is_pinned = True
            elif turn.get("role") == "assistant" and turn.get("responding_to"):
                if any(u in pinned_usernames for u in turn["responding_to"]):
                    is_pinned = True

            if is_pinned:
                pinned.append(turn)
            else:
                unpinned.append(turn)

        pinned = list(reversed(pinned[:max_messages]))

        remaining = max_messages - len(pinned)
        chronological = list(reversed(unpinned[:remaining])) if remaining > 0 else []

        combined = sorted(
            pinned + chronological,
            key=lambda t: t.get("turn_id", 0),
        )

        result = []
        for turn in combined:
            entry = dict(turn)
            text_field = "text"
            if text_field in entry and len(entry[text_field]) > char_cap:
                entry[text_field] = entry[text_field][:char_cap] + "\u2026"
            result.append(entry)

        return result

    @property
    def has_transcript(self) -> bool:
        return len(self._lru) > 0

    @property
    def transcript_file(self) -> Optional[Path]:
        return self._transcript_file

    def _append_to_disk(self, turn: dict) -> None:
        if self._transcript_file is None:
            return
        self._transcript_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._transcript_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")


class TwitchHistoryInjector(PreProcessor):
    """
    PreProcessor that injects the persistent audience chat transcript
    into the [AUDIENCE_CHAT] placeholder in the system prompt.

    Fires every turn (voice or Twitch stimulus). Collapses to nothing
    when the transcript is empty or the Twitch module hasn't connected.
    """

    PLACEHOLDER = "[AUDIENCE_CHAT]"

    PROTOCOL_PREAMBLE = (
        "You are streaming live. Your audience speaks via Twitch chat below, "
        "tagged by username. Your User speaks to you by voice \u2014 those turns "
        "appear in your conversation history. Address Twitch viewers by name "
        "when you respond to them. Address your User directly."
    )

    def __init__(
        self,
        transcript_manager: TwitchTranscriptManager,
        persona_name: str = "Assistant",
    ):
        self._manager = transcript_manager
        self._persona_name = persona_name

    @property
    def name(self) -> str:
        return "twitch_history_injector"

    def process(self, context: PipelineContext) -> PipelineContext:
        persona_name = context.persona.get("name", self._persona_name)
        audience_window = int(context.metadata.get("twitch_audience_window", 25))
        char_cap = int(context.metadata.get("twitch_audience_char_cap", 150))

        window = self._manager.get_injection_window(
            max_messages=audience_window,
            char_cap=char_cap,
        )

        if window:
            lines = [self.PROTOCOL_PREAMBLE, ""]
            for turn in window:
                if turn.get("role") == "audience":
                    username = turn.get("username", "???")
                    text = turn.get("text", "")
                    lines.append(f"{username}: {text}")
                elif turn.get("role") == "assistant":
                    text = turn.get("text", "")
                    lines.append(f"[{persona_name}]: {text}")
            content = "\n".join(lines)
        else:
            content = ""

        context.metadata["audience_chat_formatted"] = content
        return context
