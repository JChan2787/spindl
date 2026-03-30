"""
ReflectionSystem — Async memory entry generation from conversation.

Extracts salient facts from recent conversation history by sending
a transcript + reflection prompt to the LLM in a standalone call
(no character card, no system prompt — pure fact extraction).

Prompt, system message, and output delimiter are all configurable
via MemoryConfig (NANO-104). The parser is format-agnostic — it
splits on the configured delimiter and validates completeness
without enforcing Q&A structure.

Runs in a background daemon thread. Non-blocking to the main
conversation pipeline.

Based on: Generative Agents (2304.03442) / Kimjammer Neuro pattern.

NANO-043 Phase 3. NANO-104 editable prompt + format-agnostic parser.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from ..llm.base import LLMProvider
from ..llm.plugins.conversation_history import ConversationHistoryManager
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Built-in defaults — used when config fields are None (NANO-104).
DEFAULT_REFLECTION_PROMPT = (
    "Given only the conversation below, what are the 3 most salient high-level "
    "questions we can answer about the subjects discussed?\n\n"
    "Format each as a Question and Answer pair. Separate pairs with \"{qa}\".\n"
    "Output only the Q&A pairs, no explanations.\n\n"
    "Conversation:\n{transcript}"
)

DEFAULT_REFLECTION_SYSTEM_MESSAGE = (
    "You are a fact extraction assistant. Extract key facts "
    "from conversations as concise Q&A pairs. Be precise and "
    "factual. Do not add information that is not in the "
    "conversation."
)

DEFAULT_DELIMITER = "{qa}"

# Minimum character length for a memory entry to be considered valid.
MIN_ENTRY_LENGTH = 10


def format_transcript(history: list[dict]) -> str:
    """
    Format conversation history turns into a plain transcript.

    Args:
        history: List of turn dicts with 'role' and 'content' keys.

    Returns:
        Formatted transcript string.
    """
    lines = []
    for turn in history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "summary":
            lines.append(f"[Previous summary]: {content}")
    return "\n".join(lines)


def _looks_complete(text: str) -> bool:
    """
    Check if a memory entry looks like a complete, coherent response.

    A complete entry should end with sentence-ending punctuation
    (period, exclamation, question mark, or closing quote after one).

    Args:
        text: The entry text to validate.

    Returns:
        True if the text appears complete.
    """
    if not text:
        return False

    end = text.rstrip()
    if not end:
        return False

    return end[-1] in ".!?\"')"


def _parse_memory_entries(response: str, delimiter: str = DEFAULT_DELIMITER) -> list[str]:
    """
    Parse memory entries from LLM response.

    Format-agnostic: splits on the configured delimiter, strips whitespace,
    and validates minimum length and completeness. Does not enforce Q&A
    structure — any delimiter-separated text that passes validation becomes
    a memory entry.

    Args:
        response: Raw LLM response text.
        delimiter: String to split entries on.

    Returns:
        List of validated memory entry strings.
    """
    if not response or not response.strip():
        return []

    chunks = response.split(delimiter)
    entries = []

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if len(chunk) < MIN_ENTRY_LENGTH:
            logger.debug("Discarding short entry (%d chars): %.40s...", len(chunk), chunk)
            continue

        if not _looks_complete(chunk):
            logger.debug("Discarding incomplete entry: %.80s...", chunk)
            continue

        entries.append(chunk)

    return entries


class ReflectionSystem:
    """
    Async reflection system that generates memory entries from conversation.

    Monitors message count via the history manager. When the threshold
    is reached, extracts the unprocessed messages, sends a reflection
    prompt to the LLM (standalone call, no character context), parses
    the entries, and stores them as flash cards in ChromaDB.

    Prompt, system message, and delimiter are configurable at runtime
    via MemoryConfig (NANO-104).

    Runs in a daemon thread. Never blocks the main conversation.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        memory_store: MemoryStore,
        history_manager: ConversationHistoryManager,
        reflection_interval: int = 20,
        max_tokens: int = 300,
        reflection_prompt: Optional[str] = None,
        reflection_system_message: Optional[str] = None,
        reflection_delimiter: str = DEFAULT_DELIMITER,
    ):
        """
        Args:
            llm_provider: LLMProvider for generating reflections.
            memory_store: MemoryStore for storing flash cards.
            history_manager: ConversationHistoryManager to monitor.
            reflection_interval: Generate reflections every N new messages
                                (counted as individual turns, not pairs).
            max_tokens: LLM token budget for reflection generation.
            reflection_prompt: Custom prompt template. Must contain {transcript}.
                              None = use DEFAULT_REFLECTION_PROMPT.
            reflection_system_message: Custom system message for the LLM call.
                                       None = use DEFAULT_REFLECTION_SYSTEM_MESSAGE.
            reflection_delimiter: Delimiter for splitting LLM output into entries.
        """
        self._provider = llm_provider
        self._memory_store = memory_store
        self._history_manager = history_manager
        self._reflection_interval = reflection_interval
        self._max_tokens = max_tokens
        self._reflection_prompt = reflection_prompt
        self._reflection_system_message = reflection_system_message
        self._reflection_delimiter = reflection_delimiter

        # Track how many turns we've already processed
        self._processed_turn_count: int = 0

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._trigger_event = threading.Event()

        # Session tracking for metadata
        self._session_id: Optional[str] = None

    @property
    def processed_turn_count(self) -> int:
        """Number of turns already processed by reflection."""
        return self._processed_turn_count

    def start(self, session_id: Optional[str] = None) -> None:
        """
        Start the background reflection thread.

        Seeds _processed_turn_count to the current history length so that
        resumed sessions don't immediately re-reflect on already-seen turns.

        Args:
            session_id: Optional session identifier for flash card metadata.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ReflectionSystem already running")
            return

        self._session_id = session_id
        self._stop_event.clear()
        self._trigger_event.clear()

        # Seed processed count to existing history — only reflect on NEW turns
        self._processed_turn_count = self._history_manager.turn_count

        self._thread = threading.Thread(
            target=self._run_loop,
            name="reflection-system",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "ReflectionSystem started (interval=%d, max_tokens=%d)",
            self._reflection_interval,
            self._max_tokens,
        )

    def stop(self) -> None:
        """Stop the background reflection thread."""
        self._stop_event.set()
        self._trigger_event.set()  # Wake the thread so it exits
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("ReflectionSystem stopped")

    def notify(self) -> None:
        """
        Signal the reflection thread to check if reflection is needed.

        Called by ReflectionMonitor (PostProcessor) after each turn.
        Non-blocking — just sets an event flag.
        """
        self._trigger_event.set()

    def check_and_reflect(self) -> list[str]:
        """
        Check if reflection is needed and execute if so.

        Can be called directly (synchronous) or from the background thread.

        Returns:
            List of memory entry strings that were generated and stored.
            Empty list if reflection was not triggered or failed.
        """
        current_count = self._history_manager.turn_count
        new_turns = current_count - self._processed_turn_count

        if new_turns < self._reflection_interval:
            return []

        print(
            f"[Memory] Reflection triggered: {new_turns} new turns "
            f"(threshold={self._reflection_interval})",
            flush=True,
        )

        # Grab the unprocessed turns
        all_history = self._history_manager.get_history()
        unprocessed = all_history[self._processed_turn_count:]

        if not unprocessed:
            return []

        # Generate memory entries
        cards = self._generate_reflection(unprocessed)

        # Update processed count regardless of whether cards were generated
        # (avoids re-processing the same turns on failure)
        self._processed_turn_count = current_count

        if not cards:
            print("[Memory] Reflection complete: 0 entries (nothing extracted)", flush=True)
            return []

        # Store each entry in ChromaDB
        stored_cards = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for card in cards:
            metadata = {
                "type": "flash_card",
                "source": "reflection",
                "timestamp": timestamp,
            }
            if self._session_id:
                metadata["session_id"] = self._session_id

            try:
                self._memory_store.add_flash_card(card, metadata)
                stored_cards.append(card)
            except Exception as e:
                logger.warning("Failed to store flash card: %s", e)

        print(
            f"[Memory] Reflection complete: {len(stored_cards)} "
            f"{'entry' if len(stored_cards) == 1 else 'entries'} stored "
            f"(from {len(unprocessed)} turns)",
            flush=True,
        )
        logger.info(
            "Reflection complete: %d entries stored (from %d turns)",
            len(stored_cards),
            len(unprocessed),
        )
        return stored_cards

    def _generate_reflection(self, turns: list[dict]) -> list[str]:
        """
        Generate memory entries from conversation turns.

        Formats turns as transcript, sends reflection prompt to LLM,
        parses response into individual entries using configured delimiter.

        Args:
            turns: List of conversation turn dicts to reflect on.

        Returns:
            List of validated memory entry strings.
        """
        transcript = format_transcript(turns)

        # Use custom prompt or built-in default.
        # Use str.replace() instead of .format() so literal braces in the
        # prompt (e.g. "{qa}" delimiter instructions) aren't consumed.
        prompt_template = self._reflection_prompt or DEFAULT_REFLECTION_PROMPT
        prompt = prompt_template.replace("{transcript}", transcript)

        system_message = self._reflection_system_message or DEFAULT_REFLECTION_SYSTEM_MESSAGE

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=self._max_tokens,
                top_p=0.9,
            )
        except Exception as e:
            logger.error("Reflection LLM call failed: %s", e)
            return []

        # Check for truncation via finish_reason
        if response.finish_reason == "length":
            logger.warning(
                "Reflection response was truncated (hit max_tokens=%d). "
                "Last entry will be validated for completeness.",
                self._max_tokens,
            )

        return _parse_memory_entries(response.content, self._reflection_delimiter)

    def _run_loop(self) -> None:
        """
        Background thread loop.

        Waits for trigger events (from ReflectionMonitor) or times out
        after 5 seconds. On each wake, checks if reflection is needed.
        """
        logger.debug("Reflection thread started")

        while not self._stop_event.is_set():
            # Wait for trigger or timeout (5 second poll as fallback)
            self._trigger_event.wait(timeout=5.0)
            self._trigger_event.clear()

            if self._stop_event.is_set():
                break

            try:
                self.check_and_reflect()
            except Exception as e:
                logger.error("Reflection loop error: %s", e)

        logger.debug("Reflection thread exiting")
