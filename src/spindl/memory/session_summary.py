"""
SessionSummaryGenerator — On-demand narrative summary of a conversation session.

Generates a concise third-person summary of a conversation session and stores
it in the ChromaDB summaries collection. Triggered manually by the user via
the GUI Sessions page (no automatic trigger on shutdown).

Uses the same LLM call pattern as ReflectionSystem but is a one-shot operation
— no background thread, no polling, no PostProcessor.

NANO-043 Phase 4.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from ..llm.base import LLMProvider
from .memory_store import SUMMARIES, MemoryStore
from .reflection import format_transcript

logger = logging.getLogger(__name__)

# Minimum number of turns required to generate a summary.
# Less than 2 full exchanges (user+assistant pairs) is too short.
MIN_TURNS_FOR_SUMMARY = 4

SUMMARY_PROMPT = """\
Summarize this conversation session. Include:
1. Key topics discussed
2. Important facts learned about the user
3. Decisions made or preferences expressed
4. The overall tone and dynamic of the conversation

Keep the summary under 200 words. Write in third person.

Conversation:
{transcript}

Flash cards from this session:
{flash_cards}

Summary:"""


class SessionSummaryGenerator:
    """
    Generates end-of-session summaries for long-term storage in ChromaDB.

    Called on-demand via the GUI Sessions page. Not tied to agent lifecycle.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        memory_store: MemoryStore,
        max_tokens: int = 500,
    ):
        """
        Args:
            llm_provider: LLMProvider for generating the summary.
            memory_store: MemoryStore for storing the summary in ChromaDB.
            max_tokens: LLM token budget for summary generation.
        """
        self._provider = llm_provider
        self._memory_store = memory_store
        self._max_tokens = max_tokens

    def generate_and_store(
        self,
        session_turns: list[dict],
        session_id: str,
        flash_cards: Optional[list[dict]] = None,
    ) -> Optional[str]:
        """
        Generate a summary from session turns and store it in ChromaDB.

        Args:
            session_turns: List of turn dicts with 'role' and 'content' keys.
            session_id: Session identifier for metadata (e.g., filename stem).
            flash_cards: Optional flash cards generated during this session.

        Returns:
            Summary text, or None if session too short or generation failed.
        """
        if len(session_turns) < MIN_TURNS_FOR_SUMMARY:
            logger.info(
                "Session too short for summary (%d turns, need %d)",
                len(session_turns),
                MIN_TURNS_FOR_SUMMARY,
            )
            return None

        # NANO-102 Phase 3: Guard against duplicate session summaries
        try:
            existing = self._memory_store.get_all(SUMMARIES)
            if any(
                m.get("metadata", {}).get("session_id") == session_id
                for m in existing
            ):
                logger.info(
                    "Summary already exists for session %s — skipping",
                    session_id,
                )
                return None
        except Exception as e:
            logger.warning("Session summary guard check failed: %s", e)

        # Format conversation as plain transcript
        transcript = format_transcript(session_turns)

        # Format flash cards if provided
        if flash_cards:
            cards_text = "\n".join(
                f"- {card.get('content', card) if isinstance(card, dict) else card}"
                for card in flash_cards
            )
        else:
            cards_text = "(none)"

        prompt = SUMMARY_PROMPT.format(
            transcript=transcript,
            flash_cards=cards_text,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. Write concise, factual "
                    "third-person summaries of conversation sessions. Do not "
                    "add information that is not in the conversation."
                ),
            },
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
            logger.error("Session summary LLM call failed: %s", e)
            return None

        summary = response.content.strip()
        if not summary:
            logger.warning("Session summary LLM returned empty response")
            return None

        # Store in ChromaDB
        metadata = {
            "type": "session_summary",
            "source": "session_summary_generator",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }

        try:
            doc_id = self._memory_store.add_session_summary(summary, metadata)
            logger.info(
                "Session summary stored (doc_id=%s, session=%s, %d chars)",
                doc_id,
                session_id,
                len(summary),
            )
        except Exception as e:
            logger.error("Failed to store session summary: %s", e)
            return None

        return summary
