"""Tests for SessionSummaryGenerator — on-demand session narrative summaries.

NANO-043 Phase 4.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from spindl.llm.base import LLMResponse
from spindl.memory.embedding_client import EmbeddingClient
from spindl.memory.memory_store import MemoryStore
from spindl.memory.session_summary import (
    SessionSummaryGenerator,
    MIN_TURNS_FOR_SUMMARY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """Mock EmbeddingClient that returns deterministic fake embeddings."""
    client = MagicMock(spec=EmbeddingClient)

    def fake_embed(text: str) -> list[float]:
        h = hash(text) % 10000
        return [(h + i) / 10000.0 for i in range(64)]

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        return [fake_embed(t) for t in texts]

    client.embed.side_effect = fake_embed
    client.embed_batch.side_effect = fake_embed_batch
    return client


@pytest.fixture
def memory_dir() -> str:
    """Temporary directory for ChromaDB persistence."""
    d = tempfile.mkdtemp(prefix="nano_session_summary_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(mock_embedding_client: MagicMock, memory_dir: str) -> MemoryStore:
    """MemoryStore with mock embeddings. Dedup disabled — session summary tests
    verify the summary pipeline, not dedup (see test_dedup.py)."""
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        dedup_threshold=None,
        global_memory_dir=os.path.join(memory_dir, "_global"),
    )


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Mock LLMProvider that returns a predictable summary."""
    provider = MagicMock()
    provider.generate.return_value = LLMResponse(
        content=(
            "The user discussed their work as a software engineer and "
            "expressed frustration with frontend maintenance. They prefer "
            "Python over JavaScript. The conversation was casual and friendly."
        ),
        input_tokens=300,
        output_tokens=40,
        finish_reason="stop",
    )
    return provider


@pytest.fixture
def generator(
    mock_llm_provider: MagicMock, store: MemoryStore
) -> SessionSummaryGenerator:
    """SessionSummaryGenerator with mock LLM and real ChromaDB store."""
    return SessionSummaryGenerator(
        llm_provider=mock_llm_provider,
        memory_store=store,
        max_tokens=500,
    )


@pytest.fixture
def sample_turns() -> list[dict]:
    """A conversation with 6 turns (3 exchanges)."""
    return [
        {"role": "user", "content": "Hi, I'm Alex. I work as a software engineer."},
        {"role": "assistant", "content": "Nice to meet you, Alex! What kind of work do you do?"},
        {"role": "user", "content": "Mostly frontend maintenance. I'd rather do backend."},
        {"role": "assistant", "content": "That sounds frustrating! What would you rather be doing?"},
        {"role": "user", "content": "Python backend work. Or anything with actual architecture."},
        {"role": "assistant", "content": "Makes sense. Python has much better tooling for that."},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionSummaryGenerator:
    """Tests for SessionSummaryGenerator."""

    def test_generates_summary_from_turns(self, generator, sample_turns):
        """Happy path: generates summary and returns text."""
        result = generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )
        assert result is not None
        assert "software engineer" in result
        assert "frontend maintenance" in result

    def test_skips_short_sessions(self, generator):
        """Sessions with fewer than MIN_TURNS_FOR_SUMMARY turns return None."""
        short_turns = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = generator.generate_and_store(
            session_turns=short_turns,
            session_id="testchar_20260207_1100",
        )
        assert result is None

    def test_skips_empty_session(self, generator):
        """Empty session returns None."""
        result = generator.generate_and_store(
            session_turns=[],
            session_id="testchar_20260207_1100",
        )
        assert result is None

    def test_exact_threshold(self, generator):
        """Exactly MIN_TURNS_FOR_SUMMARY turns should trigger summary."""
        turns = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(MIN_TURNS_FOR_SUMMARY)
        ]
        result = generator.generate_and_store(
            session_turns=turns,
            session_id="testchar_20260207_1100",
        )
        assert result is not None

    def test_summary_stored_in_chromadb(self, generator, store, sample_turns):
        """Summary is stored in the summaries collection with correct metadata."""
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )

        summaries = store.get_all("summaries")
        assert len(summaries) == 1

        summary = summaries[0]
        assert "software engineer" in summary["content"]
        assert summary["metadata"]["type"] == "session_summary"
        assert summary["metadata"]["source"] == "session_summary_generator"
        assert summary["metadata"]["session_id"] == "testchar_20260207_1100"
        assert "timestamp" in summary["metadata"]

    def test_flash_cards_included_in_prompt(
        self, mock_llm_provider, store, sample_turns
    ):
        """When flash cards are provided, they appear in the LLM prompt."""
        generator = SessionSummaryGenerator(
            llm_provider=mock_llm_provider,
            memory_store=store,
            max_tokens=500,
        )
        flash_cards = [
            {"content": "Alex is a software engineer."},
            {"content": "Alex prefers backend work."},
        ]
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
            flash_cards=flash_cards,
        )

        call_args = mock_llm_provider.generate.call_args
        user_message = call_args.kwargs["messages"][1]["content"]
        assert "Alex is a software engineer" in user_message
        assert "Alex prefers backend work" in user_message

    def test_flash_cards_as_strings(self, mock_llm_provider, store, sample_turns):
        """Flash cards can also be plain strings instead of dicts."""
        generator = SessionSummaryGenerator(
            llm_provider=mock_llm_provider,
            memory_store=store,
            max_tokens=500,
        )
        flash_cards = [
            "Alex is a software engineer.",
            "Alex prefers backend work.",
        ]
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
            flash_cards=flash_cards,
        )

        call_args = mock_llm_provider.generate.call_args
        user_message = call_args.kwargs["messages"][1]["content"]
        assert "Alex is a software engineer" in user_message

    def test_no_flash_cards_shows_none(self, mock_llm_provider, store, sample_turns):
        """When no flash cards provided, prompt shows '(none)'."""
        generator = SessionSummaryGenerator(
            llm_provider=mock_llm_provider,
            memory_store=store,
            max_tokens=500,
        )
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )

        call_args = mock_llm_provider.generate.call_args
        user_message = call_args.kwargs["messages"][1]["content"]
        assert "(none)" in user_message

    def test_llm_failure_returns_none(self, store, sample_turns):
        """LLM exception returns None without storing anything."""
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("Connection refused")

        generator = SessionSummaryGenerator(
            llm_provider=provider,
            memory_store=store,
            max_tokens=500,
        )
        result = generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )
        assert result is None
        assert len(store.get_all("summaries")) == 0

    def test_empty_llm_response_returns_none(self, store, sample_turns):
        """Empty string from LLM returns None without storing."""
        provider = MagicMock()
        provider.generate.return_value = LLMResponse(
            content="   ",
            input_tokens=300,
            output_tokens=1,
            finish_reason="stop",
        )

        generator = SessionSummaryGenerator(
            llm_provider=provider,
            memory_store=store,
            max_tokens=500,
        )
        result = generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )
        assert result is None
        assert len(store.get_all("summaries")) == 0

    def test_summary_retrievable_via_query(self, generator, store, sample_turns):
        """Stored summary can be retrieved via semantic query."""
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )

        results = store.query("What does Alex do for work?", top_k=3)
        assert len(results) > 0
        summary_results = [r for r in results if r["collection"] == "summaries"]
        assert len(summary_results) == 1
        assert "software engineer" in summary_results[0]["content"]

    def test_uses_low_temperature(self, mock_llm_provider, store, sample_turns):
        """Summary generation uses low temperature for factual output."""
        generator = SessionSummaryGenerator(
            llm_provider=mock_llm_provider,
            memory_store=store,
            max_tokens=500,
        )
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )

        call_args = mock_llm_provider.generate.call_args
        assert call_args.kwargs["temperature"] == 0.3

    def test_system_prompt_is_summarizer_role(
        self, mock_llm_provider, store, sample_turns
    ):
        """System message establishes summarizer role, not character persona."""
        generator = SessionSummaryGenerator(
            llm_provider=mock_llm_provider,
            memory_store=store,
            max_tokens=500,
        )
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )

        call_args = mock_llm_provider.generate.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "summarizer" in system_msg.lower()

    def test_respects_max_tokens(self, mock_llm_provider, store, sample_turns):
        """max_tokens is passed through to the LLM call."""
        generator = SessionSummaryGenerator(
            llm_provider=mock_llm_provider,
            memory_store=store,
            max_tokens=250,
        )
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )

        call_args = mock_llm_provider.generate.call_args
        assert call_args.kwargs["max_tokens"] == 250

    def test_store_failure_returns_none(self, mock_llm_provider, sample_turns):
        """If ChromaDB storage fails, returns None."""
        store = MagicMock()
        store.add_session_summary.side_effect = RuntimeError("ChromaDB error")

        generator = SessionSummaryGenerator(
            llm_provider=mock_llm_provider,
            memory_store=store,
            max_tokens=500,
        )
        result = generator.generate_and_store(
            session_turns=sample_turns,
            session_id="testchar_20260207_1100",
        )
        assert result is None

    def test_multiple_summaries_accumulate(self, store, sample_turns):
        """Multiple calls store multiple summaries in the same collection."""
        # Different summary text per session (realistic — identical text would
        # be caught by NANO-102 content-hash dedup, which is correct behavior).
        provider = MagicMock()
        provider.generate.side_effect = [
            LLMResponse(
                content="Session one covered memory architecture and ChromaDB.",
                input_tokens=300, output_tokens=20, finish_reason="stop",
            ),
            LLMResponse(
                content="Session two focused on testing and deduplication.",
                input_tokens=300, output_tokens=20, finish_reason="stop",
            ),
        ]
        generator = SessionSummaryGenerator(
            llm_provider=provider, memory_store=store, max_tokens=500,
        )

        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="session_1",
        )
        generator.generate_and_store(
            session_turns=sample_turns,
            session_id="session_2",
        )

        summaries = store.get_all("summaries")
        assert len(summaries) == 2
        session_ids = {s["metadata"]["session_id"] for s in summaries}
        assert session_ids == {"session_1", "session_2"}
