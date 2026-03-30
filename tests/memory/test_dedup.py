"""Tests for NANO-102: Memory deduplication — content-hash, similarity gate, session guard."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from spindl.memory.embedding_client import EmbeddingClient
from spindl.memory.memory_store import (
    FLASHCARDS,
    GENERAL,
    SUMMARIES,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_embedding(text: str, dim: int = 64) -> list[float]:
    """Deterministic fake embedding from hash. Same text = same vector."""
    h = hash(text) % 10000
    return [(h + i) / 10000.0 for i in range(dim)]


def _make_similar_embedding(text: str, offset: float = 0.001, dim: int = 64) -> list[float]:
    """Produce a vector very close to the base text's embedding."""
    base = _make_embedding(text, dim)
    return [v + offset for v in base]


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    client = MagicMock(spec=EmbeddingClient)

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        return [_make_embedding(t) for t in texts]

    client.embed.side_effect = _make_embedding
    client.embed_batch.side_effect = fake_embed_batch
    return client


@pytest.fixture
def similar_embedding_client() -> MagicMock:
    """Embedding client where paraphrases map to nearly identical vectors."""
    client = MagicMock(spec=EmbeddingClient)

    # Map of text → vector override (paraphrases share similar vectors)
    _overrides: dict[str, list[float]] = {}

    def register_similar(base_text: str, similar_text: str):
        base_vec = _make_embedding(base_text)
        _overrides[similar_text] = [v + 0.001 for v in base_vec]

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        results = []
        for t in texts:
            if t in _overrides:
                results.append(_overrides[t])
            else:
                results.append(_make_embedding(t))
        return results

    client.embed.side_effect = lambda t: _overrides.get(t, _make_embedding(t))
    client.embed_batch.side_effect = fake_embed_batch
    client._register_similar = register_similar
    client._overrides = _overrides
    return client


@pytest.fixture
def memory_dir():
    d = tempfile.mkdtemp(prefix="nano_dedup_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(mock_embedding_client, memory_dir) -> MemoryStore:
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        dedup_threshold=0.30,
        global_memory_dir=os.path.join(memory_dir, "_global"),
    )


@pytest.fixture
def store_no_dedup(mock_embedding_client, memory_dir) -> MemoryStore:
    """MemoryStore with dedup disabled (threshold=None)."""
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        dedup_threshold=None,
        global_memory_dir=os.path.join(memory_dir, "_global"),
    )


# ---------------------------------------------------------------------------
# Phase 0: Content-hash exact dedup
# ---------------------------------------------------------------------------


class TestContentHashDedup:

    def test_exact_duplicate_blocked(self, store: MemoryStore):
        """Adding identical content twice returns same ID, count stays 1."""
        id1 = store.add_flash_card("Q: Favorite drink? A: Monster Energy")
        id2 = store.add_flash_card("Q: Favorite drink? A: Monster Energy")
        assert id1 == id2
        assert store.counts[FLASHCARDS] == 1

    def test_content_hash_deterministic(self, store: MemoryStore):
        """Same content always produces same ID."""
        content = "User prefers dark mode"
        id1 = MemoryStore._content_id(content)
        id2 = MemoryStore._content_id(content)
        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex digest

    def test_different_content_different_ids(self, store: MemoryStore):
        """Different content produces different IDs and both are stored."""
        id1 = store.add_flash_card("Fact A")
        id2 = store.add_flash_card("Fact B")
        assert id1 != id2
        assert store.counts[FLASHCARDS] == 2

    def test_exact_dedup_across_sessions(self, mock_embedding_client, memory_dir):
        """Content-hash dedup works across MemoryStore instances (persistence)."""
        store1 = MemoryStore("testchar", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        id1 = store1.add_flash_card("Persistent fact")

        store2 = MemoryStore("testchar", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        id2 = store2.add_flash_card("Persistent fact")
        assert id1 == id2
        assert store2.counts[FLASHCARDS] == 1

    def test_exact_dedup_still_active_when_semantic_dedup_disabled(self, store_no_dedup):
        """Content-hash dedup works even when threshold is None."""
        id1 = store_no_dedup.add_flash_card("Same content")
        id2 = store_no_dedup.add_flash_card("Same content")
        assert id1 == id2
        assert store_no_dedup.counts[FLASHCARDS] == 1


# ---------------------------------------------------------------------------
# Phase 1: Semantic similarity gate
# ---------------------------------------------------------------------------


class TestSimilarityGate:

    def test_dedup_allows_different_content(self, mock_embedding_client, memory_dir):
        """Genuinely different facts are both stored when vectors are far apart."""
        # Override embeddings: map specific text to controlled vectors
        vec_map = {
            "User likes Monster Energy drinks": [0.1] * 64,
            "User works as a software engineer": [0.9] * 64,
        }

        def controlled_embed_batch(texts):
            return [vec_map.get(t, [0.5] * 64) for t in texts]

        mock_embedding_client.embed_batch.side_effect = controlled_embed_batch

        store = MemoryStore(
            character_id="testchar",
            memory_dir=memory_dir,
            embedding_client=mock_embedding_client,
            dedup_threshold=0.30,
            global_memory_dir=os.path.join(memory_dir, "_global"),
        )

        id1 = store.add_flash_card("User likes Monster Energy drinks")
        id2 = store.add_flash_card("User works as a software engineer")
        assert id1 != id2
        assert store.counts[FLASHCARDS] == 2

    def test_dedup_skips_near_duplicate(self, similar_embedding_client, memory_dir):
        """Semantically similar content is deduplicated."""
        # Register paraphrase as similar embedding
        similar_embedding_client._register_similar(
            "Jay drinks Monster Energy",
            "User's preferred drink is Monster Energy",
        )

        store = MemoryStore(
            character_id="testchar",
            memory_dir=memory_dir,
            embedding_client=similar_embedding_client,
            dedup_threshold=0.30,
            global_memory_dir=os.path.join(memory_dir, "_global"),
        )

        id1 = store.add_flash_card("Jay drinks Monster Energy")
        id2 = store.add_flash_card("User's preferred drink is Monster Energy")
        # Second add should return the first entry's ID (near-duplicate)
        assert id1 == id2
        assert store.counts[FLASHCARDS] == 1

    def test_dedup_disabled_when_none(self, store_no_dedup):
        """With threshold=None, semantic dedup is disabled (different content stored)."""
        id1 = store_no_dedup.add_flash_card("Fact version 1")
        id2 = store_no_dedup.add_flash_card("Fact version 2")
        assert id1 != id2
        assert store_no_dedup.counts[FLASHCARDS] == 2

    def test_dedup_per_call_override(self, similar_embedding_client, memory_dir):
        """Per-call dedup_threshold=None bypasses similarity gate."""
        similar_embedding_client._register_similar(
            "Original fact",
            "Paraphrased fact",
        )

        store = MemoryStore(
            character_id="testchar",
            memory_dir=memory_dir,
            embedding_client=similar_embedding_client,
            dedup_threshold=0.30,
            global_memory_dir=os.path.join(memory_dir, "_global"),
        )

        store.add_flash_card("Original fact")
        # add_general bypasses semantic dedup (per ticket: manual entries skip)
        id_manual = store.add_general("Original fact different collection")
        assert store.counts[GENERAL] == 1

    def test_dedup_empty_collection_skips_query(self, store: MemoryStore):
        """No similarity query when collection is empty (count=0)."""
        # Should not raise even though there's nothing to query against
        doc_id = store.add_flash_card("First ever entry")
        assert doc_id is not None
        assert store.counts[FLASHCARDS] == 1

    def test_dedup_cross_collection_isolation(self, store: MemoryStore):
        """Same content in different collections are independent."""
        id_flash = store.add_flash_card("User likes purple")
        id_summary = store.add_session_summary("User likes purple")
        # Different collections — both should store
        assert store.counts[FLASHCARDS] == 1
        assert store.counts[SUMMARIES] == 1

    def test_manual_entry_bypasses_semantic_dedup(self, similar_embedding_client, memory_dir):
        """add_general() passes dedup_threshold=None, skipping Phase 1."""
        similar_embedding_client._register_similar(
            "User likes cats",
            "User is a cat person",
        )

        store = MemoryStore(
            character_id="testchar",
            memory_dir=memory_dir,
            embedding_client=similar_embedding_client,
            dedup_threshold=0.30,
            global_memory_dir=os.path.join(memory_dir, "_global"),
        )

        store.add_general("User likes cats")
        # Manual entry with similar content — should NOT be deduplicated
        id2 = store.add_general("User is a cat person")
        assert store.counts[GENERAL] == 2


# ---------------------------------------------------------------------------
# Phase 3: Session summary guard
# ---------------------------------------------------------------------------


class TestSessionSummaryGuard:

    def test_session_summary_guard_blocks_duplicate(self, store_no_dedup: MemoryStore):
        """generate_and_store for same session_id twice, second returns None."""
        from unittest.mock import MagicMock as Mock
        from spindl.memory.session_summary import SessionSummaryGenerator

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "This was a productive session about memory architecture."
        mock_provider.generate.return_value = mock_response

        generator = SessionSummaryGenerator(
            llm_provider=mock_provider,
            memory_store=store_no_dedup,
        )

        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm great!"},
            {"role": "user", "content": "Tell me about memory"},
        ]

        # First summary — should store
        result1 = generator.generate_and_store(turns, session_id="session_001")
        assert result1 is not None
        assert store_no_dedup.counts[SUMMARIES] == 1

        # Second summary same session_id — should skip
        result2 = generator.generate_and_store(turns, session_id="session_001")
        assert result2 is None
        assert store_no_dedup.counts[SUMMARIES] == 1

    def test_session_summary_different_sessions_allowed(self, store_no_dedup: MemoryStore):
        """Different session_ids can each have their own summary."""
        from unittest.mock import MagicMock as Mock
        from spindl.memory.session_summary import SessionSummaryGenerator

        mock_provider = Mock()
        # Different content per call so content-hash IDs differ
        responses = [Mock(content="Session one was about memory."), Mock(content="Session two was about testing.")]
        mock_provider.generate.side_effect = responses

        generator = SessionSummaryGenerator(
            llm_provider=mock_provider,
            memory_store=store_no_dedup,
        )

        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "See ya"},
            {"role": "user", "content": "One more"},
        ]

        result1 = generator.generate_and_store(turns, session_id="session_001")
        result2 = generator.generate_and_store(turns, session_id="session_002")
        assert result1 is not None
        assert result2 is not None
        assert store_no_dedup.counts[SUMMARIES] == 2
