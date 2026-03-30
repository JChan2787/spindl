"""Tests for MemoryStore — ChromaDB wrapper with per-character collections."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from spindl.memory.embedding_client import EmbeddingClient
from spindl.memory.memory_store import MemoryStore, GENERAL, FLASHCARDS, SUMMARIES


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """
    Mock EmbeddingClient that returns deterministic fake embeddings.

    Each text gets a unique-ish embedding based on its hash, ensuring
    different texts produce different vectors for meaningful similarity queries.
    """
    client = MagicMock(spec=EmbeddingClient)

    def fake_embed(text: str) -> list[float]:
        h = hash(text) % 10000
        # 64-dim fake vector — enough for ChromaDB to work with
        base = [(h + i) / 10000.0 for i in range(64)]
        return base

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        return [fake_embed(t) for t in texts]

    client.embed.side_effect = fake_embed
    client.embed_batch.side_effect = fake_embed_batch
    return client


@pytest.fixture
def memory_dir() -> str:
    """Create a temporary directory for ChromaDB persistence. Cleaned up after test."""
    d = tempfile.mkdtemp(prefix="nano_memory_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(mock_embedding_client: MagicMock, memory_dir: str) -> MemoryStore:
    """Create a MemoryStore with mock embeddings for testing.
    Dedup disabled — basic store tests verify CRUD, not dedup (see test_dedup.py)."""
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        dedup_threshold=None,
        global_memory_dir=os.path.join(memory_dir, "_global"),
    )


class TestMemoryStoreInit:
    """Tests for MemoryStore initialization."""

    def test_creates_three_collections(self, store: MemoryStore) -> None:
        """MemoryStore creates general, flashcards, and summaries collections."""
        counts = store.counts
        assert GENERAL in counts
        assert FLASHCARDS in counts
        assert SUMMARIES in counts

    def test_empty_on_creation(self, store: MemoryStore) -> None:
        """All collections start empty."""
        counts = store.counts
        assert all(c == 0 for c in counts.values())

    def test_character_id_property(self, store: MemoryStore) -> None:
        """character_id property returns the configured ID."""
        assert store.character_id == "testchar"

    def test_persistence_across_instances(
        self, mock_embedding_client: MagicMock, memory_dir: str
    ) -> None:
        """Data persists when creating a new MemoryStore pointing to same directory."""
        store1 = MemoryStore("testchar", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        store1.add_general("Alex likes mango smoothies")

        # New instance, same directory
        store2 = MemoryStore("testchar", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        assert store2.counts[GENERAL] == 1

        items = store2.get_all(GENERAL)
        assert len(items) == 1
        assert items[0]["content"] == "Alex likes mango smoothies"


class TestAddOperations:
    """Tests for adding memories to collections."""

    def test_add_flash_card(self, store: MemoryStore) -> None:
        """add_flash_card stores content and returns a document ID."""
        doc_id = store.add_flash_card("Q: What is Alex's job? A: Software engineer")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
        assert store.counts[FLASHCARDS] == 1

    def test_add_session_summary(self, store: MemoryStore) -> None:
        """add_session_summary stores content in summaries collection."""
        doc_id = store.add_session_summary("Alex discussed memory architecture tonight.")
        assert isinstance(doc_id, str)
        assert store.counts[SUMMARIES] == 1

    def test_add_general(self, store: MemoryStore) -> None:
        """add_general stores content in general collection."""
        doc_id = store.add_general("Alex's favorite color is purple")
        assert isinstance(doc_id, str)
        assert store.counts[GENERAL] == 1

    def test_add_with_metadata(self, store: MemoryStore) -> None:
        """Metadata is stored alongside content."""
        doc_id = store.add_flash_card(
            "Q: Favorite food? A: Mango smoothies",
            metadata={"session_id": "test_001", "source": "reflection"},
        )

        items = store.get_all(FLASHCARDS)
        assert len(items) == 1
        assert items[0]["metadata"]["session_id"] == "test_001"
        assert items[0]["metadata"]["source"] == "reflection"

    def test_auto_timestamp_in_metadata(self, store: MemoryStore) -> None:
        """Timestamp is auto-injected into metadata when not provided."""
        store.add_general("Some fact")
        items = store.get_all(GENERAL)
        assert "timestamp" in items[0]["metadata"]

    def test_custom_timestamp_preserved(self, store: MemoryStore) -> None:
        """Custom timestamp in metadata is not overwritten."""
        store.add_general("Some fact", metadata={"timestamp": "2026-01-01T00:00:00Z"})
        items = store.get_all(GENERAL)
        assert items[0]["metadata"]["timestamp"] == "2026-01-01T00:00:00Z"

    def test_multiple_adds_to_same_collection(self, store: MemoryStore) -> None:
        """Multiple documents can be added to the same collection."""
        store.add_flash_card("Fact 1")
        store.add_flash_card("Fact 2")
        store.add_flash_card("Fact 3")
        assert store.counts[FLASHCARDS] == 3

    def test_unique_ids_per_add(self, store: MemoryStore) -> None:
        """Each add generates a unique document ID."""
        id1 = store.add_flash_card("Fact 1")
        id2 = store.add_flash_card("Fact 2")
        assert id1 != id2


class TestQueryOperations:
    """Tests for querying memories across collections."""

    def test_query_empty_store_returns_empty(self, store: MemoryStore) -> None:
        """Querying with no data returns empty list."""
        results = store.query("anything")
        assert results == []

    def test_query_returns_relevant_results(self, store: MemoryStore) -> None:
        """Query returns stored documents with content, score, and metadata."""
        store.add_general("Alex likes mango smoothies")
        store.add_general("Alex works as a software engineer")

        results = store.query("What does Alex like to drink?")
        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("collection" in r for r in results)
        assert all("distance" in r for r in results)
        assert all("score" in r for r in results)

    def test_query_searches_all_collections(self, store: MemoryStore) -> None:
        """Query searches general, flashcards, and summaries."""
        store.add_general("General fact")
        store.add_flash_card("Flash card fact")
        store.add_session_summary("Summary fact")

        results = store.query("fact")
        collections_found = {r["collection"] for r in results}
        assert GENERAL in collections_found
        assert FLASHCARDS in collections_found
        assert SUMMARIES in collections_found

    def test_query_results_sorted_by_score(self, store: MemoryStore) -> None:
        """Results are sorted by composite score (descending — highest first)."""
        store.add_general("Fact A")
        store.add_general("Fact B")
        store.add_flash_card("Fact C")

        results = store.query("test query")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_respects_top_k(self, store: MemoryStore) -> None:
        """top_k limits results per collection."""
        for i in range(10):
            store.add_general(f"General fact {i}")

        results = store.query("fact", top_k=3)
        general_results = [r for r in results if r["collection"] == GENERAL]
        assert len(general_results) <= 3

    def test_query_result_includes_id(self, store: MemoryStore) -> None:
        """Query results include document ID for delete operations."""
        doc_id = store.add_general("Deletable fact")
        results = store.query("deletable")
        assert any(r["id"] == doc_id for r in results)


class TestQueryRetry:
    """Tests for defensive retry on transient collection errors."""

    def test_retry_succeeds_after_first_query_fails(
        self, store: MemoryStore
    ) -> None:
        """When initial query fails, retry with fresh handle returns results."""
        store.add_general("Alex likes mango smoothies")

        # Sabotage the cached collection handle so the first query raises
        original_collection = store._collections[GENERAL]
        call_count = 0
        original_query = original_collection.query

        def failing_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("HNSW segment reader: Nothing found on disk")
            return original_query(*args, **kwargs)

        original_collection.query = failing_then_ok

        # query() should catch the error, re-fetch handle, and succeed
        results = store.query("mango")
        assert len(results) == 1
        assert results[0]["content"] == "Alex likes mango smoothies"

    def test_retry_replaces_cached_handle(self, store: MemoryStore) -> None:
        """After retry, the cached collection handle is updated."""
        store.add_general("Some fact")

        old_handle = store._collections[GENERAL]
        old_handle.query = MagicMock(
            side_effect=RuntimeError("HNSW error")
        )

        store.query("anything")

        # The cached handle should have been replaced
        assert store._collections[GENERAL] is not old_handle

    def test_retry_failure_still_returns_other_collections(
        self, store: MemoryStore
    ) -> None:
        """If both initial query and retry fail, other collections still return."""
        store.add_general("General fact")
        store.add_flash_card("Flash card fact")

        # Sabotage general so both attempts fail
        store._collections[GENERAL].query = MagicMock(
            side_effect=RuntimeError("Permanent failure")
        )
        # Also need get_or_create_collection to return a broken handle
        broken = MagicMock()
        broken.query = MagicMock(side_effect=RuntimeError("Still broken"))
        broken.count = MagicMock(return_value=1)
        store._client.get_or_create_collection = MagicMock(return_value=broken)

        results = store.query("fact")
        collections_found = {r["collection"] for r in results}
        assert FLASHCARDS in collections_found
        assert GENERAL not in collections_found

    def test_no_retry_on_empty_collection(self, store: MemoryStore) -> None:
        """Empty collections are skipped entirely — no retry needed."""
        # All collections empty, query should return [] with no errors
        results = store.query("anything")
        assert results == []


class TestDeleteOperations:
    """Tests for deleting memories."""

    def test_delete_existing_document(self, store: MemoryStore) -> None:
        """delete() removes a document by ID."""
        doc_id = store.add_general("To be deleted")
        assert store.counts[GENERAL] == 1

        result = store.delete(GENERAL, doc_id)
        assert result is True
        assert store.counts[GENERAL] == 0

    def test_delete_unknown_collection_returns_false(self, store: MemoryStore) -> None:
        """delete() returns False for unknown collection names."""
        result = store.delete("nonexistent", "some-id")
        assert result is False

    def test_clear_flash_cards(self, store: MemoryStore) -> None:
        """clear_flash_cards() removes all flash cards."""
        store.add_flash_card("Card 1")
        store.add_flash_card("Card 2")
        store.add_flash_card("Card 3")
        assert store.counts[FLASHCARDS] == 3

        store.clear_flash_cards()
        assert store.counts[FLASHCARDS] == 0

    def test_clear_flash_cards_preserves_other_collections(
        self, store: MemoryStore
    ) -> None:
        """clear_flash_cards() does not affect general or summaries."""
        store.add_general("Permanent fact")
        store.add_session_summary("Session summary")
        store.add_flash_card("Ephemeral card")

        store.clear_flash_cards()

        assert store.counts[GENERAL] == 1
        assert store.counts[SUMMARIES] == 1
        assert store.counts[FLASHCARDS] == 0


class TestGetAll:
    """Tests for get_all method."""

    def test_get_all_empty_collection(self, store: MemoryStore) -> None:
        """get_all on empty collection returns empty list."""
        result = store.get_all(GENERAL)
        assert result == []

    def test_get_all_returns_all_documents(self, store: MemoryStore) -> None:
        """get_all returns every document in the collection."""
        store.add_general("Fact 1")
        store.add_general("Fact 2")
        store.add_general("Fact 3")

        items = store.get_all(GENERAL)
        assert len(items) == 3
        contents = {item["content"] for item in items}
        assert contents == {"Fact 1", "Fact 2", "Fact 3"}

    def test_get_all_unknown_collection(self, store: MemoryStore) -> None:
        """get_all on unknown collection returns empty list."""
        result = store.get_all("nonexistent")
        assert result == []

    def test_get_all_includes_metadata(self, store: MemoryStore) -> None:
        """get_all includes metadata for each document."""
        store.add_general("Fact", metadata={"source": "manual"})
        items = store.get_all(GENERAL)
        assert items[0]["metadata"]["source"] == "manual"


class TestMultipleCharacters:
    """Tests verifying per-character isolation."""

    def test_separate_characters_have_separate_data(
        self, mock_embedding_client: MagicMock, memory_dir: str
    ) -> None:
        """Different character IDs create isolated collection namespaces."""
        store_a = MemoryStore("alice", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        store_b = MemoryStore("bob", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))

        store_a.add_general("Alice's fact")
        store_b.add_general("Bob's fact")

        assert store_a.counts[GENERAL] == 1
        assert store_b.counts[GENERAL] == 1

        alice_items = store_a.get_all(GENERAL)
        bob_items = store_b.get_all(GENERAL)

        assert alice_items[0]["content"] == "Alice's fact"
        assert bob_items[0]["content"] == "Bob's fact"
