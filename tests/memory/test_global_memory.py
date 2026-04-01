"""Tests for global memory tier — NANO-105."""

import hashlib
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from spindl.memory.embedding_client import EmbeddingClient
from spindl.memory.memory_store import MemoryStore, GENERAL, GLOBAL


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """Mock EmbeddingClient with deterministic fake embeddings."""
    client = MagicMock(spec=EmbeddingClient)

    def fake_embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(64)]

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        return [fake_embed(t) for t in texts]

    client.embed.side_effect = fake_embed
    client.embed_batch.side_effect = fake_embed_batch
    return client


@pytest.fixture
def memory_dir() -> str:
    d = tempfile.mkdtemp(prefix="nano_global_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def global_dir() -> str:
    d = tempfile.mkdtemp(prefix="nano_global_mem_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(mock_embedding_client: MagicMock, memory_dir: str, global_dir: str) -> MemoryStore:
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        dedup_threshold=None,
        global_memory_dir=global_dir,
    )


class TestGlobalMemoryCRUD:
    """CRUD operations on the global memory tier."""

    def test_add_global_memory(self, store: MemoryStore) -> None:
        doc_id = store.add_global("User lives in Southern California")
        assert doc_id
        assert store.counts["global"] == 1

    def test_get_all_global(self, store: MemoryStore) -> None:
        store.add_global("User has a wife")
        store.add_global("User is Filipino-American")
        items = store.get_all_global()
        assert len(items) == 2
        contents = {item["content"] for item in items}
        assert "User has a wife" in contents
        assert "User is Filipino-American" in contents

    def test_get_all_via_collection_name(self, store: MemoryStore) -> None:
        """get_all('global') routes to get_all_global()."""
        store.add_global("Test fact")
        items = store.get_all("global")
        assert len(items) == 1

    def test_delete_global(self, store: MemoryStore) -> None:
        doc_id = store.add_global("Temporary fact")
        assert store.counts["global"] == 1
        store.delete_global(doc_id)
        assert store.counts["global"] == 0

    def test_delete_via_collection_name(self, store: MemoryStore) -> None:
        """delete('global', id) routes to global collection."""
        doc_id = store.add_global("Will be deleted")
        result = store.delete("global", doc_id)
        assert result is True
        assert store.counts["global"] == 0

    def test_edit_global(self, store: MemoryStore) -> None:
        doc_id = store.add_global("User likes cats")
        new_id = store.edit_global(doc_id, "User likes dogs")
        assert new_id != doc_id
        assert store.counts["global"] == 1
        items = store.get_all_global()
        assert items[0]["content"] == "User likes dogs"
        assert "edited_at" in items[0]["metadata"]

    def test_global_metadata(self, store: MemoryStore) -> None:
        meta = {"type": "global", "source": "gui_manual"}
        store.add_global("Fact with meta", meta)
        items = store.get_all_global()
        assert items[0]["metadata"]["type"] == "global"
        assert items[0]["metadata"]["source"] == "gui_manual"
        assert "timestamp" in items[0]["metadata"]


class TestGlobalMemoryIsolation:
    """Global memories persist across character switches."""

    def test_global_survives_character_switch(
        self, mock_embedding_client: MagicMock, memory_dir: str, global_dir: str
    ) -> None:
        store = MemoryStore(
            "alice", memory_dir, mock_embedding_client,
            dedup_threshold=None, global_memory_dir=global_dir,
        )
        store.add_global("Cross-character fact")
        store.add_general("Alice-specific fact")

        store.switch_character("bob")

        # Global still there
        assert store.counts["global"] == 1
        global_items = store.get_all_global()
        assert global_items[0]["content"] == "Cross-character fact"

        # Per-character general is now bob's (empty)
        assert store.counts[GENERAL] == 0

    def test_global_not_affected_by_switch(
        self, mock_embedding_client: MagicMock, memory_dir: str, global_dir: str
    ) -> None:
        store = MemoryStore(
            "char1", memory_dir, mock_embedding_client,
            dedup_threshold=None, global_memory_dir=global_dir,
        )
        store.add_global("Shared knowledge")

        # Switch multiple times
        store.switch_character("char2")
        store.switch_character("char3")

        assert store.counts["global"] == 1


class TestGlobalInQuery:
    """Global memories appear in unified query results."""

    def test_query_includes_global(self, store: MemoryStore) -> None:
        store.add_global("User lives in SoCal")
        store.add_general("Spindle likes tea")

        results = store.query("Where does the user live?", top_k=5)
        collections = {r["collection"] for r in results}
        assert "global" in collections
        assert GENERAL in collections

    def test_query_global_only(self, store: MemoryStore) -> None:
        """Query returns global results even with no character memories."""
        store.add_global("Important global fact")
        results = store.query("important", top_k=5)
        assert len(results) >= 1
        assert results[0]["collection"] == "global"


class TestCountsIncludeGlobal:
    """Counts property includes global tier."""

    def test_counts_has_global_key(self, store: MemoryStore) -> None:
        assert "global" in store.counts

    def test_counts_global_zero_initially(self, store: MemoryStore) -> None:
        assert store.counts["global"] == 0

    def test_counts_global_increments(self, store: MemoryStore) -> None:
        store.add_global("Fact 1")
        store.add_global("Fact 2")
        assert store.counts["global"] == 2
