"""Tests for MemoryStore.switch_character (NANO-077).

Tests cover:
- switch_character updates character_id
- switch_character re-derives collection names
- Data added before switch is NOT accessible after switch
- Data added after switch is in the new character's collections
"""

import hashlib
import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from spindl.memory.embedding_client import EmbeddingClient
from spindl.memory.memory_store import MemoryStore, GENERAL, FLASHCARDS, SUMMARIES


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """Mock EmbeddingClient with deterministic fake embeddings."""
    client = MagicMock(spec=EmbeddingClient)

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            results.append([digest[i % len(digest)] / 255.0 for i in range(64)])
        return results

    client.embed_batch.side_effect = fake_embed_batch
    return client


@pytest.fixture
def memory_dir():
    d = tempfile.mkdtemp(prefix="nano_switch_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestSwitchCharacter:

    def test_character_id_updated(self, mock_embedding_client, memory_dir) -> None:
        store = MemoryStore("spindle", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        assert store.character_id == "spindle"

        store.switch_character("mryummers")
        assert store.character_id == "mryummers"

    def test_collection_names_updated(self, mock_embedding_client, memory_dir) -> None:
        store = MemoryStore("spindle", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        store.switch_character("mryummers")

        # Verify collections are for new character by checking they exist
        for suffix in (GENERAL, FLASHCARDS, SUMMARIES):
            col = store._collections[suffix]
            assert col.name == f"mryummers_{suffix}"

    def test_old_data_not_accessible_after_switch(
        self, mock_embedding_client, memory_dir
    ) -> None:
        store = MemoryStore("spindle", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        store.add_general("Spindle likes tea")

        store.switch_character("mryummers")

        # New character's general collection should be empty
        items = store.get_all(GENERAL)
        assert len(items) == 0

    def test_new_data_lands_in_new_collections(
        self, mock_embedding_client, memory_dir
    ) -> None:
        store = MemoryStore("spindle", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        store.switch_character("mryummers")

        store.add_general("Mister Yummers is creepy")
        items = store.get_all(GENERAL)
        assert len(items) == 1
        assert items[0]["content"] == "Mister Yummers is creepy"

    def test_switch_back_recovers_old_data(
        self, mock_embedding_client, memory_dir
    ) -> None:
        store = MemoryStore("spindle", memory_dir, mock_embedding_client, global_memory_dir=os.path.join(memory_dir, "_global"))
        store.add_general("Spindle likes tea")

        store.switch_character("mryummers")
        store.add_general("Mister Yummers is creepy")

        store.switch_character("spindle")
        items = store.get_all(GENERAL)
        assert len(items) == 1
        assert items[0]["content"] == "Spindle likes tea"
