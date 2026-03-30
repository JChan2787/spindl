"""
Tests for NANO-043 Phase 6 — Memory Curation GUI socket handlers.

Tests cover all 8 memory socket handlers in GUIServer:
- request_memory_counts
- request_memories
- add_general_memory
- edit_general_memory
- delete_memory
- promote_memory
- search_memories
- clear_flashcards
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spindl.gui.server import GUIServer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_memory_store() -> MagicMock:
    """Create a mock MemoryStore with all methods used by socket handlers."""
    store = MagicMock()
    store.counts = {"general": 3, "flashcards": 5, "summaries": 2}
    store.get_all.return_value = [
        {"id": "mem-1", "content": "User likes coffee", "metadata": {"type": "general", "timestamp": "2026-02-07T12:00:00Z"}},
        {"id": "mem-2", "content": "User prefers dark mode", "metadata": {"type": "general", "timestamp": "2026-02-07T12:01:00Z"}},
    ]
    store.add_general.return_value = "new-mem-id"
    store.delete.return_value = True
    store.clear_flash_cards.return_value = None
    store.query.return_value = [
        {"id": "mem-1", "content": "User likes coffee", "collection": "general", "distance": 0.3, "metadata": {}},
        {"id": "fc-1", "content": "Q: What drink? A: Coffee", "collection": "flashcards", "distance": 0.5, "metadata": {}},
    ]
    return store


@pytest.fixture
def mock_orchestrator(mock_memory_store: MagicMock) -> MagicMock:
    """Create a mock orchestrator with a memory_store property."""
    orch = MagicMock()
    orch.memory_store = mock_memory_store
    return orch


@pytest.fixture
def server() -> GUIServer:
    """Create a GUIServer instance for testing."""
    return GUIServer(host="127.0.0.1", port=0)


@pytest.fixture
def server_with_orchestrator(server: GUIServer, mock_orchestrator: MagicMock) -> GUIServer:
    """Create a GUIServer with an attached orchestrator."""
    server._orchestrator = mock_orchestrator
    return server


# =============================================================================
# Helper to dispatch socket events through the server's registered handlers
# =============================================================================


async def _dispatch(server: GUIServer, event_name: str, data: dict) -> list[tuple]:
    """
    Simulate a socket.io event dispatch.

    Patches sio.emit to capture emitted responses, then invokes
    the registered handler directly.

    Returns list of (event, data, kwargs) tuples that were emitted.
    """
    emitted = []

    async def capture_emit(event, payload=None, **kwargs):
        emitted.append((event, payload, kwargs))

    server.sio.emit = capture_emit

    # The handlers are registered as async functions on the sio object
    # via @self.sio.event decorator. We access them through the handlers dict.
    handler = server.sio.handlers.get("/", {}).get(event_name)
    if handler is None:
        raise ValueError(f"No handler registered for event '{event_name}'")

    await handler("test-sid", data)
    return emitted


# =============================================================================
# request_memory_counts
# =============================================================================


class TestRequestMemoryCounts:
    """Tests for the request_memory_counts handler."""

    @pytest.mark.asyncio
    async def test_returns_counts_when_available(self, server_with_orchestrator: GUIServer) -> None:
        """Should return counts and enabled=True when memory store exists."""
        emitted = await _dispatch(server_with_orchestrator, "request_memory_counts", {})

        assert len(emitted) == 1
        event, data, _ = emitted[0]
        assert event == "memory_counts"
        assert data["general"] == 3
        assert data["flashcards"] == 5
        assert data["summaries"] == 2
        assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_returns_disabled_when_no_orchestrator(self, server: GUIServer) -> None:
        """Should return zeros and enabled=False when no orchestrator."""
        server._orchestrator = None
        emitted = await _dispatch(server, "request_memory_counts", {})

        assert len(emitted) == 1
        event, data, _ = emitted[0]
        assert event == "memory_counts"
        assert data["enabled"] is False
        assert data["general"] == 0

    @pytest.mark.asyncio
    async def test_returns_disabled_when_no_memory_store(self, server: GUIServer) -> None:
        """Should return disabled when orchestrator has no memory_store."""
        orch = MagicMock()
        orch.memory_store = None
        server._orchestrator = orch
        emitted = await _dispatch(server, "request_memory_counts", {})

        event, data, _ = emitted[0]
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self, server_with_orchestrator: GUIServer) -> None:
        """Should emit error payload on exception."""
        server_with_orchestrator._orchestrator.memory_store.counts = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("DB error"))
        )
        # Make counts raise
        type(server_with_orchestrator._orchestrator.memory_store).counts = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("DB error"))
        )
        emitted = await _dispatch(server_with_orchestrator, "request_memory_counts", {})

        event, data, _ = emitted[0]
        assert "error" in data


# =============================================================================
# request_memories
# =============================================================================


class TestRequestMemories:
    """Tests for the request_memories handler."""

    @pytest.mark.asyncio
    async def test_returns_memories_for_collection(self, server_with_orchestrator: GUIServer) -> None:
        """Should return memories list for the requested collection."""
        emitted = await _dispatch(server_with_orchestrator, "request_memories", {"collection": "general"})

        event, data, _ = emitted[0]
        assert event == "memory_list"
        assert data["collection"] == "general"
        assert len(data["memories"]) == 2
        assert data["memories"][0]["id"] == "mem-1"

    @pytest.mark.asyncio
    async def test_missing_collection_param(self, server_with_orchestrator: GUIServer) -> None:
        """Should return empty list when collection param is missing."""
        emitted = await _dispatch(server_with_orchestrator, "request_memories", {})

        event, data, _ = emitted[0]
        assert event == "memory_list"
        assert data["memories"] == []

    @pytest.mark.asyncio
    async def test_no_orchestrator(self, server: GUIServer) -> None:
        """Should return error when no orchestrator."""
        server._orchestrator = None
        emitted = await _dispatch(server, "request_memories", {"collection": "general"})

        event, data, _ = emitted[0]
        assert data["memories"] == []
        assert data.get("error") is not None


# =============================================================================
# add_general_memory
# =============================================================================


class TestAddGeneralMemory:
    """Tests for the add_general_memory handler."""

    @pytest.mark.asyncio
    async def test_adds_memory_successfully(self, server_with_orchestrator: GUIServer) -> None:
        """Should add a general memory and return success."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "add_general_memory",
            {"content": "User enjoys hiking"},
        )

        event, data, _ = emitted[0]
        assert event == "memory_added"
        assert data["success"] is True
        assert data["memory"]["id"] == "new-mem-id"
        assert data["memory"]["content"] == "User enjoys hiking"

    @pytest.mark.asyncio
    async def test_rejects_empty_content(self, server_with_orchestrator: GUIServer) -> None:
        """Should reject empty/whitespace-only content."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "add_general_memory",
            {"content": "   "},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False
        assert "required" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_missing_content(self, server_with_orchestrator: GUIServer) -> None:
        """Should reject when content key is missing."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "add_general_memory",
            {},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_no_orchestrator(self, server: GUIServer) -> None:
        """Should fail gracefully with no orchestrator."""
        server._orchestrator = None
        emitted = await _dispatch(
            server,
            "add_general_memory",
            {"content": "test"},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False
        assert "not available" in data["error"].lower()


# =============================================================================
# edit_general_memory
# =============================================================================


class TestEditGeneralMemory:
    """Tests for the edit_general_memory handler."""

    @pytest.mark.asyncio
    async def test_edits_memory_successfully(self, server_with_orchestrator: GUIServer) -> None:
        """Should delete old, re-add with new content, preserve timestamp."""
        store = server_with_orchestrator._orchestrator.memory_store
        emitted = await _dispatch(
            server_with_orchestrator,
            "edit_general_memory",
            {"id": "mem-1", "content": "User loves espresso"},
        )

        event, data, _ = emitted[0]
        assert event == "memory_edited"
        assert data["success"] is True
        assert data["old_id"] == "mem-1"
        assert data["memory"]["content"] == "User loves espresso"
        assert "edited_at" in data["memory"]["metadata"]

        # Verify delete was called on old
        store.delete.assert_called_once_with("general", "mem-1")
        # Verify add_general was called with new content
        store.add_general.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_missing_id(self, server_with_orchestrator: GUIServer) -> None:
        """Should reject when ID is missing."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "edit_general_memory",
            {"content": "new content"},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_rejects_empty_content(self, server_with_orchestrator: GUIServer) -> None:
        """Should reject empty content."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "edit_general_memory",
            {"id": "mem-1", "content": "  "},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False


# =============================================================================
# delete_memory
# =============================================================================


class TestDeleteMemory:
    """Tests for the delete_memory handler."""

    @pytest.mark.asyncio
    async def test_deletes_memory_successfully(self, server_with_orchestrator: GUIServer) -> None:
        """Should delete memory and return success."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "delete_memory",
            {"collection": "general", "id": "mem-1"},
        )

        event, data, _ = emitted[0]
        assert event == "memory_deleted"
        assert data["success"] is True
        assert data["collection"] == "general"
        assert data["id"] == "mem-1"

    @pytest.mark.asyncio
    async def test_rejects_missing_params(self, server_with_orchestrator: GUIServer) -> None:
        """Should reject when collection or ID is missing."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "delete_memory",
            {"collection": "general"},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_no_orchestrator(self, server: GUIServer) -> None:
        """Should return error when no orchestrator."""
        server._orchestrator = None
        emitted = await _dispatch(
            server,
            "delete_memory",
            {"collection": "general", "id": "mem-1"},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False


# =============================================================================
# promote_memory
# =============================================================================


class TestPromoteMemory:
    """Tests for the promote_memory handler."""

    @pytest.mark.asyncio
    async def test_promotes_flashcard_copy(self, server_with_orchestrator: GUIServer) -> None:
        """Should copy flash card to general (keep source)."""
        store = server_with_orchestrator._orchestrator.memory_store
        store.get_all.return_value = [
            {"id": "fc-1", "content": "Q: Drink? A: Coffee", "metadata": {"type": "flashcard"}},
        ]

        emitted = await _dispatch(
            server_with_orchestrator,
            "promote_memory",
            {"source_collection": "flashcards", "id": "fc-1", "delete_source": False},
        )

        event, data, _ = emitted[0]
        assert event == "memory_promoted"
        assert data["success"] is True
        assert data["deleted_source"] is False
        # Source should NOT be deleted
        store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_promotes_flashcard_move(self, server_with_orchestrator: GUIServer) -> None:
        """Should move flash card to general (delete source)."""
        store = server_with_orchestrator._orchestrator.memory_store
        store.get_all.return_value = [
            {"id": "fc-1", "content": "Q: Drink? A: Coffee", "metadata": {"type": "flashcard"}},
        ]

        emitted = await _dispatch(
            server_with_orchestrator,
            "promote_memory",
            {"source_collection": "flashcards", "id": "fc-1", "delete_source": True},
        )

        event, data, _ = emitted[0]
        assert data["success"] is True
        assert data["deleted_source"] is True
        store.delete.assert_called_once_with("flashcards", "fc-1")

    @pytest.mark.asyncio
    async def test_rejects_invalid_source_collection(self, server_with_orchestrator: GUIServer) -> None:
        """Should reject promotion from general (can only promote from flashcards/summaries)."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "promote_memory",
            {"source_collection": "general", "id": "mem-1"},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False
        assert "flashcards or summaries" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_source_not_found(self, server_with_orchestrator: GUIServer) -> None:
        """Should fail gracefully when source document doesn't exist."""
        store = server_with_orchestrator._orchestrator.memory_store
        store.get_all.return_value = []  # No documents found

        emitted = await _dispatch(
            server_with_orchestrator,
            "promote_memory",
            {"source_collection": "flashcards", "id": "nonexistent"},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_missing_params(self, server_with_orchestrator: GUIServer) -> None:
        """Should reject when source_collection or ID is missing."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "promote_memory",
            {"id": "fc-1"},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False


# =============================================================================
# search_memories
# =============================================================================


class TestSearchMemories:
    """Tests for the search_memories handler."""

    @pytest.mark.asyncio
    async def test_returns_search_results(self, server_with_orchestrator: GUIServer) -> None:
        """Should return search results from MemoryStore.query()."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "search_memories",
            {"query": "coffee", "top_k": 10},
        )

        event, data, _ = emitted[0]
        assert event == "memory_search_results"
        assert len(data["results"]) == 2
        assert data["query"] == "coffee"

    @pytest.mark.asyncio
    async def test_empty_query(self, server_with_orchestrator: GUIServer) -> None:
        """Should return empty results for empty query."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "search_memories",
            {"query": "  "},
        )

        event, data, _ = emitted[0]
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_no_orchestrator(self, server: GUIServer) -> None:
        """Should return error when no orchestrator."""
        server._orchestrator = None
        emitted = await _dispatch(
            server,
            "search_memories",
            {"query": "coffee"},
        )

        event, data, _ = emitted[0]
        assert data["results"] == []
        assert data.get("error") is not None


# =============================================================================
# clear_flashcards
# =============================================================================


class TestClearFlashcards:
    """Tests for the clear_flashcards handler."""

    @pytest.mark.asyncio
    async def test_clears_flashcards_successfully(self, server_with_orchestrator: GUIServer) -> None:
        """Should call clear_flash_cards() and return success."""
        emitted = await _dispatch(
            server_with_orchestrator,
            "clear_flashcards",
            {},
        )

        event, data, _ = emitted[0]
        assert event == "flashcards_cleared"
        assert data["success"] is True
        server_with_orchestrator._orchestrator.memory_store.clear_flash_cards.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_orchestrator(self, server: GUIServer) -> None:
        """Should return error when no orchestrator."""
        server._orchestrator = None
        emitted = await _dispatch(
            server,
            "clear_flashcards",
            {},
        )

        event, data, _ = emitted[0]
        assert data["success"] is False
