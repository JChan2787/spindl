"""Tests for RAGInjector — PreProcessor that queries memories for RAG injection.

NANO-043 Phase 2.
"""

import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from spindl.llm.plugins.base import PipelineContext
from spindl.memory.embedding_client import EmbeddingClient, EmbeddingError
from spindl.memory.memory_store import MemoryStore
from spindl.memory.rag_injector import RAGInjector, threshold_to_max_distance


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
    d = tempfile.mkdtemp(prefix="nano_rag_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(mock_embedding_client: MagicMock, memory_dir: str) -> MemoryStore:
    """MemoryStore with mock embeddings."""
    import os
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        global_memory_dir=os.path.join(memory_dir, "_global"),
    )


@pytest.fixture
def seeded_store(store: MemoryStore) -> MemoryStore:
    """MemoryStore pre-seeded with test memories across all collections."""
    store.add_general("Alex likes mango smoothies.")
    store.add_general("Alex has a cat named Pixel.")
    store.add_flash_card("Q: What is Alex's favorite drink? A: Coffee and mango smoothies.")
    store.add_flash_card("Q: What does Alex do for work? A: He is a software engineer at a startup.")
    store.add_session_summary("Session 1: Alex discussed his side project and his plans for the weekend.")
    return store


@pytest.fixture
def pipeline_context() -> PipelineContext:
    """Standard PipelineContext for testing."""
    return PipelineContext(
        user_input="What does Alex like to drink?",
        persona={"name": "Spindle", "system_prompt": "You are Spindle."},
        messages=[
            {"role": "system", "content": "You are Spindle."},
            {"role": "user", "content": "What does Alex like to drink?"},
        ],
    )


# ---------------------------------------------------------------------------
# Tests: RAGInjector basics
# ---------------------------------------------------------------------------

class TestRAGInjectorInit:
    """Tests for RAGInjector initialization and properties."""

    def test_name(self, store: MemoryStore) -> None:
        """Name property returns 'rag_injector'."""
        injector = RAGInjector(memory_store=store)
        assert injector.name == "rag_injector"

    def test_default_top_k(self, store: MemoryStore) -> None:
        """Default top_k is 5."""
        injector = RAGInjector(memory_store=store)
        assert injector._top_k == 5

    def test_custom_top_k(self, store: MemoryStore) -> None:
        """Custom top_k is respected."""
        injector = RAGInjector(memory_store=store, top_k=3)
        assert injector._top_k == 3


# ---------------------------------------------------------------------------
# Tests: Empty / no results
# ---------------------------------------------------------------------------

class TestRAGInjectorEmptyResults:
    """Tests for RAGInjector when no memories are available."""

    def test_empty_store_returns_empty_content(
        self, store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Empty MemoryStore produces empty rag_content and 0 token estimate."""
        injector = RAGInjector(memory_store=store)
        result = injector.process(pipeline_context)

        assert result.metadata["rag_content"] == ""
        assert result.metadata["rag_tokens_estimate"] == 0

    def test_empty_store_returns_same_context(
        self, store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """process() returns the same PipelineContext object (mutated in place)."""
        injector = RAGInjector(memory_store=store)
        result = injector.process(pipeline_context)
        assert result is pipeline_context


# ---------------------------------------------------------------------------
# Tests: Successful memory retrieval
# ---------------------------------------------------------------------------

class TestRAGInjectorWithMemories:
    """Tests for RAGInjector when memories are available."""

    def test_populates_rag_content(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Seeded store produces non-empty rag_content with default wrappers."""
        injector = RAGInjector(memory_store=seeded_store)
        result = injector.process(pipeline_context)

        assert result.metadata["rag_content"] != ""
        assert result.metadata["rag_content"].startswith(injector._rag_prefix)
        assert "End of memories." in result.metadata["rag_content"]

    def test_token_estimate_positive(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Token estimate is positive when memories are returned."""
        injector = RAGInjector(memory_store=seeded_store)
        result = injector.process(pipeline_context)

        assert result.metadata["rag_tokens_estimate"] > 0

    def test_token_estimate_is_len_div_4(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Token estimate is len(rag_content) // 4."""
        injector = RAGInjector(memory_store=seeded_store)
        result = injector.process(pipeline_context)

        expected = len(result.metadata["rag_content"]) // 4
        assert result.metadata["rag_tokens_estimate"] == expected

    def test_memories_formatted_as_bullet_list(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Memories are formatted as '- content' lines."""
        injector = RAGInjector(memory_store=seeded_store)
        result = injector.process(pipeline_context)

        lines = result.metadata["rag_content"].split("\n")
        # First line is header (default prefix), last is footer
        assert lines[0] == injector._rag_prefix
        assert lines[-1] == "End of memories."
        # Middle lines are bullet points
        for line in lines[1:-1]:
            assert line.startswith("- ")

    def test_top_k_limits_results(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """top_k=2 limits to 2 memories in output."""
        injector = RAGInjector(memory_store=seeded_store, top_k=2)
        result = injector.process(pipeline_context)

        lines = result.metadata["rag_content"].split("\n")
        # Header + 2 bullets + footer = 4 lines
        bullet_lines = [l for l in lines if l.startswith("- ")]
        assert len(bullet_lines) <= 2

    def test_does_not_modify_messages(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """RAGInjector only writes to metadata — messages are untouched."""
        original_messages = [dict(m) for m in pipeline_context.messages]
        injector = RAGInjector(memory_store=seeded_store)
        injector.process(pipeline_context)

        assert pipeline_context.messages == original_messages

    def test_query_uses_user_input(
        self, seeded_store: MemoryStore
    ) -> None:
        """RAGInjector uses context.user_input as the query text."""
        injector = RAGInjector(memory_store=seeded_store)

        with patch.object(seeded_store, "query", wraps=seeded_store.query) as mock_query:
            context = PipelineContext(
                user_input="Tell me about Alex's family",
                persona={},
                messages=[],
            )
            injector.process(context)

            mock_query.assert_called_once_with(
                query_text="Tell me about Alex's family",
                top_k=5,
            )


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------

class TestRAGInjectorGracefulDegradation:
    """Tests for RAGInjector when the embedding server is unreachable."""

    def test_connection_error_returns_empty(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Connection error from MemoryStore.query() produces empty content."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.side_effect = ConnectionError("Embedding server unreachable")

        injector = RAGInjector(memory_store=mock_store)
        result = injector.process(pipeline_context)

        assert result.metadata["rag_content"] == ""
        assert result.metadata["rag_tokens_estimate"] == 0

    def test_embedding_error_returns_empty(
        self, pipeline_context: PipelineContext
    ) -> None:
        """EmbeddingError from MemoryStore.query() produces empty content."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.side_effect = EmbeddingError("Server timeout")

        injector = RAGInjector(memory_store=mock_store)
        result = injector.process(pipeline_context)

        assert result.metadata["rag_content"] == ""
        assert result.metadata["rag_tokens_estimate"] == 0

    def test_generic_exception_returns_empty(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Any exception from MemoryStore.query() produces empty content."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.side_effect = RuntimeError("ChromaDB exploded")

        injector = RAGInjector(memory_store=mock_store)
        result = injector.process(pipeline_context)

        assert result.metadata["rag_content"] == ""
        assert result.metadata["rag_tokens_estimate"] == 0

    def test_exception_does_not_propagate(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Exceptions are caught — process() never raises."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.side_effect = Exception("Catastrophic failure")

        injector = RAGInjector(memory_store=mock_store)
        # Should not raise
        result = injector.process(pipeline_context)
        assert result is pipeline_context


# ---------------------------------------------------------------------------
# Tests: Format helper
# ---------------------------------------------------------------------------

class TestFormatMemories:
    """Tests for RAGInjector._format_memories instance method."""

    def test_single_memory(self, store: MemoryStore) -> None:
        """Single memory is formatted correctly with default wrappers."""
        injector = RAGInjector(memory_store=store)
        results = [{"content": "Alex likes mango.", "collection": "general", "distance": 0.3, "metadata": {}}]
        text = injector._format_memories(results)
        assert "Alex likes mango." in text
        assert text.startswith(injector._rag_prefix)
        assert text.endswith(injector._rag_suffix)

    def test_multiple_memories(self, store: MemoryStore) -> None:
        """Multiple memories produce multiple bullet lines."""
        injector = RAGInjector(memory_store=store)
        results = [
            {"content": "Memory one.", "collection": "general", "distance": 0.3, "metadata": {}},
            {"content": "Memory two.", "collection": "flashcards", "distance": 0.5, "metadata": {}},
            {"content": "Memory three.", "collection": "summaries", "distance": 0.7, "metadata": {}},
        ]
        text = injector._format_memories(results)
        lines = text.split("\n")
        # header + 3 bullets + footer = 5 lines
        assert len(lines) == 5
        assert lines[1] == "- Memory one."
        assert lines[2] == "- Memory two."
        assert lines[3] == "- Memory three."

    def test_empty_list(self, store: MemoryStore) -> None:
        """Empty results list still produces header and footer."""
        injector = RAGInjector(memory_store=store)
        text = injector._format_memories([])
        assert text.startswith(injector._rag_prefix)
        assert text.endswith(injector._rag_suffix)


# ---------------------------------------------------------------------------
# Tests: Pipeline integration (metadata threading)
# ---------------------------------------------------------------------------

class TestRAGInjectorMetadataThreading:
    """Tests verifying metadata is set correctly for downstream plugins."""

    def test_metadata_keys_set_on_success(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Both rag_content and rag_tokens_estimate are set on success."""
        injector = RAGInjector(memory_store=seeded_store)
        result = injector.process(pipeline_context)

        assert "rag_content" in result.metadata
        assert "rag_tokens_estimate" in result.metadata

    def test_metadata_keys_set_on_failure(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Both metadata keys are set even on failure (empty defaults)."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.side_effect = Exception("fail")

        injector = RAGInjector(memory_store=mock_store)
        result = injector.process(pipeline_context)

        assert "rag_content" in result.metadata
        assert "rag_tokens_estimate" in result.metadata

    def test_metadata_keys_set_on_empty(
        self, store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Both metadata keys are set when store returns empty results."""
        injector = RAGInjector(memory_store=store)
        result = injector.process(pipeline_context)

        assert "rag_content" in result.metadata
        assert "rag_tokens_estimate" in result.metadata

    def test_preserves_existing_metadata(
        self, seeded_store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """RAGInjector does not clobber existing metadata (e.g., codex_tokens_estimate)."""
        pipeline_context.metadata["codex_tokens_estimate"] = 150
        pipeline_context.metadata["codex_content"] = "Some codex content"

        injector = RAGInjector(memory_store=seeded_store)
        result = injector.process(pipeline_context)

        # Codex metadata preserved
        assert result.metadata["codex_tokens_estimate"] == 150
        assert result.metadata["codex_content"] == "Some codex content"
        # RAG metadata also present
        assert "rag_content" in result.metadata
        assert "rag_tokens_estimate" in result.metadata


# ---------------------------------------------------------------------------
# Tests: threshold_to_max_distance conversion
# ---------------------------------------------------------------------------

class TestThresholdConversion:
    """Tests for threshold_to_max_distance() utility."""

    def test_zero_threshold_is_fully_permissive(self) -> None:
        """threshold=0.0 → max_dist=2.0 (everything passes)."""
        assert threshold_to_max_distance(0.0) == 2.0

    def test_one_threshold_is_exact_only(self) -> None:
        """threshold=1.0 → max_dist=0.0 (only identical vectors)."""
        assert threshold_to_max_distance(1.0) == 0.0

    def test_half_threshold(self) -> None:
        """threshold=0.5 → max_dist=1.0."""
        assert threshold_to_max_distance(0.5) == 1.0

    def test_quarter_threshold(self) -> None:
        """threshold=0.25 → max_dist=1.5."""
        assert threshold_to_max_distance(0.25) == 1.5

    def test_three_quarter_threshold(self) -> None:
        """threshold=0.75 → max_dist=0.5."""
        assert threshold_to_max_distance(0.75) == 0.5


# ---------------------------------------------------------------------------
# Tests: Relevance threshold filtering
# ---------------------------------------------------------------------------

class TestRAGInjectorRelevanceThreshold:
    """Tests for distance-based relevance filtering.

    User-facing semantics: 0.0 = accept everything, 1.0 = only exact matches.
    Internally converted via threshold_to_max_distance(): max_dist = 2*(1-t).
    """

    def test_default_no_threshold(self, store: MemoryStore) -> None:
        """Default relevance_threshold is None (no filtering)."""
        injector = RAGInjector(memory_store=store)
        assert injector._relevance_threshold is None

    def test_custom_threshold(self, store: MemoryStore) -> None:
        """Custom relevance_threshold is stored."""
        injector = RAGInjector(memory_store=store, relevance_threshold=0.75)
        assert injector._relevance_threshold == 0.75

    def test_threshold_filters_distant_results(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Results with distance > converted max_distance are excluded.

        threshold=0.5 → max_dist=1.0, so distances 0.3 and 0.6 pass, 1.2 fails.
        """
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Close match", "collection": "general", "distance": 0.3, "metadata": {}},
            {"id": "2", "content": "Medium match", "collection": "general", "distance": 0.6, "metadata": {}},
            {"id": "3", "content": "Far match", "collection": "general", "distance": 1.2, "metadata": {}},
        ]

        # threshold=0.5 → max_dist = 2*(1-0.5) = 1.0
        injector = RAGInjector(memory_store=mock_store, relevance_threshold=0.5)
        result = injector.process(pipeline_context)

        # Only the two close results should appear
        assert "Close match" in result.metadata["rag_content"]
        assert "Medium match" in result.metadata["rag_content"]
        assert "Far match" not in result.metadata["rag_content"]

    def test_threshold_none_returns_all(
        self, pipeline_context: PipelineContext
    ) -> None:
        """When threshold is None, all results pass through."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Close", "collection": "general", "distance": 0.3, "metadata": {}},
            {"id": "2", "content": "Far", "collection": "general", "distance": 1.5, "metadata": {}},
        ]

        injector = RAGInjector(memory_store=mock_store, relevance_threshold=None)
        result = injector.process(pipeline_context)

        assert "Close" in result.metadata["rag_content"]
        assert "Far" in result.metadata["rag_content"]

    def test_threshold_filters_all_returns_empty(
        self, pipeline_context: PipelineContext
    ) -> None:
        """When all results exceed converted max_distance, rag_content is empty.

        threshold=0.5 → max_dist=1.0, distance 1.5 fails.
        """
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Too far", "collection": "general", "distance": 1.5, "metadata": {}},
        ]

        # threshold=0.5 → max_dist = 2*(1-0.5) = 1.0 < 1.5
        injector = RAGInjector(memory_store=mock_store, relevance_threshold=0.5)
        result = injector.process(pipeline_context)

        assert result.metadata["rag_content"] == ""
        assert result.metadata["rag_tokens_estimate"] == 0

    def test_threshold_boundary_exact_match(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Result with distance exactly equal to converted max_distance is included.

        threshold=0.625 → max_dist = 2*(1-0.625) = 0.75, distance 0.75 passes (<=).
        """
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Boundary", "collection": "general", "distance": 0.75, "metadata": {}},
        ]

        # threshold=0.625 → max_dist = 2*(1-0.625) = 0.75
        injector = RAGInjector(memory_store=mock_store, relevance_threshold=0.625)
        result = injector.process(pipeline_context)

        assert "Boundary" in result.metadata["rag_content"]

    def test_low_threshold_is_permissive(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Low threshold (close to 0) allows distant results.

        threshold=0.1 → max_dist = 2*(1-0.1) = 1.8, everything passes.
        """
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Close", "collection": "general", "distance": 0.3, "metadata": {}},
            {"id": "2", "content": "Far", "collection": "general", "distance": 1.5, "metadata": {}},
        ]

        injector = RAGInjector(memory_store=mock_store, relevance_threshold=0.1)
        result = injector.process(pipeline_context)

        assert "Close" in result.metadata["rag_content"]
        assert "Far" in result.metadata["rag_content"]

    def test_high_threshold_is_strict(
        self, pipeline_context: PipelineContext
    ) -> None:
        """High threshold (close to 1) rejects most results.

        threshold=0.9 → max_dist = 2*(1-0.9) = 0.2, only very close passes.
        """
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Very close", "collection": "general", "distance": 0.1, "metadata": {}},
            {"id": "2", "content": "Medium", "collection": "general", "distance": 0.5, "metadata": {}},
        ]

        injector = RAGInjector(memory_store=mock_store, relevance_threshold=0.9)
        result = injector.process(pipeline_context)

        assert "Very close" in result.metadata["rag_content"]
        assert "Medium" not in result.metadata["rag_content"]


# ---------------------------------------------------------------------------
# Tests: Runtime config update
# ---------------------------------------------------------------------------

class TestRAGInjectorUpdateConfig:
    """Tests for RAGInjector.update_config() runtime parameter changes."""

    def test_update_top_k(self, store: MemoryStore) -> None:
        """update_config changes top_k."""
        injector = RAGInjector(memory_store=store, top_k=5)
        injector.update_config(top_k=10)
        assert injector._top_k == 10

    def test_update_relevance_threshold(self, store: MemoryStore) -> None:
        """update_config changes relevance_threshold."""
        injector = RAGInjector(memory_store=store, relevance_threshold=0.75)
        injector.update_config(relevance_threshold=0.5)
        assert injector._relevance_threshold == 0.5

    def test_update_relevance_threshold_to_none(self, store: MemoryStore) -> None:
        """update_config can disable threshold by setting to None."""
        injector = RAGInjector(memory_store=store, relevance_threshold=0.75)
        injector.update_config(relevance_threshold=None)
        assert injector._relevance_threshold is None

    def test_update_preserves_unset_values(self, store: MemoryStore) -> None:
        """update_config with ellipsis preserves current values."""
        injector = RAGInjector(memory_store=store, top_k=5, relevance_threshold=0.75)
        injector.update_config(top_k=10)
        # relevance_threshold should be unchanged (ellipsis default)
        assert injector._relevance_threshold == 0.75
        assert injector._top_k == 10


# ---------------------------------------------------------------------------
# Tests: rag_results metadata stash (NANO-044)
# ---------------------------------------------------------------------------

class TestRAGInjectorResultsMetadata:
    """Tests for NANO-044: raw results stashed in context.metadata['rag_results']."""

    def test_rag_results_stashed_on_success(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Successful retrieval stashes raw result dicts in metadata."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Alex likes mango.", "collection": "general", "distance": 0.3, "metadata": {}},
            {"id": "2", "content": "Alex has a wife.", "collection": "flashcards", "distance": 0.5, "metadata": {}},
        ]

        injector = RAGInjector(memory_store=mock_store)
        result = injector.process(pipeline_context)

        assert "rag_results" in result.metadata
        assert len(result.metadata["rag_results"]) == 2
        assert result.metadata["rag_results"][0]["content"] == "Alex likes mango."
        assert result.metadata["rag_results"][1]["collection"] == "flashcards"

    def test_rag_results_not_stashed_on_empty(
        self, store: MemoryStore, pipeline_context: PipelineContext
    ) -> None:
        """Empty store does not stash rag_results (key absent)."""
        injector = RAGInjector(memory_store=store)
        result = injector.process(pipeline_context)

        assert "rag_results" not in result.metadata

    def test_rag_results_not_stashed_on_error(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Query error does not stash rag_results (key absent)."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.side_effect = Exception("fail")

        injector = RAGInjector(memory_store=mock_store)
        result = injector.process(pipeline_context)

        assert "rag_results" not in result.metadata

    def test_rag_results_respects_threshold(
        self, pipeline_context: PipelineContext
    ) -> None:
        """rag_results only contains results that passed threshold filter."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Close", "collection": "general", "distance": 0.3, "metadata": {}},
            {"id": "2", "content": "Far", "collection": "general", "distance": 1.5, "metadata": {}},
        ]

        # threshold=0.5 → max_dist=1.0, so distance 1.5 is filtered out
        injector = RAGInjector(memory_store=mock_store, relevance_threshold=0.5)
        result = injector.process(pipeline_context)

        assert len(result.metadata["rag_results"]) == 1
        assert result.metadata["rag_results"][0]["content"] == "Close"

    def test_rag_results_respects_top_k(
        self, pipeline_context: PipelineContext
    ) -> None:
        """rag_results is trimmed to top_k."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": str(i), "content": f"Memory {i}", "collection": "general", "distance": 0.1 * i, "metadata": {}}
            for i in range(10)
        ]

        injector = RAGInjector(memory_store=mock_store, top_k=3)
        result = injector.process(pipeline_context)

        assert len(result.metadata["rag_results"]) == 3


# ---------------------------------------------------------------------------
# Tests: Configurable injection wrappers (NANO-045d)
# ---------------------------------------------------------------------------

class TestRAGInjectorCustomWrappers:
    """Tests for NANO-045d: configurable RAG prefix/suffix strings."""

    def test_default_prefix(self, store: MemoryStore) -> None:
        """Default prefix is the directive string, not legacy 'Relevant memories:'."""
        injector = RAGInjector(memory_store=store)
        assert "Use them to inform your response" in injector._rag_prefix

    def test_default_suffix(self, store: MemoryStore) -> None:
        """Default suffix is 'End of memories.'."""
        injector = RAGInjector(memory_store=store)
        assert injector._rag_suffix == "End of memories."

    def test_custom_prefix(self, store: MemoryStore) -> None:
        """Custom rag_prefix is used in constructor."""
        injector = RAGInjector(memory_store=store, rag_prefix="REMEMBER:")
        assert injector._rag_prefix == "REMEMBER:"

    def test_custom_suffix(self, store: MemoryStore) -> None:
        """Custom rag_suffix is used in constructor."""
        injector = RAGInjector(memory_store=store, rag_suffix="END.")
        assert injector._rag_suffix == "END."

    def test_custom_prefix_in_formatted_output(self, store: MemoryStore) -> None:
        """Custom prefix appears in _format_memories output."""
        injector = RAGInjector(memory_store=store, rag_prefix="CUSTOM PREFIX:")
        results = [{"content": "A memory.", "collection": "general", "distance": 0.3, "metadata": {}}]
        text = injector._format_memories(results)
        assert text.startswith("CUSTOM PREFIX:")

    def test_custom_suffix_in_formatted_output(self, store: MemoryStore) -> None:
        """Custom suffix appears in _format_memories output."""
        injector = RAGInjector(memory_store=store, rag_suffix="-- DONE --")
        results = [{"content": "A memory.", "collection": "general", "distance": 0.3, "metadata": {}}]
        text = injector._format_memories(results)
        assert text.endswith("-- DONE --")

    def test_empty_suffix_omitted(self, store: MemoryStore) -> None:
        """Empty string suffix is not appended (no trailing newline)."""
        injector = RAGInjector(memory_store=store, rag_suffix="")
        results = [{"content": "A memory.", "collection": "general", "distance": 0.3, "metadata": {}}]
        text = injector._format_memories(results)
        assert text.endswith("- A memory.")

    def test_update_config_changes_prefix(self, store: MemoryStore) -> None:
        """update_config can change rag_prefix at runtime."""
        injector = RAGInjector(memory_store=store)
        injector.update_config(rag_prefix="NEW PREFIX:")
        assert injector._rag_prefix == "NEW PREFIX:"

    def test_update_config_changes_suffix(self, store: MemoryStore) -> None:
        """update_config can change rag_suffix at runtime."""
        injector = RAGInjector(memory_store=store)
        injector.update_config(rag_suffix="NEW SUFFIX.")
        assert injector._rag_suffix == "NEW SUFFIX."

    def test_update_config_ellipsis_preserves_wrappers(self, store: MemoryStore) -> None:
        """update_config with ellipsis default preserves wrapper strings."""
        injector = RAGInjector(memory_store=store, rag_prefix="ORIGINAL")
        injector.update_config(top_k=10)  # rag_prefix/suffix default to ellipsis
        assert injector._rag_prefix == "ORIGINAL"

    def test_update_config_none_resets_to_default(self, store: MemoryStore) -> None:
        """update_config with None resets wrapper to class default."""
        injector = RAGInjector(memory_store=store, rag_prefix="CUSTOM")
        injector.update_config(rag_prefix=None)
        assert injector._rag_prefix == RAGInjector._DEFAULT_RAG_PREFIX

    def test_custom_wrappers_in_process(
        self, pipeline_context: PipelineContext
    ) -> None:
        """Custom wrappers appear in the full process() output."""
        mock_store = MagicMock(spec=MemoryStore)
        mock_store.query.return_value = [
            {"id": "1", "content": "Test memory.", "collection": "general", "distance": 0.3, "metadata": {}},
        ]
        injector = RAGInjector(
            memory_store=mock_store,
            rag_prefix="[MEMORIES BEGIN]",
            rag_suffix="[MEMORIES END]",
        )
        result = injector.process(pipeline_context)
        assert result.metadata["rag_content"].startswith("[MEMORIES BEGIN]")
        assert result.metadata["rag_content"].endswith("[MEMORIES END]")
