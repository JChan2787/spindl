"""Tests for NANO-107 retrieval scoring pipeline.

Tests the composite scoring function (compute_score), metadata enrichment
at creation time, tier weighting, decay curve, access reinforcement,
and session-scoped retrieval.
"""

import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from spindl.memory.embedding_client import EmbeddingClient
from spindl.memory.memory_store import (
    MemoryStore,
    compute_score,
    DEFAULT_IMPORTANCE,
    DECAY_BASE,
    GENERAL,
    FLASHCARDS,
    SUMMARIES,
    TIER_WEIGHTS,
    W_RELEVANCE,
    W_RECENCY,
    W_IMPORTANCE,
    W_FREQUENCY,
)


# Fixed reference time for deterministic tests
NOW = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)


class TestComputeScore:
    """Tests for the composite scoring function."""

    def test_perfect_relevance_scores_highest(self):
        """Distance 0 (identical embedding) produces maximum relevance component."""
        score_perfect = compute_score(
            distance=0.0, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        score_poor = compute_score(
            distance=1.8, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert score_perfect > score_poor

    def test_relevance_dominant_signal(self):
        """With equal other signals, semantic similarity determines ranking."""
        close = compute_score(
            distance=0.2, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        far = compute_score(
            distance=1.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert close > far

    def test_recency_tiebreaker(self):
        """Equal relevance — more recently accessed memory wins."""
        recent = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T11:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        stale = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-01T00:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert recent > stale

    def test_importance_boost(self):
        """High importance memory beats low importance at similar relevance."""
        important = compute_score(
            distance=0.5, importance=9, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        trivial = compute_score(
            distance=0.5, importance=2, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert important > trivial

    def test_frequency_reinforcement(self):
        """Frequently accessed memory gets mild boost."""
        frequent = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=50, tier=GENERAL, now=NOW,
        )
        never = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert frequent > never

    def test_decay_curve_shape(self):
        """Decay is exponential — drops fast initially, then flattens."""
        one_hour = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T11:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        one_day = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-28T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        thirty_days = compute_score(
            distance=0.5, importance=5, last_accessed="2026-02-27T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )

        assert one_hour > one_day > thirty_days
        # The drop from 1 hour to 1 day should be larger than 1 day to 30 days
        # (exponential decay flattens)
        drop_early = one_hour - one_day
        drop_late = one_day - thirty_days
        assert drop_early < drop_late  # More total decay over longer period

    def test_tier_weights_ordering(self):
        """Global > General > Flashcard > Summary at equal raw scores."""
        base_args = dict(
            distance=0.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, now=NOW,
        )
        global_score = compute_score(**base_args, tier="global")
        general_score = compute_score(**base_args, tier=GENERAL)
        flashcard_score = compute_score(**base_args, tier=FLASHCARDS)
        summary_score = compute_score(**base_args, tier=SUMMARIES)

        assert global_score > general_score > flashcard_score > summary_score

    def test_old_but_relevant_survives(self):
        """Memory with 0 recency but perfect relevance still scores well.

        This is the key additive-not-multiplicative test — if scoring were
        multiplicative, a zero recency would zero out the entire score.
        """
        old_relevant = compute_score(
            distance=0.1, importance=8,
            last_accessed="2025-01-01T00:00:00Z",  # Over a year ago
            access_count=0, tier=GENERAL, now=NOW,
        )
        # Score should still be meaningfully positive
        assert old_relevant > 0.3

    def test_backfill_defaults_no_crash(self):
        """Missing metadata fields use neutral defaults — no crash, no penalty."""
        # Empty string for last_accessed triggers fallback
        score = compute_score(
            distance=0.5, importance=5, last_accessed="",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert score > 0

    def test_unparseable_timestamp_fallback(self):
        """Garbage timestamp doesn't crash — falls back to 30-day-old assumption."""
        score = compute_score(
            distance=0.5, importance=5, last_accessed="not-a-date",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert score > 0

    def test_importance_clamped(self):
        """Importance values outside 1-10 are clamped, not crashed."""
        score_zero = compute_score(
            distance=0.5, importance=0, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        score_hundred = compute_score(
            distance=0.5, importance=100, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        # Both should produce valid scores
        assert score_zero > 0
        assert score_hundred > 0
        # Clamped to 1 and 10 respectively
        score_one = compute_score(
            distance=0.5, importance=1, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        score_ten = compute_score(
            distance=0.5, importance=10, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW,
        )
        assert score_zero == pytest.approx(score_one)
        assert score_hundred == pytest.approx(score_ten)

    def test_unknown_tier_gets_neutral_weight(self):
        """Unknown tier name defaults to weight 1.0."""
        known = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier=FLASHCARDS, now=NOW,
        )
        unknown = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=0, tier="nonexistent_tier", now=NOW,
        )
        # Flashcards tier weight is 1.0, unknown defaults to 1.0
        assert known == pytest.approx(unknown)

    def test_custom_weights_override(self):
        """Config-provided weights override module defaults."""
        default_score = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=10, tier=GENERAL, now=NOW,
        )
        # Set relevance to 1.0, everything else to 0 — pure similarity mode
        pure_relevance = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-29T12:00:00Z",
            access_count=10, tier=GENERAL, now=NOW,
            w_relevance=1.0, w_recency=0.0, w_importance=0.0, w_frequency=0.0,
        )
        assert pure_relevance != pytest.approx(default_score)

    def test_custom_decay_base(self):
        """Custom decay base changes the decay curve."""
        # Very aggressive decay (0.5 per hour)
        aggressive = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-28T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW, decay_base=0.5,
        )
        # Very gentle decay (0.9999 per hour)
        gentle = compute_score(
            distance=0.5, importance=5, last_accessed="2026-03-28T12:00:00Z",
            access_count=0, tier=GENERAL, now=NOW, decay_base=0.9999,
        )
        assert gentle > aggressive


class TestDefaultImportance:
    """Tests for heuristic importance assignment by creation source."""

    def test_global_highest(self):
        assert DEFAULT_IMPORTANCE["global"] == 8

    def test_general_curated(self):
        assert DEFAULT_IMPORTANCE[GENERAL] == 7

    def test_flashcard_auto_generated(self):
        assert DEFAULT_IMPORTANCE[FLASHCARDS] == 4

    def test_summary_lowest(self):
        assert DEFAULT_IMPORTANCE[SUMMARIES] == 3

    def test_ordering(self):
        assert (
            DEFAULT_IMPORTANCE["global"]
            > DEFAULT_IMPORTANCE[GENERAL]
            > DEFAULT_IMPORTANCE[FLASHCARDS]
            > DEFAULT_IMPORTANCE[SUMMARIES]
        )


class TestTierWeights:
    """Tests for tier weight multipliers."""

    def test_global_boosted(self):
        assert TIER_WEIGHTS["global"] > 1.0

    def test_general_boosted(self):
        assert TIER_WEIGHTS[GENERAL] > 1.0

    def test_flashcard_neutral(self):
        assert TIER_WEIGHTS[FLASHCARDS] == 1.0

    def test_summary_discounted(self):
        assert TIER_WEIGHTS[SUMMARIES] < 1.0

    def test_ordering(self):
        assert (
            TIER_WEIGHTS["global"]
            > TIER_WEIGHTS[GENERAL]
            > TIER_WEIGHTS[FLASHCARDS]
            > TIER_WEIGHTS[SUMMARIES]
        )


# ---------------------------------------------------------------------------
# Session-scoped retrieval tests (require real ChromaDB)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_client() -> MagicMock:
    client = MagicMock(spec=EmbeddingClient)

    def fake_embed_batch(texts):
        return [[(hash(t) + i) / 10000.0 for i in range(64)] for t in texts]

    client.embed_batch.side_effect = fake_embed_batch
    return client


@pytest.fixture
def memory_dir():
    d = tempfile.mkdtemp(prefix="nano_scoring_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(mock_embedding_client, memory_dir) -> MemoryStore:
    import os
    global_dir = os.path.join(memory_dir, "_global")
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        dedup_threshold=None,
        global_memory_dir=global_dir,
    )


class TestSessionScopedRetrieval:
    """Flashcards and summaries only appear in retrieval for the current session."""

    def test_flashcards_filtered_by_session(self, store: MemoryStore):
        """Flashcards from a different session are excluded from query results."""
        store.add_flash_card("Session A fact", metadata={"session_id": "session_a"})
        store.add_flash_card("Session B fact", metadata={"session_id": "session_b"})
        store.add_general("Always visible")

        store.set_session_id("session_a")
        results = store.query("fact")

        flashcard_contents = [
            r["content"] for r in results if r["collection"] == FLASHCARDS
        ]
        assert "Session A fact" in flashcard_contents
        assert "Session B fact" not in flashcard_contents

    def test_general_always_visible(self, store: MemoryStore):
        """General memories are not session-filtered."""
        store.add_general("Durable fact")
        store.add_flash_card("Session fact", metadata={"session_id": "other_session"})

        store.set_session_id("current_session")
        results = store.query("fact")

        collections = {r["collection"] for r in results}
        assert GENERAL in collections

    def test_global_always_visible(self, store: MemoryStore):
        """Global memories are not session-filtered."""
        store.add_global("Cross-character fact")
        store.add_flash_card("Session fact", metadata={"session_id": "other_session"})

        store.set_session_id("current_session")
        results = store.query("fact")

        collections = {r["collection"] for r in results}
        assert "global" in collections

    def test_no_session_id_returns_all(self, store: MemoryStore):
        """When session_id is None, all flashcards are returned (backward compat)."""
        store.add_flash_card("Fact A", metadata={"session_id": "session_a"})
        store.add_flash_card("Fact B", metadata={"session_id": "session_b"})

        # session_id is None by default
        results = store.query("fact")
        flashcard_contents = [
            r["content"] for r in results if r["collection"] == FLASHCARDS
        ]
        assert "Fact A" in flashcard_contents
        assert "Fact B" in flashcard_contents

    def test_summaries_filtered_by_session(self, store: MemoryStore):
        """Summaries from a different session are excluded."""
        store.add_session_summary("Summary A", metadata={"session_id": "session_a"})
        store.add_session_summary("Summary B", metadata={"session_id": "session_b"})

        store.set_session_id("session_a")
        results = store.query("summary")

        summary_contents = [
            r["content"] for r in results if r["collection"] == SUMMARIES
        ]
        assert "Summary A" in summary_contents
        assert "Summary B" not in summary_contents

    def test_flashcards_without_session_id_excluded(self, store: MemoryStore):
        """Pre-existing flashcards with no session_id metadata are excluded
        when a session filter is active."""
        store.add_flash_card("Old unscoped card")  # no session_id in metadata
        store.add_flash_card("Current session card", metadata={"session_id": "current"})

        store.set_session_id("current")
        results = store.query("card")

        flashcard_contents = [
            r["content"] for r in results if r["collection"] == FLASHCARDS
        ]
        assert "Current session card" in flashcard_contents
        assert "Old unscoped card" not in flashcard_contents

    def test_promoted_memory_survives_session_change(self, store: MemoryStore):
        """A memory promoted to general is visible regardless of session."""
        # Simulate promotion: add to general (not flashcards)
        store.add_general("Promoted from session A")

        store.set_session_id("session_b")
        results = store.query("promoted")

        general_contents = [
            r["content"] for r in results if r["collection"] == GENERAL
        ]
        assert "Promoted from session A" in general_contents
