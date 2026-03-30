"""
MemoryStore — Per-character ChromaDB memory storage.

Manages three collections per character:
  - {character_id}_general    — Human-curated permanent facts
  - {character_id}_flashcards — Reflection-generated Q&A flash cards
  - {character_id}_summaries  — End-of-session narrative summaries

Uses a custom embedding function that delegates to EmbeddingClient,
bridging the external embedding server with ChromaDB's storage.

Part of NANO-043 Phase 1.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .curation_client import CurationClient as CurationClientProtocol

import chromadb
import numpy as np

from .embedding_client import EmbeddingClient, EmbeddingError

logger = logging.getLogger(__name__)

# Collection suffixes (per-character)
GENERAL = "general"
FLASHCARDS = "flashcards"
SUMMARIES = "summaries"

# Global collection name (cross-character, NANO-105)
GLOBAL = "global_memories"

# Sentinel for "use instance default" in _add() dedup_threshold parameter
_SENTINEL = object()

# --- Retrieval Scoring (NANO-107) ---
# Composite scoring: score = w_rel * relevance + w_rec * decay(time) + w_imp * importance + w_freq * frequency
# Applied as tier_weight * raw_score for final ranking.

# Weights (must sum to 1.0 for normalized scoring)
W_RELEVANCE = 0.5
W_RECENCY = 0.2
W_IMPORTANCE = 0.2
W_FREQUENCY = 0.1

# Ebbinghaus decay base — 0.9975^hours gives half-life ~277 hours (~11.5 days)
DECAY_BASE = 0.9975

# Log-scale cap for access frequency — diminishing returns after this many accesses
MAX_ACCESS_CAP = 100

# Per-tier weight multipliers — curated tiers get a boost
TIER_WEIGHTS: dict[str, float] = {
    "global": 1.15,
    GENERAL: 1.10,
    FLASHCARDS: 1.00,
    SUMMARIES: 0.95,
}

# Default importance by creation source
DEFAULT_IMPORTANCE: dict[str, int] = {
    "global": 8,
    GENERAL: 7,
    FLASHCARDS: 4,
    SUMMARIES: 3,
}


def compute_score(
    distance: float,
    importance: int,
    last_accessed: str,
    access_count: int,
    tier: str,
    now: Optional[datetime] = None,
    w_relevance: float = W_RELEVANCE,
    w_recency: float = W_RECENCY,
    w_importance: float = W_IMPORTANCE,
    w_frequency: float = W_FREQUENCY,
    decay_base: float = DECAY_BASE,
) -> float:
    """
    Composite retrieval score combining relevance, recency, importance, and frequency.

    Synthesis of Park et al. (Generative Agents) three-term retrieval function
    and MemoryBank access-frequency reinforcement. Additive, not multiplicative —
    a zero in any single term does not kill the result.

    Args:
        distance: L2 distance from ChromaDB (0 = identical, 2 = maximally different).
        importance: 1-10 rating (higher = more important).
        last_accessed: ISO 8601 timestamp of last access.
        access_count: Number of times this memory has been retrieved into a prompt.
        tier: Collection tier name (global, general, flashcards, summaries).
        now: Current time. None = use UTC now.
        w_relevance: Weight for semantic similarity signal.
        w_recency: Weight for time-decay signal.
        w_importance: Weight for importance rating signal.
        w_frequency: Weight for access frequency signal.
        decay_base: Base for Ebbinghaus exponential decay (per hour).

    Returns:
        Composite score (higher = better). Tier-weighted.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # 1. Relevance: L2 distance [0, 2] → similarity [1, 0]
    relevance = 1.0 - (distance / 2.0)

    # 2. Recency: Ebbinghaus exponential decay since last access
    try:
        last_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
        hours_since = max((now - last_dt).total_seconds() / 3600.0, 0.0)
    except (ValueError, AttributeError):
        hours_since = 720.0  # ~30 days fallback for unparseable timestamps

    recency = decay_base ** hours_since

    # 3. Importance: normalize to [0, 1]
    importance_norm = max(min(importance, 10), 1) / 10.0

    # 4. Frequency: log-scaled access count
    frequency = math.log1p(access_count) / math.log1p(MAX_ACCESS_CAP)

    # 5. Weighted sum
    raw = (
        w_relevance * relevance
        + w_recency * recency
        + w_importance * importance_norm
        + w_frequency * frequency
    )

    # 6. Tier boost
    tier_weight = TIER_WEIGHTS.get(tier, 1.0)
    return raw * tier_weight


class ExternalEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    ChromaDB embedding function that delegates to an external server
    via EmbeddingClient. Bridges llama.cpp / Ollama / OpenAI embeddings
    into ChromaDB's storage layer.
    """

    def __init__(self, client: EmbeddingClient):
        self._client = client

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        vectors = self._client.embed_batch(input)
        return [np.array(v, dtype=np.float32) for v in vectors]

    def name(self) -> str:
        return "spindl_external_embedding"

    def get_config(self) -> dict:
        return {"base_url": self._client.base_url}

    @classmethod
    def build_from_config(cls, config: dict) -> "ExternalEmbeddingFunction":
        client = EmbeddingClient(base_url=config["base_url"])
        return cls(client)


class MemoryStore:
    """Per-character and global ChromaDB memory storage."""

    def __init__(
        self,
        character_id: str,
        memory_dir: str,
        embedding_client: EmbeddingClient,
        dedup_threshold: Optional[float] = 0.30,
        curation_client: Optional["CurationClientProtocol"] = None,
        global_memory_dir: Optional[str] = None,
        scoring_config: Optional[dict] = None,
    ):
        """
        Initialize ChromaDB collections for a character + global tier.

        Creates/opens per-character collections:
          - {character_id}_general
          - {character_id}_flashcards
          - {character_id}_summaries

        And one global collection (cross-character, user-entered only):
          - global_memories

        Args:
            character_id: Character identifier (e.g., "spindle").
            memory_dir: Path to character's memory directory for ChromaDB persistence.
            embedding_client: EmbeddingClient for generating embeddings.
            dedup_threshold: L2 distance below which entries are considered duplicates.
                None disables semantic dedup (content-hash exact dedup still active).
            curation_client: Optional CurationClient for LLM-assisted dedup decisions.
            global_memory_dir: Path to global memory directory. If None, derived
                as sibling to character memory dir (../global/).
            scoring_config: Optional dict with keys w_relevance, w_recency,
                w_importance, w_frequency, decay_base. None = use module defaults.
        """
        self._character_id = character_id
        self._dedup_threshold = dedup_threshold
        self._curation_client = curation_client
        self._scoring_config = scoring_config or {}
        self._session_id: Optional[str] = None  # NANO-107: session-scoped retrieval
        self._embedding_fn = ExternalEmbeddingFunction(embedding_client)

        logger.info(
            "Initializing MemoryStore for '%s' at %s", character_id, memory_dir
        )
        self._client = chromadb.PersistentClient(path=memory_dir)

        self._collections = {
            GENERAL: self._client.get_or_create_collection(
                name=f"{character_id}_{GENERAL}",
                embedding_function=self._embedding_fn,
            ),
            FLASHCARDS: self._client.get_or_create_collection(
                name=f"{character_id}_{FLASHCARDS}",
                embedding_function=self._embedding_fn,
            ),
            SUMMARIES: self._client.get_or_create_collection(
                name=f"{character_id}_{SUMMARIES}",
                embedding_function=self._embedding_fn,
            ),
        }

        # Global tier — cross-character, separate ChromaDB client (NANO-105)
        if global_memory_dir is None:
            global_memory_dir = str(Path(memory_dir).parent / "global")
        self._global_memory_dir = global_memory_dir

        logger.info("Initializing global memory at %s", global_memory_dir)
        self._global_client = chromadb.PersistentClient(path=global_memory_dir)
        self._global_collection = self._global_client.get_or_create_collection(
            name=GLOBAL,
            embedding_function=self._embedding_fn,
        )

    def add_flash_card(
        self, content: str, metadata: Optional[dict] = None
    ) -> str:
        """
        Store a Q&A reflection flash card.

        Args:
            content: The Q&A text to store.
            metadata: Optional metadata dict (timestamp, session_id, etc.).

        Returns:
            Document ID assigned to this flash card.
        """
        return self._add(FLASHCARDS, content, metadata)

    def add_session_summary(
        self, content: str, metadata: Optional[dict] = None
    ) -> str:
        """
        Store an end-of-session summary.

        Args:
            content: The narrative summary text.
            metadata: Optional metadata dict.

        Returns:
            Document ID assigned to this summary.
        """
        return self._add(SUMMARIES, content, metadata)

    def add_general(
        self, content: str, metadata: Optional[dict] = None
    ) -> str:
        """
        Store a user-curated general memory.

        Manual entries bypass semantic dedup (Phase 1) and LLM curation (Phase 2).
        Content-hash exact dedup (Phase 0) still applies.

        Args:
            content: The permanent fact or memory text.
            metadata: Optional metadata dict.

        Returns:
            Document ID assigned to this memory.
        """
        return self._add(GENERAL, content, metadata, dedup_threshold=None)

    # --- Global tier methods (NANO-105) ---

    def add_global(
        self, content: str, metadata: Optional[dict] = None
    ) -> str:
        """
        Store a user-entered global memory (cross-character).

        Bypasses semantic dedup — content-hash exact dedup only.

        Args:
            content: The memory text.
            metadata: Optional metadata dict.

        Returns:
            Document ID assigned to this memory.
        """
        return self._add_to_collection(
            self._global_collection, "global", content, metadata, dedup_threshold=None
        )

    def edit_global(self, doc_id: str, new_content: str) -> str:
        """
        Edit a global memory by deleting and re-adding with new content.

        Args:
            doc_id: The document ID to replace.
            new_content: The updated memory text.

        Returns:
            New document ID (content-hash based).
        """
        # Get existing metadata
        try:
            existing = self._global_collection.get(ids=[doc_id], include=["metadatas"])
            old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        except Exception:
            old_meta = {}

        # Delete old, add new
        self._global_collection.delete(ids=[doc_id])
        meta = {**old_meta, "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        new_id = self._content_id(new_content)
        if "timestamp" not in meta:
            meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._global_collection.add(
            ids=[new_id],
            documents=[new_content],
            metadatas=[meta],
        )
        return new_id

    def delete_global(self, doc_id: str) -> bool:
        """Delete a global memory by ID."""
        self._global_collection.delete(ids=[doc_id])
        return True

    def get_all_global(self) -> list[dict]:
        """Get all global memories."""
        count = self._global_collection.count()
        if count == 0:
            return []
        result = self._global_collection.get()
        items = []
        for i, doc_id in enumerate(result["ids"]):
            items.append(
                {
                    "id": doc_id,
                    "content": result["documents"][i],
                    "metadata": result["metadatas"][i] or {},
                }
            )
        return items

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Query all collections for relevant memories with composite scoring.

        Queries global, general, flash cards, and summaries collections,
        merges results, and ranks by composite score (NANO-107):
          score = w_rel * relevance + w_rec * decay(time) + w_imp * importance + w_freq * frequency
        Tier weights boost curated collections over auto-generated ones.

        Args:
            query_text: The semantic search query.
            top_k: Maximum number of results per collection.

        Returns:
            List of dicts with keys: content, collection, distance, score, metadata.
            Sorted by score (descending — highest score first).
        """
        results = []
        now = datetime.now(timezone.utc)

        for collection_type, collection in self._collections.items():
            count = collection.count()
            if count == 0:
                continue

            try:
                query_result = collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k, collection.count()),
                )
            except Exception as e:
                logger.warning(
                    "Failed to query %s_%s: %s — retrying with fresh handle",
                    self._character_id,
                    collection_type,
                    e,
                )
                try:
                    collection_name = f"{self._character_id}_{collection_type}"
                    fresh = self._client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=self._embedding_fn,
                    )
                    self._collections[collection_type] = fresh
                    query_result = fresh.query(
                        query_texts=[query_text],
                        n_results=min(top_k, fresh.count()),
                    )
                except Exception as retry_err:
                    logger.error(
                        "Retry also failed for %s_%s: %s",
                        self._character_id,
                        collection_type,
                        retry_err,
                    )
                    continue

            ids = query_result["ids"][0]
            documents = query_result["documents"][0]
            distances = query_result["distances"][0]
            metadatas = query_result["metadatas"][0]

            for i, doc_id in enumerate(ids):
                meta = metadatas[i] or {}

                # NANO-107: session-scoped retrieval — flashcards and summaries
                # only return results from the current session.
                if (
                    self._session_id
                    and collection_type in (FLASHCARDS, SUMMARIES)
                    and meta.get("session_id") != self._session_id
                ):
                    continue

                dist = distances[i]
                score = compute_score(
                    distance=dist,
                    importance=meta.get("importance", 5),
                    last_accessed=meta.get("last_accessed", meta.get("timestamp", "")),
                    access_count=meta.get("access_count", 0),
                    tier=collection_type,
                    now=now,
                    **self._scoring_config,
                )
                results.append(
                    {
                        "id": doc_id,
                        "content": documents[i],
                        "collection": collection_type,
                        "distance": dist,
                        "score": score,
                        "metadata": meta,
                    }
                )

        # Query global collection (NANO-105)
        try:
            global_count = self._global_collection.count()
            if global_count > 0:
                global_result = self._global_collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k, global_count),
                )
                for i, doc_id in enumerate(global_result["ids"][0]):
                    meta = global_result["metadatas"][0][i] or {}
                    dist = global_result["distances"][0][i]
                    score = compute_score(
                        distance=dist,
                        importance=meta.get("importance", 5),
                        last_accessed=meta.get("last_accessed", meta.get("timestamp", "")),
                        access_count=meta.get("access_count", 0),
                        tier="global",
                        now=now,
                        **self._scoring_config,
                    )
                    results.append(
                        {
                            "id": doc_id,
                            "content": global_result["documents"][0][i],
                            "collection": "global",
                            "distance": dist,
                            "score": score,
                            "metadata": meta,
                        }
                    )
        except Exception as e:
            logger.warning("Failed to query global memories: %s", e)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def reinforce(self, results: list[dict]) -> None:
        """
        Bump access_count and last_accessed for memories that were injected into a prompt.

        Called by RAGInjector after filtering — only memories that actually reach
        the prompt get credit. Uses ChromaDB metadata-only update (no re-embedding).

        NANO-107: access reinforcement.

        Args:
            results: List of result dicts from query(), each with id, collection, metadata.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for result in results:
            try:
                collection = self._resolve_collection(result["collection"])
                if collection is None:
                    continue
                meta = result["metadata"].copy()
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta["last_accessed"] = now
                collection.update(ids=[result["id"]], metadatas=[meta])
            except Exception as e:
                logger.debug("Failed to reinforce %s: %s", result["id"][:12], e)

    def _resolve_collection(self, tier: str):
        """Resolve a tier name to a ChromaDB collection handle."""
        if tier == "global":
            return self._global_collection
        return self._collections.get(tier)

    def delete(self, collection_name: str, doc_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            collection_name: One of "global", "general", "flashcards", "summaries".
            doc_id: The document ID to delete.

        Returns:
            True if deletion was attempted (ChromaDB doesn't confirm existence).
        """
        if collection_name == "global":
            self._global_collection.delete(ids=[doc_id])
            return True

        if collection_name not in self._collections:
            logger.warning("Unknown collection: %s", collection_name)
            return False

        self._collections[collection_name].delete(ids=[doc_id])
        return True

    def clear_flash_cards(self) -> None:
        """Clear all flash cards (e.g., for testing or reset)."""
        name = f"{self._character_id}_{FLASHCARDS}"
        self._client.delete_collection(name)
        self._collections[FLASHCARDS] = self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
        )
        logger.info("Cleared flash cards for '%s'", self._character_id)

    def get_all(self, collection_name: str) -> list[dict]:
        """
        Get all memories from a specific collection.

        Args:
            collection_name: One of "global", "general", "flashcards", "summaries".

        Returns:
            List of dicts with keys: id, content, metadata.
        """
        if collection_name == "global":
            return self.get_all_global()

        if collection_name not in self._collections:
            logger.warning("Unknown collection: %s", collection_name)
            return []

        collection = self._collections[collection_name]
        count = collection.count()
        if count == 0:
            return []

        result = collection.get()
        items = []
        for i, doc_id in enumerate(result["ids"]):
            items.append(
                {
                    "id": doc_id,
                    "content": result["documents"][i],
                    "metadata": result["metadatas"][i] or {},
                }
            )
        return items

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Set the current session ID for session-scoped retrieval.

        Flashcards and summaries are only retrieved if their session_id
        matches. Global and general memories are always retrieved.
        None = no session filter (retrieve all).
        """
        self._session_id = session_id
        logger.info("MemoryStore session_id set to: %s", session_id)

    def switch_character(self, new_character_id: str) -> None:
        """
        Switch to a different character's memory collections.

        Updates the character_id and re-derives collection handles without
        re-creating the ChromaDB client connection (which is expensive).
        Global collection is preserved — it is cross-character (NANO-105).
        Used by runtime character switching (NANO-077).

        Args:
            new_character_id: The new character identifier (e.g., "mryummers")
        """
        logger.info(
            "Switching MemoryStore from '%s' to '%s'",
            self._character_id,
            new_character_id,
        )
        self._character_id = new_character_id
        self._collections = {
            GENERAL: self._client.get_or_create_collection(
                name=f"{new_character_id}_{GENERAL}",
                embedding_function=self._embedding_fn,
            ),
            FLASHCARDS: self._client.get_or_create_collection(
                name=f"{new_character_id}_{FLASHCARDS}",
                embedding_function=self._embedding_fn,
            ),
            SUMMARIES: self._client.get_or_create_collection(
                name=f"{new_character_id}_{SUMMARIES}",
                embedding_function=self._embedding_fn,
            ),
        }
        # Global collection intentionally NOT re-created — persists across characters

    @property
    def character_id(self) -> str:
        """The character ID this store manages."""
        return self._character_id

    @property
    def counts(self) -> dict[str, int]:
        """Document counts per collection, including global."""
        counts = {name: col.count() for name, col in self._collections.items()}
        counts["global"] = self._global_collection.count()
        return counts

    @staticmethod
    def _content_id(content: str) -> str:
        """Deterministic document ID from content hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _add(
        self,
        collection_type: str,
        content: str,
        metadata: Optional[dict] = None,
        dedup_threshold: Optional[float] = _SENTINEL,
    ) -> str:
        """
        Internal: add a document to a per-character collection with deduplication.

        Delegates to _add_to_collection with the resolved collection handle.

        Args:
            collection_type: One of GENERAL, FLASHCARDS, SUMMARIES.
            content: The text to store.
            metadata: Optional metadata dict (timestamp auto-injected if missing).
            dedup_threshold: Per-call override. Pass None to disable semantic dedup
                for this call. Omit (or _SENTINEL) to use instance default.

        Returns:
            Document ID (existing ID if deduplicated, new hash ID if stored).
        """
        collection = self._collections[collection_type]
        return self._add_to_collection(
            collection, collection_type, content, metadata, dedup_threshold
        )

    def _add_to_collection(
        self,
        collection: "chromadb.Collection",
        label: str,
        content: str,
        metadata: Optional[dict] = None,
        dedup_threshold: Optional[float] = _SENTINEL,
    ) -> str:
        """
        Internal: add a document to any collection with deduplication.

        Three-layer dedup pipeline:
          Phase 0: Content-hash ID — identical content maps to same ID (free, instant).
          Phase 1: Similarity gate — L2 distance query catches paraphrases.
          Phase 2: LLM curation — ambiguous cases routed to frontier model (if enabled).

        Args:
            collection: ChromaDB collection handle.
            label: Human-readable label for logging (e.g., "general", "global").
            content: The text to store.
            metadata: Optional metadata dict (timestamp auto-injected if missing).
            dedup_threshold: Per-call override. Pass None to disable semantic dedup
                for this call. Omit (or _SENTINEL) to use instance default.

        Returns:
            Document ID (existing ID if deduplicated, new hash ID if stored).
        """
        doc_id = self._content_id(content)

        meta = metadata.copy() if metadata else {}
        if "timestamp" not in meta:
            meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # --- NANO-107: Retrieval scoring metadata ---
        if "importance" not in meta:
            meta["importance"] = DEFAULT_IMPORTANCE.get(label, 5)
        if "access_count" not in meta:
            meta["access_count"] = 0
        if "last_accessed" not in meta:
            meta["last_accessed"] = meta["timestamp"]

        # --- Phase 0: exact duplicate via content-hash ID ---
        try:
            existing = collection.get(ids=[doc_id], include=[])
            if existing["ids"]:
                print(f"[Memory] Exact duplicate blocked in {label}: {content[:60]}...", flush=True)
                return doc_id
        except Exception:
            pass  # ID doesn't exist — continue

        # --- Phase 1: semantic similarity gate ---
        threshold = (
            self._dedup_threshold if dedup_threshold is _SENTINEL else dedup_threshold
        )

        if threshold is not None and collection.count() > 0:
            try:
                results = collection.query(query_texts=[content], n_results=1)
                if results["distances"] and results["distances"][0]:
                    nearest_dist = results["distances"][0][0]
                    nearest_id = results["ids"][0][0]

                    if nearest_dist < threshold:
                        # --- Phase 2: LLM curation (if enabled) ---
                        if self._curation_client is not None:
                            decision = self._run_curation(
                                content, label, results, meta
                            )
                            if decision is not None:
                                return decision

                        # No curation or curation disabled — simple skip
                        print(
                            f"[Memory] Dedup: skipping near-duplicate in {label} "
                            f"(dist={nearest_dist:.4f} < {threshold:.4f}): {content[:60]}...",
                            flush=True,
                        )
                        return nearest_id
            except Exception as e:
                logger.warning("Dedup similarity query failed in %s: %s", label, e)

        # --- No duplicate found — insert with content-hash ID ---
        collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta],
        )

        print(f"[Memory] Stored in {label}: {content[:60]}...", flush=True)
        return doc_id

    def _run_curation(
        self,
        candidate: str,
        collection_type: str,
        query_results: dict,
        metadata: dict,
    ) -> Optional[str]:
        """
        Route an ambiguous near-duplicate to the LLM curation judge.

        Returns:
            doc_id if the curation decision was handled (ADD/SKIP/UPDATE/DELETE),
            None to fall back to default SKIP behavior.
        """
        try:
            # Build existing entries list for the judge
            existing = []
            for i, eid in enumerate(query_results["ids"][0]):
                doc_content = query_results["documents"][0][i] if query_results.get("documents") else ""
                dist = query_results["distances"][0][i] if query_results.get("distances") else 0.0
                existing.append({"id": eid, "content": doc_content, "distance": dist})

            decision = self._curation_client.classify(candidate, existing)

            if decision.action == "ADD":
                doc_id = self._content_id(candidate)
                self._collections[collection_type].add(
                    ids=[doc_id],
                    documents=[candidate],
                    metadatas=[metadata],
                )
                print(f"[Memory] Curation: ADD in {collection_type} — new info despite similarity: {candidate[:60]}...", flush=True)
                return doc_id

            elif decision.action == "SKIP":
                print(f"[Memory] Curation: SKIP in {collection_type} — already captured: {candidate[:60]}...", flush=True)
                return query_results["ids"][0][0]

            elif decision.action == "UPDATE" and decision.target_id and decision.merged_text:
                collection = self._collections[collection_type]
                update_meta = metadata.copy()
                update_meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                update_meta["supersedes"] = candidate[:100]
                collection.update(
                    ids=[decision.target_id],
                    documents=[decision.merged_text],
                    metadatas=[update_meta],
                )
                print(f"[Memory] Curation: UPDATE in {collection_type} — merged into {decision.target_id}: {decision.merged_text[:60]}...", flush=True)
                return decision.target_id

            elif decision.action == "DELETE" and decision.target_id:
                collection = self._collections[collection_type]
                collection.delete(ids=[decision.target_id])
                # Insert the new entry as the replacement
                doc_id = self._content_id(candidate)
                collection.add(
                    ids=[doc_id],
                    documents=[candidate],
                    metadatas=[metadata],
                )
                print(f"[Memory] Curation: DELETE {decision.target_id} + ADD in {collection_type}: {candidate[:60]}...", flush=True)
                return doc_id

        except Exception as e:
            print(f"[Memory] Curation failed in {collection_type}, falling back to default: {e}", flush=True)

        return None
