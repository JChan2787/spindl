"""
RAG Injector — PreProcessor that queries ChromaDB and stages
retrieved memories for injection into the [RAG_CONTEXT] placeholder.

NANO-043 Phase 2.

Follows the CodexActivatorPlugin pattern:
  1. PreProcessor queries memories, stores results in context.metadata
  2. Pipeline._inject_rag_content() replaces [RAG_CONTEXT] in system prompt

Must be registered AFTER CodexActivatorPlugin and BEFORE BudgetEnforcer
so that RAG token costs are included in the budget calculation.
"""

import logging
from typing import Optional

from ..llm.plugins.base import PipelineContext, PreProcessor
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)

# L2 distance ceiling used for threshold conversion.
# With normalized embeddings, L2 distance ranges from 0 (identical)
# to 2.0 (maximally different). This constant defines the upper bound.
_MAX_L2_DISTANCE = 2.0


def threshold_to_max_distance(threshold: float) -> float:
    """Convert user-facing relevance threshold to L2 max-distance.

    User semantics: 0.0 = accept everything, 1.0 = only exact matches.
    Internal:       max_distance = 2.0 * (1.0 - threshold).

    Examples:
        0.0  → 2.0  (everything passes)
        0.25 → 1.5  (very loose)
        0.5  → 1.0  (moderate)
        0.75 → 0.5  (strict)
        1.0  → 0.0  (only identical)
    """
    return _MAX_L2_DISTANCE * (1.0 - threshold)


class RAGInjector(PreProcessor):
    """
    PreProcessor that queries ChromaDB for relevant memories and
    stages them in context.metadata for pipeline injection.

    Stores:
        context.metadata["rag_content"]          — formatted memory text
        context.metadata["rag_tokens_estimate"]  — rough token count (len/4)
        context.metadata["rag_results"]          — raw result dicts for GUI display

    Graceful degradation: if the embedding server is unreachable or
    the query fails, stores empty string and 0 tokens. The pipeline
    collapses [RAG_CONTEXT] to nothing and the conversation proceeds
    without memories.
    """

    # Default wrapper strings (NANO-045d)
    _DEFAULT_RAG_PREFIX = (
        "The following are relevant memories about the user and past "
        "conversations. Use them to inform your response:"
    )
    _DEFAULT_RAG_SUFFIX = "End of memories."

    def __init__(
        self,
        memory_store: MemoryStore,
        top_k: int = 5,
        relevance_threshold: Optional[float] = None,
        rag_prefix: Optional[str] = None,
        rag_suffix: Optional[str] = None,
    ):
        """
        Args:
            memory_store: MemoryStore for the active character.
            top_k: Maximum number of memories to retrieve per query.
            relevance_threshold: Relevance strictness from 0.0 (accept
                everything) to 1.0 (only exact matches). None = no filtering.
            rag_prefix: Header text before memory list. None = use default.
            rag_suffix: Footer text after memory list. None = use default.
        """
        self._memory_store = memory_store
        self._top_k = top_k
        self._relevance_threshold = relevance_threshold
        self._rag_prefix = rag_prefix if rag_prefix is not None else self._DEFAULT_RAG_PREFIX
        self._rag_suffix = rag_suffix if rag_suffix is not None else self._DEFAULT_RAG_SUFFIX

    @property
    def name(self) -> str:
        return "rag_injector"

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Query memories relevant to the user's input and stage results.

        Uses context.user_input as the semantic search query.
        """
        try:
            results = self._memory_store.query(
                query_text=context.user_input,
                top_k=self._top_k,
            )
        except Exception as e:
            logger.warning("RAG query failed (memories unavailable): %s", e)
            context.metadata["rag_content"] = ""
            context.metadata["rag_tokens_estimate"] = 0
            return context

        if not results:
            context.metadata["rag_content"] = ""
            context.metadata["rag_tokens_estimate"] = 0
            return context

        # Filter by distance threshold — discard semantically distant results.
        # User-facing threshold: 0.0 = accept everything, 1.0 = only exact matches.
        # Internally converted to L2 max-distance for the comparison.
        if self._relevance_threshold is not None:
            max_dist = threshold_to_max_distance(self._relevance_threshold)
            before_count = len(results)
            results = [
                r for r in results if r["distance"] <= max_dist
            ]
        # MemoryStore.query() applies top_k per collection, so the merged
        # result set can exceed top_k. Apply a global trim here.
        results = results[:self._top_k]

        if not results:
            context.metadata["rag_content"] = ""
            context.metadata["rag_tokens_estimate"] = 0
            return context

        # Format memories for injection
        rag_text = self._format_memories(results)

        context.metadata["rag_content"] = rag_text
        context.metadata["rag_tokens_estimate"] = len(rag_text) // 4
        context.metadata["rag_results"] = results

        # NANO-107: reinforce memories that actually made it into the prompt
        try:
            self._memory_store.reinforce(results)
        except Exception as e:
            logger.debug("Memory reinforcement failed (non-fatal): %s", e)

        logger.debug(
            "RAG injected %d memories (~%d tokens)",
            len(results),
            context.metadata["rag_tokens_estimate"],
        )

        return context

    def update_config(
        self,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = ...,
        rag_prefix: Optional[str] = ...,
        rag_suffix: Optional[str] = ...,
    ) -> None:
        """
        Update RAG query parameters at runtime (no pipeline rebuild needed).

        Args:
            top_k: New max results. None = keep current.
            relevance_threshold: Relevance strictness 0.0–1.0. None = disable.
                                 Ellipsis (...) = keep current.
            rag_prefix: New prefix string. Ellipsis (...) = keep current.
            rag_suffix: New suffix string. Ellipsis (...) = keep current.
        """
        if top_k is not None:
            self._top_k = top_k
        if relevance_threshold is not ...:
            self._relevance_threshold = relevance_threshold
        if rag_prefix is not ...:
            self._rag_prefix = rag_prefix if rag_prefix is not None else self._DEFAULT_RAG_PREFIX
        if rag_suffix is not ...:
            self._rag_suffix = rag_suffix if rag_suffix is not None else self._DEFAULT_RAG_SUFFIX

    def _format_memories(self, results: list[dict]) -> str:
        """
        Format retrieved memories into injection text.

        Uses configurable prefix/suffix strings (NANO-045d).

        Args:
            results: List of memory dicts from MemoryStore.query().
                     Each has: content, collection, distance, metadata.

        Returns:
            Formatted string for [RAG_CONTEXT] placeholder.
        """
        lines = [self._rag_prefix]
        for mem in results:
            lines.append(f"- {mem['content']}")
        if self._rag_suffix:
            lines.append(self._rag_suffix)
        return "\n".join(lines)
