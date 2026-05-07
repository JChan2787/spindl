"""
Cross-activation plugin — multi-hop retrieval from RAG→Codex.

Reads recalled memories from RAG results, feeds them as supplemental
scan text to CodexManager.activate(), deduplicates against first-pass
activations, and merges new entries into the codex metadata.

Must be registered AFTER both CodexActivator and RAGInjector,
and BEFORE BudgetEnforcer.

NANO-127.
"""

import logging

from .base import PipelineContext, PreProcessor
from ...codex import CodexManager
from ...utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class CrossActivatorPlugin(PreProcessor):
    """
    PreProcessor that cross-activates Codex entries from RAG results.

    Flow:
    1. Read rag_content from context.metadata (populated by RAGInjector)
    2. Run CodexManager.activate() against the memory text
    3. Deduplicate against first-pass codex_results (by entry_id)
    4. Merge new activations into codex_content and codex_tokens_estimate

    Registration order:
        SummarizationTrigger → CodexActivator → RAGInjector →
        CrossActivator → BudgetEnforcer → HistoryInjector
    """

    def __init__(self, codex_manager: CodexManager, enabled: bool = False):
        self._manager = codex_manager
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "cross_activator"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def process(self, context: PipelineContext) -> PipelineContext:
        if not self._enabled:
            return context

        rag_content = context.metadata.get("rag_content", "")
        if not rag_content:
            return context

        cross_results = self._manager.activate(rag_content)
        if not cross_results:
            return context

        first_pass = context.metadata.get("codex_results", [])
        first_pass_ids = {r.entry_id for r in first_pass if r.activated}

        new_results = [
            r for r in cross_results
            if r.activated and r.entry_id not in first_pass_ids
        ]

        if not new_results:
            logger.debug("Cross-activation: all entries already activated in first pass")
            return context

        merged_results = [r for r in first_pass if r.activated] + new_results
        merged_content = self._manager.get_activated_content(merged_results)
        prev_estimate = context.metadata.get("codex_tokens_estimate", 0)
        token_estimate = count_tokens(merged_content) if merged_content else 0

        context.metadata["codex_results"] = merged_results
        context.metadata["codex_content"] = merged_content
        context.metadata["codex_tokens_estimate"] = token_estimate

        cross_names = [r.entry_name or f"entry_{r.entry_id}" for r in new_results]
        print(
            f"[NANO-127] Cross-activated {len(new_results)} codex entries from RAG: "
            f"{cross_names} (+{token_estimate - prev_estimate} tokens)",
            flush=True,
        )

        return context


def create_cross_activator(
    codex_manager: CodexManager,
    enabled: bool = False,
) -> CrossActivatorPlugin:
    """
    Factory function to create a CrossActivatorPlugin.

    Args:
        codex_manager: CodexManager instance (same as CodexActivatorPlugin)
        enabled: Whether cross-activation is active (default: False)

    Returns:
        Configured CrossActivatorPlugin instance
    """
    return CrossActivatorPlugin(
        codex_manager=codex_manager,
        enabled=enabled,
    )
