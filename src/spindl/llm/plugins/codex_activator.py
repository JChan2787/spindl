"""
Codex activation plugin for injecting lorebook content into prompts.

Scans user input for keywords and injects matching codex entries
at their configured positions in the prompt.

Must be registered BEFORE BudgetEnforcer so codex tokens are
included in budget calculations.
"""

import logging
from typing import Optional

from .base import PipelineContext, PreProcessor
from ...codex import CodexManager, ActivationResult


logger = logging.getLogger(__name__)


class CodexActivatorPlugin(PreProcessor):
    """
    PreProcessor that activates codex entries based on user input.

    Scans the user's input for keywords matching codex entries.
    Activated entries are stored in context.metadata for:
    1. BudgetEnforcer to include in token calculations
    2. CodexCooldownPlugin to update state after the turn

    Injection happens via context.metadata["codex_content"] which
    the PromptBuilder uses to add content at configured positions.

    Registration order:
        SummarizationTrigger → CodexActivator → BudgetEnforcer → HistoryInjector
    """

    def __init__(
        self,
        codex_manager: CodexManager,
        scan_assistant_response: bool = False,
    ):
        """
        Initialize the codex activator.

        Args:
            codex_manager: CodexManager instance (shared with CodexCooldownPlugin)
            scan_assistant_response: If True, also scan previous assistant
                response for recursive activation (default: False)
        """
        self._manager = codex_manager
        self._scan_assistant_response = scan_assistant_response

    @property
    def name(self) -> str:
        return "codex_activator"

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Scan input for keywords and activate matching codex entries.

        Args:
            context: Pipeline context

        Returns:
            Context with codex metadata populated:
                - codex_results: List of ActivationResult objects
                - codex_content: Combined content for injection (single field)
                - codex_tokens_estimate: Rough token count for budget
        """
        # Build scan text
        scan_text = context.user_input

        # Optionally include previous assistant response for recursive scanning
        if self._scan_assistant_response and context.messages:
            # Find last assistant message
            for msg in reversed(context.messages):
                if msg.get("role") == "assistant":
                    scan_text = f"{msg.get('content', '')}\n{scan_text}"
                    break

        # Activate entries
        results = self._manager.activate(scan_text)

        # Get all activated content (position field ignored - single injection point)
        # insertion_order and priority still control ordering within the block
        codex_content = self._manager.get_activated_content(results)

        # Estimate tokens (rough: 4 chars per token average)
        # This is used by BudgetEnforcer for quick budget check
        token_estimate = len(codex_content) // 4 if codex_content else 0

        # Store in metadata for pipeline to inject
        context.metadata["codex_results"] = results
        context.metadata["codex_content"] = codex_content
        context.metadata["codex_tokens_estimate"] = token_estimate

        if results:
            activated_names = [
                r.entry_name or f"entry_{r.entry_id}"
                for r in results
                if r.activated
            ]
            logger.info(
                "Codex activated %d entries: %s (est. %d tokens)",
                len(activated_names),
                activated_names,
                token_estimate,
            )

        return context


def create_codex_activator(
    codex_manager: CodexManager,
    scan_assistant_response: bool = False,
) -> CodexActivatorPlugin:
    """
    Factory function to create a CodexActivatorPlugin.

    Args:
        codex_manager: CodexManager instance
        scan_assistant_response: Enable recursive scanning of assistant output

    Returns:
        Configured CodexActivatorPlugin instance

    Usage:
        from spindl.codex import CodexManager
        from spindl.llm.plugins import create_codex_activator, create_codex_cooldown

        # Create shared manager
        manager = CodexManager(characters_dir="./characters")
        manager.load_character("spindle")

        # Create plugin pair
        activator = create_codex_activator(manager)
        cooldown = create_codex_cooldown(manager)

        # Register in correct order
        pipeline.register_pre_processor(summarization_trigger)
        pipeline.register_pre_processor(activator)  # BEFORE budget enforcer!
        pipeline.register_pre_processor(budget_enforcer)
        pipeline.register_pre_processor(history_injector)
        pipeline.register_post_processor(history_recorder)
        pipeline.register_post_processor(cooldown)  # AFTER history recorder
    """
    return CodexActivatorPlugin(
        codex_manager=codex_manager,
        scan_assistant_response=scan_assistant_response,
    )
