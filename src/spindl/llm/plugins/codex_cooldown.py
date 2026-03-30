"""
Codex cooldown plugin for managing timed effects after LLM responses.

Advances the codex turn counter after each conversation turn,
enabling sticky and cooldown effects to function correctly.
"""

import logging

from .base import PipelineContext, PostProcessor
from ...codex import CodexManager


logger = logging.getLogger(__name__)


class CodexCooldownPlugin(PostProcessor):
    """
    PostProcessor that manages codex timed effects.

    After each LLM response:
    1. Advances the turn counter in CodexState
    2. This causes sticky entries to expire when their duration ends
    3. This causes cooldown entries to become re-activatable

    Must share the same CodexManager instance as CodexActivatorPlugin.

    Registration order:
        ... → HistoryRecorder → CodexCooldownPlugin
    """

    def __init__(self, codex_manager: CodexManager):
        """
        Initialize the codex cooldown plugin.

        Args:
            codex_manager: CodexManager instance (shared with CodexActivatorPlugin)
        """
        self._manager = codex_manager

    @property
    def name(self) -> str:
        return "codex_cooldown"

    def process(self, context: PipelineContext, response: str) -> str:
        """
        Advance turn counter and manage timed effects.

        Args:
            context: Pipeline context (contains codex_results from activator)
            response: LLM response (passed through unchanged)

        Returns:
            Unchanged response
        """
        # Get previous state for logging
        state_before = self._manager.state.to_dict()

        # Advance turn counter
        self._manager.advance_turn()

        # Get new state
        state_after = self._manager.state.to_dict()

        # Log state changes
        sticky_before = set(state_before.get("active_sticky", []))
        sticky_after = set(state_after.get("active_sticky", []))
        cooldown_before = set(state_before.get("on_cooldown", []))
        cooldown_after = set(state_after.get("on_cooldown", []))

        expired_sticky = sticky_before - sticky_after
        expired_cooldown = cooldown_before - cooldown_after

        if expired_sticky:
            logger.debug("Codex sticky expired for entries: %s", list(expired_sticky))

        if expired_cooldown:
            logger.debug("Codex cooldown expired for entries: %s", list(expired_cooldown))

        logger.debug(
            "Codex turn advanced to %d (sticky: %d, cooldown: %d)",
            state_after["current_turn"],
            len(sticky_after),
            len(cooldown_after),
        )

        return response


def create_codex_cooldown(codex_manager: CodexManager) -> CodexCooldownPlugin:
    """
    Factory function to create a CodexCooldownPlugin.

    Args:
        codex_manager: CodexManager instance (same as CodexActivatorPlugin)

    Returns:
        Configured CodexCooldownPlugin instance

    Usage:
        See create_codex_activator() for full registration example.
    """
    return CodexCooldownPlugin(codex_manager=codex_manager)


def create_codex_plugins(
    characters_dir: str = "./characters",
    character_id: str | None = None,
    match_whole_words: bool = False,
    max_entries_per_turn: int | None = None,
    scan_assistant_response: bool = False,
) -> tuple[CodexManager, "CodexActivatorPlugin", CodexCooldownPlugin]:
    """
    Factory function to create a complete codex plugin set.

    Creates a shared CodexManager and both plugins in one call.

    Args:
        characters_dir: Directory containing character folders
        character_id: Character to load (if provided, loads immediately)
        match_whole_words: Default whole-word matching mode
        max_entries_per_turn: Max entries to activate per turn
        scan_assistant_response: Enable recursive scanning

    Returns:
        Tuple of (CodexManager, CodexActivatorPlugin, CodexCooldownPlugin)

    Usage:
        manager, activator, cooldown = create_codex_plugins(
            characters_dir="./characters",
            character_id="spindle",
        )

        # Register in correct order
        pipeline.register_pre_processor(summarization_trigger)
        pipeline.register_pre_processor(activator)
        pipeline.register_pre_processor(budget_enforcer)
        pipeline.register_pre_processor(history_injector)
        pipeline.register_post_processor(history_recorder)
        pipeline.register_post_processor(cooldown)
    """
    from .codex_activator import CodexActivatorPlugin

    # Create shared manager
    manager = CodexManager(
        characters_dir=characters_dir,
        match_whole_words=match_whole_words,
        max_entries_per_turn=max_entries_per_turn,
    )

    # Load character if specified
    if character_id:
        manager.load_character(character_id)

    # Create plugin pair
    activator = CodexActivatorPlugin(
        codex_manager=manager,
        scan_assistant_response=scan_assistant_response,
    )
    cooldown = CodexCooldownPlugin(codex_manager=manager)

    return manager, activator, cooldown
