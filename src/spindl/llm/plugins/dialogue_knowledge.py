"""
Dialogue knowledge injection plugin (NANO-116 Phase B.2).

PreProcessor that fills the [CHARACTER_KNOWLEDGE] placeholder in the
system prompt with accumulated game character dialogue — summary blob
+ raw unsummarized tail when over budget, all raw lines when under.

Mirrors the TwitchHistoryInjector pattern from NANO-115.
"""

from typing import Optional

from .base import PipelineContext, PreProcessor


class DialogueKnowledgeInjector(PreProcessor):
    """
    PreProcessor that injects game dialogue character knowledge into
    the [CHARACTER_KNOWLEDGE] placeholder in the system prompt.

    Fires every turn. Collapses to empty string when no dialogue content
    exists or the game-state module hasn't received dialogue events.
    """

    PLACEHOLDER = "[CHARACTER_KNOWLEDGE]"

    PROTOCOL_PREAMBLE = (
        "The following is your accumulated knowledge of in-game characters "
        "from dialogue you've observed during this playthrough. Use this "
        "context to inform your commentary — reference characters by name, "
        "note what they've said or done, and track narrative progression."
    )

    def __init__(self, token_budget_chars: int = 2000):
        self._token_budget_chars = token_budget_chars
        self._dialogue_store = None  # Set by orchestrator wiring

    @property
    def name(self) -> str:
        return "dialogue_knowledge_injector"

    @property
    def token_budget_chars(self) -> int:
        return self._token_budget_chars

    @token_budget_chars.setter
    def token_budget_chars(self, value: int) -> None:
        self._token_budget_chars = max(500, value)

    def set_dialogue_store(self, store: "Optional[object]") -> None:
        """Bind the DialogueStore instance. Called during orchestrator startup."""
        self._dialogue_store = store

    def process(self, context: PipelineContext) -> PipelineContext:
        """Fill [CHARACTER_KNOWLEDGE] placeholder with dialogue content."""
        content = ""

        if self._dialogue_store is not None:
            # Import here to avoid circular imports at module level
            from ...stimuli.game_state.dialogue_store import DialogueStore

            store: DialogueStore = self._dialogue_store  # type: ignore[assignment]
            raw_content = store.get_injection_content(self._token_budget_chars)
            if raw_content:
                content = f"{self.PROTOCOL_PREAMBLE}\n\n{raw_content}"

        context.metadata["character_knowledge_formatted"] = content
        return context
