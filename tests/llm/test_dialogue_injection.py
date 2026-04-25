"""Tests for DialogueKnowledgeInjector (NANO-116 Phase B.2)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.llm.plugins.base import PipelineContext
from spindl.llm.plugins.dialogue_knowledge import DialogueKnowledgeInjector


def _make_context(system_content: str = "") -> PipelineContext:
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    return PipelineContext(
        user_input="test input",
        persona={"name": "Spindle"},
        messages=messages,
        metadata={},
    )


class TestDialogueKnowledgeInjectorBasics:
    """Basic PreProcessor behavior."""

    def test_name(self):
        injector = DialogueKnowledgeInjector()
        assert injector.name == "dialogue_knowledge_injector"

    def test_no_store_produces_empty(self):
        injector = DialogueKnowledgeInjector()
        context = _make_context("[CHARACTER_KNOWLEDGE]")
        result = injector.process(context)
        assert result.metadata["character_knowledge_formatted"] == ""

    def test_empty_store_produces_empty(self):
        injector = DialogueKnowledgeInjector()
        mock_store = MagicMock()
        mock_store.get_injection_content.return_value = ""
        injector.set_dialogue_store(mock_store)

        context = _make_context("[CHARACTER_KNOWLEDGE]")
        result = injector.process(context)
        assert result.metadata["character_knowledge_formatted"] == ""

    def test_content_injected_with_preamble(self):
        injector = DialogueKnowledgeInjector(token_budget_chars=5000)
        mock_store = MagicMock()
        mock_store.get_injection_content.return_value = "Diana: Watch out!\nKen: Stay close."
        injector.set_dialogue_store(mock_store)

        context = _make_context("[CHARACTER_KNOWLEDGE]")
        result = injector.process(context)
        formatted = result.metadata["character_knowledge_formatted"]

        assert "accumulated knowledge" in formatted
        assert "Diana: Watch out!" in formatted
        assert "Ken: Stay close." in formatted

    def test_token_budget_passed_to_store(self):
        injector = DialogueKnowledgeInjector(token_budget_chars=3000)
        mock_store = MagicMock()
        mock_store.get_injection_content.return_value = "content"
        injector.set_dialogue_store(mock_store)

        injector.process(_make_context("[CHARACTER_KNOWLEDGE]"))
        mock_store.get_injection_content.assert_called_once_with(3000)


class TestDialogueKnowledgeInjectorBudget:
    """Token budget property."""

    def test_default_budget(self):
        injector = DialogueKnowledgeInjector()
        assert injector.token_budget_chars == 2000

    def test_custom_budget(self):
        injector = DialogueKnowledgeInjector(token_budget_chars=5000)
        assert injector.token_budget_chars == 5000

    def test_budget_setter_clamps(self):
        injector = DialogueKnowledgeInjector()
        injector.token_budget_chars = 100  # Below minimum
        assert injector.token_budget_chars == 500

    def test_budget_setter_normal(self):
        injector = DialogueKnowledgeInjector()
        injector.token_budget_chars = 8000
        assert injector.token_budget_chars == 8000


class TestSourceLabeling:
    """Verify tag_user_input produces [Message Type - Game State]."""

    def test_game_state_tag(self):
        from spindl.orchestrator.callbacks import tag_user_input
        from spindl.llm.build_context import InputModality

        result = tag_user_input("test", InputModality.STIMULUS, stimulus_source="game_state")
        assert result.startswith("[Message Type - Game State]")

    def test_twitch_tag_unchanged(self):
        from spindl.orchestrator.callbacks import tag_user_input
        from spindl.llm.build_context import InputModality

        result = tag_user_input("test", InputModality.STIMULUS, stimulus_source="twitch")
        assert result.startswith("[Message Type - Twitch Chat]")

    def test_other_stimulus_falls_through(self):
        from spindl.orchestrator.callbacks import tag_user_input
        from spindl.llm.build_context import InputModality

        result = tag_user_input("test", InputModality.STIMULUS, stimulus_source="patience")
        assert result.startswith("[Message Type - Stimuli]")
