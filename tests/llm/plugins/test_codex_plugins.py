"""Tests for codex activation and cooldown plugins."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from pathlib import Path

from spindl.llm.plugins import (
    PipelineContext,
    CodexActivatorPlugin,
    CodexCooldownPlugin,
    create_codex_activator,
    create_codex_cooldown,
    create_codex_plugins,
)
from spindl.codex import CodexManager, ActivationResult
from spindl.characters.models import CharacterBookEntry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_characters_dir():
    """Create a temporary characters directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chars_dir = Path(tmpdir) / "characters"
        chars_dir.mkdir()

        # Create _global directory with codex
        global_dir = chars_dir / "_global"
        global_dir.mkdir()

        global_codex = {
            "name": "Global Codex",
            "entries": [
                {
                    "keys": ["coffee", "caffeine"],
                    "content": "User enjoys coffee, especially dark roast.",
                    "enabled": True,
                    "insertion_order": 1,
                    "id": 1,
                    "name": "coffee_preference",
                    "position": "after_char",
                },
                {
                    "keys": ["secret", "hidden"],
                    "content": "There is a secret door behind the bookshelf.",
                    "enabled": True,
                    "insertion_order": 2,
                    "id": 2,
                    "name": "secret_door",
                    "position": "before_char",
                },
                {
                    "keys": ["dragon"],
                    "content": "Dragons are mythical fire-breathing creatures.",
                    "enabled": True,
                    "insertion_order": 3,
                    "id": 3,
                    "name": "dragon_lore",
                    "sticky": 2,  # Stays active for 2 turns
                },
                {
                    "keys": ["magic"],
                    "content": "Magic requires years of training.",
                    "enabled": True,
                    "insertion_order": 4,
                    "id": 4,
                    "name": "magic_info",
                    "cooldown": 3,  # Can't re-activate for 3 turns
                },
            ],
        }

        with open(global_dir / "codex.json", "w") as f:
            json.dump(global_codex, f)

        yield chars_dir


@pytest.fixture
def codex_manager(temp_characters_dir):
    """Create a CodexManager with test entries loaded."""
    manager = CodexManager(characters_dir=temp_characters_dir)
    manager.load_global_codex()
    return manager


@pytest.fixture
def pipeline_context():
    """Create a basic pipeline context."""
    return PipelineContext(
        user_input="Tell me about coffee and dragons.",
        persona={"name": "Test", "system_prompt": "You are a helpful assistant."},
        messages=[],
        metadata={},
    )


# =============================================================================
# CodexActivatorPlugin Tests
# =============================================================================


class TestCodexActivatorPlugin:
    """Tests for CodexActivatorPlugin."""

    def test_name(self, codex_manager):
        """Plugin should have correct name."""
        plugin = CodexActivatorPlugin(codex_manager)
        assert plugin.name == "codex_activator"

    def test_activates_matching_entries(self, codex_manager, pipeline_context):
        """Should activate entries when keywords match."""
        plugin = CodexActivatorPlugin(codex_manager)

        result = plugin.process(pipeline_context)

        # Should have activated entries
        assert "codex_results" in result.metadata
        results = result.metadata["codex_results"]
        assert len(results) >= 2  # coffee and dragon should match

        # Check specific entries
        entry_names = [r.entry_name for r in results if r.activated]
        assert "coffee_preference" in entry_names
        assert "dragon_lore" in entry_names

    def test_stores_combined_content(self, codex_manager, pipeline_context):
        """Should store all activated content in single codex_content field."""
        # Update input to trigger entries with different positions
        pipeline_context.user_input = "Tell me about coffee and the secret door."
        plugin = CodexActivatorPlugin(codex_manager)

        result = plugin.process(pipeline_context)

        # Should have combined content field
        assert "codex_content" in result.metadata

        # Both entries should be in combined content
        codex_content = result.metadata["codex_content"].lower()
        assert "coffee" in codex_content
        assert "secret" in codex_content

    def test_estimates_tokens(self, codex_manager, pipeline_context):
        """Should estimate token count for activated entries."""
        plugin = CodexActivatorPlugin(codex_manager)

        result = plugin.process(pipeline_context)

        assert "codex_tokens_estimate" in result.metadata
        # Should be non-zero since we matched entries
        assert result.metadata["codex_tokens_estimate"] > 0

    def test_no_matches_empty_metadata(self, codex_manager):
        """Should handle no matches gracefully."""
        context = PipelineContext(
            user_input="Tell me about quantum physics.",  # No matching keywords
            persona={"name": "Test"},
            messages=[],
            metadata={},
        )
        plugin = CodexActivatorPlugin(codex_manager)

        result = plugin.process(context)

        assert result.metadata["codex_results"] == []
        assert result.metadata["codex_content"] == ""
        assert result.metadata["codex_tokens_estimate"] == 0

    def test_scan_assistant_response(self, codex_manager):
        """Should scan previous assistant response when enabled."""
        context = PipelineContext(
            user_input="What else?",  # No keywords
            persona={"name": "Test"},
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "I was just making coffee."},
            ],
            metadata={},
        )
        plugin = CodexActivatorPlugin(codex_manager, scan_assistant_response=True)

        result = plugin.process(context)

        # Should find "coffee" in the assistant response
        entry_names = [r.entry_name for r in result.metadata["codex_results"] if r.activated]
        assert "coffee_preference" in entry_names


class TestCodexCooldownPlugin:
    """Tests for CodexCooldownPlugin."""

    def test_name(self, codex_manager):
        """Plugin should have correct name."""
        plugin = CodexCooldownPlugin(codex_manager)
        assert plugin.name == "codex_cooldown"

    def test_advances_turn(self, codex_manager, pipeline_context):
        """Should advance turn counter after processing."""
        initial_turn = codex_manager.state.current_turn
        plugin = CodexCooldownPlugin(codex_manager)

        plugin.process(pipeline_context, "Test response")

        assert codex_manager.state.current_turn == initial_turn + 1

    def test_passes_response_unchanged(self, codex_manager, pipeline_context):
        """Should return response unchanged."""
        plugin = CodexCooldownPlugin(codex_manager)
        response = "This is the LLM response."

        result = plugin.process(pipeline_context, response)

        assert result == response

    def test_sticky_expires(self, codex_manager, pipeline_context):
        """Sticky entries should expire after their duration."""
        # Activate dragon entry (has sticky=2)
        activator = CodexActivatorPlugin(codex_manager)
        cooldown = CodexCooldownPlugin(codex_manager)

        # Turn 0: Activate dragon
        pipeline_context.user_input = "Tell me about the dragon."
        activator.process(pipeline_context)

        # Dragon should be active due to match
        results = codex_manager.activate("Tell me about the dragon.")
        dragon_result = next((r for r in results if r.entry_name == "dragon_lore"), None)
        assert dragon_result is not None
        assert dragon_result.activated

        # Advance turn
        cooldown.process(pipeline_context, "Response 1")

        # Turn 1: Dragon should still be active (sticky)
        results = codex_manager.activate("What else?")  # No keyword match
        dragon_result = next((r for r in results if r.entry_name == "dragon_lore"), None)
        assert dragon_result is not None
        assert dragon_result.activated
        assert dragon_result.reason == "sticky_active"

        # Advance to turn 2
        cooldown.process(pipeline_context, "Response 2")

        # Turn 2: Dragon sticky should have expired
        results = codex_manager.activate("What else?")
        dragon_result = next((r for r in results if r.entry_name == "dragon_lore"), None)
        # Should not be in results (not activated)
        assert dragon_result is None or not dragon_result.activated


class TestCooldownEffect:
    """Tests for cooldown timed effect."""

    def test_cooldown_prevents_reactivation(self, codex_manager, pipeline_context):
        """Cooldown should prevent re-activation for N turns."""
        activator = CodexActivatorPlugin(codex_manager)
        cooldown = CodexCooldownPlugin(codex_manager)

        # Turn 0: Activate magic entry (has cooldown=3)
        pipeline_context.user_input = "Tell me about magic."
        result = activator.process(pipeline_context)

        # Should activate
        magic_result = next(
            (r for r in result.metadata["codex_results"] if r.entry_name == "magic_info"),
            None,
        )
        assert magic_result is not None
        assert magic_result.activated

        # Advance turn
        cooldown.process(pipeline_context, "Response")

        # Turn 1: Try to activate again - should be on cooldown
        result = activator.process(pipeline_context)
        magic_result = next(
            (r for r in result.metadata["codex_results"] if r.entry_name == "magic_info"),
            None,
        )
        # Should not be in activated results (blocked by cooldown)
        assert magic_result is None or not magic_result.activated

        # Advance 3 more turns to clear cooldown
        for _ in range(3):
            cooldown.process(pipeline_context, "Response")

        # Turn 4: Should be able to activate again
        result = activator.process(pipeline_context)
        magic_result = next(
            (r for r in result.metadata["codex_results"] if r.entry_name == "magic_info"),
            None,
        )
        assert magic_result is not None
        assert magic_result.activated


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for plugin factory functions."""

    def test_create_codex_activator(self, codex_manager):
        """create_codex_activator should return configured plugin."""
        plugin = create_codex_activator(codex_manager)

        assert isinstance(plugin, CodexActivatorPlugin)
        assert plugin.name == "codex_activator"

    def test_create_codex_cooldown(self, codex_manager):
        """create_codex_cooldown should return configured plugin."""
        plugin = create_codex_cooldown(codex_manager)

        assert isinstance(plugin, CodexCooldownPlugin)
        assert plugin.name == "codex_cooldown"

    def test_create_codex_plugins(self, temp_characters_dir):
        """create_codex_plugins should return manager and both plugins."""
        manager, activator, cooldown = create_codex_plugins(
            characters_dir=str(temp_characters_dir),
        )

        assert isinstance(manager, CodexManager)
        assert isinstance(activator, CodexActivatorPlugin)
        assert isinstance(cooldown, CodexCooldownPlugin)

    def test_create_codex_plugins_loads_character(self, temp_characters_dir):
        """create_codex_plugins should load character if specified."""
        # Create a test character
        char_dir = temp_characters_dir / "test_char"
        char_dir.mkdir()

        card = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Test Character",
                "description": "A test character.",
                "personality": "Friendly.",
                "scenario": "",
                "first_mes": "Hello!",
                "mes_example": "",
                "extensions": {},
            },
        }
        with open(char_dir / "card.json", "w") as f:
            json.dump(card, f)

        manager, _, _ = create_codex_plugins(
            characters_dir=str(temp_characters_dir),
            character_id="test_char",
        )

        assert manager._current_character_id == "test_char"


# =============================================================================
# Integration with BudgetEnforcer Tests
# =============================================================================


class TestBudgetIntegration:
    """Tests for codex integration with budget enforcer."""

    def test_codex_tokens_available_for_budget(self, codex_manager, pipeline_context):
        """Codex tokens should be stored in metadata for budget enforcer."""
        activator = CodexActivatorPlugin(codex_manager)

        result = activator.process(pipeline_context)

        # BudgetEnforcer looks for this key
        assert "codex_tokens_estimate" in result.metadata
        assert isinstance(result.metadata["codex_tokens_estimate"], int)

    def test_budget_enforcer_reads_codex_tokens(self, codex_manager, pipeline_context):
        """BudgetEnforcer should use codex tokens in calculations."""
        from spindl.llm.plugins import BudgetEnforcer
        from spindl.llm.plugins.conversation_history import ConversationHistoryManager

        # Mock LLM provider
        mock_provider = Mock()
        mock_provider.get_properties.return_value = Mock(context_length=4096)
        mock_provider.count_tokens.side_effect = lambda text: len(text) // 4

        # Create history manager
        history_manager = Mock(spec=ConversationHistoryManager)
        history_manager.get_history.return_value = []
        history_manager._history = []

        # First, run codex activator to populate metadata
        activator = CodexActivatorPlugin(codex_manager)
        context = activator.process(pipeline_context)

        # Now create budget enforcer and calculate budget
        enforcer = BudgetEnforcer(
            llm_provider=mock_provider,
            manager=history_manager,
        )

        budget = enforcer._calculate_budget(context)

        # Budget should include codex tokens
        assert "codex_tokens" in budget
        assert budget["codex_tokens"] == context.metadata["codex_tokens_estimate"]

        # Available for history should be reduced by codex
        # n_ctx - system - codex - user - reserve = available
        expected_available = (
            4096
            - budget["system_tokens"]
            - budget["codex_tokens"]
            - budget["user_tokens"]
            - budget["reserve"]
        )
        assert budget["available_for_history"] == max(0, expected_available)
