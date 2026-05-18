"""Tests for CrossActivatorPlugin — NANO-127 multi-hop retrieval."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import tempfile
import json
from pathlib import Path

from spindl.llm.plugins.base import PipelineContext
from spindl.llm.plugins.cross_activator import (
    CrossActivatorPlugin,
    create_cross_activator,
)
from spindl.codex import CodexManager, ActivationResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_characters_dir():
    """Create a temporary characters directory with test codex entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chars_dir = Path(tmpdir) / "characters"
        chars_dir.mkdir()

        global_dir = chars_dir / "_global"
        global_dir.mkdir()

        global_codex = {
            "name": "Global Codex",
            "entries": [
                {
                    "keys": ["Walkers", "walker"],
                    "content": "Walkers are hostile entities that roam the collapsed sectors.",
                    "enabled": True,
                    "insertion_order": 1,
                    "id": 1,
                    "name": "walkers_lore",
                    "position": "after_char",
                },
                {
                    "keys": ["Diana"],
                    "content": "Diana is the player's AI companion. She provides tactical support.",
                    "enabled": True,
                    "insertion_order": 2,
                    "id": 2,
                    "name": "diana_companion",
                    "position": "after_char",
                },
                {
                    "keys": ["Cradle"],
                    "content": "The Cradle is the central hub area where survivors gather.",
                    "enabled": True,
                    "insertion_order": 3,
                    "id": 3,
                    "name": "cradle_location",
                    "position": "after_char",
                },
                {
                    "keys": ["coffee"],
                    "content": "User enjoys dark roast coffee.",
                    "enabled": True,
                    "insertion_order": 4,
                    "id": 4,
                    "name": "coffee_preference",
                    "position": "after_char",
                },
                {
                    "keys": ["SectorGuard"],
                    "content": "SectorGuard is a faction that controls resource distribution.",
                    "enabled": True,
                    "insertion_order": 5,
                    "id": 5,
                    "name": "sectorguard_faction",
                    "position": "after_char",
                },
                {
                    "keys": ["magic"],
                    "content": "Magic requires training.",
                    "enabled": True,
                    "insertion_order": 6,
                    "id": 6,
                    "name": "magic_info",
                    "cooldown": 3,
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
def base_context():
    """Context with no RAG or codex metadata — simulates empty pipeline."""
    return PipelineContext(
        user_input="tell me about our first stream",
        persona={"name": "Spindle", "system_prompt": "You are a streaming co-host."},
        messages=[],
        metadata={},
    )


@pytest.fixture
def context_with_rag():
    """Context simulating RAGInjector having run (rag_content populated)."""
    return PipelineContext(
        user_input="tell me about our first stream",
        persona={"name": "Spindle", "system_prompt": "You are a streaming co-host."},
        messages=[],
        metadata={
            "rag_content": (
                "We fought Walkers near the collapsed overpass in SectorGuard territory. "
                "Diana provided covering fire while we retreated to the Cradle."
            ),
            "codex_results": [],
            "codex_content": "",
            "codex_tokens_estimate": 0,
        },
    )


@pytest.fixture
def context_with_rag_and_first_pass(codex_manager):
    """Context where first-pass codex already activated some entries."""
    # Simulate first-pass having activated "diana_companion" via keyword in user_input
    first_pass_results = codex_manager.activate("Diana is great")
    activated = [r for r in first_pass_results if r.activated]
    content = codex_manager.get_activated_content(activated) if activated else ""

    return PipelineContext(
        user_input="Diana is great, tell me about our first stream",
        persona={"name": "Spindle", "system_prompt": "You are a streaming co-host."},
        messages=[],
        metadata={
            "rag_content": (
                "We fought Walkers near the collapsed overpass in SectorGuard territory. "
                "Diana provided covering fire while we retreated to the Cradle."
            ),
            "codex_results": first_pass_results,
            "codex_content": content,
            "codex_tokens_estimate": len(content) // 4,
        },
    )


# =============================================================================
# Basic Behavior
# =============================================================================


class TestCrossActivatorBasics:
    """Core behavior tests."""

    def test_name(self, codex_manager):
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        assert plugin.name == "cross_activator"

    def test_disabled_returns_context_unchanged(self, codex_manager, context_with_rag):
        plugin = CrossActivatorPlugin(codex_manager, enabled=False)
        result = plugin.process(context_with_rag)
        assert result.metadata["codex_results"] == []
        assert result.metadata["codex_content"] == ""
        assert result.metadata["codex_tokens_estimate"] == 0

    def test_no_rag_content_returns_unchanged(self, codex_manager, base_context):
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(base_context)
        assert "codex_results" not in result.metadata

    def test_empty_rag_content_returns_unchanged(self, codex_manager):
        context = PipelineContext(
            user_input="hello",
            persona={"name": "Test"},
            messages=[],
            metadata={"rag_content": "", "codex_results": [], "codex_content": ""},
        )
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context)
        assert result.metadata["codex_results"] == []

    def test_enabled_property_settable(self, codex_manager):
        plugin = CrossActivatorPlugin(codex_manager, enabled=False)
        assert not plugin.enabled
        plugin.enabled = True
        assert plugin.enabled


# =============================================================================
# Cross-Activation Logic
# =============================================================================


class TestCrossActivation:
    """Tests for the actual cross-activation behavior."""

    def test_activates_entries_from_rag_content(self, codex_manager, context_with_rag):
        """RAG content mentioning Walkers/Diana/Cradle/SectorGuard should activate those entries."""
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context_with_rag)

        activated_names = [r.entry_name for r in result.metadata["codex_results"] if r.activated]
        assert "walkers_lore" in activated_names
        assert "diana_companion" in activated_names
        assert "cradle_location" in activated_names
        assert "sectorguard_faction" in activated_names

    def test_does_not_activate_unrelated_entries(self, codex_manager, context_with_rag):
        """Entries whose keywords don't appear in RAG content stay inactive."""
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context_with_rag)

        activated_names = [r.entry_name for r in result.metadata["codex_results"] if r.activated]
        assert "coffee_preference" not in activated_names

    def test_updates_codex_content(self, codex_manager, context_with_rag):
        """codex_content should contain the activated entries' text."""
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context_with_rag)

        content = result.metadata["codex_content"]
        assert "Walkers are hostile entities" in content
        assert "Diana is the player" in content
        assert "Cradle is the central hub" in content

    def test_updates_token_estimate(self, codex_manager, context_with_rag):
        """Token estimate should be non-zero after cross-activation."""
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context_with_rag)

        assert result.metadata["codex_tokens_estimate"] > 0

    def test_token_estimate_uses_real_tokenizer(self, codex_manager, context_with_rag):
        """Token estimate should come from count_tokens, not len//4."""
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context_with_rag)

        content = result.metadata["codex_content"]
        estimate = result.metadata["codex_tokens_estimate"]
        heuristic = len(content) // 4

        from spindl.utils.tokens import count_tokens
        real_count = count_tokens(content)
        assert estimate == real_count


# =============================================================================
# Deduplication
# =============================================================================


class TestDeduplication:
    """Tests for dedup against first-pass results."""

    def test_deduplicates_against_first_pass(self, codex_manager, context_with_rag_and_first_pass):
        """Entries already activated in first pass should not appear twice."""
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context_with_rag_and_first_pass)

        activated = result.metadata["codex_results"]
        activated_ids = [r.entry_id for r in activated if r.activated]
        # No duplicate IDs
        assert len(activated_ids) == len(set(activated_ids))

    def test_all_first_pass_duplicates_returns_unchanged(self, codex_manager):
        """If RAG content only triggers entries already in first pass, no update."""
        # First pass already has walkers_lore activated
        first_pass = codex_manager.activate("Walkers are nearby")
        activated_first = [r for r in first_pass if r.activated]
        content = codex_manager.get_activated_content(activated_first)

        context = PipelineContext(
            user_input="Walkers are nearby",
            persona={"name": "Test"},
            messages=[],
            metadata={
                "rag_content": "The Walkers attacked at dawn.",  # Only triggers walkers_lore
                "codex_results": first_pass,
                "codex_content": content,
                "codex_tokens_estimate": 10,
            },
        )

        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context)

        # Should not have changed the token estimate
        assert result.metadata["codex_tokens_estimate"] == 10

    def test_merges_new_with_existing(self, codex_manager, context_with_rag_and_first_pass):
        """New cross-activated entries should merge with first-pass results."""
        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context_with_rag_and_first_pass)

        activated = [r for r in result.metadata["codex_results"] if r.activated]
        names = [r.entry_name for r in activated]

        # First pass had diana_companion, cross should add walkers, cradle, sectorguard
        assert "diana_companion" in names
        assert "walkers_lore" in names
        assert "cradle_location" in names
        assert "sectorguard_faction" in names


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases from the ticket's documented list."""

    def test_empty_codex_returns_unchanged(self):
        """If codex has no entries, cross-activation is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chars_dir = Path(tmpdir) / "characters"
            chars_dir.mkdir()
            global_dir = chars_dir / "_global"
            global_dir.mkdir()
            with open(global_dir / "codex.json", "w") as f:
                json.dump({"name": "Empty", "entries": []}, f)

            manager = CodexManager(characters_dir=chars_dir)
            manager.load_global_codex()

            context = PipelineContext(
                user_input="hello",
                persona={"name": "Test"},
                messages=[],
                metadata={
                    "rag_content": "Walkers attacked at dawn near the Cradle.",
                    "codex_results": [],
                    "codex_content": "",
                    "codex_tokens_estimate": 0,
                },
            )

            plugin = CrossActivatorPlugin(manager, enabled=True)
            result = plugin.process(context)

            assert result.metadata["codex_tokens_estimate"] == 0

    def test_rag_wrapper_text_does_not_trigger(self, codex_manager):
        """The RAG prefix/suffix boilerplate shouldn't activate entries."""
        context = PipelineContext(
            user_input="hello",
            persona={"name": "Test"},
            messages=[],
            metadata={
                "rag_content": (
                    "The following are relevant memories about the user and past conversations. "
                    "Use them to inform your response:\n\nEnd of memories."
                ),
                "codex_results": [],
                "codex_content": "",
                "codex_tokens_estimate": 0,
            },
        )

        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(context)

        assert result.metadata["codex_tokens_estimate"] == 0

    def test_cooldown_respected_on_cross_activation(self, codex_manager):
        """An entry on cooldown should not activate via cross-activation."""
        from spindl.llm.plugins.codex_activator import CodexActivatorPlugin
        from spindl.llm.plugins.codex_cooldown import CodexCooldownPlugin

        # Activate magic_info (cooldown=3) via first pass
        context = PipelineContext(
            user_input="Tell me about magic",
            persona={"name": "Test"},
            messages=[],
            metadata={},
        )
        activator = CodexActivatorPlugin(codex_manager)
        activator.process(context)

        # Advance turn to start cooldown
        cooldown = CodexCooldownPlugin(codex_manager)
        cooldown.process(context, "response")

        # Now try cross-activation with RAG mentioning magic
        cross_context = PipelineContext(
            user_input="something else",
            persona={"name": "Test"},
            messages=[],
            metadata={
                "rag_content": "The ancient magic was powerful beyond measure.",
                "codex_results": [],
                "codex_content": "",
                "codex_tokens_estimate": 0,
            },
        )

        plugin = CrossActivatorPlugin(codex_manager, enabled=True)
        result = plugin.process(cross_context)

        activated_names = [r.entry_name for r in result.metadata.get("codex_results", []) if r.activated]
        assert "magic_info" not in activated_names


# =============================================================================
# Factory Function
# =============================================================================


class TestFactory:
    """Tests for create_cross_activator factory."""

    def test_creates_plugin(self, codex_manager):
        plugin = create_cross_activator(codex_manager, enabled=True)
        assert isinstance(plugin, CrossActivatorPlugin)
        assert plugin.name == "cross_activator"
        assert plugin.enabled

    def test_default_disabled(self, codex_manager):
        plugin = create_cross_activator(codex_manager)
        assert not plugin.enabled
