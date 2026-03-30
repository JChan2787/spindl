"""
Unit tests for _build_token_breakdown() in OrchestratorCallbacks (NANO-045b).

Tests both block-aware distribution and legacy header-parsing fallback.
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Fixtures
# =============================================================================


def _make_callbacks():
    """Create a minimal OrchestratorCallbacks for testing _build_token_breakdown."""
    from spindl.orchestrator.callbacks import OrchestratorCallbacks

    callbacks = OrchestratorCallbacks.__new__(OrchestratorCallbacks)
    return callbacks


def _make_messages(system_content: str, user_content: str = "Hello") -> list[dict]:
    """Build a standard [system, user] message list."""
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _make_block_contents() -> list[dict]:
    """Sample block_contents matching default 14-block layout."""
    return [
        {"id": "persona_name", "label": "Agent Name", "section": "Agent", "chars": 30, "deferred": False},
        {"id": "persona_appearance", "label": "Appearance", "section": None, "chars": 60, "deferred": False},
        {"id": "persona_personality", "label": "Personality", "section": None, "chars": 40, "deferred": False},
        {"id": "scenario", "label": "Scenario", "section": None, "chars": 0, "deferred": False},
        {"id": "example_dialogue", "label": "Example Dialogue", "section": None, "chars": 0, "deferred": False},
        {"id": "modality_context", "label": "Modality", "section": "Context", "chars": 50, "deferred": False},
        {"id": "voice_state", "label": "Voice State", "section": None, "chars": 20, "deferred": False},
        {"id": "codex_context", "label": "Codex", "section": None, "chars": 100, "deferred": False},
        {"id": "rag_context", "label": "Memories", "section": None, "chars": 0, "deferred": False},
        {"id": "persona_rules", "label": "Rules", "section": "Rules", "chars": 80, "deferred": False},
        {"id": "modality_rules", "label": "Modality Rules", "section": None, "chars": 30, "deferred": False},
        {"id": "conversation_summary", "label": "Summary", "section": "Conversation", "chars": 25, "deferred": False},
        {"id": "recent_history", "label": "Chat History", "section": None, "chars": 200, "deferred": False},
        {"id": "closing_instruction", "label": "Closing", "section": "Input", "chars": 15, "deferred": False},
    ]


# =============================================================================
# Block-Aware Distribution Tests
# =============================================================================


class TestBlockAwareDistribution:
    """Tests for _build_token_breakdown with block_contents."""

    def test_blocks_key_present(self):
        """Breakdown includes 'blocks' key when block_contents provided."""
        cb = _make_callbacks()
        msgs = _make_messages("x" * 100)
        result = cb._build_token_breakdown(msgs, 200, 50, block_contents=_make_block_contents())
        assert "blocks" in result
        assert isinstance(result["blocks"], list)

    def test_block_count_matches_input(self):
        """Output blocks list has same length as input block_contents."""
        cb = _make_callbacks()
        blocks = _make_block_contents()
        msgs = _make_messages("x" * 100)
        result = cb._build_token_breakdown(msgs, 200, 50, block_contents=blocks)
        assert len(result["blocks"]) == len(blocks)

    def test_block_ids_preserved(self):
        """Block IDs are preserved in output."""
        cb = _make_callbacks()
        blocks = _make_block_contents()
        msgs = _make_messages("x" * 100)
        result = cb._build_token_breakdown(msgs, 200, 50, block_contents=blocks)
        output_ids = [b["id"] for b in result["blocks"]]
        input_ids = [b["id"] for b in blocks]
        assert output_ids == input_ids

    def test_block_tokens_proportional(self):
        """Block tokens are distributed proportionally by char count."""
        cb = _make_callbacks()
        # Two blocks: 100 chars and 100 chars = 50/50 split
        blocks = [
            {"id": "a", "label": "A", "section": "Agent", "chars": 100, "deferred": False},
            {"id": "b", "label": "B", "section": None, "chars": 100, "deferred": False},
        ]
        msgs = _make_messages("x" * 200)
        result = cb._build_token_breakdown(msgs, 200, 0, block_contents=blocks)
        # System tokens estimated from char ratio: 200/205 * 200 ≈ 195
        system_tokens = result["system"]
        assert system_tokens > 0
        a_tokens = result["blocks"][0]["tokens"]
        b_tokens = result["blocks"][1]["tokens"]
        # Both should be roughly equal (within rounding)
        assert a_tokens == b_tokens

    def test_zero_char_block_gets_zero_tokens(self):
        """Blocks with 0 chars get 0 tokens."""
        cb = _make_callbacks()
        blocks = _make_block_contents()
        msgs = _make_messages("x" * 500)
        result = cb._build_token_breakdown(msgs, 1000, 100, block_contents=blocks)
        by_id = {b["id"]: b for b in result["blocks"]}
        # rag_context has chars=0
        assert by_id["rag_context"]["tokens"] == 0

    def test_legacy_sections_rollup(self):
        """Legacy sections are computed from block accumulation."""
        cb = _make_callbacks()
        blocks = _make_block_contents()
        msgs = _make_messages("x" * 500)
        result = cb._build_token_breakdown(msgs, 1000, 100, block_contents=blocks)

        by_id = {b["id"]: b for b in result["blocks"]}

        # Agent section = persona_name + persona_appearance + persona_personality
        expected_agent = (
            by_id["persona_name"]["tokens"]
            + by_id["persona_appearance"]["tokens"]
            + by_id["persona_personality"]["tokens"]
        )
        assert result["sections"]["agent"] == expected_agent

        # Context section = modality_context + voice_state + codex_context + rag_context
        expected_context = (
            by_id["modality_context"]["tokens"]
            + by_id["voice_state"]["tokens"]
            + by_id["codex_context"]["tokens"]
            + by_id["rag_context"]["tokens"]
        )
        assert result["sections"]["context"] == expected_context

        # Rules section = persona_rules + modality_rules
        expected_rules = (
            by_id["persona_rules"]["tokens"]
            + by_id["modality_rules"]["tokens"]
        )
        assert result["sections"]["rules"] == expected_rules

        # Conversation section = conversation_summary + recent_history
        expected_convo = (
            by_id["conversation_summary"]["tokens"]
            + by_id["recent_history"]["tokens"]
        )
        assert result["sections"]["conversation"] == expected_convo

    def test_total_prompt_completion_correct(self):
        """Top-level total/prompt/completion are always correct."""
        cb = _make_callbacks()
        msgs = _make_messages("x" * 100)
        result = cb._build_token_breakdown(msgs, 300, 75, block_contents=_make_block_contents())
        assert result["total"] == 375
        assert result["prompt"] == 300
        assert result["completion"] == 75

    def test_system_user_split(self):
        """System/user split uses character ratio of system vs user content."""
        cb = _make_callbacks()
        # System = 400 chars, User = 100 chars → 80/20 split
        msgs = _make_messages("x" * 400, "y" * 100)
        result = cb._build_token_breakdown(msgs, 500, 0, block_contents=_make_block_contents())
        assert result["system"] == 400  # int(500 * 400/500) = 400
        assert result["user"] == 100


# =============================================================================
# Legacy Fallback Tests
# =============================================================================


class TestLegacyFallback:
    """Tests for _build_token_breakdown without block_contents (legacy mode)."""

    def test_no_blocks_key(self):
        """Legacy mode does not include 'blocks' key."""
        cb = _make_callbacks()
        system = "### Agent\nYou are Bot.\n\n### Context\n\n### Rules\n\n### Conversation\n"
        msgs = _make_messages(system)
        result = cb._build_token_breakdown(msgs, 200, 50)
        assert "blocks" not in result

    def test_sections_populated(self):
        """Legacy mode populates sections from header parsing."""
        cb = _make_callbacks()
        system = (
            "### Agent\n"
            "You are Bot. A robot.\n\n"
            "### Context\n"
            "Text mode.\n\n"
            "### Rules\n"
            "Be concise.\n\n"
            "### Conversation\n"
            "No history yet.\n"
        )
        msgs = _make_messages(system)
        result = cb._build_token_breakdown(msgs, 400, 0)
        # All 4 sections should have some tokens
        assert result["sections"]["agent"] > 0
        assert result["sections"]["context"] > 0
        assert result["sections"]["rules"] > 0
        assert result["sections"]["conversation"] > 0

    def test_none_block_contents_uses_legacy(self):
        """Explicit None for block_contents triggers legacy path."""
        cb = _make_callbacks()
        system = "### Agent\nBot\n\n### Context\n\n### Rules\n\n### Conversation\n"
        msgs = _make_messages(system)
        result = cb._build_token_breakdown(msgs, 200, 50, block_contents=None)
        assert "blocks" not in result
        assert result["sections"]["agent"] > 0

    def test_empty_list_block_contents_uses_legacy(self):
        """Empty block_contents list triggers legacy path."""
        cb = _make_callbacks()
        system = "### Agent\nBot\n\n### Conversation\n"
        msgs = _make_messages(system)
        result = cb._build_token_breakdown(msgs, 200, 50, block_contents=[])
        assert "blocks" not in result


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases for _build_token_breakdown."""

    def test_zero_prompt_tokens(self):
        """Zero prompt tokens yields zero for all sections and blocks."""
        cb = _make_callbacks()
        msgs = _make_messages("x" * 100)
        result = cb._build_token_breakdown(msgs, 0, 50, block_contents=_make_block_contents())
        assert result["system"] == 0
        assert result["user"] == 0
        for block in result["blocks"]:
            assert block["tokens"] == 0

    def test_empty_messages(self):
        """Empty message list doesn't crash."""
        cb = _make_callbacks()
        result = cb._build_token_breakdown([], 100, 50, block_contents=_make_block_contents())
        assert result["system"] == 0
        assert result["user"] == 0

    def test_all_zero_char_blocks(self):
        """All blocks with zero chars get zero tokens."""
        cb = _make_callbacks()
        blocks = [
            {"id": "a", "label": "A", "section": "Agent", "chars": 0, "deferred": False},
            {"id": "b", "label": "B", "section": None, "chars": 0, "deferred": False},
        ]
        msgs = _make_messages("x" * 100)
        result = cb._build_token_breakdown(msgs, 200, 0, block_contents=blocks)
        for block in result["blocks"]:
            assert block["tokens"] == 0
