"""
Tests for NANO-042 Phase 6: ReasoningExtractor PostProcessor.

Tests cover:
- Strips inline <think>...</think> blocks from response
- Stashes extracted reasoning in context.metadata
- Skips extraction when provider already separated reasoning
- Handles multiple <think> blocks
- Handles empty <think> blocks
- Handles no <think> blocks (passthrough)
- Handles multiline <think> content
"""

import pytest

from spindl.llm.plugins.reasoning_extractor import ReasoningExtractor
from spindl.llm.plugins.base import PipelineContext


def _make_context(reasoning: str = None) -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    ctx = PipelineContext(
        user_input="test",
        persona={"id": "test", "name": "Test"},
        messages=[],
    )
    if reasoning:
        ctx.metadata["reasoning"] = reasoning
    return ctx


class TestReasoningExtractor:
    """Tests for ReasoningExtractor PostProcessor."""

    def setup_method(self):
        self.extractor = ReasoningExtractor()

    def test_name(self):
        """Plugin name should be 'reasoning_extractor'."""
        assert self.extractor.name == "reasoning_extractor"

    def test_strips_single_think_block(self):
        """Should strip a single <think>...</think> block and extract reasoning."""
        ctx = _make_context()
        response = "<think>Let me consider this carefully.</think>The answer is 42."
        result = self.extractor.process(ctx, response)

        assert result == "The answer is 42."
        assert ctx.metadata["reasoning"] == "Let me consider this carefully."

    def test_strips_multiline_think_block(self):
        """Should handle multiline content inside <think> blocks."""
        ctx = _make_context()
        response = (
            "<think>\nStep 1: Read the question\n"
            "Step 2: Think about it\n</think>\n"
            "Here's my answer."
        )
        result = self.extractor.process(ctx, response)

        assert result == "Here's my answer."
        assert "Step 1: Read the question" in ctx.metadata["reasoning"]
        assert "Step 2: Think about it" in ctx.metadata["reasoning"]

    def test_strips_multiple_think_blocks(self):
        """Should handle multiple <think> blocks, joining them."""
        ctx = _make_context()
        response = (
            "<think>First thought.</think>Part one. "
            "<think>Second thought.</think>Part two."
        )
        result = self.extractor.process(ctx, response)

        assert result == "Part one. Part two."
        assert "First thought." in ctx.metadata["reasoning"]
        assert "Second thought." in ctx.metadata["reasoning"]

    def test_empty_think_block(self):
        """Should handle empty <think></think> blocks gracefully."""
        ctx = _make_context()
        response = "<think></think>Just the answer."
        result = self.extractor.process(ctx, response)

        assert result == "Just the answer."
        # Empty block should not set reasoning
        assert "reasoning" not in ctx.metadata

    def test_no_think_blocks_passthrough(self):
        """Should return response unchanged when no <think> blocks present."""
        ctx = _make_context()
        response = "A perfectly normal response."
        result = self.extractor.process(ctx, response)

        assert result == "A perfectly normal response."
        assert "reasoning" not in ctx.metadata

    def test_skips_when_reasoning_already_set(self):
        """Should not extract when provider already separated reasoning."""
        ctx = _make_context(reasoning="Provider-parsed reasoning.")
        response = "<think>This should be ignored.</think>Response text."
        result = self.extractor.process(ctx, response)

        # Should NOT strip - provider reasoning takes precedence
        assert result == "<think>This should be ignored.</think>Response text."
        assert ctx.metadata["reasoning"] == "Provider-parsed reasoning."

    def test_whitespace_only_think_block(self):
        """Should handle <think> blocks with only whitespace."""
        ctx = _make_context()
        response = "<think>   \n  </think>Clean response."
        result = self.extractor.process(ctx, response)

        assert result == "Clean response."
        # Whitespace-only should not set reasoning
        assert "reasoning" not in ctx.metadata

    def test_think_block_at_end(self):
        """Should handle <think> block at the end of response."""
        ctx = _make_context()
        response = "The answer is 42.<think>I figured it out by multiplying 6 by 7.</think>"
        result = self.extractor.process(ctx, response)

        assert result == "The answer is 42."
        assert ctx.metadata["reasoning"] == "I figured it out by multiplying 6 by 7."
