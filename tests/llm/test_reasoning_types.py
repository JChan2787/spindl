"""
Tests for NANO-042: Reasoning/thinking type support.

Tests cover:
- LLMResponse reasoning fields (backward compatibility + population)
- StreamChunk reasoning passthrough from LLMResponse (base class default)
"""

import pytest

from spindl.llm.base import LLMResponse, StreamChunk


# =============================================================================
# LLMResponse Reasoning Fields
# =============================================================================


class TestLLMResponseReasoning:
    """Tests for LLMResponse reasoning/thinking fields (NANO-042 Phase 1)."""

    def test_reasoning_defaults_to_none(self):
        """Reasoning field should default to None when not provided."""
        response = LLMResponse(
            content="Hello!",
            input_tokens=10,
            output_tokens=5,
        )
        assert response.reasoning is None
        assert response.reasoning_tokens is None

    def test_reasoning_populated(self):
        """Reasoning field should hold thinking content when provided."""
        response = LLMResponse(
            content="The answer is 42.",
            input_tokens=10,
            output_tokens=20,
            reasoning="Let me think about this step by step...",
            reasoning_tokens=15,
        )
        assert response.reasoning == "Let me think about this step by step..."
        assert response.reasoning_tokens == 15

    def test_reasoning_with_tool_calls(self):
        """Reasoning should coexist with tool calls."""
        from spindl.llm.base import ToolCall

        response = LLMResponse(
            content="",
            input_tokens=10,
            output_tokens=20,
            tool_calls=[ToolCall(id="1", name="screen_vision", arguments={})],
            finish_reason="tool_calls",
            reasoning="I should use the vision tool to see the screen.",
            reasoning_tokens=12,
        )
        assert response.reasoning is not None
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls"

    def test_backward_compatible_construction(self):
        """Existing code that constructs LLMResponse without reasoning should still work."""
        # Positional args only (content, input_tokens, output_tokens)
        response = LLMResponse("Hello", 10, 5)
        assert response.content == "Hello"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.reasoning is None
        assert response.reasoning_tokens is None

    def test_reasoning_empty_string(self):
        """Empty string reasoning should be stored as-is (not coerced to None)."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            reasoning="",
        )
        assert response.reasoning == ""

    def test_reasoning_tokens_without_reasoning_text(self):
        """reasoning_tokens can be set even if reasoning text is None."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            reasoning_tokens=0,
        )
        assert response.reasoning is None
        assert response.reasoning_tokens == 0


# =============================================================================
# StreamChunk Reasoning Passthrough
# =============================================================================


class TestStreamChunkReasoningPassthrough:
    """Tests for base class generate_stream() passing reasoning to StreamChunk."""

    def test_stream_chunk_reasoning_already_exists(self):
        """StreamChunk should have had reasoning field before NANO-042."""
        chunk = StreamChunk(content="test", reasoning="thinking...")
        assert chunk.reasoning == "thinking..."

    def test_stream_chunk_reasoning_tokens_already_exists(self):
        """StreamChunk should have had reasoning_tokens field before NANO-042."""
        chunk = StreamChunk(content="test", reasoning_tokens=42)
        assert chunk.reasoning_tokens == 42
