"""
Tests for NANO-042 Phase 2: Reasoning through pipeline, events, and history.

Tests cover:
- TokenUsage reasoning_tokens from LLMResponse
- PipelineResult reasoning field
- ResponseReadyEvent reasoning field
- HistoryRecorder stores reasoning in JSONL
- HistoryInjector does NOT replay reasoning to LLM
"""

import json
import pytest
from pathlib import Path

from spindl.llm.base import LLMResponse
from spindl.llm.pipeline import TokenUsage, PipelineResult
from spindl.core.events import ResponseReadyEvent


# =============================================================================
# TokenUsage Reasoning Tokens
# =============================================================================


class TestTokenUsageReasoning:
    """Tests for TokenUsage.reasoning_tokens field (NANO-042 Phase 2)."""

    def test_reasoning_tokens_default_none(self):
        """reasoning_tokens should default to None."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.reasoning_tokens is None

    def test_reasoning_tokens_populated(self):
        """reasoning_tokens should hold the count when provided."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, reasoning_tokens=30)
        assert usage.reasoning_tokens == 30

    def test_from_llm_response_with_reasoning_tokens(self):
        """from_llm_response should propagate reasoning_tokens."""
        response = LLMResponse(
            content="Hello",
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=30,
        )
        usage = TokenUsage.from_llm_response(response)
        assert usage.reasoning_tokens == 30

    def test_from_llm_response_without_reasoning_tokens(self):
        """from_llm_response should leave reasoning_tokens as None."""
        response = LLMResponse(
            content="Hello",
            input_tokens=100,
            output_tokens=50,
        )
        usage = TokenUsage.from_llm_response(response)
        assert usage.reasoning_tokens is None


# =============================================================================
# PipelineResult Reasoning
# =============================================================================


class TestPipelineResultReasoning:
    """Tests for PipelineResult.reasoning field (NANO-042 Phase 2)."""

    def test_reasoning_default_none(self):
        """reasoning should default to None."""
        result = PipelineResult(
            content="Hello",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            messages=[],
        )
        assert result.reasoning is None

    def test_reasoning_populated(self):
        """reasoning should hold thinking content when provided."""
        result = PipelineResult(
            content="The answer is 42.",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            messages=[],
            reasoning="Let me think step by step...",
        )
        assert result.reasoning == "Let me think step by step..."


# =============================================================================
# ResponseReadyEvent Reasoning
# =============================================================================


class TestResponseReadyEventReasoning:
    """Tests for ResponseReadyEvent.reasoning field (NANO-042 Phase 2)."""

    def test_reasoning_default_none(self):
        """reasoning should default to None."""
        event = ResponseReadyEvent(text="Hello")
        assert event.reasoning is None

    def test_reasoning_populated(self):
        """reasoning should hold thinking content when provided."""
        event = ResponseReadyEvent(
            text="The answer is 42.",
            reasoning="Step 1: Consider the question...",
        )
        assert event.reasoning == "Step 1: Consider the question..."

    def test_backward_compatible(self):
        """Existing code that constructs without reasoning should still work."""
        event = ResponseReadyEvent(
            text="Hello",
            user_input="Hi",
            activated_codex_entries=[{"name": "test"}],
        )
        assert event.text == "Hello"
        assert event.reasoning is None


# =============================================================================
# ConversationHistory Reasoning Storage
# =============================================================================


class TestHistoryReasoningStorage:
    """Tests for reasoning storage in JSONL history (NANO-042 Phase 2)."""

    def test_store_turn_with_reasoning(self, tmp_path):
        """store_turn should include reasoning in assistant turn when provided."""
        from spindl.llm.plugins.conversation_history import ConversationHistoryManager

        manager = ConversationHistoryManager(
            conversations_dir=str(tmp_path),
        )
        manager.ensure_session("test_char")
        manager.stash_user_input("What is 2+2?")
        manager.store_turn("The answer is 4.", reasoning="Let me add 2 and 2...")

        # Read the JSONL file
        session_file = manager.session_file
        lines = session_file.read_text().strip().split("\n")
        assert len(lines) == 2

        assistant_turn = json.loads(lines[1])
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"] == "The answer is 4."
        assert assistant_turn["reasoning"] == "Let me add 2 and 2..."

    def test_store_turn_without_reasoning(self, tmp_path):
        """store_turn should NOT include reasoning key when not provided."""
        from spindl.llm.plugins.conversation_history import ConversationHistoryManager

        manager = ConversationHistoryManager(
            conversations_dir=str(tmp_path),
        )
        manager.ensure_session("test_char")
        manager.stash_user_input("Hello")
        manager.store_turn("Hi there!")

        session_file = manager.session_file
        lines = session_file.read_text().strip().split("\n")
        assistant_turn = json.loads(lines[1])
        assert "reasoning" not in assistant_turn

    def test_history_injector_excludes_reasoning(self, tmp_path):
        """HistoryInjector should NOT include reasoning when building messages."""
        from spindl.llm.plugins.conversation_history import (
            ConversationHistoryManager,
            HistoryInjector,
        )
        from spindl.llm.plugins.base import PipelineContext

        manager = ConversationHistoryManager(
            conversations_dir=str(tmp_path),
        )
        # Use same persona_id that the injector will derive from context.persona["id"]
        manager.ensure_session("test_char")

        # Store a turn with reasoning
        manager.stash_user_input("What is 2+2?")
        manager.store_turn("The answer is 4.", reasoning="Let me think about addition...")

        # Create injector and process — persona must have matching "id"
        injector = HistoryInjector(manager)
        context = PipelineContext(
            user_input="Next question",
            persona={"id": "test_char", "name": "TestBot"},
            messages=[
                {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
                {"role": "user", "content": "Next question"},
            ],
        )

        result = injector.process(context)
        system_content = result.messages[0]["content"]

        # Reasoning should NOT appear in the injected history
        assert "Let me think about addition" not in system_content
        # But the content should be there
        assert "The answer is 4." in system_content
