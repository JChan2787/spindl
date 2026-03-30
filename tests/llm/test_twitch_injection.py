"""Tests for Twitch chat prompt block injection in LLMPipeline (NANO-056b).

Validates that Twitch chat content is injected into the system prompt
via _inject_twitch_content(), deferred block patching updates char counts,
and stimulus_metadata threads through pipeline.run().
"""

from unittest.mock import MagicMock

import pytest

from spindl.llm.pipeline import LLMPipeline
from spindl.llm.plugins.base import PipelineContext
from spindl.llm.prompt_builder import PromptBuilder


@pytest.fixture
def mock_pipeline() -> LLMPipeline:
    """Minimal pipeline with mock provider and builder."""
    provider = MagicMock()
    builder = MagicMock(spec=PromptBuilder)
    return LLMPipeline(provider=provider, prompt_builder=builder)


@pytest.fixture
def context_with_twitch() -> PipelineContext:
    """PipelineContext with system prompt containing [TWITCH_CONTEXT] placeholder."""
    return PipelineContext(
        user_input="Respond to the Twitch chat messages.",
        persona={},
        messages=[
            {"role": "system", "content": "System prompt.\n[TWITCH_CONTEXT]\nMore content."},
            {"role": "user", "content": "Respond to the Twitch chat messages."},
        ],
    )


# ---------------------------------------------------------------------------
# Tests: _inject_twitch_content
# ---------------------------------------------------------------------------

class TestTwitchInjection:
    """Tests for Twitch content injection into system prompt."""

    def test_replaces_placeholder(
        self, mock_pipeline: LLMPipeline, context_with_twitch: PipelineContext
    ) -> None:
        """Twitch content replaces [TWITCH_CONTEXT] placeholder."""
        context_with_twitch.metadata["twitch_content"] = "viewer1: hello\nviewer2: world"

        mock_pipeline._inject_twitch_content(context_with_twitch)

        system = context_with_twitch.messages[0]["content"]
        assert "viewer1: hello" in system
        assert "viewer2: world" in system
        assert "[TWITCH_CONTEXT]" not in system

    def test_empty_collapses_placeholder(
        self, mock_pipeline: LLMPipeline, context_with_twitch: PipelineContext
    ) -> None:
        """Empty twitch_content collapses placeholder to empty string."""
        context_with_twitch.metadata["twitch_content"] = ""

        mock_pipeline._inject_twitch_content(context_with_twitch)

        system = context_with_twitch.messages[0]["content"]
        assert "[TWITCH_CONTEXT]" not in system

    def test_missing_metadata_collapses(
        self, mock_pipeline: LLMPipeline, context_with_twitch: PipelineContext
    ) -> None:
        """Missing twitch_content metadata is treated as empty string."""
        mock_pipeline._inject_twitch_content(context_with_twitch)

        system = context_with_twitch.messages[0]["content"]
        assert "[TWITCH_CONTEXT]" not in system

    def test_no_system_message_noop(
        self, mock_pipeline: LLMPipeline
    ) -> None:
        """Gracefully handles context with no system message."""
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[{"role": "user", "content": "test"}],
        )
        context.metadata["twitch_content"] = "viewer: hello"

        # Should not raise
        mock_pipeline._inject_twitch_content(context)

    def test_excess_newlines_collapsed(
        self, mock_pipeline: LLMPipeline, context_with_twitch: PipelineContext
    ) -> None:
        """Three or more consecutive newlines are collapsed to two."""
        context_with_twitch.messages[0]["content"] = (
            "Before.\n\n\n[TWITCH_CONTEXT]\n\n\nAfter."
        )
        context_with_twitch.metadata["twitch_content"] = ""

        mock_pipeline._inject_twitch_content(context_with_twitch)

        system = context_with_twitch.messages[0]["content"]
        assert "\n\n\n" not in system


# ---------------------------------------------------------------------------
# Tests: Deferred block patching
# ---------------------------------------------------------------------------

class TestTwitchDeferredPatching:
    """Tests for _update_deferred_block_contents with twitch_context."""

    def test_patches_twitch_block(
        self, mock_pipeline: LLMPipeline
    ) -> None:
        """Twitch block entry gets real char count and content."""
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[],
        )
        twitch_content = "Recent chat:\nviewer1: hello\nPick one."
        context.metadata["twitch_content"] = twitch_content
        context.metadata["block_contents"] = [
            {"id": "twitch_context", "chars": 0, "deferred": True, "content": ""},
        ]

        mock_pipeline._update_deferred_block_contents(context)

        entry = context.metadata["block_contents"][0]
        assert entry["chars"] == len(twitch_content)
        assert entry["content"] == twitch_content
        assert entry["deferred"] is False

    def test_empty_twitch_patches_zero_chars(
        self, mock_pipeline: LLMPipeline
    ) -> None:
        """When no Twitch content, block patches with 0 chars."""
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[],
        )
        context.metadata["block_contents"] = [
            {"id": "twitch_context", "chars": 0, "deferred": True, "content": ""},
        ]

        mock_pipeline._update_deferred_block_contents(context)

        entry = context.metadata["block_contents"][0]
        assert entry["chars"] == 0
        assert entry["content"] == ""
        assert entry["deferred"] is False


# ---------------------------------------------------------------------------
# Tests: stimulus_metadata threading through pipeline.run()
# ---------------------------------------------------------------------------

class TestStimulusMetadataThreading:
    """Tests for stimulus_metadata flowing through to injection."""

    def test_twitch_content_populates_and_injects(
        self, mock_pipeline: LLMPipeline
    ) -> None:
        """Simulates the full flow: metadata populates context, injection replaces placeholder."""
        context = PipelineContext(
            user_input="Respond to the Twitch chat messages.",
            persona={},
            messages=[
                {"role": "system", "content": "System.\n[TWITCH_CONTEXT]\nEnd."},
                {"role": "user", "content": "Respond to the Twitch chat messages."},
            ],
        )

        # Simulate what pipeline.run() does with stimulus_metadata
        stimulus_metadata = {"twitch_content": "viewer: hello", "message_count": 1}
        if stimulus_metadata and "twitch_content" in stimulus_metadata:
            context.metadata["twitch_content"] = stimulus_metadata["twitch_content"]

        mock_pipeline._inject_twitch_content(context)

        system = context.messages[0]["content"]
        assert "viewer: hello" in system
        assert "[TWITCH_CONTEXT]" not in system

    def test_none_metadata_collapses_placeholder(
        self, mock_pipeline: LLMPipeline
    ) -> None:
        """When stimulus_metadata is None, placeholder collapses to empty."""
        context = PipelineContext(
            user_input="Hello",
            persona={},
            messages=[
                {"role": "system", "content": "System.\n[TWITCH_CONTEXT]\nEnd."},
                {"role": "user", "content": "Hello"},
            ],
        )

        # Simulate None metadata — no twitch_content key set
        stimulus_metadata = None
        if stimulus_metadata and "twitch_content" in stimulus_metadata:
            context.metadata["twitch_content"] = stimulus_metadata["twitch_content"]

        mock_pipeline._inject_twitch_content(context)

        system = context.messages[0]["content"]
        assert "[TWITCH_CONTEXT]" not in system

    def test_metadata_without_twitch_content_key(
        self, mock_pipeline: LLMPipeline
    ) -> None:
        """stimulus_metadata without 'twitch_content' key doesn't populate context."""
        context = PipelineContext(
            user_input="Hello",
            persona={},
            messages=[
                {"role": "system", "content": "System.\n[TWITCH_CONTEXT]\nEnd."},
                {"role": "user", "content": "Hello"},
            ],
        )

        stimulus_metadata = {"message_count": 3, "channel": "testchannel"}
        if stimulus_metadata and "twitch_content" in stimulus_metadata:
            context.metadata["twitch_content"] = stimulus_metadata["twitch_content"]

        mock_pipeline._inject_twitch_content(context)

        system = context.messages[0]["content"]
        assert "[TWITCH_CONTEXT]" not in system
