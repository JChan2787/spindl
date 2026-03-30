"""Tests for codex injection wrappers in LLMPipeline (NANO-045d).

Validates configurable prefix/suffix wrapping of codex content
when injected into the system prompt via _inject_codex_content().
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
def context_with_codex() -> PipelineContext:
    """PipelineContext with system prompt containing [CODEX_CONTEXT] placeholder."""
    return PipelineContext(
        user_input="Hello",
        persona={},
        messages=[
            {"role": "system", "content": "System prompt.\n[CODEX_CONTEXT]\nMore content."},
            {"role": "user", "content": "Hello"},
        ],
    )


# ---------------------------------------------------------------------------
# Tests: Default codex wrappers
# ---------------------------------------------------------------------------

class TestCodexWrappersDefaults:
    """Tests for default codex prefix/suffix values."""

    def test_default_prefix(self, mock_pipeline: LLMPipeline) -> None:
        """Default codex_prefix is the directive string."""
        assert "always true" in mock_pipeline._codex_prefix

    def test_default_suffix(self, mock_pipeline: LLMPipeline) -> None:
        """Default codex_suffix is empty string."""
        assert mock_pipeline._codex_suffix == ""


# ---------------------------------------------------------------------------
# Tests: Codex wrapping in _inject_codex_content
# ---------------------------------------------------------------------------

class TestCodexInjection:
    """Tests for codex content wrapping during injection."""

    def test_prefix_wraps_codex_content(
        self, mock_pipeline: LLMPipeline, context_with_codex: PipelineContext
    ) -> None:
        """Codex content is prefixed with _codex_prefix."""
        mock_pipeline._codex_prefix = "FACTS:"
        mock_pipeline._codex_suffix = ""
        context_with_codex.metadata["codex_content"] = "The sky is blue."

        mock_pipeline._inject_codex_content(context_with_codex)

        system = context_with_codex.messages[0]["content"]
        assert "FACTS:\nThe sky is blue." in system

    def test_suffix_wraps_codex_content(
        self, mock_pipeline: LLMPipeline, context_with_codex: PipelineContext
    ) -> None:
        """Codex content is suffixed with _codex_suffix."""
        mock_pipeline._codex_prefix = ""
        mock_pipeline._codex_suffix = "END FACTS."
        context_with_codex.metadata["codex_content"] = "The sky is blue."

        mock_pipeline._inject_codex_content(context_with_codex)

        system = context_with_codex.messages[0]["content"]
        assert "The sky is blue.\nEND FACTS." in system

    def test_both_prefix_and_suffix(
        self, mock_pipeline: LLMPipeline, context_with_codex: PipelineContext
    ) -> None:
        """Both prefix and suffix wrap content."""
        mock_pipeline._codex_prefix = "[BEGIN]"
        mock_pipeline._codex_suffix = "[END]"
        context_with_codex.metadata["codex_content"] = "Fact."

        mock_pipeline._inject_codex_content(context_with_codex)

        system = context_with_codex.messages[0]["content"]
        assert "[BEGIN]\nFact.\n[END]" in system

    def test_empty_codex_content_no_wrapping(
        self, mock_pipeline: LLMPipeline, context_with_codex: PipelineContext
    ) -> None:
        """Empty codex content is not wrapped (placeholder collapses)."""
        mock_pipeline._codex_prefix = "SHOULD NOT APPEAR"
        mock_pipeline._codex_suffix = "ALSO SHOULD NOT"
        context_with_codex.metadata["codex_content"] = ""

        mock_pipeline._inject_codex_content(context_with_codex)

        system = context_with_codex.messages[0]["content"]
        assert "SHOULD NOT APPEAR" not in system
        assert "ALSO SHOULD NOT" not in system

    def test_no_codex_metadata(
        self, mock_pipeline: LLMPipeline, context_with_codex: PipelineContext
    ) -> None:
        """Missing codex_content metadata is treated as empty string."""
        mock_pipeline._codex_prefix = "PREFIX"
        mock_pipeline._inject_codex_content(context_with_codex)

        system = context_with_codex.messages[0]["content"]
        assert "PREFIX" not in system
        assert "[CODEX_CONTEXT]" not in system


# ---------------------------------------------------------------------------
# Tests: set_codex_wrappers runtime update
# ---------------------------------------------------------------------------

class TestSetCodexWrappers:
    """Tests for set_codex_wrappers() runtime method."""

    def test_update_prefix(self, mock_pipeline: LLMPipeline) -> None:
        """set_codex_wrappers changes _codex_prefix."""
        mock_pipeline.set_codex_wrappers(codex_prefix="NEW PREFIX")
        assert mock_pipeline._codex_prefix == "NEW PREFIX"

    def test_update_suffix(self, mock_pipeline: LLMPipeline) -> None:
        """set_codex_wrappers changes _codex_suffix."""
        mock_pipeline.set_codex_wrappers(codex_suffix="NEW SUFFIX")
        assert mock_pipeline._codex_suffix == "NEW SUFFIX"

    def test_ellipsis_preserves_values(self, mock_pipeline: LLMPipeline) -> None:
        """Ellipsis default preserves current wrapper values."""
        original_prefix = mock_pipeline._codex_prefix
        mock_pipeline.set_codex_wrappers(codex_suffix="CHANGED")
        assert mock_pipeline._codex_prefix == original_prefix

    def test_none_clears_to_empty(self, mock_pipeline: LLMPipeline) -> None:
        """None value clears wrapper to empty string."""
        mock_pipeline.set_codex_wrappers(codex_prefix=None)
        assert mock_pipeline._codex_prefix == ""
