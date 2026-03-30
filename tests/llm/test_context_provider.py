"""Tests for ContextProvider base class and helpers."""

import pytest
from typing import Optional

from spindl.llm.build_context import BuildContext, InputModality
from spindl.llm.context_provider import (
    ContextProvider,
    build_section,
    cleanup_formatting,
)


class TestBuildSection:
    """Tests for build_section helper function."""

    def test_builds_section_with_content(self) -> None:
        """build_section creates formatted section with header."""
        result = build_section("Context", "This is voice input.")
        assert result == "### Context\nThis is voice input.\n"

    def test_collapses_with_none(self) -> None:
        """build_section returns empty string for None content."""
        result = build_section("Context", None)
        assert result == ""

    def test_collapses_with_empty_string(self) -> None:
        """build_section returns empty string for empty content."""
        result = build_section("Context", "")
        assert result == ""

    def test_collapses_with_whitespace(self) -> None:
        """build_section returns empty string for whitespace-only content."""
        result = build_section("Context", "   ")
        assert result == ""
        result = build_section("Context", "\n\n")
        assert result == ""
        result = build_section("Context", "\t  \n")
        assert result == ""

    def test_strips_content_whitespace(self) -> None:
        """build_section strips leading/trailing whitespace from content."""
        result = build_section("Rules", "  Rule 1\n  Rule 2  ")
        assert result == "### Rules\nRule 1\n  Rule 2\n"


class TestCleanupFormatting:
    """Tests for cleanup_formatting helper function."""

    def test_collapses_triple_newlines(self) -> None:
        """cleanup_formatting collapses 3+ newlines to 2."""
        text = "Section 1\n\n\nSection 2"
        assert cleanup_formatting(text) == "Section 1\n\nSection 2"

    def test_collapses_many_newlines(self) -> None:
        """cleanup_formatting handles many consecutive newlines."""
        text = "A\n\n\n\n\nB"
        assert cleanup_formatting(text) == "A\n\nB"

    def test_preserves_double_newlines(self) -> None:
        """cleanup_formatting preserves intended paragraph breaks."""
        text = "Section 1\n\nSection 2"
        assert cleanup_formatting(text) == "Section 1\n\nSection 2"

    def test_preserves_single_newlines(self) -> None:
        """cleanup_formatting preserves single line breaks."""
        text = "Line 1\nLine 2\nLine 3"
        assert cleanup_formatting(text) == "Line 1\nLine 2\nLine 3"

    def test_strips_outer_whitespace(self) -> None:
        """cleanup_formatting strips leading/trailing whitespace."""
        text = "\n\nContent\n\n"
        assert cleanup_formatting(text) == "Content"

    def test_handles_mixed_patterns(self) -> None:
        """cleanup_formatting handles complex mixed patterns."""
        text = """### Agent
You are Spindle.



### Context

This is a test.


### Rules

No asterisks.



End."""
        expected = """### Agent
You are Spindle.

### Context

This is a test.

### Rules

No asterisks.

End."""
        assert cleanup_formatting(text) == expected


class TestContextProviderInterface:
    """Tests for ContextProvider abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """ContextProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ContextProvider()  # type: ignore

    def test_concrete_implementation_works(self) -> None:
        """A concrete ContextProvider implementation works correctly."""

        class TestProvider(ContextProvider):
            @property
            def placeholder(self) -> str:
                return "[TEST_PLACEHOLDER]"

            def provide(self, context: BuildContext) -> Optional[str]:
                return f"Content for {context.input_content}"

        provider = TestProvider()
        ctx = BuildContext(input_content="testing")

        assert provider.placeholder == "[TEST_PLACEHOLDER]"
        assert provider.provide(ctx) == "Content for testing"

    def test_provider_can_return_none(self) -> None:
        """A provider can return None to collapse its section."""

        class ConditionalProvider(ContextProvider):
            @property
            def placeholder(self) -> str:
                return "[CONDITIONAL]"

            def provide(self, context: BuildContext) -> Optional[str]:
                if context.input_modality == InputModality.VOICE:
                    return "Voice mode active"
                return None

        provider = ConditionalProvider()

        voice_ctx = BuildContext(input_content="test", input_modality=InputModality.VOICE)
        text_ctx = BuildContext(input_content="test", input_modality=InputModality.TEXT)

        assert provider.provide(voice_ctx) == "Voice mode active"
        assert provider.provide(text_ctx) is None

    def test_provider_can_access_all_context_fields(self) -> None:
        """Provider can access all BuildContext fields."""

        class FullContextProvider(ContextProvider):
            @property
            def placeholder(self) -> str:
                return "[FULL]"

            def provide(self, context: BuildContext) -> Optional[str]:
                parts = [
                    f"Input: {context.input_content}",
                    f"Modality: {context.input_modality.value}",
                    f"Trigger: {context.state_trigger or 'none'}",
                    f"Persona: {context.persona.get('name', 'unknown')}",
                    f"Messages: {len(context.recent_messages)}",
                ]
                return "\n".join(parts)

        provider = FullContextProvider()
        ctx = BuildContext(
            input_content="Hello",
            input_modality=InputModality.VOICE,
            state_trigger="barge_in",
            persona={"name": "Spindle"},
            recent_messages=[],
        )

        result = provider.provide(ctx)
        assert "Input: Hello" in result
        assert "Modality: voice" in result
        assert "Trigger: barge_in" in result
        assert "Persona: Spindle" in result
        assert "Messages: 0" in result
