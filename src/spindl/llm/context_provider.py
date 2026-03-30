"""
ContextProvider - Abstract base class for modular context sources.

Each provider is responsible for a specific placeholder in the prompt template.
Providers receive a BuildContext and return content (or None to collapse the section).

Design Principle: Adding new context sources = adding new providers, not changing
core prompt builder logic.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional

from .build_context import BuildContext


class ContextProvider(ABC):
    """
    Abstract base class for all context providers.

    Each provider fills a specific placeholder in the prompt template.
    Return None or empty string from provide() to collapse the section entirely.
    """

    @property
    @abstractmethod
    def placeholder(self) -> str:
        """
        The placeholder this provider fills.

        Example: "[PERSONA_NAME]", "[VOICE_STATE]", "[CURRENT_INPUT]"
        """
        ...

    @abstractmethod
    def provide(self, context: BuildContext) -> Optional[str]:
        """
        Generate content for this provider's placeholder.

        Args:
            context: BuildContext containing all state for prompt building.

        Returns:
            Content string to substitute for the placeholder.
            Return None or empty string to collapse/remove this section.
        """
        ...


def build_section(header: str, content: Optional[str]) -> str:
    """
    Build a formatted section with header, or return empty string if no content.

    This is the section collapsing pattern from Troupeteer. Empty content
    results in the entire section being omitted, preventing orphaned headers.

    Args:
        header: Section header text (without "###" prefix).
        content: Section content, or None/empty to collapse.

    Returns:
        Formatted section string, or empty string if content is empty.

    Example:
        >>> build_section("Context", "This is voice input.")
        '### Context\\nThis is voice input.\\n'

        >>> build_section("Context", None)
        ''

        >>> build_section("Context", "   ")
        ''
    """
    if not content or not content.strip():
        return ""
    return f"### {header}\n{content.strip()}\n"


def cleanup_formatting(text: str) -> str:
    """
    Clean up prompt formatting by collapsing excessive newlines.

    Prevents triple+ newlines that can occur when sections are collapsed.
    Matches Troupeteer's cleanup pattern.

    Args:
        text: Raw prompt text with potential excessive newlines.

    Returns:
        Cleaned text with max 2 consecutive newlines.
    """
    # Collapse 3+ newlines to 2
    return re.sub(r"\n{3,}", "\n\n", text).strip()
