"""
Input provider - Inject the current user input.

This provider fills the [CURRENT_INPUT] placeholder with the formatted
current user input from BuildContext.
"""

from typing import Optional

from ..build_context import BuildContext
from ..context_provider import ContextProvider


class CurrentInputProvider(ContextProvider):
    """
    Provides current user input for [CURRENT_INPUT] placeholder.

    Formats the input with a "User:" label to match history formatting.
    This is a required field - always returns content (never collapses).
    """

    @property
    def placeholder(self) -> str:
        return "[CURRENT_INPUT]"

    def provide(self, context: BuildContext) -> Optional[str]:
        if not context.input_content:
            # Edge case: empty input still needs label for template consistency
            return "User:\n"

        return f"User:\n{context.input_content.strip()}"
