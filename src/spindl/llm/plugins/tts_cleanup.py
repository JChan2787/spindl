"""
TTS Cleanup Plugin for LLM pipeline.

Strips LLM artifacts (asterisks, parentheticals, quotes) for clean TTS output.
"""

import re

from .base import PipelineContext, PostProcessor


class TTSCleanupPlugin(PostProcessor):
    """
    Strip asterisks, parentheticals, and quotes for TTS.

    LLMs often produce text with formatting that sounds awkward when
    read aloud by TTS. This plugin cleans common artifacts:

    - *action markers* -> removed entirely
    - (parenthetical asides) -> removed entirely
    - "quoted speech" -> keeps content, strips quotes

    Examples:
        Input:  '*sighs* Hello there. (she waves) "Nice to meet you!"'
        Output: 'Hello there. Nice to meet you!'
    """

    @property
    def name(self) -> str:
        return "tts_cleanup"

    def process(self, context: PipelineContext, response: str) -> str:
        """
        Clean response for voice synthesis.

        Args:
            context: Pipeline context (unused, but available for future extensions)
            response: Raw LLM response text

        Returns:
            Cleaned text suitable for TTS
        """
        text = response

        # Remove *action markers* (e.g., *sighs*, *laughs nervously*)
        text = re.sub(r'\*[^*]+\*', '', text)

        # Remove (parenthetical asides) (e.g., (she said softly), (pauses))
        text = re.sub(r'\([^)]+\)', '', text)

        # Remove "quotes" but keep content (e.g., "Hello" -> Hello)
        text = re.sub(r'"([^"]*)"', r'\1', text)

        # Collapse multiple spaces into one and strip edges
        text = re.sub(r'\s+', ' ', text).strip()

        return text
