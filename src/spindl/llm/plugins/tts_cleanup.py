"""
TTS Cleanup Plugin for LLM pipeline.

Produces a TTS-safe version of the LLM response and stashes it in
context.metadata["tts_text"].  The original response is returned
*unchanged* so that chat display and conversation logs keep the full
formatting (action markers, code blocks, emojis, markdown, etc.).

NANO-109: Dual-output — display text vs. speech text.
"""

import re

from .base import PipelineContext, PostProcessor

# Pre-compiled emoji pattern — covers Emoticons, Misc Symbols, Dingbats,
# Transport, Flags, Supplemental, Chess, and ZWJ sequences.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U00002600-\U000026FF"  # Misc Symbols
    "\U0000200D"             # ZWJ
    "\U00002B50"             # Star
    "]+",
)


class TTSCleanupPlugin(PostProcessor):
    """
    Produce clean text for TTS without mutating the display response.

    Stashes the cleaned version in ``context.metadata["tts_text"]`` and
    returns the original response so downstream consumers (chat bubble,
    conversation history) see the raw LLM output.

    Cleaning pipeline (order matters):

    1. Strip fenced code blocks (triple-backtick)
    2. Strip inline code (single-backtick)
    3. Strip markdown headers (# through ######)
    4. Strip markdown list bullets (* or -)
    5. Unwrap **bold** emphasis — keep the words
    6. Strip multi-word *action markers* — remove entirely
    7. Unwrap remaining single-word *emphasis* — keep the words
    8. Strip emojis
    9. Strip (parenthetical asides)
    10. Unwrap "quoted speech" — keep the words
    11. Collapse whitespace
    """

    @property
    def name(self) -> str:
        return "tts_cleanup"

    def process(self, context: PipelineContext, response: str) -> str:
        """
        Build TTS-safe text from *response*, stash it, return original.

        Args:
            context: Pipeline context — metadata["tts_text"] is written.
            response: Raw LLM response text.

        Returns:
            The *original* response, unmodified.
        """
        text = response

        # 1. Fenced code blocks — strip entirely (TTS should not read code)
        text = re.sub(r"```[\s\S]*?```", "", text)

        # 2. Inline code — strip entirely
        text = re.sub(r"`[^`]+`", "", text)

        # 3. Markdown headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # 4. Markdown list bullets (* item / - item)
        text = re.sub(r"^\*\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^-\s+", "", text, flags=re.MULTILINE)

        # 5. **bold emphasis** — unwrap, keep content
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

        # 6. Multi-word *action markers* — always RP stage directions
        #    e.g. *laughs nervously*, *waves excitedly at viewers*
        #    Opening * must be at start-of-line or after whitespace to avoid
        #    matching closing * of a prior pair as an opening delimiter.
        text = re.sub(r"(?:^|(?<=\s))\*([^*]*\s[^*]*)\*", "", text, flags=re.MULTILINE)

        # 7. Remaining single-word *emphasis* — unwrap, keep the word
        #    e.g. *really*, *like*, *wink* (single-word actions also unwrap,
        #    but TTS reading "wink" aloud is harmless)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)

        # 8. Emojis
        text = _EMOJI_RE.sub("", text)

        # 9. (Parenthetical asides) — e.g. (she said softly), (pauses)
        text = re.sub(r"\([^)]+\)", "", text)

        # 10. "Quoted speech" — unwrap, keep content
        text = re.sub(r'"([^"]*)"', r"\1", text)

        # 11. Collapse whitespace and trim
        text = re.sub(r"\s+", " ", text).strip()

        context.metadata["tts_text"] = text

        return response
