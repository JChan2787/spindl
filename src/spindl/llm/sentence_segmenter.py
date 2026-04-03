"""
Sentence segmenter for streaming LLM responses (NANO-111).

Consumes StreamChunk objects from a streaming LLM provider and yields
complete sentences as they form. Designed for the streaming TTS pipeline:
each yielded sentence is independently cleaned and sent to TTS.

Design decisions:
- First chunk splits on comma for fastest time-to-first-audio
  (borrowed from Open-LLM-VTuber's faster_first_response pattern)
- Subsequent chunks split on sentence-ending punctuation (.!?)
- Handles common abbreviations (Mr., Dr., etc.), ellipsis
- Excludes <think>...</think> reasoning blocks from sentence output
- Flushes remaining buffer on final chunk
"""

import re
from dataclasses import dataclass
from typing import Iterator

from .base import StreamChunk


@dataclass
class SentenceChunk:
    """A complete sentence extracted from the streaming token flow."""

    text: str
    """The sentence text (cleaned of leading/trailing whitespace)."""

    index: int
    """Sentence order index (0-based). Used for TTS ordering."""

    is_final: bool
    """True if this is the last sentence in the response."""


# Common abbreviations that end with a period but aren't sentence boundaries
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "dept", "est", "vol", "vs", "etc", "inc", "ltd", "corp", "approx",
    "govt", "assn", "bros", "no", "gen", "rep", "sen", "rev", "hon",
    "sgt", "cpl", "pvt", "capt", "cmdr", "lt", "col", "maj",
    "fig", "eq", "ref", "sec", "ch", "pt",
    "e.g", "i.e", "a.m", "p.m",
})

# Commas and clause-level delimiters (for first-chunk fast split)
_COMMAS = {",", ";", "，", "、", "；"}

# Sentence-ending punctuation
_END_PUNCTUATION = re.compile(r'[.!?。！？]')

# Pattern: sentence-ending punctuation followed by whitespace or end of string
# Negative lookbehind for single uppercase letter (initials like "J. Chan")
_SENTENCE_BOUNDARY = re.compile(
    r'(?<![A-Z])'           # not preceded by single uppercase (initials)
    r'([.!?。！？])'        # sentence-ending punctuation (captured)
    r'(?:\s|$)'             # followed by whitespace or end
)


class SentenceSegmenter:
    """
    Accumulates streaming LLM tokens and yields complete sentences.

    First sentence splits on commas for fastest time-to-first-audio.
    Subsequent sentences split on sentence-ending punctuation.

    Usage:
        segmenter = SentenceSegmenter()
        for chunk in provider.generate_stream(messages):
            for sentence in segmenter.feed(chunk):
                # sentence.text is a complete sentence ready for TTS
                ...
    """

    def __init__(self, faster_first_response: bool = True):
        self._buffer: str = ""
        self._sentence_index: int = 0
        self._in_think_block: bool = False
        self._is_first_sentence: bool = True
        self._faster_first_response: bool = faster_first_response

    def feed(self, chunk: StreamChunk) -> Iterator[SentenceChunk]:
        """
        Feed a stream chunk, yield any complete sentences.

        Args:
            chunk: StreamChunk from LLM provider

        Yields:
            SentenceChunk for each complete sentence detected
        """
        text = chunk.content or ""

        # Handle reasoning/think blocks — accumulate but don't yield as sentences
        if text:
            text = self._filter_think_blocks(text)

        if text:
            self._buffer += text

        # On final chunk, flush whatever remains
        if chunk.is_final:
            if self._buffer.strip():
                yield SentenceChunk(
                    text=self._buffer.strip(),
                    index=self._sentence_index,
                    is_final=True,
                )
                self._sentence_index += 1
            self._buffer = ""
            return

        # Try to extract complete sentences from buffer
        yield from self._extract_sentences()

    def _extract_sentences(self) -> Iterator[SentenceChunk]:
        """Extract complete sentences from the current buffer."""

        # First sentence: split on comma for fastest time-to-first-audio
        if self._is_first_sentence and self._faster_first_response:
            result = self._try_comma_split()
            if result is not None:
                self._is_first_sentence = False
                yield result
                # Continue to check for more sentences in remaining buffer

        # Standard sentence boundary detection
        while True:
            match = _SENTENCE_BOUNDARY.search(self._buffer)
            if not match:
                return

            boundary_pos = match.start(1)

            # Check if this is actually an abbreviation
            if match.group(1) == ".":
                before = self._buffer[:boundary_pos].rstrip()
                last_word = before.split()[-1].lower().rstrip(".") if before.split() else ""
                if last_word in _ABBREVIATIONS:
                    # Not a real boundary — look for the next one
                    check_pos = match.end()
                    if check_pos >= len(self._buffer):
                        return
                    remaining = self._buffer[check_pos:]
                    next_match = _SENTENCE_BOUNDARY.search(remaining)
                    if not next_match:
                        return
                    boundary_pos = check_pos + next_match.start(1)
                    match = next_match

                # Check for ellipsis (...)
                if boundary_pos + 3 <= len(self._buffer) and self._buffer[boundary_pos:boundary_pos + 3] == "...":
                    after_ellipsis = boundary_pos + 3
                    if after_ellipsis >= len(self._buffer):
                        return
                    # Skip past ellipsis and continue looking
                    remaining = self._buffer[after_ellipsis:]
                    next_match = _SENTENCE_BOUNDARY.search(remaining)
                    if not next_match:
                        return
                    boundary_pos = after_ellipsis + next_match.start(1)
                    match = next_match

            # We have a real sentence boundary
            split_pos = match.end()
            sentence = self._buffer[:split_pos].strip()
            self._buffer = self._buffer[split_pos:]

            if sentence:
                self._is_first_sentence = False
                yield SentenceChunk(
                    text=sentence,
                    index=self._sentence_index,
                    is_final=False,
                )
                self._sentence_index += 1

    def _try_comma_split(self) -> SentenceChunk | None:
        """
        Try to split the buffer at the first comma (for faster first response).

        Returns:
            SentenceChunk if a comma was found, None otherwise.
        """
        for i, char in enumerate(self._buffer):
            if char in _COMMAS:
                # Split at the comma (include the comma in the chunk)
                sentence = self._buffer[:i + 1].strip()
                self._buffer = self._buffer[i + 1:].lstrip()
                if sentence:
                    chunk = SentenceChunk(
                        text=sentence,
                        index=self._sentence_index,
                        is_final=False,
                    )
                    self._sentence_index += 1
                    return chunk
        return None

    def _filter_think_blocks(self, text: str) -> str:
        """
        Filter out <think>...</think> reasoning blocks from text.

        These blocks are for display only, not for TTS or sentence splitting.
        Handles blocks that span multiple chunks.
        """
        result = []
        i = 0
        while i < len(text):
            if self._in_think_block:
                close_idx = text.find("</think>", i)
                if close_idx == -1:
                    break
                else:
                    i = close_idx + len("</think>")
                    self._in_think_block = False
            else:
                open_idx = text.find("<think>", i)
                if open_idx == -1:
                    result.append(text[i:])
                    break
                else:
                    result.append(text[i:open_idx])
                    i = open_idx + len("<think>")
                    self._in_think_block = True

        return "".join(result)

    def reset(self) -> None:
        """Reset segmenter state for a new response."""
        self._buffer = ""
        self._sentence_index = 0
        self._in_think_block = False
        self._is_first_sentence = True
