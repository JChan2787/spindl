"""
Tests for NANO-109: TTSCleanupPlugin dual-output and upgraded regex chain.

Tests cover:
- Returns original response unchanged (display text preserved)
- Stashes cleaned text in context.metadata["tts_text"]
- Strips fenced code blocks
- Strips inline code
- Strips markdown headers
- Strips markdown list bullets (* and -)
- Unwraps **bold emphasis** (keeps words)
- Strips standalone *action markers*
- Unwraps inline *emphasis* (keeps words)
- Strips emojis
- Strips (parenthetical asides)
- Unwraps "quoted speech" (keeps words)
- Collapses whitespace
- Handles combined formatting in a single response
- Handles real conversation log patterns
"""

import pytest

from spindl.llm.plugins.tts_cleanup import TTSCleanupPlugin
from spindl.llm.plugins.base import PipelineContext


def _make_context() -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    return PipelineContext(
        user_input="test",
        persona={"id": "test", "name": "Test"},
        messages=[],
    )


class TestTTSCleanupPlugin:
    """Tests for TTSCleanupPlugin PostProcessor."""

    def setup_method(self):
        self.plugin = TTSCleanupPlugin()

    def test_name(self):
        """Plugin name should be 'tts_cleanup'."""
        assert self.plugin.name == "tts_cleanup"

    # --- Core dual-output behavior ---

    def test_returns_original_response_unchanged(self):
        """process() must return the original response text for display."""
        ctx = _make_context()
        raw = "*sighs* Hello **world**! `code` (quietly)"
        result = self.plugin.process(ctx, raw)
        assert result == raw

    def test_stashes_tts_text_in_metadata(self):
        """Cleaned text must be stashed in context.metadata['tts_text']."""
        ctx = _make_context()
        self.plugin.process(ctx, "*laughs nervously* Hello there.")
        assert "tts_text" in ctx.metadata
        assert ctx.metadata["tts_text"] == "Hello there."

    # --- Fenced code blocks ---

    def test_strips_fenced_code_block(self):
        ctx = _make_context()
        self.plugin.process(ctx, 'Here:\n```python\nprint("hi")\n```\nDone.')
        assert ctx.metadata["tts_text"] == "Here: Done."

    def test_strips_multiple_code_blocks(self):
        ctx = _make_context()
        self.plugin.process(ctx, "A ```x``` B ```y``` C")
        assert ctx.metadata["tts_text"] == "A B C"

    # --- Inline code ---

    def test_strips_inline_code(self):
        ctx = _make_context()
        self.plugin.process(ctx, "Run `npm install` to start.")
        assert ctx.metadata["tts_text"] == "Run to start."

    # --- Markdown headers ---

    def test_strips_markdown_headers(self):
        ctx = _make_context()
        self.plugin.process(ctx, "### Section Title\nSome content.")
        assert ctx.metadata["tts_text"] == "Section Title Some content."

    def test_strips_h1_through_h6(self):
        ctx = _make_context()
        self.plugin.process(ctx, "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6")
        assert ctx.metadata["tts_text"] == "H1 H2 H3 H4 H5 H6"

    # --- Markdown list bullets ---

    def test_strips_asterisk_list_bullet(self):
        ctx = _make_context()
        self.plugin.process(ctx, "* First item\n* Second item")
        assert ctx.metadata["tts_text"] == "First item Second item"

    def test_strips_dash_list_bullet(self):
        ctx = _make_context()
        self.plugin.process(ctx, "- First item\n- Second item")
        assert ctx.metadata["tts_text"] == "First item Second item"

    def test_strips_asterisk_bullet_with_bold(self):
        """Real pattern from seraphina conversation logs."""
        ctx = _make_context()
        self.plugin.process(ctx, "*   **Generate creative text formats** like poems")
        assert ctx.metadata["tts_text"] == "Generate creative text formats like poems"

    # --- Double emphasis ---

    def test_unwraps_bold_emphasis(self):
        ctx = _make_context()
        self.plugin.process(ctx, "This is **important** information.")
        assert ctx.metadata["tts_text"] == "This is important information."

    def test_unwraps_bold_names(self):
        """Real pattern from conversation logs — bolded character names."""
        ctx = _make_context()
        self.plugin.process(ctx, "**Spindle** says hello to **User**.")
        assert ctx.metadata["tts_text"] == "Spindle says hello to User."

    # --- Action markers ---

    def test_strips_standalone_single_word_action_unwraps(self):
        """Single-word *action* is unwrapped (kept) — harmless for TTS."""
        ctx = _make_context()
        self.plugin.process(ctx, "*sighs* I don't know.")
        assert ctx.metadata["tts_text"] == "sighs I don't know."

    def test_strips_multiword_action_marker(self):
        ctx = _make_context()
        self.plugin.process(ctx, "*laughs nervously* That's funny.")
        assert ctx.metadata["tts_text"] == "That's funny."

    def test_strips_action_at_end_single_word_unwraps(self):
        """Single-word *action* at end is unwrapped — TTS says 'winks', harmless."""
        ctx = _make_context()
        self.plugin.process(ctx, "Hello there! *winks*")
        assert ctx.metadata["tts_text"] == "Hello there! winks"

    def test_strips_multiword_action_after_ellipsis(self):
        """Real pattern from blue_birb logs — multi-word action stripped."""
        ctx = _make_context()
        self.plugin.process(ctx, "1, 2, 3... *punches air* Next!")
        assert ctx.metadata["tts_text"] == "1, 2, 3... Next!"

    def test_strips_multiword_paragraph_action(self):
        """Real pattern — multi-word action on its own line."""
        ctx = _make_context()
        self.plugin.process(ctx, "*flutters angrily*\n\nWho are you?!")
        assert ctx.metadata["tts_text"] == "Who are you?!"

    # --- Inline emphasis ---

    def test_unwraps_inline_emphasis(self):
        ctx = _make_context()
        self.plugin.process(ctx, "What you'd *like* to know?")
        assert ctx.metadata["tts_text"] == "What you'd like to know?"

    def test_unwraps_inline_emphasis_mid_sentence(self):
        ctx = _make_context()
        self.plugin.process(ctx, "I *really* don't think so.")
        assert ctx.metadata["tts_text"] == "I really don't think so."

    # --- Emojis ---

    def test_strips_emojis(self):
        ctx = _make_context()
        self.plugin.process(ctx, "Hello! 😊 How are you? 🎉")
        assert ctx.metadata["tts_text"] == "Hello! How are you?"

    def test_strips_flag_emojis(self):
        ctx = _make_context()
        self.plugin.process(ctx, "Welcome 🇺🇸 everyone!")
        assert ctx.metadata["tts_text"] == "Welcome everyone!"

    # --- Parentheticals ---

    def test_strips_parenthetical_aside(self):
        ctx = _make_context()
        self.plugin.process(ctx, "Hello there. (she waves) Nice to meet you!")
        assert ctx.metadata["tts_text"] == "Hello there. Nice to meet you!"

    def test_strips_stage_direction_parenthetical(self):
        ctx = _make_context()
        self.plugin.process(ctx, "(pauses) Well then.")
        assert ctx.metadata["tts_text"] == "Well then."

    # --- Quoted speech ---

    def test_unwraps_quoted_speech(self):
        ctx = _make_context()
        self.plugin.process(ctx, 'She said "hello" to me.')
        assert ctx.metadata["tts_text"] == "She said hello to me."

    # --- Whitespace ---

    def test_collapses_whitespace(self):
        ctx = _make_context()
        self.plugin.process(ctx, "Hello    there.\n\n\nHow   are  you?")
        assert ctx.metadata["tts_text"] == "Hello there. How are you?"

    # --- Combined real-world patterns ---

    def test_real_pattern_blue_birb(self):
        """Actual response from blue_birb conversation logs."""
        ctx = _make_context()
        raw = "Ach, ye tryin' to test me, ye wee shite? I'll punch yer face intae the ground, ye numbskull! 1, 2, 3... *punches air* Yer testicles be next, ye wee scunner!"
        self.plugin.process(ctx, raw)
        assert ctx.metadata["tts_text"] == "Ach, ye tryin' to test me, ye wee shite? I'll punch yer face intae the ground, ye numbskull! 1, 2, 3... Yer testicles be next, ye wee scunner!"

    def test_real_pattern_spindle_wink(self):
        """Actual response from spindle conversation logs. Single-word *wink* unwraps."""
        ctx = _make_context()
        raw = "Oh, a stream? Now that sounds like a party I don't want to miss! Let me guess, you're planning to blow the internet up with your awesomeness, right? I mean, who wouldn't want to watch that? *wink*"
        self.plugin.process(ctx, raw)
        # Single-word action unwraps — asterisks gone, word kept
        assert "*wink*" not in ctx.metadata["tts_text"]
        assert ctx.metadata["tts_text"].endswith("want to watch that? wink")

    def test_real_pattern_seraphina_list(self):
        """Actual response from seraphina conversation — list with bold labels."""
        ctx = _make_context()
        raw = "I can help with:\n\n*   **Generate creative text formats** like poems\n*   **Answer your questions** in an informative way"
        self.plugin.process(ctx, raw)
        expected = "I can help with: Generate creative text formats like poems Answer your questions in an informative way"
        assert ctx.metadata["tts_text"] == expected

    def test_combined_all_patterns(self):
        """A single response hitting every cleanup pass."""
        ctx = _make_context()
        raw = (
            "### Heading\n"
            "*waves* Hello **friend**! 😊\n"
            "I *really* think `code_here` is great.\n"
            '(she said softly) "Indeed!"\n'
            "```python\nprint('hi')\n```\n"
            "- Item one\n"
            "* Item two"
        )
        self.plugin.process(ctx, raw)
        tts = ctx.metadata["tts_text"]
        # Verify formatting artifacts are gone
        assert "```" not in tts
        assert "`" not in tts
        assert "###" not in tts
        assert "**" not in tts
        assert "*waves*" not in tts  # asterisks gone
        assert "😊" not in tts
        assert "(she said softly)" not in tts
        # Verify content survived
        assert "Heading" in tts
        assert "Hello" in tts
        assert "friend" in tts
        assert "really" in tts
        assert "Indeed!" in tts
        assert "Item one" in tts
        assert "Item two" in tts
        assert "waves" in tts  # single-word action unwrapped, word kept

    def test_empty_response(self):
        """Empty string should produce empty tts_text."""
        ctx = _make_context()
        result = self.plugin.process(ctx, "")
        assert result == ""
        assert ctx.metadata["tts_text"] == ""

    def test_plain_text_passthrough(self):
        """Response with no formatting should pass through unchanged."""
        ctx = _make_context()
        plain = "Just a normal sentence with no special formatting."
        self.plugin.process(ctx, plain)
        assert ctx.metadata["tts_text"] == plain
