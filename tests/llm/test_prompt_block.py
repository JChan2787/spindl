"""
Unit tests for PromptBlock model and block-based prompt assembly (NANO-045a).

Tests the full block lifecycle:
- Default block registry creation
- Config-driven customization (ordering, disabling, overrides, wrappers)
- Block-based assembly producing correct prompt output
- Byte-identical output when using default blocks vs legacy template
- Late-stage injection placeholder preservation
"""

import pytest

from spindl.llm.build_context import BuildContext, InputModality
from spindl.llm.context_provider import cleanup_formatting
from spindl.llm.prompt_block import (
    PromptBlock,
    create_default_blocks,
    load_block_config,
)
from spindl.llm.prompt_builder import PromptBuilder
from spindl.llm.prompt_library import CONVERSATION_TEMPLATE
from spindl.llm.providers.registry import create_default_registry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def structured_persona() -> dict:
    """Persona with all structured fields populated."""
    return {
        "id": "spindle",
        "name": "Spindle",
        "description": "A robot spider with eight articulated metal legs.",
        "personality": "Helpful, concise, slightly playful.",
        "rules": [
            "NO asterisks or action markers",
            "Keep responses concise",
        ],
    }


@pytest.fixture
def text_context(structured_persona: dict) -> BuildContext:
    """Text mode BuildContext with no block config (legacy path)."""
    return BuildContext(
        input_content="Hello, how are you?",
        input_modality=InputModality.TEXT,
        persona=structured_persona,
    )


@pytest.fixture
def default_blocks() -> list[PromptBlock]:
    """Fresh default block list."""
    return create_default_blocks()


@pytest.fixture
def default_registry():
    """Default provider registry."""
    return create_default_registry()


# =============================================================================
# PromptBlock Dataclass Tests
# =============================================================================


class TestPromptBlock:
    """Tests for PromptBlock dataclass."""

    def test_basic_creation(self):
        """Create a minimal block."""
        block = PromptBlock(id="test", label="Test", order=0)
        assert block.id == "test"
        assert block.label == "Test"
        assert block.order == 0
        assert block.enabled is True
        assert block.placeholder is None
        assert block.section_header is None
        assert block.content_wrapper is None
        assert block.user_override is None
        assert block.is_static is False
        assert block.static_content is None
        assert block.tight_header is False

    def test_static_block(self):
        """Create a static block."""
        block = PromptBlock(
            id="closing",
            label="Closing",
            order=11,
            is_static=True,
            static_content="Respond as {persona_name}.",
        )
        assert block.is_static is True
        assert block.static_content == "Respond as {persona_name}."

    def test_block_with_wrapper(self):
        """Create a block with content wrapper."""
        block = PromptBlock(
            id="name",
            label="Name",
            order=0,
            placeholder="[PERSONA_NAME]",
            content_wrapper="You are {content}.",
        )
        assert block.content_wrapper == "You are {content}."


# =============================================================================
# Default Block Registry Tests
# =============================================================================


class TestDefaultBlocks:
    """Tests for create_default_blocks()."""

    def test_returns_16_blocks(self, default_blocks: list[PromptBlock]):
        """Default registry has exactly 16 blocks."""
        assert len(default_blocks) == 16

    def test_all_blocks_enabled(self, default_blocks: list[PromptBlock]):
        """All default blocks are enabled."""
        for block in default_blocks:
            assert block.enabled is True, f"Block {block.id} should be enabled"

    def test_order_is_sequential(self, default_blocks: list[PromptBlock]):
        """Default blocks are ordered 0-15."""
        orders = [b.order for b in default_blocks]
        assert orders == list(range(16))

    def test_block_ids_unique(self, default_blocks: list[PromptBlock]):
        """All block IDs are unique."""
        ids = [b.id for b in default_blocks]
        assert len(ids) == len(set(ids))

    def test_expected_block_ids(self, default_blocks: list[PromptBlock]):
        """All expected block IDs are present."""
        ids = {b.id for b in default_blocks}
        expected = {
            "persona_name", "persona_appearance", "persona_personality",
            "scenario", "example_dialogue", "modality_context", "voice_state",
            "codex_context", "rag_context", "twitch_context", "audience_chat",
            "persona_rules", "modality_rules", "conversation_summary",
            "recent_history", "closing_instruction",
        }
        assert ids == expected

    def test_section_headers(self, default_blocks: list[PromptBlock]):
        """Correct blocks have section headers."""
        header_blocks = {b.id: b.section_header for b in default_blocks if b.section_header}
        assert header_blocks == {
            "persona_name": "Agent",
            "modality_context": "Context",
            "persona_rules": "Rules",
            "conversation_summary": "Conversation",
            "closing_instruction": "Input",
        }

    def test_late_stage_blocks_are_static(self, default_blocks: list[PromptBlock]):
        """Codex, RAG, Twitch, and history blocks are static with placeholder content."""
        block_map = {b.id: b for b in default_blocks}

        for block_id, expected_content in [
            ("codex_context", "[CODEX_CONTEXT]"),
            ("rag_context", "[RAG_CONTEXT]"),
            ("twitch_context", "[TWITCH_CONTEXT]"),
            ("audience_chat", "[AUDIENCE_CHAT]"),
            ("recent_history", "[RECENT_HISTORY]"),
        ]:
            block = block_map[block_id]
            assert block.is_static is True, f"{block_id} should be static"
            assert block.static_content == expected_content

    def test_persona_name_has_wrapper(self, default_blocks: list[PromptBlock]):
        """persona_name block has 'You are {content}.' wrapper."""
        block = next(b for b in default_blocks if b.id == "persona_name")
        assert block.content_wrapper == "You are {content}."
        assert block.tight_header is True

    def test_closing_instruction_is_static(self, default_blocks: list[PromptBlock]):
        """closing_instruction is static with persona name interpolation."""
        block = next(b for b in default_blocks if b.id == "closing_instruction")
        assert block.is_static is True
        assert block.static_content == "Respond as {persona_name}."


# =============================================================================
# Config Loading Tests
# =============================================================================


class TestLoadBlockConfig:
    """Tests for load_block_config()."""

    def test_empty_config_preserves_defaults(self, default_blocks: list[PromptBlock]):
        """Empty config dict returns defaults unchanged."""
        result = load_block_config({}, default_blocks)
        assert len(result) == 16
        assert [b.order for b in result] == list(range(16))

    def test_reorder_blocks(self):
        """Config order reorders blocks."""
        config = {
            "order": [
                "persona_rules",
                "modality_rules",
                "persona_name",
                "persona_appearance",
                "persona_personality",
                "modality_context",
                "voice_state",
                "codex_context",
                "rag_context",
                "conversation_summary",
                "recent_history",
                "closing_instruction",
            ]
        }
        result = load_block_config(config)
        assert result[0].id == "persona_rules"
        assert result[1].id == "modality_rules"
        assert result[2].id == "persona_name"

    def test_disable_blocks(self):
        """Config disabled list disables blocks."""
        config = {"disabled": ["voice_state", "modality_rules"]}
        result = load_block_config(config)
        block_map = {b.id: b for b in result}
        assert block_map["voice_state"].enabled is False
        assert block_map["modality_rules"].enabled is False
        assert block_map["persona_name"].enabled is True  # not disabled

    def test_override_content(self):
        """Config overrides replace block content."""
        config = {"overrides": {"persona_rules": "Custom rules here."}}
        result = load_block_config(config)
        block = next(b for b in result if b.id == "persona_rules")
        assert block.user_override == "Custom rules here."

    def test_custom_wrapper(self):
        """Config wrappers customize block wrapping."""
        config = {"wrappers": {"persona_name": "You play the role of {content}."}}
        result = load_block_config(config)
        block = next(b for b in result if b.id == "persona_name")
        assert block.content_wrapper == "You play the role of {content}."

    def test_unknown_block_in_order_ignored(self):
        """Unknown block ID in order list is silently ignored."""
        config = {"order": ["persona_name", "nonexistent_block", "persona_appearance"]}
        result = load_block_config(config)
        # persona_name is order 0, persona_appearance is order 1
        # all other blocks appended after
        assert result[0].id == "persona_name"
        assert result[1].id == "persona_appearance"
        assert len(result) == 16  # no extra blocks created

    def test_unknown_block_in_disabled_ignored(self):
        """Unknown block ID in disabled list is silently ignored."""
        config = {"disabled": ["nonexistent_block"]}
        result = load_block_config(config)
        assert all(b.enabled for b in result)

    def test_missing_block_in_order_appended(self):
        """Blocks not in config order are appended at the end."""
        config = {"order": ["persona_name", "closing_instruction"]}
        result = load_block_config(config)
        assert result[0].id == "persona_name"
        assert result[1].id == "closing_instruction"
        # Remaining 14 blocks appended after
        assert len(result) == 16
        remaining_ids = [b.id for b in result[2:]]
        assert "persona_appearance" in remaining_ids
        assert "persona_rules" in remaining_ids


# =============================================================================
# Block-Based Assembly Tests (THE CORE)
# =============================================================================


class TestBlockAssembly:
    """Tests for _build_with_blocks() in PromptBuilder."""

    def _build_with_blocks(
        self,
        persona: dict,
        user_input: str = "Hello",
        blocks: list[PromptBlock] = None,
        modality: InputModality = InputModality.TEXT,
    ) -> list[dict]:
        """Helper to build messages using block mode."""
        if blocks is None:
            blocks = create_default_blocks()
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)
        context = BuildContext(
            input_content=user_input,
            input_modality=modality,
            persona=persona,
            block_config=blocks,
        )
        return builder.build(persona, user_input, build_context=context)

    def _build_legacy(
        self,
        persona: dict,
        user_input: str = "Hello",
        modality: InputModality = InputModality.TEXT,
    ) -> list[dict]:
        """Helper to build messages using legacy template path."""
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)
        context = BuildContext(
            input_content=user_input,
            input_modality=modality,
            persona=persona,
        )
        return builder.build(persona, user_input, build_context=context)

    def test_default_blocks_byte_identical(self, structured_persona: dict):
        """THE ACID TEST: Default blocks produce byte-identical output to legacy template."""
        legacy = self._build_legacy(structured_persona)
        block_based = self._build_with_blocks(structured_persona)

        assert legacy[0]["content"] == block_based[0]["content"], (
            f"System prompts differ!\n"
            f"=== LEGACY ===\n{legacy[0]['content']}\n"
            f"=== BLOCK ===\n{block_based[0]['content']}"
        )
        assert legacy[1]["content"] == block_based[1]["content"]

    def test_default_blocks_byte_identical_voice(self, structured_persona: dict):
        """Acid test for voice modality too."""
        legacy = self._build_legacy(structured_persona, modality=InputModality.VOICE)
        block_based = self._build_with_blocks(
            structured_persona, modality=InputModality.VOICE
        )
        assert legacy[0]["content"] == block_based[0]["content"]

    def test_default_blocks_byte_identical_minimal_persona(self):
        """Acid test with minimal persona (most fields empty)."""
        persona = {"name": "Bot"}
        legacy = self._build_legacy(persona)
        block_based = self._build_with_blocks(persona)
        assert legacy[0]["content"] == block_based[0]["content"]

    def test_default_blocks_byte_identical_empty_persona(self):
        """Acid test with empty persona dict."""
        persona = {}
        legacy = self._build_legacy(persona)
        block_based = self._build_with_blocks(persona)
        assert legacy[0]["content"] == block_based[0]["content"]

    def test_scenario_injected_when_populated(self):
        """Scenario text appears in assembled prompt when populated."""
        persona = {
            "name": "Spindle",
            "description": "A robot spider.",
            "personality": "Helpful.",
            "scenario": "A dark forest at midnight.",
        }
        messages = self._build_with_blocks(persona)
        system = messages[0]["content"]
        assert "A dark forest at midnight." in system

    def test_scenario_collapses_when_empty(self, structured_persona: dict):
        """No scenario placeholder residue when scenario is empty."""
        messages = self._build_with_blocks(structured_persona)
        system = messages[0]["content"]
        assert "[SCENARIO]" not in system

    def test_scenario_position_between_personality_and_context(self):
        """Scenario appears after personality but before Context section."""
        persona = {
            "name": "Spindle",
            "personality": "Helpful.",
            "scenario": "A dark forest at midnight.",
        }
        messages = self._build_with_blocks(persona)
        system = messages[0]["content"]
        personality_pos = system.index("Helpful.")
        scenario_pos = system.index("A dark forest at midnight.")
        context_pos = system.index("### Context")
        assert personality_pos < scenario_pos < context_pos

    def test_default_blocks_byte_identical_with_scenario(self):
        """Acid test: blocks with scenario are byte-identical to legacy template."""
        persona = {
            "name": "Spindle",
            "description": "A robot spider.",
            "personality": "Helpful.",
            "scenario": "A dark forest at midnight.",
            "rules": ["- Stay in character"],
        }
        legacy = self._build_legacy(persona)
        block_based = self._build_with_blocks(persona)
        assert legacy[0]["content"] == block_based[0]["content"], (
            f"System prompts differ!\n"
            f"=== LEGACY ===\n{legacy[0]['content']}\n"
            f"=== BLOCK ===\n{block_based[0]['content']}"
        )

    def test_message_structure(self, structured_persona: dict):
        """Block assembly produces [system, user] message structure."""
        messages = self._build_with_blocks(structured_persona)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_reorder_blocks(self, structured_persona: dict):
        """Reordered blocks change prompt section order."""
        # Put Rules before Context
        config = {
            "order": [
                "persona_name",
                "persona_appearance",
                "persona_personality",
                "persona_rules",
                "modality_rules",
                "modality_context",
                "voice_state",
                "codex_context",
                "rag_context",
                "conversation_summary",
                "recent_history",
                "closing_instruction",
            ]
        }
        blocks = load_block_config(config)
        messages = self._build_with_blocks(structured_persona, blocks=blocks)
        system = messages[0]["content"]

        # Rules should appear before Context
        rules_pos = system.index("### Rules")
        context_pos = system.index("### Context")
        assert rules_pos < context_pos

    def test_disable_block(self, structured_persona: dict):
        """Disabled block is absent from output."""
        config = {"disabled": ["persona_appearance"]}
        blocks = load_block_config(config)
        messages = self._build_with_blocks(structured_persona, blocks=blocks)
        system = messages[0]["content"]

        assert "robot spider" not in system
        # Other content still present
        assert "Spindle" in system
        assert "playful" in system

    def test_disable_all_blocks_in_section(self, structured_persona: dict):
        """Disabling all blocks under a header collapses the header."""
        config = {"disabled": ["modality_context", "voice_state", "codex_context", "rag_context"]}
        blocks = load_block_config(config)
        messages = self._build_with_blocks(structured_persona, blocks=blocks)
        system = messages[0]["content"]

        assert "### Context" not in system
        # Other sections still present
        assert "### Agent" in system
        assert "### Rules" in system

    def test_user_override(self, structured_persona: dict):
        """User override replaces provider content."""
        config = {"overrides": {"persona_rules": "My custom rules."}}
        blocks = load_block_config(config)
        messages = self._build_with_blocks(structured_persona, blocks=blocks)
        system = messages[0]["content"]

        assert "My custom rules." in system
        # Original rules should NOT be present
        assert "asterisks" not in system

    def test_late_stage_placeholders_preserved(self, structured_persona: dict):
        """Codex, RAG, Twitch, and history placeholders survive in assembled prompt."""
        messages = self._build_with_blocks(structured_persona)
        system = messages[0]["content"]

        assert "[CODEX_CONTEXT]" in system
        assert "[RAG_CONTEXT]" in system
        assert "[TWITCH_CONTEXT]" in system
        assert "[RECENT_HISTORY]" in system

    def test_closing_instruction_dynamic(self, structured_persona: dict):
        """Closing instruction uses correct persona name."""
        messages = self._build_with_blocks(structured_persona)
        system = messages[0]["content"]
        assert "Respond as Spindle." in system

    def test_closing_instruction_different_name(self):
        """Closing instruction adapts to different persona names."""
        persona = {"name": "Aria"}
        messages = self._build_with_blocks(persona)
        system = messages[0]["content"]
        assert "Respond as Aria." in system
        assert "You are Aria." in system

    def test_no_block_config_uses_legacy(self, structured_persona: dict):
        """When block_config is None, legacy template path is used."""
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)
        # No block_config = None (default)
        context = BuildContext(
            input_content="Hello",
            input_modality=InputModality.TEXT,
            persona=structured_persona,
        )
        assert context.block_config is None
        messages = builder.build(structured_persona, "Hello", build_context=context)
        # Should still produce valid output via legacy path
        assert len(messages) == 2
        assert "Spindle" in messages[0]["content"]

    def test_custom_wrapper_persona_name(self, structured_persona: dict):
        """Custom wrapper changes persona name presentation."""
        config = {"wrappers": {"persona_name": "You play the role of {content}."}}
        blocks = load_block_config(config)
        messages = self._build_with_blocks(structured_persona, blocks=blocks)
        system = messages[0]["content"]

        assert "You play the role of Spindle." in system
        assert "You are Spindle." not in system

    def test_custom_wrapper_closing(self, structured_persona: dict):
        """Custom wrapper on closing instruction."""
        config = {"wrappers": {"closing_instruction": "Always respond in character as {content}."}}
        blocks = load_block_config(config)
        messages = self._build_with_blocks(structured_persona, blocks=blocks)
        system = messages[0]["content"]

        assert "Always respond in character as Spindle." in system
        assert "Respond as Spindle." not in system

    def test_section_headers_present(self, structured_persona: dict):
        """All section headers appear in assembled prompt."""
        messages = self._build_with_blocks(structured_persona)
        system = messages[0]["content"]

        assert "### Agent" in system
        assert "### Context" in system
        assert "### Rules" in system
        assert "### Conversation" in system
        assert "### Input" in system

    def test_disabled_late_stage_block(self, structured_persona: dict):
        """Disabling a late-stage block removes its placeholder."""
        config = {"disabled": ["codex_context"]}
        blocks = load_block_config(config)
        messages = self._build_with_blocks(structured_persona, blocks=blocks)
        system = messages[0]["content"]

        assert "[CODEX_CONTEXT]" not in system
        # Other placeholders still present
        assert "[RAG_CONTEXT]" in system
        assert "[RECENT_HISTORY]" in system


# =============================================================================
# Pipeline Integration Tests
# =============================================================================


class TestPipelineBlockConfig:
    """Tests for block config wiring through the pipeline."""

    def test_pipeline_default_no_blocks(self):
        """Pipeline without set_block_config uses None."""
        from spindl.llm.pipeline import LLMPipeline
        from unittest.mock import MagicMock

        provider = MagicMock()
        builder = PromptBuilder(providers=create_default_registry())
        pipeline = LLMPipeline(provider, builder)

        assert pipeline._block_config is None

    def test_pipeline_set_block_config(self):
        """Pipeline stores resolved blocks after set_block_config."""
        from spindl.llm.pipeline import LLMPipeline
        from unittest.mock import MagicMock

        provider = MagicMock()
        builder = PromptBuilder(providers=create_default_registry())
        pipeline = LLMPipeline(provider, builder)

        config = {"disabled": ["voice_state"]}
        pipeline.set_block_config(config)

        assert pipeline._block_config is not None
        assert len(pipeline._block_config) == 16
        block_map = {b.id: b for b in pipeline._block_config}
        assert block_map["voice_state"].enabled is False

    def test_pipeline_clear_block_config(self):
        """Pipeline clears blocks when set_block_config(None)."""
        from spindl.llm.pipeline import LLMPipeline
        from unittest.mock import MagicMock

        provider = MagicMock()
        builder = PromptBuilder(providers=create_default_registry())
        pipeline = LLMPipeline(provider, builder)

        pipeline.set_block_config({"disabled": ["voice_state"]})
        assert pipeline._block_config is not None

        pipeline.set_block_config(None)
        assert pipeline._block_config is None


# =============================================================================
# Block Contents Capture Tests (NANO-045b)
# =============================================================================


class TestBlockContentsCapture:
    """Tests for per-block content data capture during _build_with_blocks()."""

    def _build_and_get_context(
        self,
        persona: dict,
        user_input: str = "Hello",
        blocks: list[PromptBlock] = None,
        modality: InputModality = InputModality.TEXT,
    ) -> BuildContext:
        """Helper: build messages and return the BuildContext with block_contents."""
        if blocks is None:
            blocks = create_default_blocks()
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)
        context = BuildContext(
            input_content=user_input,
            input_modality=modality,
            persona=persona,
            block_config=blocks,
        )
        builder.build(persona, user_input, build_context=context)
        return context

    def test_block_contents_populated(self, structured_persona: dict):
        """Block contents list is populated after build."""
        ctx = self._build_and_get_context(structured_persona)
        assert ctx.block_contents is not None
        assert len(ctx.block_contents) > 0

    def test_block_contents_count_matches_enabled(self, structured_persona: dict):
        """block_contents has one entry per enabled block."""
        ctx = self._build_and_get_context(structured_persona)
        # All 16 default blocks are enabled
        assert len(ctx.block_contents) == 16

    def test_block_ids_match_defaults(self, structured_persona: dict):
        """block_contents IDs match the default block IDs in order."""
        ctx = self._build_and_get_context(structured_persona)
        expected_ids = [b.id for b in create_default_blocks()]
        actual_ids = [bc["id"] for bc in ctx.block_contents]
        assert actual_ids == expected_ids

    def test_block_contents_have_required_fields(self, structured_persona: dict):
        """Each block_contents entry has id, label, section, chars, deferred."""
        ctx = self._build_and_get_context(structured_persona)
        required = {"id", "label", "section", "chars", "deferred"}
        for entry in ctx.block_contents:
            assert required.issubset(entry.keys()), (
                f"Block {entry.get('id', '?')} missing fields: "
                f"{required - entry.keys()}"
            )

    def test_deferred_flags_correct(self, structured_persona: dict):
        """Only injection blocks are flagged as deferred."""
        ctx = self._build_and_get_context(structured_persona)
        deferred_ids = {e["id"] for e in ctx.block_contents if e["deferred"]}
        assert deferred_ids == {"codex_context", "rag_context", "twitch_context", "audience_chat", "recent_history"}

    def test_provider_blocks_have_nonzero_chars(self, structured_persona: dict):
        """Provider-backed blocks with content have positive char counts."""
        ctx = self._build_and_get_context(structured_persona)
        by_id = {e["id"]: e for e in ctx.block_contents}
        # persona_name has wrapper "You are {content}." + section header
        assert by_id["persona_name"]["chars"] > 0
        assert by_id["persona_appearance"]["chars"] > 0
        assert by_id["persona_personality"]["chars"] > 0

    def test_section_headers_captured(self, structured_persona: dict):
        """Blocks with section headers have their section name captured."""
        ctx = self._build_and_get_context(structured_persona)
        by_id = {e["id"]: e for e in ctx.block_contents}
        assert by_id["persona_name"]["section"] == "Agent"
        assert by_id["modality_context"]["section"] == "Context"
        assert by_id["persona_rules"]["section"] == "Rules"
        assert by_id["conversation_summary"]["section"] == "Conversation"
        assert by_id["closing_instruction"]["section"] == "Input"
        # Non-header blocks have None
        assert by_id["persona_appearance"]["section"] is None
        assert by_id["persona_personality"]["section"] is None

    def test_disabled_block_excluded(self, structured_persona: dict):
        """Disabled blocks don't appear in block_contents."""
        blocks = load_block_config({"disabled": ["voice_state"]})
        ctx = self._build_and_get_context(structured_persona, blocks=blocks)
        ids = {e["id"] for e in ctx.block_contents}
        assert "voice_state" not in ids
        assert len(ctx.block_contents) == 15

    def test_legacy_mode_no_block_contents(self, structured_persona: dict):
        """Legacy mode (no block_config) leaves block_contents as None."""
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)
        context = BuildContext(
            input_content="Hello",
            input_modality=InputModality.TEXT,
            persona=structured_persona,
            # No block_config
        )
        builder.build(structured_persona, "Hello", build_context=context)
        assert context.block_contents is None

    def test_closing_instruction_has_chars(self, structured_persona: dict):
        """Static closing_instruction block has chars from persona name interpolation."""
        ctx = self._build_and_get_context(structured_persona)
        by_id = {e["id"]: e for e in ctx.block_contents}
        closing = by_id["closing_instruction"]
        assert closing["chars"] > 0
        assert not closing["deferred"]
