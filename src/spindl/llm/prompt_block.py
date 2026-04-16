"""
PromptBlock - Configurable block model for prompt composition.

NANO-045a: Replaces the static CONVERSATION_TEMPLATE with an ordered,
configurable list of blocks. Each block maps to a provider (or is static)
and can be reordered, disabled, overridden, or wrapper-customized via config.

Design Principle: Without prompt_blocks config, output is byte-identical
to the legacy template path. The block model is opt-in.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptBlock:
    """
    A discrete section of the assembled prompt.

    Each block corresponds to one provider's output (or static text)
    and carries metadata for ordering, toggling, and customization.

    Attributes:
        id: Unique identifier (e.g., "persona_name", "rag_context").
        label: Human-readable label for GUI display.
        order: Assembly position (0-indexed). Lower = earlier in prompt.
        enabled: Whether this block is included in the assembled prompt.
        placeholder: The template placeholder this block fills (e.g., "[PERSONA_NAME]").
                     Used to match the block to its provider. None for static blocks.
        section_header: Section header text (e.g., "Agent"). Emits "### Agent"
                       before this block's content. None if no header.
        content_wrapper: Format string wrapping provider output. Uses {content}
                        as interpolation token. E.g., "You are {content}."
                        None means raw provider output.
        user_override: When set, replaces provider output entirely.
        is_static: True for blocks with no provider (e.g., closing_instruction).
        static_content: Content for static blocks. May contain {persona_name}
                       for dynamic name insertion.
        tight_header: When True, section header joins content with \\n (no blank
                     line). When False (default), joins with \\n\\n. Only affects
                     blocks that have a section_header.
    """

    id: str
    label: str
    order: int
    enabled: bool = True
    placeholder: Optional[str] = None
    section_header: Optional[str] = None
    content_wrapper: Optional[str] = None
    user_override: Optional[str] = None
    is_static: bool = False
    static_content: Optional[str] = None
    tight_header: bool = False


def create_default_blocks() -> list[PromptBlock]:
    """
    Create the default block registry matching current CONVERSATION_TEMPLATE output.

    Block ordering and section headers reproduce the exact structure of the
    static template. Late-stage injection blocks (codex, RAG, history) use
    their placeholder strings as static content — they get replaced by
    pipeline inject methods post-build, same as today.

    Returns:
        List of 15 PromptBlock instances in default order.
    """
    return [
        PromptBlock(
            id="persona_name",
            label="Agent Name",
            order=0,
            placeholder="[PERSONA_NAME]",
            section_header="Agent",
            content_wrapper="You are {content}.",
            tight_header=True,
        ),
        PromptBlock(
            id="persona_appearance",
            label="Appearance",
            order=1,
            placeholder="[PERSONA_APPEARANCE]",
        ),
        PromptBlock(
            id="persona_personality",
            label="Personality",
            order=2,
            placeholder="[PERSONA_PERSONALITY]",
        ),
        PromptBlock(
            id="scenario",
            label="Scenario",
            order=3,
            placeholder="[SCENARIO]",
        ),
        PromptBlock(
            id="example_dialogue",
            label="Example Dialogue",
            order=4,
            placeholder="[EXAMPLE_DIALOGUE]",
        ),
        PromptBlock(
            id="modality_context",
            label="Modality Context",
            order=5,
            placeholder="[MODALITY_CONTEXT]",
            section_header="Context",
        ),
        PromptBlock(
            id="voice_state",
            label="Voice State",
            order=6,
            placeholder="[STATE_CONTEXT]",
        ),
        PromptBlock(
            id="codex_context",
            label="Codex",
            order=7,
            is_static=True,
            static_content="[CODEX_CONTEXT]",
        ),
        PromptBlock(
            id="rag_context",
            label="Memories",
            order=8,
            is_static=True,
            static_content="[RAG_CONTEXT]",
        ),
        PromptBlock(
            id="twitch_context",
            label="Twitch Chat",
            order=9,
            is_static=True,
            static_content="[TWITCH_CONTEXT]",
        ),
        PromptBlock(
            id="audience_chat",
            label="Audience Chat",
            order=10,
            is_static=True,
            static_content="[AUDIENCE_CHAT]",
        ),
        PromptBlock(
            id="persona_rules",
            label="Character Rules",
            order=11,
            placeholder="[PERSONA_RULES]",
            section_header="Rules",
        ),
        PromptBlock(
            id="modality_rules",
            label="Modality Rules",
            order=12,
            placeholder="[MODALITY_RULES]",
        ),
        PromptBlock(
            id="conversation_summary",
            label="Summary",
            order=13,
            placeholder="[CONVERSATION_SUMMARY]",
            section_header="Conversation",
        ),
        PromptBlock(
            id="recent_history",
            label="Chat History",
            order=14,
            is_static=True,
            static_content="[RECENT_HISTORY]",
        ),
        PromptBlock(
            id="closing_instruction",
            label="Closing Instruction",
            order=15,
            section_header="Input",
            is_static=True,
            static_content="Respond as {persona_name}.",
        ),
    ]


def load_block_config(
    config: dict,
    defaults: Optional[list[PromptBlock]] = None,
) -> list[PromptBlock]:
    """
    Merge prompt_blocks config with default blocks.

    Applies ordering, disable/enable, content overrides, and wrapper
    customization from the config dict onto the default block set.

    Args:
        config: The `prompt_blocks` section from spindl.yaml. Expected keys:
                - order: List of block IDs defining assembly order
                - disabled: List of block IDs to disable
                - overrides: Dict of block_id -> content string
                - wrappers: Dict of block_id -> wrapper format string
        defaults: Base block list. If None, uses create_default_blocks().

    Returns:
        List of PromptBlock instances with config applied, sorted by order.
    """
    if defaults is None:
        defaults = create_default_blocks()

    # Index defaults by ID for fast lookup
    block_map: dict[str, PromptBlock] = {b.id: b for b in defaults}

    # Apply ordering
    order_list = config.get("order")
    if order_list:
        # Assign order based on position in config list
        seen_ids = set()
        for idx, block_id in enumerate(order_list):
            if block_id in block_map:
                block_map[block_id].order = idx
                seen_ids.add(block_id)

        # Blocks not in config order get appended after the last explicit one
        next_order = len(order_list)
        for block in defaults:
            if block.id not in seen_ids:
                block.order = next_order
                next_order += 1

    # Apply disabled list
    disabled = config.get("disabled", [])
    for block_id in disabled:
        if block_id in block_map:
            block_map[block_id].enabled = False

    # Apply content overrides
    overrides = config.get("overrides", {})
    for block_id, content in overrides.items():
        if block_id in block_map:
            block_map[block_id].user_override = content

    # Apply wrapper customization
    wrappers = config.get("wrappers", {})
    for block_id, wrapper in wrappers.items():
        if block_id in block_map:
            block_map[block_id].content_wrapper = wrapper

    # Return sorted by order
    return sorted(block_map.values(), key=lambda b: b.order)
