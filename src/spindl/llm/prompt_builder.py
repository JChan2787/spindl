"""
PromptBuilder - Modular prompt assembly with provider orchestration.

Session 3 rewrite of NANO-014. Supports three modes:
- Legacy mode: Simple concatenation (original behavior)
- Provider mode: Template-based building with registered ContextProviders
- Block mode: Configurable block-based assembly (NANO-045a)

The legacy interface is fully preserved for backward compatibility with
existing code (LLMPipeline, demos, etc.).
"""

from typing import Optional, Union

from .build_context import BuildContext, InputModality
from .context_provider import ContextProvider, cleanup_formatting
from .prompt_block import PromptBlock
from .prompt_library import CONVERSATION_TEMPLATE
from .providers.registry import ProviderRegistry
from .providers.input_provider import CurrentInputProvider


class PromptBuilder:
    """
    Builds message lists for LLM consumption.

    Supports three modes:
    1. Legacy mode (no providers): Original simple concatenation
    2. Provider mode: Template-based building with registered providers
    3. Block mode (NANO-045a): Configurable block-based assembly when
       BuildContext.block_config is set

    The legacy interface `build(persona, user_input)` is preserved exactly
    for backward compatibility with LLMPipeline and existing demos.
    """

    def __init__(
        self,
        providers: Optional[Union[list[ContextProvider], ProviderRegistry]] = None,
    ) -> None:
        """
        Initialize the prompt builder.

        Args:
            providers: Optional list of ContextProviders or a ProviderRegistry.
                       If None, uses legacy mode (original behavior).
        """
        if providers is None:
            self._providers: list[ContextProvider] = []
        elif isinstance(providers, ProviderRegistry):
            self._providers = list(providers)
        else:
            self._providers = list(providers)

    def build_prompt(self, template: str, context: BuildContext) -> str:
        """
        Build prompt string by substituting placeholders with provider content.

        This is the new template-based building method. Iterates through all
        registered providers, substitutes their placeholders, and cleans up
        formatting.

        Args:
            template: Template string with placeholders (e.g., CONVERSATION_TEMPLATE)
            context: BuildContext containing all state for prompt building

        Returns:
            Assembled prompt string with placeholders substituted and
            formatting cleaned up.
        """
        result = template

        for provider in self._providers:
            content = provider.provide(context)
            if content:
                result = result.replace(provider.placeholder, content)
            else:
                # Empty content: replace placeholder with empty string (collapse)
                result = result.replace(provider.placeholder, "")

        return cleanup_formatting(result)

    def build(
        self,
        persona: dict,
        user_input: str,
        context_injection: Optional[str] = None,
        build_context: Optional[BuildContext] = None,
    ) -> list[dict]:
        """
        Build OpenAI-style messages.

        This method is backward compatible with the original implementation.
        When no providers are registered, it uses the original legacy behavior.
        When providers are registered, it uses template-based building.

        In provider mode, the system prompt is built from the template (excluding
        user input), and the user input is returned as a separate message. This
        maintains compatibility with HistoryInjector which expects [system, user]
        message structure.

        Args:
            persona: Persona config dict with 'system_prompt' key (legacy) or
                     structured fields for providers (name, appearance, etc.)
            user_input: Current user utterance
            context_injection: Optional context to append to system prompt
                               (used by RAG plugin in future)
            build_context: Optional BuildContext with full state (modality, trigger, etc.)
                          If provided in provider mode, uses this instead of constructing
                          a minimal context. Enables voice state injection, etc.

        Returns:
            List of message dicts:
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "..."}]

        Raises:
            ValueError: If in legacy mode and persona missing 'system_prompt' key
        """
        if not self._providers:
            # Legacy mode - original behavior
            return self._build_legacy(persona, user_input, context_injection)

        # Provider mode - template-based building
        return self._build_with_providers(
            persona, user_input, context_injection, build_context
        )

    def _build_legacy(
        self,
        persona: dict,
        user_input: str,
        context_injection: Optional[str],
    ) -> list[dict]:
        """
        Original legacy behavior for backward compatibility.

        Uses persona['system_prompt'] directly without template processing.
        """
        if "system_prompt" not in persona:
            raise ValueError("Persona must contain 'system_prompt' key")

        system_content = persona["system_prompt"]

        # Append context injection if provided (for future RAG integration)
        if context_injection:
            system_content = f"{system_content}\n\n{context_injection}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]

    def _build_with_providers(
        self,
        persona: dict,
        user_input: str,
        context_injection: Optional[str],
        build_context: Optional[BuildContext] = None,
    ) -> list[dict]:
        """
        Provider-based building with template substitution.

        Builds the system prompt using registered providers (excluding
        CurrentInputProvider), then returns user input as separate message.
        This maintains [system, user] structure for HistoryInjector compatibility.

        If BuildContext.block_config is set, delegates to _build_with_blocks()
        for configurable block-based assembly (NANO-045a).
        """
        # Use provided BuildContext or create minimal one from args
        if build_context is not None:
            context = build_context
        else:
            # Build context from legacy args (backward compatibility)
            context = BuildContext(
                input_content=user_input,
                input_modality=InputModality.TEXT,
                persona=persona,
            )

        # NANO-045a: Block mode — delegate if block_config is present
        if context.block_config is not None:
            return self._build_with_blocks(
                context, user_input, context_injection
            )

        # Filter out CurrentInputProvider - user input is separate in this mode
        system_providers = [
            p for p in self._providers
            if not isinstance(p, CurrentInputProvider)
        ]

        # Build system prompt using template
        result = CONVERSATION_TEMPLATE
        for provider in system_providers:
            content = provider.provide(context)
            if content:
                result = result.replace(provider.placeholder, content)
            else:
                result = result.replace(provider.placeholder, "")

        # Remove any remaining [CURRENT_INPUT] placeholder
        result = result.replace("[CURRENT_INPUT]", "")

        # [RAG_CONTEXT] placeholder is preserved here — it survives into the
        # pipeline where _inject_rag_content() replaces it with retrieved
        # memories (or collapses it to empty string if memory is disabled).
        # Same pattern as [CODEX_CONTEXT]. See: NANO-043 Phase 2.

        system_content = cleanup_formatting(result)

        # Append context injection if provided
        if context_injection:
            system_content = f"{system_content}\n\n{context_injection}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]

    def _build_with_blocks(
        self,
        context: BuildContext,
        user_input: str,
        context_injection: Optional[str],
    ) -> list[dict]:
        """
        Block-based prompt assembly (NANO-045a).

        Assembles the system prompt from an ordered list of PromptBlock instances.
        Each block produces content via its linked provider, static content, or
        user override. Section headers are emitted before header-bearing blocks.

        Late-stage injection blocks (codex, RAG, history) emit their placeholder
        strings as content — pipeline inject methods replace them post-build.

        Args:
            context: BuildContext with block_config set (list of PromptBlock).
            user_input: Current user utterance (becomes separate user message).
            context_injection: Optional context appended to system prompt.

        Returns:
            [system, user] message list.
        """
        blocks: list[PromptBlock] = context.block_config

        # Index providers by placeholder for fast lookup
        provider_map: dict[str, ContextProvider] = {}
        for p in self._providers:
            if not isinstance(p, CurrentInputProvider):
                provider_map[p.placeholder] = p

        # Resolve persona name for static content interpolation
        persona_name = context.persona.get("name", "Assistant")

        # IDs of blocks whose real content arrives after preprocessing (NANO-045b)
        _DEFERRED_BLOCK_IDS = {"codex_context", "rag_context", "recent_history", "twitch_context", "audience_chat"}

        # Assemble blocks in order, capturing per-block content data (NANO-045b)
        parts: list[str] = []
        block_contents: list[dict] = []
        for block in sorted(blocks, key=lambda b: b.order):
            if not block.enabled:
                continue

            # Resolve content
            content: Optional[str] = None

            if block.user_override is not None:
                # User override takes precedence
                content = block.user_override
            elif block.is_static:
                if block.content_wrapper:
                    # Static block with custom wrapper: wrapper replaces
                    # static_content entirely. {content} = persona name
                    # (the dynamic part of static blocks).
                    content = block.content_wrapper.replace(
                        "{content}", persona_name
                    )
                elif block.static_content:
                    # Static block with default content and {persona_name} interpolation
                    content = block.static_content.replace(
                        "{persona_name}", persona_name
                    )
            elif block.placeholder and block.placeholder in provider_map:
                # Provider-backed block
                content = provider_map[block.placeholder].provide(context)
                # Apply content wrapper if present and content is non-empty
                if content and block.content_wrapper:
                    content = block.content_wrapper.replace("{content}", content)

            has_content = content and content.strip()

            # Section headers are emitted even when content is empty (matching
            # template behavior where ### headers are static text). Headers only
            # vanish when the block itself is disabled.
            emitted_text: Optional[str] = None
            if block.section_header:
                if has_content:
                    joiner = "\n" if block.tight_header else "\n\n"
                    emitted_text = f"### {block.section_header}{joiner}{content.strip()}"
                else:
                    # Header only — content is empty but header stays
                    emitted_text = f"### {block.section_header}"
                parts.append(emitted_text)
            elif has_content:
                emitted_text = content.strip()
                parts.append(emitted_text)

            # NANO-045b: Record per-block content data for token counting
            block_contents.append({
                "id": block.id,
                "label": block.label,
                "section": block.section_header,
                "chars": len(emitted_text) if emitted_text else 0,
                "deferred": block.id in _DEFERRED_BLOCK_IDS,
                "content": emitted_text or "",
            })

        # NANO-045b: Stash block contents on BuildContext for pipeline
        context.block_contents = block_contents

        system_content = cleanup_formatting("\n\n".join(parts))

        # Append context injection if provided
        if context_injection:
            system_content = f"{system_content}\n\n{context_injection}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]
