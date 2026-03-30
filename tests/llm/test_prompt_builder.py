"""
Unit tests for PromptBuilder - Session 3 of NANO-014.

Tests both legacy mode (backward compatibility) and provider mode
(new template-based building).
"""

import pytest

from spindl.llm.prompt_builder import PromptBuilder
from spindl.llm.build_context import BuildContext, InputModality, Message
from spindl.llm.context_provider import ContextProvider
from spindl.llm.prompt_library import CONVERSATION_TEMPLATE
from spindl.llm.providers.registry import ProviderRegistry, create_default_registry
from spindl.llm.providers.persona_provider import (
    PersonaNameProvider,
    PersonaAppearanceProvider,
    PersonaPersonalityProvider,
    PersonaRulesProvider,
)
from spindl.llm.providers.modality_provider import (
    ModalityContextProvider,
    ModalityRulesProvider,
)
from spindl.llm.providers.input_provider import CurrentInputProvider


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def legacy_persona() -> dict:
    """Persona with system_prompt for legacy mode."""
    return {
        "id": "test",
        "name": "TestBot",
        "system_prompt": "You are TestBot, a helpful assistant.",
    }


@pytest.fixture
def structured_persona() -> dict:
    """Persona with structured fields for provider mode."""
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
def basic_context(structured_persona: dict) -> BuildContext:
    """Basic BuildContext for testing."""
    return BuildContext(
        input_content="Hello, how are you?",
        input_modality=InputModality.TEXT,
        persona=structured_persona,
    )


@pytest.fixture
def voice_context(structured_persona: dict) -> BuildContext:
    """Voice mode BuildContext."""
    return BuildContext(
        input_content="Hello",
        input_modality=InputModality.VOICE,
        persona=structured_persona,
        state_trigger="barge_in",
    )


# =============================================================================
# Constructor Tests
# =============================================================================


class TestPromptBuilderConstructor:
    """Tests for PromptBuilder initialization."""

    def test_no_providers_default(self):
        """No providers = legacy mode."""
        builder = PromptBuilder()
        assert builder._providers == []

    def test_none_providers_explicit(self):
        """Explicit None = legacy mode."""
        builder = PromptBuilder(providers=None)
        assert builder._providers == []

    def test_list_of_providers(self):
        """Accept list of ContextProviders."""
        providers = [PersonaNameProvider(), PersonaAppearanceProvider()]
        builder = PromptBuilder(providers=providers)
        assert len(builder._providers) == 2
        assert isinstance(builder._providers[0], PersonaNameProvider)
        assert isinstance(builder._providers[1], PersonaAppearanceProvider)

    def test_empty_list_providers(self):
        """Empty list = legacy mode."""
        builder = PromptBuilder(providers=[])
        assert builder._providers == []

    def test_registry_providers(self):
        """Accept ProviderRegistry."""
        registry = ProviderRegistry()
        registry.register(PersonaNameProvider())
        registry.register(PersonaAppearanceProvider())

        builder = PromptBuilder(providers=registry)
        assert len(builder._providers) == 2

    def test_default_registry(self):
        """Accept default registry with all providers."""
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)
        assert len(builder._providers) == 11  # 11 default providers (RecentHistoryProvider removed)


# =============================================================================
# Legacy Mode Tests
# =============================================================================


class TestLegacyMode:
    """Tests for backward-compatible legacy mode."""

    def test_basic_build(self, legacy_persona: dict):
        """Basic build returns [system, user] messages."""
        builder = PromptBuilder()
        messages = builder.build(legacy_persona, "Hello")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are TestBot, a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_build_with_context_injection(self, legacy_persona: dict):
        """Context injection appended to system prompt."""
        builder = PromptBuilder()
        messages = builder.build(
            legacy_persona,
            "What's the weather?",
            context_injection="[RAG] Weather is sunny today.",
        )

        assert len(messages) == 2
        assert "[RAG] Weather is sunny today." in messages[0]["content"]
        assert messages[0]["content"].endswith("[RAG] Weather is sunny today.")

    def test_missing_system_prompt_raises(self):
        """ValueError if persona missing system_prompt."""
        builder = PromptBuilder()
        bad_persona = {"id": "test", "name": "Test"}

        with pytest.raises(ValueError) as exc_info:
            builder.build(bad_persona, "Hello")

        assert "system_prompt" in str(exc_info.value)

    def test_preserves_system_prompt_exactly(self):
        """System prompt used verbatim without modification."""
        multiline_prompt = """You are a helpful assistant.

You follow these rules:
- Be concise
- Be helpful"""

        persona = {"system_prompt": multiline_prompt}
        builder = PromptBuilder()
        messages = builder.build(persona, "Hi")

        assert messages[0]["content"] == multiline_prompt

    def test_empty_user_input(self, legacy_persona: dict):
        """Empty user input is preserved."""
        builder = PromptBuilder()
        messages = builder.build(legacy_persona, "")

        assert messages[1]["content"] == ""


# =============================================================================
# Provider Mode Tests
# =============================================================================


class TestProviderMode:
    """Tests for template-based provider mode."""

    def test_basic_provider_build(self, structured_persona: dict):
        """Basic build with providers."""
        providers = [PersonaNameProvider(), PersonaAppearanceProvider()]
        builder = PromptBuilder(providers=providers)

        messages = builder.build(structured_persona, "Hello")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_persona_name_substituted(self, structured_persona: dict):
        """[PERSONA_NAME] placeholder is substituted."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        messages = builder.build(structured_persona, "Hello")

        system = messages[0]["content"]
        assert "Spindle" in system
        assert "[PERSONA_NAME]" not in system

    def test_appearance_substituted(self, structured_persona: dict):
        """[PERSONA_APPEARANCE] placeholder is substituted."""
        builder = PromptBuilder(providers=[PersonaAppearanceProvider()])
        messages = builder.build(structured_persona, "Hello")

        system = messages[0]["content"]
        assert "robot spider" in system
        assert "[PERSONA_APPEARANCE]" not in system

    def test_personality_substituted(self, structured_persona: dict):
        """[PERSONA_PERSONALITY] placeholder is substituted."""
        builder = PromptBuilder(providers=[PersonaPersonalityProvider()])
        messages = builder.build(structured_persona, "Hello")

        system = messages[0]["content"]
        assert "playful" in system
        assert "[PERSONA_PERSONALITY]" not in system

    def test_rules_substituted(self, structured_persona: dict):
        """[PERSONA_RULES] placeholder is substituted."""
        builder = PromptBuilder(providers=[PersonaRulesProvider()])
        messages = builder.build(structured_persona, "Hello")

        system = messages[0]["content"]
        assert "asterisks" in system
        assert "[PERSONA_RULES]" not in system

    def test_empty_provider_collapses(self):
        """Empty provider content collapses (placeholder removed)."""
        # Persona without appearance field
        persona = {"name": "Test"}
        builder = PromptBuilder(providers=[PersonaAppearanceProvider()])

        messages = builder.build(persona, "Hello")
        system = messages[0]["content"]

        assert "[PERSONA_APPEARANCE]" not in system

    def test_current_input_excluded(self, structured_persona: dict):
        """CurrentInputProvider is excluded - user input separate."""
        providers = [PersonaNameProvider(), CurrentInputProvider()]
        builder = PromptBuilder(providers=providers)

        messages = builder.build(structured_persona, "Hello there!")

        system = messages[0]["content"]
        # User input should NOT be in system prompt (it's separate)
        assert "Hello there!" not in system
        # But should be in user message
        assert messages[1]["content"] == "Hello there!"

    def test_rag_placeholder_preserved(self, structured_persona: dict):
        """[RAG_CONTEXT] placeholder preserved for pipeline injection (NANO-043 Phase 2)."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        messages = builder.build(structured_persona, "Hello")

        system = messages[0]["content"]
        assert "[RAG_CONTEXT]" in system

    def test_context_injection_appended(self, structured_persona: dict):
        """Context injection works in provider mode."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        messages = builder.build(
            structured_persona,
            "Hello",
            context_injection="Additional context here.",
        )

        system = messages[0]["content"]
        assert "Additional context here." in system

    def test_full_provider_set(self, structured_persona: dict):
        """Build with all default providers."""
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)

        messages = builder.build(structured_persona, "Hello Spindle!")

        system = messages[0]["content"]
        # Check key substitutions happened
        assert "Spindle" in system
        assert "robot spider" in system
        assert "playful" in system
        assert "asterisks" in system
        # No remaining placeholders
        assert "[PERSONA_NAME]" not in system
        assert "[PERSONA_APPEARANCE]" not in system
        assert "[PERSONA_PERSONALITY]" not in system
        assert "[PERSONA_RULES]" not in system
        assert "[CURRENT_INPUT]" not in system


# =============================================================================
# build_prompt() Method Tests
# =============================================================================


class TestBuildPrompt:
    """Tests for the new build_prompt() method."""

    def test_basic_substitution(self, basic_context: BuildContext):
        """Basic placeholder substitution."""
        providers = [PersonaNameProvider()]
        builder = PromptBuilder(providers=providers)

        template = "Hello, I am [PERSONA_NAME]."
        result = builder.build_prompt(template, basic_context)

        assert result == "Hello, I am Spindle."

    def test_multiple_placeholders(self, basic_context: BuildContext):
        """Multiple placeholders in template."""
        providers = [PersonaNameProvider(), PersonaAppearanceProvider()]
        builder = PromptBuilder(providers=providers)

        template = "I am [PERSONA_NAME]. [PERSONA_APPEARANCE]"
        result = builder.build_prompt(template, basic_context)

        assert "Spindle" in result
        assert "robot spider" in result
        assert "[PERSONA_NAME]" not in result
        assert "[PERSONA_APPEARANCE]" not in result

    def test_empty_content_collapses(self):
        """Empty provider content collapses placeholder."""
        persona = {"name": "Test"}  # No appearance
        context = BuildContext(input_content="Hi", persona=persona)

        builder = PromptBuilder(providers=[PersonaAppearanceProvider()])
        template = "Start [PERSONA_APPEARANCE] End"
        result = builder.build_prompt(template, context)

        assert "[PERSONA_APPEARANCE]" not in result
        assert "Start" in result
        assert "End" in result

    def test_formatting_cleanup(self, basic_context: BuildContext):
        """Triple+ newlines collapsed to double."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])

        template = "Line 1\n\n\n\nLine 2"
        result = builder.build_prompt(template, basic_context)

        assert "\n\n\n" not in result
        assert "Line 1\n\nLine 2" == result

    def test_full_template(self, basic_context: BuildContext):
        """Build with full CONVERSATION_TEMPLATE."""
        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)

        result = builder.build_prompt(CONVERSATION_TEMPLATE, basic_context)

        # Template structure preserved
        assert "### Agent" in result
        assert "### Rules" in result
        # Content substituted
        assert "Spindle" in result
        assert "robot spider" in result

    def test_current_input_included(self, basic_context: BuildContext):
        """CurrentInputProvider includes user input in template."""
        providers = [PersonaNameProvider(), CurrentInputProvider()]
        builder = PromptBuilder(providers=providers)

        template = "Agent: [PERSONA_NAME]\nInput: [CURRENT_INPUT]"
        result = builder.build_prompt(template, basic_context)

        assert "Spindle" in result
        assert "Hello, how are you?" in result

    def test_unregistered_placeholder_unchanged(self, basic_context: BuildContext):
        """Unregistered placeholders remain (no provider for them)."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])

        template = "[PERSONA_NAME] says [UNKNOWN_PLACEHOLDER]"
        result = builder.build_prompt(template, basic_context)

        assert "Spindle" in result
        # Unknown placeholder stays
        assert "[UNKNOWN_PLACEHOLDER]" in result


# =============================================================================
# Voice Mode Tests
# =============================================================================


class TestVoiceMode:
    """Tests for voice modality context."""

    def test_voice_modality_context(self, voice_context: BuildContext):
        """Voice modality context injected."""
        providers = [ModalityContextProvider()]
        builder = PromptBuilder(providers=providers)

        template = "[MODALITY_CONTEXT]"
        result = builder.build_prompt(template, voice_context)

        assert "voice conversation" in result
        assert "TTS" in result

    def test_voice_modality_rules(self, voice_context: BuildContext):
        """Voice modality rules injected."""
        providers = [ModalityRulesProvider()]
        builder = PromptBuilder(providers=providers)

        template = "[MODALITY_RULES]"
        result = builder.build_prompt(template, voice_context)

        assert "asterisks" in result
        assert "concise" in result


# =============================================================================
# HistoryInjector Compatibility Tests
# =============================================================================


class TestHistoryInjectorCompatibility:
    """Tests ensuring compatibility with HistoryInjector plugin."""

    def test_message_structure_legacy(self, legacy_persona: dict):
        """Legacy mode: [system, user] structure preserved."""
        builder = PromptBuilder()
        messages = builder.build(legacy_persona, "Hello")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_message_structure_provider(self, structured_persona: dict):
        """Provider mode: [system, user] structure preserved."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        messages = builder.build(structured_persona, "Hello")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_user_content_accessible(self, structured_persona: dict):
        """User input accessible at messages[-1]['content']."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        user_input = "This is the user's message"
        messages = builder.build(structured_persona, user_input)

        assert messages[-1]["content"] == user_input


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_persona(self):
        """Provider mode with empty persona dict."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        messages = builder.build({}, "Hello")

        # Should not crash, name falls back to "Assistant"
        assert len(messages) == 2
        assert "Assistant" in messages[0]["content"]

    def test_whitespace_only_input(self, structured_persona: dict):
        """Whitespace-only user input preserved."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        messages = builder.build(structured_persona, "   ")

        assert messages[1]["content"] == "   "

    def test_special_characters_in_input(self, structured_persona: dict):
        """Special characters in input preserved."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])
        special_input = "Hello! @#$%^&*() こんにちは 🎉"
        messages = builder.build(structured_persona, special_input)

        assert messages[1]["content"] == special_input

    def test_placeholder_like_content(self, basic_context: BuildContext):
        """Content that looks like a placeholder handled correctly."""
        # Persona with [brackets] in content
        context = basic_context.with_updates(
            persona={"name": "[ROBOT]", "description": "Has [METAL] parts"}
        )
        providers = [PersonaNameProvider(), PersonaAppearanceProvider()]
        builder = PromptBuilder(providers=providers)

        template = "[PERSONA_NAME]: [PERSONA_APPEARANCE]"
        result = builder.build_prompt(template, context)

        # Content with brackets preserved
        assert "[ROBOT]" in result
        assert "[METAL]" in result

    def test_repeated_placeholder(self, basic_context: BuildContext):
        """Placeholder appears multiple times in template."""
        builder = PromptBuilder(providers=[PersonaNameProvider()])

        template = "[PERSONA_NAME] says hi. [PERSONA_NAME] is helpful."
        result = builder.build_prompt(template, basic_context)

        # Both occurrences replaced
        assert result.count("Spindle") == 2
        assert "[PERSONA_NAME]" not in result
