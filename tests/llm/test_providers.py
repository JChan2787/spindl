"""
Tests for all context providers.

Tests cover:
- Persona providers (name, appearance, personality, rules)
- Modality providers (context, rules)
- Voice state provider
- History providers (summary, recent)
- Input provider
- Provider registry
"""

import pytest

from spindl.llm.build_context import BuildContext, InputModality, Message
from spindl.llm.providers import (
    CurrentInputProvider,
    ExampleDialogueProvider,
    ModalityContextProvider,
    ModalityRulesProvider,
    PersonaAppearanceProvider,
    PersonaNameProvider,
    PersonaPersonalityProvider,
    PersonaRulesProvider,
    ProviderRegistry,
    RecentHistoryProvider,
    ScenarioProvider,
    SummaryProvider,
    VoiceStateProvider,
    create_default_registry,
)


# =============================================================================
# Persona Provider Tests
# =============================================================================


class TestPersonaNameProvider:
    """Tests for PersonaNameProvider."""

    def test_placeholder(self):
        """Placeholder should be [PERSONA_NAME]."""
        provider = PersonaNameProvider()
        assert provider.placeholder == "[PERSONA_NAME]"

    def test_provide_with_name(self):
        """Should return persona name when present."""
        context = BuildContext(
            input_content="Hello",
            persona={"name": "Spindle"},
        )
        provider = PersonaNameProvider()
        assert provider.provide(context) == "Spindle"

    def test_provide_fallback(self):
        """Should return 'Assistant' when name is missing."""
        context = BuildContext(input_content="Hello", persona={})
        provider = PersonaNameProvider()
        assert provider.provide(context) == "Assistant"

    def test_provide_empty_persona(self):
        """Should return 'Assistant' when persona dict is empty."""
        context = BuildContext(input_content="Hello")
        provider = PersonaNameProvider()
        assert provider.provide(context) == "Assistant"


class TestPersonaAppearanceProvider:
    """Tests for PersonaAppearanceProvider."""

    def test_placeholder(self):
        """Placeholder should be [PERSONA_APPEARANCE]."""
        provider = PersonaAppearanceProvider()
        assert provider.placeholder == "[PERSONA_APPEARANCE]"

    def test_provide_with_appearance(self):
        """Should return appearance when present."""
        context = BuildContext(
            input_content="Hello",
            persona={"description": "A robot spider with eight legs."},
        )
        provider = PersonaAppearanceProvider()
        assert provider.provide(context) == "A robot spider with eight legs."

    def test_provide_collapses_when_missing(self):
        """Should return None when appearance is missing."""
        context = BuildContext(input_content="Hello", persona={"name": "Spindle"})
        provider = PersonaAppearanceProvider()
        assert provider.provide(context) is None

    def test_provide_collapses_when_whitespace(self):
        """Should return None when appearance is only whitespace."""
        context = BuildContext(
            input_content="Hello",
            persona={"description": "   \n  "},
        )
        provider = PersonaAppearanceProvider()
        assert provider.provide(context) is None

    def test_provide_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        context = BuildContext(
            input_content="Hello",
            persona={"description": "  Robot spider.  \n"},
        )
        provider = PersonaAppearanceProvider()
        assert provider.provide(context) == "Robot spider."


class TestPersonaPersonalityProvider:
    """Tests for PersonaPersonalityProvider."""

    def test_placeholder(self):
        """Placeholder should be [PERSONA_PERSONALITY]."""
        provider = PersonaPersonalityProvider()
        assert provider.placeholder == "[PERSONA_PERSONALITY]"

    def test_provide_with_personality(self):
        """Should return personality when present."""
        context = BuildContext(
            input_content="Hello",
            persona={"personality": "Helpful and playful."},
        )
        provider = PersonaPersonalityProvider()
        assert provider.provide(context) == "Helpful and playful."

    def test_provide_collapses_when_missing(self):
        """Should return None when personality is missing."""
        context = BuildContext(input_content="Hello", persona={})
        provider = PersonaPersonalityProvider()
        assert provider.provide(context) is None


class TestScenarioProvider:
    """Tests for ScenarioProvider."""

    def test_placeholder(self):
        """Placeholder should be [SCENARIO]."""
        provider = ScenarioProvider()
        assert provider.placeholder == "[SCENARIO]"

    def test_provide_with_scenario(self):
        """Should return scenario when present."""
        context = BuildContext(
            input_content="Hello",
            persona={"scenario": "A dark forest at midnight."},
        )
        provider = ScenarioProvider()
        assert provider.provide(context) == "A dark forest at midnight."

    def test_provide_collapses_when_missing(self):
        """Should return None when scenario is missing."""
        context = BuildContext(input_content="Hello", persona={})
        provider = ScenarioProvider()
        assert provider.provide(context) is None

    def test_provide_collapses_when_empty(self):
        """Should return None when scenario is empty string."""
        context = BuildContext(
            input_content="Hello",
            persona={"scenario": ""},
        )
        provider = ScenarioProvider()
        assert provider.provide(context) is None

    def test_provide_collapses_when_whitespace(self):
        """Should return None when scenario is only whitespace."""
        context = BuildContext(
            input_content="Hello",
            persona={"scenario": "   \n  "},
        )
        provider = ScenarioProvider()
        assert provider.provide(context) is None

    def test_provide_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        context = BuildContext(
            input_content="Hello",
            persona={"scenario": "  A dark forest.  \n"},
        )
        provider = ScenarioProvider()
        assert provider.provide(context) == "A dark forest."


class TestExampleDialogueProvider:
    """Tests for ExampleDialogueProvider."""

    def test_placeholder(self):
        """Placeholder should be [EXAMPLE_DIALOGUE]."""
        provider = ExampleDialogueProvider()
        assert provider.placeholder == "[EXAMPLE_DIALOGUE]"

    def test_provide_with_first_mes(self):
        """Should return first_mes content."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "first_mes": "Hello, I am Spindle!",
            },
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        assert result == "Hello, I am Spindle!"

    def test_provide_with_all_fields(self):
        """Should combine all three fields with double newlines."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "first_mes": "Hello there!",
                "alternate_greetings": ["Hi!", "Hey."],
                "mes_example": "{{char}}: How can I help?",
            },
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        assert "Hello there!" in result
        assert "Hi!" in result
        assert "Hey." in result
        assert "Spindle: How can I help?" in result
        # Fields separated by double newlines
        parts = result.split("\n\n")
        assert len(parts) == 4

    def test_provide_collapses_when_all_empty(self):
        """Should return None when all fields are empty."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "first_mes": "",
                "alternate_greetings": [],
                "mes_example": "",
            },
        )
        provider = ExampleDialogueProvider()
        assert provider.provide(context) is None

    def test_provide_collapses_when_missing(self):
        """Should return None when none of the fields are present."""
        context = BuildContext(
            input_content="Hello",
            persona={"name": "Spindle"},
        )
        provider = ExampleDialogueProvider()
        assert provider.provide(context) is None

    def test_char_substitution(self):
        """Should replace {{char}} with character name."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "first_mes": "{{char}} waves a leg.",
            },
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        assert result == "Spindle waves a leg."
        assert "{{char}}" not in result

    def test_user_substitution(self):
        """Should replace {{user}} with 'User'."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "first_mes": "Hello {{user}}, welcome!",
            },
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        assert result == "Hello User, welcome!"
        assert "{{user}}" not in result

    def test_start_tag_cleanup(self):
        """Should replace <START> tags with --- separator."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "mes_example": "<START>\n{{char}}: Hello!\n{{user}}: Hi!",
            },
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        assert "<START>" not in result
        assert "Spindle: Hello!" in result
        assert "User: Hi!" in result

    def test_start_tag_only_collapses(self):
        """mes_example with only <START> tags should not contribute."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "mes_example": "<START>",
            },
        )
        provider = ExampleDialogueProvider()
        assert provider.provide(context) is None

    def test_partial_fields(self):
        """Should work with only some fields populated."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "first_mes": "",
                "alternate_greetings": ["Greetings!"],
                "mes_example": "",
            },
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        assert result == "Greetings!"

    def test_whitespace_only_fields_collapse(self):
        """Fields with only whitespace should not contribute."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "first_mes": "   \n  ",
                "alternate_greetings": ["  ", ""],
            },
        )
        provider = ExampleDialogueProvider()
        assert provider.provide(context) is None

    def test_empty_alternate_greetings_filtered(self):
        """Empty strings in alternate_greetings list should be skipped."""
        context = BuildContext(
            input_content="Hello",
            persona={
                "name": "Spindle",
                "alternate_greetings": ["", "Hello!", "", "Hey!", ""],
            },
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        parts = result.split("\n\n")
        assert len(parts) == 2
        assert parts[0] == "Hello!"
        assert parts[1] == "Hey!"

    def test_wrappers_prefix_only(self):
        """set_wrappers prefix should appear before content."""
        context = BuildContext(
            input_content="Hello",
            persona={"name": "Spindle", "first_mes": "Hello there!"},
        )
        provider = ExampleDialogueProvider()
        provider.set_wrappers(prefix="STYLE EXAMPLES:")
        result = provider.provide(context)
        assert result.startswith("STYLE EXAMPLES:\n")
        assert "Hello there!" in result

    def test_wrappers_suffix_only(self):
        """set_wrappers suffix should appear after content."""
        context = BuildContext(
            input_content="Hello",
            persona={"name": "Spindle", "first_mes": "Hello there!"},
        )
        provider = ExampleDialogueProvider()
        provider.set_wrappers(suffix="END EXAMPLES.")
        result = provider.provide(context)
        assert result.endswith("\nEND EXAMPLES.")
        assert "Hello there!" in result

    def test_wrappers_both(self):
        """set_wrappers with both prefix and suffix wraps content."""
        context = BuildContext(
            input_content="Hello",
            persona={"name": "Spindle", "first_mes": "Hello there!"},
        )
        provider = ExampleDialogueProvider()
        provider.set_wrappers(prefix="BEGIN:", suffix="END.")
        result = provider.provide(context)
        assert result.startswith("BEGIN:\n")
        assert result.endswith("\nEND.")
        assert "Hello there!" in result

    def test_wrappers_not_applied_when_empty_content(self):
        """Wrappers should not appear when content is None (collapses)."""
        context = BuildContext(
            input_content="Hello",
            persona={"name": "Spindle"},
        )
        provider = ExampleDialogueProvider()
        provider.set_wrappers(prefix="BEGIN:", suffix="END.")
        assert provider.provide(context) is None

    def test_wrappers_default_empty(self):
        """Default wrappers are empty strings — no wrapping."""
        context = BuildContext(
            input_content="Hello",
            persona={"name": "Spindle", "first_mes": "Hello there!"},
        )
        provider = ExampleDialogueProvider()
        result = provider.provide(context)
        assert result == "Hello there!"


class TestPersonaRulesProvider:
    """Tests for PersonaRulesProvider."""

    def test_placeholder(self):
        """Placeholder should be [PERSONA_RULES]."""
        provider = PersonaRulesProvider()
        assert provider.placeholder == "[PERSONA_RULES]"

    def test_provide_with_string_rules(self):
        """Should return string rules as-is."""
        context = BuildContext(
            input_content="Hello",
            persona={"rules": "- Rule one\n- Rule two"},
        )
        provider = PersonaRulesProvider()
        assert provider.provide(context) == "- Rule one\n- Rule two"

    def test_provide_with_list_rules(self):
        """Should format list rules as bullet points."""
        context = BuildContext(
            input_content="Hello",
            persona={"rules": ["Rule one", "Rule two"]},
        )
        provider = PersonaRulesProvider()
        result = provider.provide(context)
        assert result == "- Rule one\n- Rule two"

    def test_provide_list_preserves_existing_bullets(self):
        """Should not double-bullet rules that already have bullets."""
        context = BuildContext(
            input_content="Hello",
            persona={"rules": ["- Already bulleted", "Not bulleted"]},
        )
        provider = PersonaRulesProvider()
        result = provider.provide(context)
        assert result == "- Already bulleted\n- Not bulleted"

    def test_provide_collapses_when_missing(self):
        """Should return None when rules are missing."""
        context = BuildContext(input_content="Hello", persona={})
        provider = PersonaRulesProvider()
        assert provider.provide(context) is None

    def test_provide_collapses_empty_list(self):
        """Should return None for empty rules list."""
        context = BuildContext(
            input_content="Hello",
            persona={"rules": []},
        )
        provider = PersonaRulesProvider()
        assert provider.provide(context) is None


# =============================================================================
# Modality Provider Tests
# =============================================================================


class TestModalityContextProvider:
    """Tests for ModalityContextProvider."""

    def test_placeholder(self):
        """Placeholder should be [MODALITY_CONTEXT]."""
        provider = ModalityContextProvider()
        assert provider.placeholder == "[MODALITY_CONTEXT]"

    def test_provide_voice_modality(self):
        """Should return voice context for VOICE modality."""
        context = BuildContext(
            input_content="Hello",
            input_modality=InputModality.VOICE,
        )
        provider = ModalityContextProvider()
        result = provider.provide(context)
        assert "voice conversation" in result.lower()
        assert "TTS" in result

    def test_provide_text_modality(self):
        """Should return text context for TEXT modality."""
        context = BuildContext(
            input_content="Hello",
            input_modality=InputModality.TEXT,
        )
        provider = ModalityContextProvider()
        result = provider.provide(context)
        assert "text conversation" in result.lower()


class TestModalityRulesProvider:
    """Tests for ModalityRulesProvider."""

    def test_placeholder(self):
        """Placeholder should be [MODALITY_RULES]."""
        provider = ModalityRulesProvider()
        assert provider.placeholder == "[MODALITY_RULES]"

    def test_provide_voice_rules(self):
        """Should return voice rules for VOICE modality."""
        context = BuildContext(
            input_content="Hello",
            input_modality=InputModality.VOICE,
        )
        provider = ModalityRulesProvider()
        result = provider.provide(context)
        assert "asterisks" in result.lower()
        assert "concise" in result.lower()

    def test_provide_text_rules(self):
        """Should return text rules for TEXT modality."""
        context = BuildContext(
            input_content="Hello",
            input_modality=InputModality.TEXT,
        )
        provider = ModalityRulesProvider()
        result = provider.provide(context)
        assert "formatting" in result.lower()


# =============================================================================
# Voice State Provider Tests
# =============================================================================


class TestVoiceStateProvider:
    """Tests for VoiceStateProvider."""

    def test_placeholder(self):
        """Placeholder should be [STATE_CONTEXT]."""
        provider = VoiceStateProvider()
        assert provider.placeholder == "[STATE_CONTEXT]"

    def test_provide_barge_in(self):
        """Should return barge-in injection for barge_in trigger."""
        context = BuildContext(
            input_content="Hello",
            state_trigger="barge_in",
        )
        provider = VoiceStateProvider()
        result = provider.provide(context)
        assert "interrupted" in result.lower()

    def test_provide_empty_transcription(self):
        """Should return empty transcription injection."""
        context = BuildContext(
            input_content="",
            state_trigger="empty_transcription",
        )
        provider = VoiceStateProvider()
        result = provider.provide(context)
        assert "sound" in result.lower()
        assert "words" in result.lower()

    def test_provide_error(self):
        """Should return error injection for error trigger."""
        context = BuildContext(
            input_content="Hello",
            state_trigger="error",
        )
        provider = VoiceStateProvider()
        result = provider.provide(context)
        assert "error" in result.lower()

    def test_provide_collapses_no_trigger(self):
        """Should return None when no state_trigger."""
        context = BuildContext(input_content="Hello")
        provider = VoiceStateProvider()
        assert provider.provide(context) is None

    def test_provide_collapses_normal_trigger(self):
        """Should return None for normal triggers like vad_speech_start."""
        context = BuildContext(
            input_content="Hello",
            state_trigger="vad_speech_start",
        )
        provider = VoiceStateProvider()
        assert provider.provide(context) is None


# =============================================================================
# History Provider Tests
# =============================================================================


class TestSummaryProvider:
    """Tests for SummaryProvider."""

    def test_placeholder(self):
        """Placeholder should be [CONVERSATION_SUMMARY]."""
        provider = SummaryProvider()
        assert provider.placeholder == "[CONVERSATION_SUMMARY]"

    def test_provide_with_summary(self):
        """Should return summary when present."""
        context = BuildContext(
            input_content="Hello",
            summary="User discussed weather and plans.",
        )
        provider = SummaryProvider()
        assert provider.provide(context) == "User discussed weather and plans."

    def test_provide_collapses_when_missing(self):
        """Should return None when summary is None."""
        context = BuildContext(input_content="Hello")
        provider = SummaryProvider()
        assert provider.provide(context) is None

    def test_provide_collapses_when_whitespace(self):
        """Should return None when summary is only whitespace."""
        context = BuildContext(input_content="Hello", summary="   \n  ")
        provider = SummaryProvider()
        assert provider.provide(context) is None


class TestRecentHistoryProvider:
    """Tests for RecentHistoryProvider."""

    def test_placeholder(self):
        """Placeholder should be [RECENT_HISTORY]."""
        provider = RecentHistoryProvider()
        assert provider.placeholder == "[RECENT_HISTORY]"

    def test_provide_with_messages(self):
        """Should format messages as labeled turns."""
        messages = [
            Message(role="user", content="What's the weather?"),
            Message(role="assistant", content="I don't have weather data."),
        ]
        context = BuildContext(
            input_content="Hello",
            recent_messages=messages,
            persona={"name": "Spindle"},
        )
        provider = RecentHistoryProvider()
        result = provider.provide(context)

        assert "User:" in result
        assert "What's the weather?" in result
        assert "Spindle:" in result
        assert "I don't have weather data." in result

    def test_provide_uses_persona_name(self):
        """Should use persona name for assistant messages."""
        messages = [
            Message(role="assistant", content="Hello!"),
        ]
        context = BuildContext(
            input_content="Hi",
            recent_messages=messages,
            persona={"name": "CustomBot"},
        )
        provider = RecentHistoryProvider()
        result = provider.provide(context)
        assert "CustomBot:" in result

    def test_provide_fallback_name(self):
        """Should use 'Assistant' when persona name is missing."""
        messages = [
            Message(role="assistant", content="Hello!"),
        ]
        context = BuildContext(
            input_content="Hi",
            recent_messages=messages,
            persona={},
        )
        provider = RecentHistoryProvider()
        result = provider.provide(context)
        assert "Assistant:" in result

    def test_provide_collapses_when_empty(self):
        """Should return None when no messages."""
        context = BuildContext(input_content="Hello", recent_messages=[])
        provider = RecentHistoryProvider()
        assert provider.provide(context) is None

    def test_provide_multiple_turns(self):
        """Should separate turns with double newlines."""
        messages = [
            Message(role="user", content="First"),
            Message(role="assistant", content="Second"),
            Message(role="user", content="Third"),
        ]
        context = BuildContext(
            input_content="Hello",
            recent_messages=messages,
            persona={"name": "Bot"},
        )
        provider = RecentHistoryProvider()
        result = provider.provide(context)

        # Should have two \n\n separators for 3 messages
        parts = result.split("\n\n")
        assert len(parts) == 3


# =============================================================================
# Input Provider Tests
# =============================================================================


class TestCurrentInputProvider:
    """Tests for CurrentInputProvider."""

    def test_placeholder(self):
        """Placeholder should be [CURRENT_INPUT]."""
        provider = CurrentInputProvider()
        assert provider.placeholder == "[CURRENT_INPUT]"

    def test_provide_formats_input(self):
        """Should format input with User: label."""
        context = BuildContext(input_content="What time is it?")
        provider = CurrentInputProvider()
        result = provider.provide(context)
        assert result == "User:\nWhat time is it?"

    def test_provide_strips_whitespace(self):
        """Should strip whitespace from input."""
        context = BuildContext(input_content="  Hello world  \n")
        provider = CurrentInputProvider()
        result = provider.provide(context)
        assert result == "User:\nHello world"

    def test_provide_handles_empty_input(self):
        """Should return label even for empty input."""
        context = BuildContext(input_content="")
        provider = CurrentInputProvider()
        result = provider.provide(context)
        assert result == "User:\n"


# =============================================================================
# Provider Registry Tests
# =============================================================================


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def test_register_single(self):
        """Should register single provider."""
        registry = ProviderRegistry()
        provider = PersonaNameProvider()
        registry.register(provider)

        assert len(registry) == 1
        assert list(registry)[0] is provider

    def test_register_all(self):
        """Should register multiple providers."""
        registry = ProviderRegistry()
        providers = [PersonaNameProvider(), PersonaAppearanceProvider()]
        registry.register_all(providers)

        assert len(registry) == 2

    def test_iteration_order(self):
        """Should iterate in registration order."""
        registry = ProviderRegistry()
        p1 = PersonaNameProvider()
        p2 = PersonaAppearanceProvider()
        p3 = PersonaPersonalityProvider()

        registry.register(p1)
        registry.register(p2)
        registry.register(p3)

        providers_list = list(registry)
        assert providers_list[0] is p1
        assert providers_list[1] is p2
        assert providers_list[2] is p3

    def test_providers_property_returns_copy(self):
        """Should return copy of providers list."""
        registry = ProviderRegistry()
        registry.register(PersonaNameProvider())

        providers = registry.providers
        providers.append(PersonaAppearanceProvider())

        # Original should be unaffected
        assert len(registry) == 1


class TestCreateDefaultRegistry:
    """Tests for create_default_registry factory."""

    def test_creates_registry(self):
        """Should create a ProviderRegistry instance."""
        registry = create_default_registry()
        assert isinstance(registry, ProviderRegistry)

    def test_registers_all_providers(self):
        """Should register all standard providers."""
        registry = create_default_registry()

        # Should have 11 providers total (RecentHistoryProvider removed;
        # [RECENT_HISTORY] is filled by HistoryInjector PreProcessor instead)
        assert len(registry) == 11

    def test_provider_placeholders(self):
        """Should have correct placeholders for all providers."""
        registry = create_default_registry()

        placeholders = {p.placeholder for p in registry}

        expected = {
            "[PERSONA_NAME]",
            "[PERSONA_APPEARANCE]",
            "[PERSONA_PERSONALITY]",
            "[SCENARIO]",
            "[EXAMPLE_DIALOGUE]",
            "[PERSONA_RULES]",
            "[MODALITY_CONTEXT]",
            "[MODALITY_RULES]",
            "[STATE_CONTEXT]",
            "[CONVERSATION_SUMMARY]",
            # [RECENT_HISTORY] not in registry — filled by HistoryInjector PreProcessor
            "[CURRENT_INPUT]",
        }

        assert placeholders == expected
