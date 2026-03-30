"""
Tests for CharacterCard model — migration logic.

NANO-041b: Verifies that to_persona_dict() correctly sources the
'description' key from data.description with fallback to nano.appearance.

NANO-054a: Verifies tts_voice_config migration from bare voice/language fields.
"""

import pytest

from spindl.characters.models import (
    CharacterCard,
    CharacterCardData,
    SpindlExtensions,
)


class TestDescriptionMigration:
    """Tests for description migration in to_persona_dict()."""

    def test_data_description_becomes_persona_description(self):
        """data.description should map to persona dict 'description' key."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                description="Tall and dark.",
                extensions={
                    "spindl": {"id": "test"}
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["description"] == "Tall and dark."

    def test_nano_appearance_fallback(self):
        """nano.appearance should fall back to 'description' key when data.description is empty."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                description="",
                extensions={
                    "spindl": {
                        "id": "test",
                        "appearance": "A robot spider.",
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["description"] == "A robot spider."

    def test_data_description_wins_over_nano_appearance(self):
        """When both are populated, data.description takes precedence."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                description="The canonical description.",
                extensions={
                    "spindl": {
                        "id": "test",
                        "appearance": "The old appearance.",
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["description"] == "The canonical description."
        assert "appearance" not in persona

    def test_neither_populated(self):
        """When neither is populated, no 'description' key in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                description="",
                extensions={
                    "spindl": {
                        "id": "test",
                        "appearance": "",
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        assert "description" not in persona
        assert "appearance" not in persona

    def test_no_nano_extensions(self):
        """Card with data.description but no nano extensions still works."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                description="Has a description.",
            )
        )
        persona = card.to_persona_dict()
        assert persona["description"] == "Has a description."

    def test_appearance_key_no_longer_emitted(self):
        """The 'appearance' key should never appear in persona dict output."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                description="",
                extensions={
                    "spindl": {
                        "id": "test",
                        "appearance": "Spider form.",
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        # Should use 'description' key, not 'appearance'
        assert "appearance" not in persona
        assert persona["description"] == "Spider form."


class TestScenarioInPersonaDict:
    """Tests for scenario field in to_persona_dict()."""

    def test_scenario_included_when_populated(self):
        """data.scenario should map to persona dict 'scenario' key."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                scenario="A dark forest at midnight.",
                extensions={
                    "spindl": {"id": "test"}
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["scenario"] == "A dark forest at midnight."

    def test_scenario_excluded_when_empty(self):
        """Empty scenario should not appear in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                scenario="",
                extensions={
                    "spindl": {"id": "test"}
                },
            )
        )
        persona = card.to_persona_dict()
        assert "scenario" not in persona

    def test_scenario_excluded_when_default(self):
        """Default scenario (not set) should not appear in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={
                    "spindl": {"id": "test"}
                },
            )
        )
        persona = card.to_persona_dict()
        assert "scenario" not in persona


class TestExampleDialogueInPersonaDict:
    """Tests for example dialogue fields in to_persona_dict()."""

    def test_first_mes_included(self):
        """first_mes should map to persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                first_mes="Hello there!",
            )
        )
        persona = card.to_persona_dict()
        assert persona["first_mes"] == "Hello there!"

    def test_mes_example_included(self):
        """mes_example should map to persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                mes_example="<START>\n{{char}}: Hi!",
            )
        )
        persona = card.to_persona_dict()
        assert persona["mes_example"] == "<START>\n{{char}}: Hi!"

    def test_alternate_greetings_included(self):
        """alternate_greetings should map to persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                alternate_greetings=["Hey!", "Yo!"],
            )
        )
        persona = card.to_persona_dict()
        assert persona["alternate_greetings"] == ["Hey!", "Yo!"]

    def test_empty_fields_excluded(self):
        """Empty example dialogue fields should not appear in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                first_mes="",
                mes_example="",
            )
        )
        persona = card.to_persona_dict()
        assert "first_mes" not in persona
        assert "mes_example" not in persona
        assert "alternate_greetings" not in persona

    def test_empty_alternate_greetings_excluded(self):
        """Empty alternate_greetings list should not appear in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                alternate_greetings=[],
            )
        )
        persona = card.to_persona_dict()
        assert "alternate_greetings" not in persona


class TestFromPersonaDict:
    """Tests for from_persona_dict() with description field."""

    def test_description_maps_to_data_description(self):
        """'description' in persona dict should map to data.description."""
        card = CharacterCard.from_persona_dict({
            "name": "Test",
            "description": "Tall and dark.",
        })
        assert card.data.description == "Tall and dark."

    def test_missing_description_defaults_empty(self):
        """Missing description should default to empty string."""
        card = CharacterCard.from_persona_dict({
            "name": "Test",
            "personality": "Friendly.",
        })
        assert card.data.description == ""


class TestTTSVoiceConfigMigration:
    """Tests for tts_voice_config migration in to_persona_dict() (NANO-054a)."""

    def test_tts_voice_config_emitted_when_present(self):
        """Card with tts_voice_config should emit it directly in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={
                    "spindl": {
                        "id": "test",
                        "tts_voice_config": {"voice": "ryan", "speaker": "Ryan"},
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["tts_voice_config"] == {"voice": "ryan", "speaker": "Ryan"}

    def test_legacy_voice_language_migrated(self):
        """Card with bare voice/language (no tts_voice_config) should construct tts_voice_config."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={
                    "spindl": {
                        "id": "test",
                        "voice": "af_bella",
                        "language": "a",
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["tts_voice_config"] == {"voice": "af_bella", "language": "a"}
        # Bare fields should NOT appear at top level
        assert "voice" not in persona
        assert "language" not in persona

    def test_tts_voice_config_wins_over_bare_fields(self):
        """When both tts_voice_config and bare fields exist, tts_voice_config wins."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={
                    "spindl": {
                        "id": "test",
                        "voice": "af_bella",
                        "language": "a",
                        "tts_voice_config": {"voice": "ryan", "language": "en"},
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["tts_voice_config"] == {"voice": "ryan", "language": "en"}

    def test_no_tts_config_when_no_voice_no_language(self):
        """Card with neither voice nor language should not emit tts_voice_config."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={
                    "spindl": {"id": "test"}
                },
            )
        )
        persona = card.to_persona_dict()
        assert "tts_voice_config" not in persona

    def test_bare_voice_only_migrated(self):
        """Card with only voice (no language) should still construct tts_voice_config."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={
                    "spindl": {
                        "id": "test",
                        "voice": "af_bella",
                    }
                },
            )
        )
        persona = card.to_persona_dict()
        assert persona["tts_voice_config"] == {"voice": "af_bella"}

    def test_from_persona_dict_preserves_tts_voice_config(self):
        """Round-trip: from_persona_dict() should preserve tts_voice_config."""
        card = CharacterCard.from_persona_dict({
            "name": "Test",
            "tts_voice_config": {"voice": "ryan", "speaker": "Ryan"},
        })
        assert card.data.spindl is not None
        assert card.data.spindl.tts_voice_config == {"voice": "ryan", "speaker": "Ryan"}


class TestAvatarVrmField:
    """Tests for avatar_vrm field in SpindlExtensions (NANO-097)."""

    def test_avatar_vrm_in_to_persona_dict(self):
        """avatar_vrm should appear in persona dict when set."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test", "avatar_vrm": "model.vrm"}},
            )
        )
        persona = card.to_persona_dict()
        assert persona["avatar_vrm"] == "model.vrm"

    def test_avatar_vrm_absent_when_not_set(self):
        """avatar_vrm should not appear in persona dict when not set."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test"}},
            )
        )
        persona = card.to_persona_dict()
        assert "avatar_vrm" not in persona

    def test_avatar_vrm_roundtrip_from_persona_dict(self):
        """avatar_vrm should survive persona dict round-trip."""
        card = CharacterCard.from_persona_dict({
            "name": "Test",
            "id": "test",
            "avatar_vrm": "model.vrm",
        })
        persona = card.to_persona_dict()
        assert persona["avatar_vrm"] == "model.vrm"

    def test_avatar_vrm_none_excluded(self):
        """Explicitly None avatar_vrm should not appear in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test", "avatar_vrm": None}},
            )
        )
        persona = card.to_persona_dict()
        assert "avatar_vrm" not in persona

    def test_avatar_vrm_parsed_from_extensions(self):
        """avatar_vrm should be accessible via card.data.spindl."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test", "avatar_vrm": "avatar.vrm"}},
            )
        )
        assert card.data.spindl is not None
        assert card.data.spindl.avatar_vrm == "avatar.vrm"


class TestAvatarExpressionsField:
    """Tests for avatar_expressions field in SpindlExtensions (NANO-098 Session 2 fix)."""

    def test_avatar_expressions_in_to_persona_dict(self):
        """avatar_expressions should appear in persona dict when set."""
        composites = {"surprised": {"aa": 0.4, "happy": 0.3}}
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test", "avatar_expressions": composites}},
            )
        )
        persona = card.to_persona_dict()
        assert persona["avatar_expressions"] == composites

    def test_avatar_expressions_absent_when_not_set(self):
        """avatar_expressions should not appear in persona dict when not set."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test"}},
            )
        )
        persona = card.to_persona_dict()
        assert "avatar_expressions" not in persona

    def test_avatar_expressions_roundtrip(self):
        """avatar_expressions should survive persona dict round-trip."""
        composites = {"surprised": {"oh": 0.5, "ee": 0.2}}
        card = CharacterCard.from_persona_dict({
            "name": "Test",
            "id": "test",
            "avatar_expressions": composites,
        })
        persona = card.to_persona_dict()
        assert persona["avatar_expressions"] == composites


class TestAvatarAnimationsField:
    """Tests for avatar_animations field in SpindlExtensions (NANO-098 Session 3)."""

    SAMPLE_CONFIG = {
        "default": "Breathing Idle",
        "emotions": {
            "amused": {"threshold": 0.75, "clip": "Happy"},
            "melancholy": {"threshold": 0.8, "clip": "Sad Idle"},
        },
    }

    def test_avatar_animations_in_to_persona_dict(self):
        """avatar_animations should appear in persona dict when set."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test", "avatar_animations": self.SAMPLE_CONFIG}},
            )
        )
        persona = card.to_persona_dict()
        assert persona["avatar_animations"] == self.SAMPLE_CONFIG

    def test_avatar_animations_absent_when_not_set(self):
        """avatar_animations should not appear in persona dict when not set."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test"}},
            )
        )
        persona = card.to_persona_dict()
        assert "avatar_animations" not in persona

    def test_avatar_animations_roundtrip(self):
        """avatar_animations should survive persona dict round-trip."""
        card = CharacterCard.from_persona_dict({
            "name": "Test",
            "id": "test",
            "avatar_animations": self.SAMPLE_CONFIG,
        })
        persona = card.to_persona_dict()
        assert persona["avatar_animations"] == self.SAMPLE_CONFIG

    def test_avatar_animations_none_excluded(self):
        """Explicitly None avatar_animations should not appear in persona dict."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test", "avatar_animations": None}},
            )
        )
        persona = card.to_persona_dict()
        assert "avatar_animations" not in persona

    def test_avatar_animations_parsed_from_extensions(self):
        """avatar_animations should be accessible via card.data.spindl."""
        card = CharacterCard(
            data=CharacterCardData(
                name="Test",
                extensions={"spindl": {"id": "test", "avatar_animations": self.SAMPLE_CONFIG}},
            )
        )
        assert card.data.spindl is not None
        assert card.data.spindl.avatar_animations == self.SAMPLE_CONFIG
