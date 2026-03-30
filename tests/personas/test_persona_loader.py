"""
Tests for PersonaLoader - validates both legacy and structured persona formats.

NANO-014 Session 4: Persona YAML Enhancement
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from spindl.personas.persona_loader import PersonaLoader


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_personas_dir(tmp_path):
    """Create a temporary personas directory for testing."""
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    return personas_dir


@pytest.fixture
def loader(temp_personas_dir):
    """Create a PersonaLoader with temp directory."""
    return PersonaLoader(str(temp_personas_dir))


def write_persona(personas_dir: Path, persona_id: str, data: dict):
    """Helper to write a persona YAML file."""
    persona_path = personas_dir / f"{persona_id}.yaml"
    with open(persona_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)


# =============================================================================
# Legacy Format Tests
# =============================================================================


class TestLegacyFormat:
    """Tests for legacy persona format (system_prompt only)."""

    def test_load_legacy_persona(self, temp_personas_dir, loader):
        """Load persona with only system_prompt (legacy format)."""
        write_persona(
            temp_personas_dir,
            "legacy_bot",
            {
                "id": "legacy_bot",
                "name": "Legacy Bot",
                "system_prompt": "You are a helpful assistant.",
            },
        )

        persona = loader.load("legacy_bot")

        assert persona["id"] == "legacy_bot"
        assert persona["name"] == "Legacy Bot"
        assert persona["system_prompt"] == "You are a helpful assistant."

    def test_has_legacy_prompt_true(self, temp_personas_dir, loader):
        """has_legacy_prompt returns True for legacy format."""
        write_persona(
            temp_personas_dir,
            "legacy_bot",
            {
                "id": "legacy_bot",
                "name": "Legacy Bot",
                "system_prompt": "You are a helpful assistant.",
            },
        )

        persona = loader.load("legacy_bot")
        assert loader.has_legacy_prompt(persona) is True

    def test_has_structured_fields_false_for_legacy(self, temp_personas_dir, loader):
        """has_structured_fields returns False for legacy format."""
        write_persona(
            temp_personas_dir,
            "legacy_bot",
            {
                "id": "legacy_bot",
                "name": "Legacy Bot",
                "system_prompt": "You are a helpful assistant.",
            },
        )

        persona = loader.load("legacy_bot")
        assert loader.has_structured_fields(persona) is False


# =============================================================================
# Structured Format Tests
# =============================================================================


class TestStructuredFormat:
    """Tests for structured persona format (appearance, personality, rules)."""

    def test_load_structured_persona_all_fields(self, temp_personas_dir, loader):
        """Load persona with all structured fields."""
        write_persona(
            temp_personas_dir,
            "structured_bot",
            {
                "id": "structured_bot",
                "name": "Structured Bot",
                "appearance": "A sleek chrome robot with blue LEDs.",
                "personality": "Helpful and curious.",
                "rules": [
                    "NO asterisks",
                    "Keep responses concise",
                ],
            },
        )

        persona = loader.load("structured_bot")

        assert persona["id"] == "structured_bot"
        assert persona["name"] == "Structured Bot"
        assert persona["appearance"] == "A sleek chrome robot with blue LEDs."
        assert persona["personality"] == "Helpful and curious."
        assert persona["rules"] == ["NO asterisks", "Keep responses concise"]

    def test_load_structured_persona_appearance_only(self, temp_personas_dir, loader):
        """Load persona with only appearance (partial structured)."""
        write_persona(
            temp_personas_dir,
            "partial_bot",
            {
                "id": "partial_bot",
                "name": "Partial Bot",
                "appearance": "A floating orb of light.",
            },
        )

        persona = loader.load("partial_bot")

        assert persona["id"] == "partial_bot"
        assert persona["appearance"] == "A floating orb of light."
        assert persona.get("personality") is None
        assert persona.get("rules") is None

    def test_load_structured_persona_rules_only(self, temp_personas_dir, loader):
        """Load persona with only rules (partial structured)."""
        write_persona(
            temp_personas_dir,
            "rules_only",
            {
                "id": "rules_only",
                "name": "Rules Bot",
                "rules": ["Always be helpful"],
            },
        )

        persona = loader.load("rules_only")
        assert persona["rules"] == ["Always be helpful"]

    def test_has_structured_fields_true(self, temp_personas_dir, loader):
        """has_structured_fields returns True for structured format."""
        write_persona(
            temp_personas_dir,
            "structured_bot",
            {
                "id": "structured_bot",
                "name": "Structured Bot",
                "appearance": "Shiny robot.",
            },
        )

        persona = loader.load("structured_bot")
        assert loader.has_structured_fields(persona) is True

    def test_has_legacy_prompt_false_for_structured(self, temp_personas_dir, loader):
        """has_legacy_prompt returns False for structured-only format."""
        write_persona(
            temp_personas_dir,
            "structured_bot",
            {
                "id": "structured_bot",
                "name": "Structured Bot",
                "appearance": "Shiny robot.",
            },
        )

        persona = loader.load("structured_bot")
        assert loader.has_legacy_prompt(persona) is False


# =============================================================================
# Hybrid Format Tests (Both Legacy and Structured)
# =============================================================================


class TestHybridFormat:
    """Tests for hybrid persona format (both system_prompt and structured fields)."""

    def test_load_hybrid_persona(self, temp_personas_dir, loader):
        """Load persona with both legacy and structured fields."""
        write_persona(
            temp_personas_dir,
            "hybrid_bot",
            {
                "id": "hybrid_bot",
                "name": "Hybrid Bot",
                "system_prompt": "You are Hybrid Bot, a helpful assistant.",
                "appearance": "A holographic projection.",
                "personality": "Friendly and efficient.",
                "rules": ["Be concise"],
            },
        )

        persona = loader.load("hybrid_bot")

        assert persona["system_prompt"] == "You are Hybrid Bot, a helpful assistant."
        assert persona["appearance"] == "A holographic projection."
        assert persona["personality"] == "Friendly and efficient."
        assert persona["rules"] == ["Be concise"]

    def test_has_both_formats_true(self, temp_personas_dir, loader):
        """Both helper methods return True for hybrid format."""
        write_persona(
            temp_personas_dir,
            "hybrid_bot",
            {
                "id": "hybrid_bot",
                "name": "Hybrid Bot",
                "system_prompt": "Legacy prompt here.",
                "appearance": "Structured appearance.",
            },
        )

        persona = loader.load("hybrid_bot")

        assert loader.has_legacy_prompt(persona) is True
        assert loader.has_structured_fields(persona) is True


# =============================================================================
# Validation Error Tests
# =============================================================================


class TestValidationErrors:
    """Tests for validation errors and edge cases."""

    def test_missing_id_raises_error(self, temp_personas_dir, loader):
        """Missing 'id' field raises ValueError."""
        write_persona(
            temp_personas_dir,
            "no_id",
            {
                "name": "No ID Bot",
                "system_prompt": "A bot without an id.",
            },
        )

        with pytest.raises(ValueError, match="missing required fields"):
            loader.load("no_id")

    def test_missing_name_raises_error(self, temp_personas_dir, loader):
        """Missing 'name' field raises ValueError."""
        write_persona(
            temp_personas_dir,
            "no_name",
            {
                "id": "no_name",
                "system_prompt": "A bot without a name.",
            },
        )

        with pytest.raises(ValueError, match="missing required fields"):
            loader.load("no_name")

    def test_no_content_raises_error(self, temp_personas_dir, loader):
        """Persona with no system_prompt and no structured fields raises ValueError."""
        write_persona(
            temp_personas_dir,
            "empty_bot",
            {
                "id": "empty_bot",
                "name": "Empty Bot",
                # No system_prompt, no appearance/personality/rules
            },
        )

        with pytest.raises(ValueError, match="must have either 'system_prompt'"):
            loader.load("empty_bot")

    def test_empty_system_prompt_with_no_structured_raises_error(
        self, temp_personas_dir, loader
    ):
        """Empty string system_prompt without structured fields raises error."""
        write_persona(
            temp_personas_dir,
            "empty_prompt",
            {
                "id": "empty_prompt",
                "name": "Empty Prompt Bot",
                "system_prompt": "",  # Empty string is falsy
            },
        )

        with pytest.raises(ValueError, match="must have either 'system_prompt'"):
            loader.load("empty_prompt")

    def test_empty_structured_fields_raises_error(self, temp_personas_dir, loader):
        """Empty structured fields (all falsy) raises error."""
        write_persona(
            temp_personas_dir,
            "empty_structured",
            {
                "id": "empty_structured",
                "name": "Empty Structured Bot",
                "appearance": "",
                "personality": "",
                "rules": [],
            },
        )

        with pytest.raises(ValueError, match="must have either 'system_prompt'"):
            loader.load("empty_structured")

    def test_persona_not_found_raises_error(self, loader):
        """Loading nonexistent persona raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load("nonexistent_persona")

    def test_invalid_yaml_raises_error(self, temp_personas_dir, loader):
        """Invalid YAML syntax raises ValueError."""
        persona_path = temp_personas_dir / "bad_yaml.yaml"
        with open(persona_path, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        with pytest.raises(ValueError, match="Failed to parse"):
            loader.load("bad_yaml")

    def test_non_dict_yaml_raises_error(self, temp_personas_dir, loader):
        """YAML that parses to non-dict raises ValueError."""
        persona_path = temp_personas_dir / "list_yaml.yaml"
        with open(persona_path, "w") as f:
            f.write("- item1\n- item2\n")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            loader.load("list_yaml")


# =============================================================================
# List Personas Tests
# =============================================================================


class TestListPersonas:
    """Tests for list_personas method."""

    def test_list_empty_directory(self, loader):
        """Empty personas directory returns empty list."""
        assert loader.list_personas() == []

    def test_list_multiple_personas(self, temp_personas_dir, loader):
        """List all personas in directory."""
        write_persona(
            temp_personas_dir,
            "bot_a",
            {"id": "bot_a", "name": "Bot A", "system_prompt": "A"},
        )
        write_persona(
            temp_personas_dir,
            "bot_b",
            {"id": "bot_b", "name": "Bot B", "appearance": "B"},
        )

        personas = loader.list_personas()
        assert sorted(personas) == ["bot_a", "bot_b"]

    def test_list_nonexistent_directory(self, tmp_path):
        """Nonexistent personas directory returns empty list."""
        loader = PersonaLoader(str(tmp_path / "nonexistent"))
        assert loader.list_personas() == []


# =============================================================================
# Real Spindle Persona Test
# =============================================================================


class TestSpindlePersona:
    """Test loading the actual spindle.yaml persona."""

    def test_load_spindle_from_project(self):
        """Load spindle.yaml from the actual project personas directory."""
        # Use the real project path
        loader = PersonaLoader("./personas")

        # This may fail if running from wrong directory - that's OK
        try:
            persona = loader.load("spindle")
        except FileNotFoundError:
            pytest.skip("Not running from spindl-project root")

        # Verify required fields
        assert persona["id"] == "spindle"
        assert persona["name"] == "Spindle"

        # Verify structured fields (NANO-014, migrated to description in NANO-041b)
        assert "description" in persona
        assert "purple hair" in persona["description"].lower()

        assert "personality" in persona
        assert "upbeat" in persona["personality"].lower()

        assert "rules" in persona
        assert isinstance(persona["rules"], list)
        assert len(persona["rules"]) >= 5

        # Verify legacy system_prompt still present (backward compat)
        assert "system_prompt" in persona
        assert "Spindle" in persona["system_prompt"]

        # Verify helper methods
        assert loader.has_structured_fields(persona) is True
        assert loader.has_legacy_prompt(persona) is True

    def test_spindle_with_build_context_and_providers(self):
        """Integration test: Load spindle, create BuildContext, run providers."""
        from spindl.llm.build_context import BuildContext
        from spindl.llm.providers import (
            PersonaAppearanceProvider,
            PersonaPersonalityProvider,
            PersonaRulesProvider,
        )

        loader = PersonaLoader("./personas")

        try:
            persona = loader.load("spindle")
        except FileNotFoundError:
            pytest.skip("Not running from spindl-project root")

        # Create BuildContext with loaded persona
        context = BuildContext(
            input_content="Hello Spindle!",
            persona=persona,
        )

        # Verify providers can extract structured fields
        appearance_provider = PersonaAppearanceProvider()
        result = appearance_provider.provide(context)
        assert result is not None
        assert "purple hair" in result.lower()

        personality_provider = PersonaPersonalityProvider()
        result = personality_provider.provide(context)
        assert result is not None
        assert "upbeat" in result.lower()

        rules_provider = PersonaRulesProvider()
        result = rules_provider.provide(context)
        assert result is not None
        assert "asterisks" in result.lower()  # One of the rules


# =============================================================================
# Rules Format Tests
# =============================================================================


class TestRulesFormat:
    """Tests for different rules formats in personas."""

    def test_rules_as_list(self, temp_personas_dir, loader):
        """Rules as list of strings."""
        write_persona(
            temp_personas_dir,
            "list_rules",
            {
                "id": "list_rules",
                "name": "List Rules Bot",
                "rules": [
                    "Rule one",
                    "Rule two",
                    "Rule three",
                ],
            },
        )

        persona = loader.load("list_rules")
        assert persona["rules"] == ["Rule one", "Rule two", "Rule three"]

    def test_rules_as_string(self, temp_personas_dir, loader):
        """Rules as single string (alternative format)."""
        write_persona(
            temp_personas_dir,
            "string_rules",
            {
                "id": "string_rules",
                "name": "String Rules Bot",
                "rules": "- Always be helpful\n- Never lie",
            },
        )

        persona = loader.load("string_rules")
        assert "Always be helpful" in persona["rules"]

    def test_rules_with_dashes(self, temp_personas_dir, loader):
        """Rules list items that already have dashes."""
        write_persona(
            temp_personas_dir,
            "dash_rules",
            {
                "id": "dash_rules",
                "name": "Dash Rules Bot",
                "rules": [
                    "- Already has dash",
                    "No dash here",
                ],
            },
        )

        persona = loader.load("dash_rules")
        # PersonaRulesProvider handles dash normalization, not PersonaLoader
        assert persona["rules"] == ["- Already has dash", "No dash here"]
