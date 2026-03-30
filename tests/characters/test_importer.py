"""Tests for character card importer."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from spindl.characters import CharacterImporter, ImportError, ImportResult


class TestCharacterImporter:
    """Tests for CharacterImporter class."""

    @pytest.fixture
    def temp_characters_dir(self):
        """Create temporary characters directory."""
        with TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def importer(self, temp_characters_dir):
        """Create importer with temp directory."""
        return CharacterImporter(temp_characters_dir)

    @pytest.fixture
    def valid_v2_card(self):
        """Valid ST V2 character card."""
        return {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Test Character",
                "description": "A test character for unit tests",
                "personality": "Helpful and friendly",
                "scenario": "",
                "first_mes": "Hello!",
                "mes_example": "",
                "creator_notes": "",
                "system_prompt": "You are a helpful assistant.",
                "post_history_instructions": "",
                "alternate_greetings": [],
                "tags": ["test", "unit-test"],
                "creator": "Test Suite",
                "character_version": "1.0",
                "extensions": {},
            },
        }

    @pytest.fixture
    def v2_card_with_spindl(self, valid_v2_card):
        """V2 card with spindl extensions."""
        card = valid_v2_card.copy()
        card["data"] = valid_v2_card["data"].copy()
        card["data"]["extensions"] = {
            "spindl": {
                "id": "custom_id",
                "voice": "af_bella",
                "language": "a",
                "rules": ["Be helpful", "Be concise"],
            }
        }
        return card

    @pytest.fixture
    def v2_card_with_codex(self, valid_v2_card):
        """V2 card with embedded codex."""
        card = valid_v2_card.copy()
        card["data"] = valid_v2_card["data"].copy()
        card["data"]["character_book"] = {
            "entries": [
                {
                    "keys": ["secret", "password"],
                    "content": "The secret password is 12345",
                    "enabled": True,
                    "id": 0,
                },
                {
                    "keys": ["location"],
                    "content": "The character lives in a castle",
                    "enabled": True,
                    "id": 1,
                },
            ]
        }
        return card

    # === Basic Import Tests ===

    def test_import_valid_v2_dict(self, importer, valid_v2_card, temp_characters_dir):
        """Import valid V2 card from dict."""
        result = importer.import_dict(valid_v2_card)

        assert isinstance(result, ImportResult)
        assert result.character_id == "test_character"
        assert result.card.data.name == "Test Character"
        assert not result.was_overwrite

        # Verify file was created
        card_path = Path(temp_characters_dir) / "test_character" / "card.json"
        assert card_path.exists()

    def test_import_valid_v2_json_string(self, importer, valid_v2_card):
        """Import valid V2 card from JSON string."""
        json_str = json.dumps(valid_v2_card)
        result = importer.import_json(json_str)

        assert result.character_id == "test_character"
        assert result.card.data.name == "Test Character"

    def test_import_from_file(self, importer, valid_v2_card, temp_characters_dir):
        """Import from JSON file."""
        # Create temp file
        file_path = Path(temp_characters_dir) / "import_test.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(valid_v2_card, f)

        result = importer.import_file(file_path)

        assert result.character_id == "test_character"

    # === ID Resolution Tests ===

    def test_id_from_spindl_extension(self, importer, v2_card_with_spindl):
        """ID is extracted from spindl extension."""
        result = importer.import_dict(v2_card_with_spindl)
        assert result.character_id == "custom_id"

    def test_id_override(self, importer, valid_v2_card):
        """Override ID with explicit parameter."""
        result = importer.import_dict(valid_v2_card, character_id="my_custom_id")
        assert result.character_id == "my_custom_id"

    def test_id_sanitization(self, importer):
        """IDs are sanitized properly."""
        card = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Test Character!!! With Spaces & Special Ch@rs",
                "description": "",
                "personality": "",
                "scenario": "",
                "first_mes": "",
                "mes_example": "",
                "extensions": {},
            },
        }
        result = importer.import_dict(card)
        # Should be lowercase, spaces to underscores, special chars removed, multiple underscores collapsed
        assert result.character_id == "test_character_with_spaces_special_chrs"

    # === spindl Extensions Tests ===

    def test_spindl_id_ensured(self, importer, valid_v2_card, temp_characters_dir):
        """spindl.id is set in the saved card."""
        result = importer.import_dict(valid_v2_card)

        # Read back the saved card
        card_path = Path(temp_characters_dir) / result.character_id / "card.json"
        with open(card_path) as f:
            saved = json.load(f)

        assert "spindl" in saved["data"]["extensions"]
        assert saved["data"]["extensions"]["spindl"]["id"] == "test_character"

    def test_preserves_spindl_extensions(self, importer, v2_card_with_spindl, temp_characters_dir):
        """Existing spindl extensions are preserved."""
        result = importer.import_dict(v2_card_with_spindl)

        card_path = Path(temp_characters_dir) / result.character_id / "card.json"
        with open(card_path) as f:
            saved = json.load(f)

        nano = saved["data"]["extensions"]["spindl"]
        assert nano["voice"] == "af_bella"
        assert nano["language"] == "a"
        assert "Be helpful" in nano["rules"]

    # === Codex/Character Book Tests ===

    def test_codex_preserved(self, importer, v2_card_with_codex, temp_characters_dir):
        """Character book entries are preserved."""
        result = importer.import_dict(v2_card_with_codex)

        card_path = Path(temp_characters_dir) / result.character_id / "card.json"
        with open(card_path) as f:
            saved = json.load(f)

        assert "character_book" in saved["data"]
        assert len(saved["data"]["character_book"]["entries"]) == 2

    # === Conflict Handling Tests ===

    def test_conflict_raises_error(self, importer, valid_v2_card):
        """Importing duplicate raises FileExistsError."""
        importer.import_dict(valid_v2_card)

        with pytest.raises(FileExistsError) as exc_info:
            importer.import_dict(valid_v2_card)

        assert "already exists" in str(exc_info.value)

    def test_overwrite_succeeds(self, importer, valid_v2_card):
        """Overwrite replaces existing character."""
        result1 = importer.import_dict(valid_v2_card)

        # Modify and reimport
        modified = valid_v2_card.copy()
        modified["data"] = valid_v2_card["data"].copy()
        modified["data"]["description"] = "Modified description"

        result2 = importer.import_dict(modified, overwrite=True)

        assert result2.character_id == result1.character_id
        assert result2.card.data.description == "Modified description"

    # === V1 Upgrade Tests ===

    def test_upgrade_v1_card(self, importer):
        """V1 cards are upgraded to V2 format."""
        v1_card = {
            "name": "V1 Character",
            "description": "A V1 format character",
            "personality": "Friendly",
            "scenario": "",
            "first_mes": "Hi there!",
            "mes_example": "",
        }

        result = importer.import_dict(v1_card)

        assert result.card.spec == "chara_card_v2"
        assert result.card.data.name == "V1 Character"
        assert result.card.data.personality == "Friendly"

    def test_upgrade_partial_v2(self, importer):
        """Cards with partial V2 structure are handled."""
        partial = {
            "spec": "chara_card_v2",
            # Missing spec_version
            "data": {
                "name": "Partial V2",
                "description": "",
                "personality": "",
                "scenario": "",
                "first_mes": "",
                "mes_example": "",
                # Missing extensions
            },
        }

        result = importer.import_dict(partial)

        assert result.card.spec == "chara_card_v2"
        assert result.card.spec_version == "2.0"
        assert result.card.data.extensions == {"spindl": {"id": "partial_v2"}}

    # === Validation Tests ===

    def test_missing_name_raises_error(self, importer):
        """Cards without name raise ImportError."""
        invalid = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "description": "No name here",
            },
        }

        with pytest.raises(ImportError) as exc_info:
            importer.import_dict(invalid)

        assert "name" in str(exc_info.value).lower()

    def test_missing_data_block_raises_error(self, importer):
        """Cards without data block raise ImportError."""
        invalid = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
        }

        with pytest.raises(ImportError) as exc_info:
            importer.import_dict(invalid)

        assert "data" in str(exc_info.value).lower()

    def test_invalid_json_raises_error(self, importer):
        """Invalid JSON raises ImportError."""
        with pytest.raises(ImportError) as exc_info:
            importer.import_json("not valid json {{{")

        assert "invalid json" in str(exc_info.value).lower()

    def test_file_not_found_raises_error(self, importer):
        """Missing file raises ImportError."""
        with pytest.raises(ImportError) as exc_info:
            importer.import_file("/nonexistent/path.json")

        assert "not found" in str(exc_info.value).lower()

    # === Validation Only Tests ===

    def test_validate_only_valid(self, importer, valid_v2_card):
        """Validate returns valid result for good card."""
        result = importer.validate_only(valid_v2_card)

        assert result.valid
        assert result.card is not None
        assert result.card.data.name == "Test Character"
        assert len(result.errors) == 0

    def test_validate_only_invalid(self, importer):
        """Validate returns errors for bad card."""
        invalid = {"spec": "chara_card_v2", "data": {}}

        result = importer.validate_only(invalid)

        assert not result.valid
        assert result.card is None
        assert len(result.errors) > 0

    def test_validate_only_warnings(self, importer):
        """Validate returns warnings for suboptimal cards."""
        minimal = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Minimal",
                "description": "",
                "personality": "",
                "scenario": "",
                "first_mes": "",
                "mes_example": "",
                "extensions": {},
            },
        }

        result = importer.validate_only(minimal)

        assert result.valid
        assert len(result.warnings) > 0  # Should warn about missing personality
