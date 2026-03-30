"""Tests for character card exporter."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from spindl.characters import (
    CharacterExporter,
    CharacterImporter,
    CharacterLoader,
    CharacterCard,
    ExportError,
)


class TestCharacterExporter:
    """Tests for CharacterExporter class."""

    @pytest.fixture
    def temp_characters_dir(self):
        """Create temporary characters directory."""
        with TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_card(self):
        """Sample character card for testing."""
        return {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": "Export Test",
                "description": "A character for export testing",
                "personality": "Friendly and helpful",
                "scenario": "",
                "first_mes": "Hello!",
                "mes_example": "",
                "creator_notes": "Test notes",
                "system_prompt": "You are a helpful assistant.",
                "post_history_instructions": "",
                "alternate_greetings": ["Hi there!"],
                "tags": ["test"],
                "creator": "Test Suite",
                "character_version": "1.0",
                "extensions": {
                    "spindl": {
                        "id": "export_test",
                        "voice": "af_bella",
                        "language": "a",
                        "rules": ["Be helpful"],
                        "generation": {"temperature": 0.8},
                    }
                },
                "character_book": {
                    "entries": [
                        {
                            "keys": ["secret"],
                            "content": "Secret content",
                            "enabled": True,
                            "id": 0,
                        }
                    ]
                },
            },
        }

    @pytest.fixture
    def exporter_with_character(self, temp_characters_dir, sample_card):
        """Create exporter with a test character already saved."""
        # Save the character using importer
        importer = CharacterImporter(temp_characters_dir)
        result = importer.import_dict(sample_card)

        exporter = CharacterExporter(temp_characters_dir)
        return exporter, result.character_id

    # === Basic Export Tests ===

    def test_export_to_dict(self, exporter_with_character):
        """Export character to dict."""
        exporter, char_id = exporter_with_character

        result = exporter.export_to_dict(char_id)

        assert result["spec"] == "chara_card_v2"
        assert result["data"]["name"] == "Export Test"

    def test_export_to_json(self, exporter_with_character):
        """Export character to JSON string."""
        exporter, char_id = exporter_with_character

        json_str = exporter.export_to_json(char_id)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["spec"] == "chara_card_v2"
        assert data["data"]["name"] == "Export Test"

    def test_export_to_json_pretty(self, exporter_with_character):
        """Export with pretty printing."""
        exporter, char_id = exporter_with_character

        json_str = exporter.export_to_json(char_id, pretty=True)

        # Pretty printed JSON has newlines and indentation
        assert "\n" in json_str
        assert "  " in json_str  # Indentation

    def test_export_to_json_compact(self, exporter_with_character):
        """Export without pretty printing."""
        exporter, char_id = exporter_with_character

        json_str = exporter.export_to_json(char_id, pretty=False)

        # Compact JSON has no unnecessary whitespace
        assert "\n" not in json_str

    def test_export_to_file(self, exporter_with_character, temp_characters_dir):
        """Export character to file."""
        exporter, char_id = exporter_with_character
        output_path = Path(temp_characters_dir) / "exports" / "test.json"

        result_path = exporter.export_to_file(char_id, output_path)

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert data["data"]["name"] == "Export Test"

    # === Extension Filtering Tests ===

    def test_export_includes_spindl_by_default(self, exporter_with_character):
        """spindl extensions are included by default."""
        exporter, char_id = exporter_with_character

        result = exporter.export_to_dict(char_id)

        assert "spindl" in result["data"]["extensions"]
        assert result["data"]["extensions"]["spindl"]["voice"] == "af_bella"

    def test_export_excludes_spindl(self, exporter_with_character):
        """Can exclude spindl extensions."""
        exporter, char_id = exporter_with_character

        result = exporter.export_to_dict(char_id, include_spindl=False)

        # Extensions should still exist (per ST spec) but be empty
        assert result["data"]["extensions"] == {}

    def test_export_includes_codex_by_default(self, exporter_with_character):
        """Codex/character_book is included by default."""
        exporter, char_id = exporter_with_character

        result = exporter.export_to_dict(char_id)

        assert "character_book" in result["data"]
        assert len(result["data"]["character_book"]["entries"]) == 1

    def test_export_excludes_codex(self, exporter_with_character):
        """Can exclude codex entries."""
        exporter, char_id = exporter_with_character

        result = exporter.export_to_dict(char_id, include_codex=False)

        assert "character_book" not in result["data"]

    def test_export_pure_st_compatible(self, exporter_with_character):
        """Export for pure ST compatibility (no spindl, with codex)."""
        exporter, char_id = exporter_with_character

        result = exporter.export_to_dict(char_id, include_spindl=False)

        # Should have spec fields
        assert result["spec"] == "chara_card_v2"
        assert result["spec_version"] == "2.0"

        # Should have data fields
        assert result["data"]["name"] == "Export Test"
        assert result["data"]["personality"] == "Friendly and helpful"

        # Should have empty extensions (ST spec requires it to exist)
        assert result["data"]["extensions"] == {}

        # Should still have character_book
        assert "character_book" in result["data"]

    # === Error Handling Tests ===

    def test_export_nonexistent_character(self, temp_characters_dir):
        """Exporting nonexistent character raises ExportError."""
        exporter = CharacterExporter(temp_characters_dir)

        with pytest.raises(ExportError) as exc_info:
            exporter.export_to_dict("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    # === Filename Generation Tests ===

    def test_get_export_filename(self, exporter_with_character):
        """Generated filename includes character ID."""
        exporter, char_id = exporter_with_character

        filename = exporter.get_export_filename(char_id)

        assert char_id in filename
        assert filename.endswith(".json")
        assert "v2" in filename

    # === Batch Export Tests ===

    def test_batch_export(self, temp_characters_dir, sample_card):
        """Batch export multiple characters."""
        importer = CharacterImporter(temp_characters_dir)
        exporter = CharacterExporter(temp_characters_dir)

        # Create multiple characters
        char_ids = []
        for i in range(3):
            card = sample_card.copy()
            card["data"] = sample_card["data"].copy()
            card["data"]["name"] = f"Character {i}"
            card["data"]["extensions"] = {"spindl": {"id": f"char_{i}"}}

            result = importer.import_dict(card)
            char_ids.append(result.character_id)

        # Batch export
        output_dir = Path(temp_characters_dir) / "batch_export"
        result = exporter.batch_export(output_dir)

        assert result.success_count == 3
        assert result.failure_count == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.json"))) == 3

    def test_batch_export_specific_ids(self, temp_characters_dir, sample_card):
        """Batch export only specific characters."""
        importer = CharacterImporter(temp_characters_dir)
        exporter = CharacterExporter(temp_characters_dir)

        # Create multiple characters
        for i in range(3):
            card = sample_card.copy()
            card["data"] = sample_card["data"].copy()
            card["data"]["name"] = f"Character {i}"
            card["data"]["extensions"] = {"spindl": {"id": f"char_{i}"}}
            importer.import_dict(card)

        # Export only specific ones
        output_dir = Path(temp_characters_dir) / "partial_export"
        result = exporter.batch_export(output_dir, character_ids=["char_0", "char_2"])

        assert result.success_count == 2
        assert len(list(output_dir.glob("*.json"))) == 2

    # === Roundtrip Tests ===

    def test_import_export_roundtrip(self, temp_characters_dir, sample_card):
        """Exported card can be reimported with same data."""
        importer = CharacterImporter(temp_characters_dir)
        exporter = CharacterExporter(temp_characters_dir)

        # Import original
        import_result = importer.import_dict(sample_card)

        # Export
        exported_json = exporter.export_to_json(import_result.character_id)

        # Create new importer for different dir
        with TemporaryDirectory() as tmpdir2:
            importer2 = CharacterImporter(tmpdir2)
            reimport_result = importer2.import_json(exported_json)

            # Compare
            assert reimport_result.card.data.name == "Export Test"
            assert reimport_result.card.data.personality == "Friendly and helpful"

            # spindl extensions preserved
            nano = reimport_result.card.data.spindl
            assert nano.voice == "af_bella"
            assert nano.rules == ["Be helpful"]

            # Codex preserved
            book = reimport_result.card.data.character_book
            assert len(book.entries) == 1
            assert book.entries[0].keys == ["secret"]
