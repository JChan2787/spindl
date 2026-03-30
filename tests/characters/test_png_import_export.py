"""Tests for PNG character card import/export (NANO-101)."""

import base64
import json
import struct
import zlib
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from spindl.characters import (
    CharacterImporter,
    CharacterExporter,
    CharacterLoader,
    ImportError,
    extract_chara_from_png,
    embed_chara_in_png,
)


# ============================================
# Helpers: Minimal PNG construction
# ============================================


def _make_minimal_png() -> bytes:
    """Create a minimal valid 1x1 red PNG."""
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR: 1x1, 8-bit RGB
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr_chunk = (
        struct.pack(">I", len(ihdr_data))
        + b"IHDR"
        + ihdr_data
        + struct.pack(">I", ihdr_crc)
    )

    # IDAT: 1x1 red pixel (filter byte 0, then RGB)
    raw_row = b"\x00\xff\x00\x00"  # filter=None, R=255, G=0, B=0
    compressed = zlib.compress(raw_row)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat_chunk = (
        struct.pack(">I", len(compressed))
        + b"IDAT"
        + compressed
        + struct.pack(">I", idat_crc)
    )

    # IEND
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    return signature + ihdr_chunk + idat_chunk + iend_chunk


def _make_png_with_chara(card_dict: dict) -> bytes:
    """Create a minimal PNG with an embedded 'chara' tEXt chunk."""
    json_str = json.dumps(card_dict)
    b64_data = base64.b64encode(json_str.encode("utf-8"))
    return embed_chara_in_png(_make_minimal_png(), json_str)


VALID_V2_CARD = {
    "spec": "chara_card_v2",
    "spec_version": "2.0",
    "data": {
        "name": "PNG Test Character",
        "description": "A character embedded in a PNG",
        "personality": "Helpful",
        "scenario": "",
        "first_mes": "Hello from the PNG!",
        "mes_example": "",
        "creator_notes": "",
        "system_prompt": "You are a test character.",
        "post_history_instructions": "",
        "alternate_greetings": [],
        "tags": ["png-test"],
        "creator": "Test Suite",
        "character_version": "1.0",
        "extensions": {},
    },
}

VALID_V2_CARD_WITH_CODEX = {
    **VALID_V2_CARD,
    "data": {
        **VALID_V2_CARD["data"],
        "character_book": {
            "entries": [
                {
                    "keys": ["secret"],
                    "content": "The secret is embedded in PNG",
                    "enabled": True,
                    "id": 0,
                },
            ],
        },
    },
}


# ============================================
# PNG tEXt Extraction Tests
# ============================================


class TestExtractCharaFromPng:
    """Tests for extract_chara_from_png()."""

    def test_extract_valid_chara(self):
        """Extract character JSON from a PNG with a chara tEXt chunk."""
        png = _make_png_with_chara(VALID_V2_CARD)
        result = extract_chara_from_png(png)

        assert result is not None
        parsed = json.loads(result)
        assert parsed["data"]["name"] == "PNG Test Character"

    def test_extract_preserves_codex(self):
        """Character book entries survive extraction."""
        png = _make_png_with_chara(VALID_V2_CARD_WITH_CODEX)
        result = extract_chara_from_png(png)

        parsed = json.loads(result)
        assert "character_book" in parsed["data"]
        assert len(parsed["data"]["character_book"]["entries"]) == 1
        assert parsed["data"]["character_book"]["entries"][0]["content"] == "The secret is embedded in PNG"

    def test_extract_returns_none_for_plain_png(self):
        """Returns None for a PNG without character data."""
        png = _make_minimal_png()
        assert extract_chara_from_png(png) is None

    def test_extract_returns_none_for_non_png(self):
        """Returns None for non-PNG data."""
        assert extract_chara_from_png(b"not a png") is None
        assert extract_chara_from_png(b"") is None

    def test_extract_returns_none_for_jpeg(self):
        """Returns None for JPEG data (common user error)."""
        jpeg_sig = b"\xff\xd8\xff\xe0"
        assert extract_chara_from_png(jpeg_sig + b"\x00" * 100) is None


# ============================================
# PNG tEXt Embedding Tests
# ============================================


class TestEmbedCharaInPng:
    """Tests for embed_chara_in_png()."""

    def test_embed_creates_valid_png(self):
        """Embedding produces valid PNG bytes."""
        base_png = _make_minimal_png()
        result = embed_chara_in_png(base_png, json.dumps(VALID_V2_CARD))

        assert result[:8] == b"\x89PNG\r\n\x1a\n"
        # Should end with IEND chunk
        assert result[-12:-8] == b"\x00\x00\x00\x00"  # IEND has 0 length
        assert result[-8:-4] == b"IEND"

    def test_roundtrip(self):
        """Embed then extract produces identical JSON."""
        json_str = json.dumps(VALID_V2_CARD)
        base_png = _make_minimal_png()

        embedded = embed_chara_in_png(base_png, json_str)
        extracted = extract_chara_from_png(embedded)

        assert extracted is not None
        assert json.loads(extracted) == VALID_V2_CARD

    def test_replaces_existing_chara(self):
        """Embedding into a PNG that already has chara replaces it."""
        # First embed
        png_v1 = embed_chara_in_png(_make_minimal_png(), json.dumps({"version": 1}))

        # Second embed with different data
        png_v2 = embed_chara_in_png(png_v1, json.dumps(VALID_V2_CARD))

        extracted = extract_chara_from_png(png_v2)
        parsed = json.loads(extracted)
        assert parsed["data"]["name"] == "PNG Test Character"

        # Should only have one chara chunk (not both)
        count = 0
        offset = 8
        while offset < len(png_v2):
            if offset + 8 > len(png_v2):
                break
            length = struct.unpack(">I", png_v2[offset : offset + 4])[0]
            chunk_type = png_v2[offset + 4 : offset + 8]
            if chunk_type == b"tEXt":
                chunk_data = png_v2[offset + 8 : offset + 8 + length]
                null_idx = chunk_data.find(b"\x00")
                if null_idx != -1 and chunk_data[:null_idx] == b"chara":
                    count += 1
            if chunk_type == b"IEND":
                break
            offset += 12 + length
        assert count == 1

    def test_raises_on_non_png(self):
        """Raises ValueError for non-PNG input."""
        with pytest.raises(ValueError, match="not a valid PNG"):
            embed_chara_in_png(b"not a png", json.dumps(VALID_V2_CARD))

    def test_unicode_content(self):
        """Handles Unicode characters in JSON correctly."""
        card = {**VALID_V2_CARD}
        card["data"] = {**VALID_V2_CARD["data"], "name": "テストキャラクター"}
        json_str = json.dumps(card, ensure_ascii=False)

        embedded = embed_chara_in_png(_make_minimal_png(), json_str)
        extracted = extract_chara_from_png(embedded)

        parsed = json.loads(extracted)
        assert parsed["data"]["name"] == "テストキャラクター"


# ============================================
# PNG Import Tests (Importer integration)
# ============================================


class TestPngImport:
    """Tests for CharacterImporter.import_png()."""

    @pytest.fixture
    def temp_characters_dir(self):
        with TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def importer(self, temp_characters_dir):
        return CharacterImporter(temp_characters_dir)

    def test_import_png_file(self, importer, temp_characters_dir):
        """Import character from PNG file."""
        png_bytes = _make_png_with_chara(VALID_V2_CARD)
        png_path = Path(temp_characters_dir) / "test_card.png"
        png_path.write_bytes(png_bytes)

        result = importer.import_file(png_path)

        assert result.character_id == "png_test_character"
        assert result.card.data.name == "PNG Test Character"
        assert result.has_avatar

    def test_import_png_saves_avatar(self, importer, temp_characters_dir):
        """PNG import saves the file as avatar.png."""
        png_bytes = _make_png_with_chara(VALID_V2_CARD)
        png_path = Path(temp_characters_dir) / "test_card.png"
        png_path.write_bytes(png_bytes)

        result = importer.import_file(png_path)

        avatar_path = Path(temp_characters_dir) / result.character_id / "avatar.png"
        assert avatar_path.exists()
        assert avatar_path.read_bytes() == png_bytes

    def test_import_png_saves_card_json(self, importer, temp_characters_dir):
        """PNG import saves the extracted card.json."""
        png_bytes = _make_png_with_chara(VALID_V2_CARD)
        png_path = Path(temp_characters_dir) / "test_card.png"
        png_path.write_bytes(png_bytes)

        result = importer.import_file(png_path)

        card_path = Path(temp_characters_dir) / result.character_id / "card.json"
        assert card_path.exists()

        with open(card_path) as f:
            saved = json.load(f)
        assert saved["data"]["name"] == "PNG Test Character"

    def test_import_png_with_codex(self, importer, temp_characters_dir):
        """PNG import preserves character_book entries."""
        png_bytes = _make_png_with_chara(VALID_V2_CARD_WITH_CODEX)
        png_path = Path(temp_characters_dir) / "test_card.png"
        png_path.write_bytes(png_bytes)

        result = importer.import_file(png_path)

        card_path = Path(temp_characters_dir) / result.character_id / "card.json"
        with open(card_path) as f:
            saved = json.load(f)

        assert "character_book" in saved["data"]
        assert len(saved["data"]["character_book"]["entries"]) == 1

    def test_import_plain_png_raises_error(self, importer, temp_characters_dir):
        """PNG without character data raises ImportError."""
        png_path = Path(temp_characters_dir) / "plain.png"
        png_path.write_bytes(_make_minimal_png())

        with pytest.raises(ImportError, match="does not contain character data"):
            importer.import_file(png_path)

    def test_import_png_overwrite(self, importer, temp_characters_dir):
        """PNG import respects overwrite flag."""
        png_bytes = _make_png_with_chara(VALID_V2_CARD)
        png_path = Path(temp_characters_dir) / "test_card.png"
        png_path.write_bytes(png_bytes)

        # First import
        importer.import_file(png_path)

        # Should fail without overwrite
        with pytest.raises(FileExistsError):
            importer.import_file(png_path)

        # Should succeed with overwrite
        result = importer.import_file(png_path, overwrite=True)
        assert result.character_id == "png_test_character"

    def test_import_file_rejects_unsupported_extension(self, importer, temp_characters_dir):
        """import_file rejects unsupported file types."""
        txt_path = Path(temp_characters_dir) / "card.txt"
        txt_path.write_text("not a card")

        with pytest.raises(ImportError, match="Unsupported file type"):
            importer.import_file(txt_path)

    def test_import_png_with_custom_id(self, importer, temp_characters_dir):
        """PNG import accepts custom character_id."""
        png_bytes = _make_png_with_chara(VALID_V2_CARD)
        png_path = Path(temp_characters_dir) / "test_card.png"
        png_path.write_bytes(png_bytes)

        result = importer.import_file(png_path, character_id="my_custom_png")

        assert result.character_id == "my_custom_png"


# ============================================
# PNG Export Tests (Exporter integration)
# ============================================


class TestPngExport:
    """Tests for CharacterExporter.export_to_png()."""

    @pytest.fixture
    def temp_characters_dir(self):
        with TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def setup_character(self, temp_characters_dir):
        """Set up a character with avatar for export tests."""
        importer = CharacterImporter(temp_characters_dir)
        png_bytes = _make_png_with_chara(VALID_V2_CARD)
        png_path = Path(temp_characters_dir) / "import.png"
        png_path.write_bytes(png_bytes)
        result = importer.import_file(png_path)
        return result.character_id

    @pytest.fixture
    def exporter(self, temp_characters_dir):
        return CharacterExporter(temp_characters_dir)

    def test_export_to_png_bytes(self, exporter, setup_character):
        """Export character as PNG bytes."""
        png_bytes = exporter.export_to_png(setup_character)

        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        # Should contain the embedded character data
        extracted = extract_chara_from_png(png_bytes)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["data"]["name"] == "PNG Test Character"

    def test_export_to_png_file(self, exporter, setup_character, temp_characters_dir):
        """Export character as PNG file."""
        output_path = Path(temp_characters_dir) / "export.png"
        result_path = exporter.export_to_png_file(setup_character, output_path)

        assert result_path.exists()
        png_bytes = result_path.read_bytes()
        extracted = extract_chara_from_png(png_bytes)
        assert extracted is not None

    def test_export_roundtrip(self, exporter, setup_character, temp_characters_dir):
        """Export PNG then re-import it — data survives roundtrip."""
        # Export
        png_bytes = exporter.export_to_png(setup_character)

        # Re-import
        reimport_path = Path(temp_characters_dir) / "roundtrip.png"
        reimport_path.write_bytes(png_bytes)

        importer = CharacterImporter(temp_characters_dir)
        result = importer.import_file(reimport_path, character_id="roundtrip_test")

        assert result.card.data.name == "PNG Test Character"
        assert result.card.data.personality == "Helpful"
        assert result.has_avatar

    def test_export_without_spindl(self, exporter, setup_character):
        """Export PNG without SpindL extensions."""
        png_bytes = exporter.export_to_png(setup_character, include_spindl=False)
        extracted = extract_chara_from_png(png_bytes)
        parsed = json.loads(extracted)

        assert "spindl" not in parsed["data"].get("extensions", {})

    def test_export_without_avatar_raises(self, temp_characters_dir):
        """Export as PNG raises when no avatar exists."""
        # Create character without avatar
        importer = CharacterImporter(temp_characters_dir)
        importer.import_dict(VALID_V2_CARD, character_id="no_avatar_char")

        exporter = CharacterExporter(temp_characters_dir)
        from spindl.characters import ExportError
        with pytest.raises(ExportError, match="no avatar"):
            exporter.export_to_png("no_avatar_char")

    def test_export_filename_png(self, exporter, setup_character):
        """get_export_filename returns .png extension."""
        filename = exporter.get_export_filename(setup_character, format="png")
        assert filename.endswith(".png")

    def test_export_filename_json(self, exporter, setup_character):
        """get_export_filename returns .json extension (default)."""
        filename = exporter.get_export_filename(setup_character)
        assert filename.endswith(".json")
