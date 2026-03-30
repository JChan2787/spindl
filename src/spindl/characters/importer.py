"""
Character Card V2 importer.

Imports ST Character Card V2 JSON/PNG files into spindl's character system,
enabling ecosystem interoperability with thousands of existing ST characters.

Supports:
- Raw JSON files (.json)
- PNG files with embedded JSON in tEXt chunks (.png) — the standard
  distribution format on Chub.ai and SillyTavern exports.
"""

import base64
import json
import struct
import zlib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .loader import CharacterLoader
from .models import CharacterCard


# PNG magic bytes
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def extract_chara_from_png(png_bytes: bytes) -> str | None:
    """
    Extract character card JSON from a PNG's tEXt chunks.

    SillyTavern embeds character data as:
        tEXt chunk → keyword: "chara" → text: base64(JSON)

    Also checks for 'ccv3' keyword (Character Card V3 draft spec).

    Args:
        png_bytes: Raw PNG file bytes

    Returns:
        Decoded JSON string, or None if no character data found
    """
    if not png_bytes.startswith(_PNG_SIGNATURE):
        return None

    offset = 8  # Skip PNG signature

    while offset < len(png_bytes):
        if offset + 8 > len(png_bytes):
            break

        length = struct.unpack(">I", png_bytes[offset : offset + 4])[0]
        chunk_type = png_bytes[offset + 4 : offset + 8]

        if offset + 12 + length > len(png_bytes):
            break

        chunk_data = png_bytes[offset + 8 : offset + 8 + length]

        if chunk_type == b"tEXt":
            # tEXt format: keyword\0text
            null_idx = chunk_data.find(b"\x00")
            if null_idx != -1:
                keyword = chunk_data[:null_idx].decode("latin-1")
                text = chunk_data[null_idx + 1 :]

                if keyword in ("chara", "ccv3"):
                    try:
                        return base64.b64decode(text).decode("utf-8")
                    except Exception:
                        # Try treating text as raw UTF-8 (non-standard but seen in wild)
                        try:
                            return text.decode("utf-8")
                        except Exception:
                            pass

        elif chunk_type == b"IEND":
            break

        # Advance: 4 (length) + 4 (type) + length (data) + 4 (CRC)
        offset += 12 + length

    return None


def embed_chara_in_png(png_bytes: bytes, json_str: str) -> bytes:
    """
    Embed character card JSON into a PNG as a tEXt chunk.

    Inserts a tEXt chunk with keyword 'chara' and base64-encoded JSON
    before the IEND chunk. Removes any existing 'chara' tEXt chunk first.

    Args:
        png_bytes: Raw PNG file bytes
        json_str: Character card JSON string to embed

    Returns:
        New PNG bytes with embedded character data

    Raises:
        ValueError: Input is not a valid PNG
    """
    if not png_bytes.startswith(_PNG_SIGNATURE):
        raise ValueError("Input is not a valid PNG file")

    # Build the tEXt chunk data: keyword\0base64_data
    text_payload = b"chara\x00" + base64.b64encode(json_str.encode("utf-8"))
    text_crc = zlib.crc32(b"tEXt" + text_payload) & 0xFFFFFFFF
    text_chunk = (
        struct.pack(">I", len(text_payload))
        + b"tEXt"
        + text_payload
        + struct.pack(">I", text_crc)
    )

    # Walk chunks, collect everything except existing 'chara' tEXt, insert before IEND
    result = bytearray(_PNG_SIGNATURE)
    offset = 8

    while offset < len(png_bytes):
        if offset + 8 > len(png_bytes):
            break

        length = struct.unpack(">I", png_bytes[offset : offset + 4])[0]
        chunk_type = png_bytes[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length

        if chunk_end > len(png_bytes):
            break

        raw_chunk = png_bytes[offset:chunk_end]

        if chunk_type == b"IEND":
            # Insert our tEXt chunk before IEND
            result.extend(text_chunk)
            result.extend(raw_chunk)
            break
        elif chunk_type == b"tEXt":
            # Skip existing 'chara' chunks
            chunk_data = png_bytes[offset + 8 : offset + 8 + length]
            null_idx = chunk_data.find(b"\x00")
            if null_idx != -1:
                keyword = chunk_data[:null_idx].decode("latin-1")
                if keyword == "chara":
                    offset = chunk_end
                    continue
            result.extend(raw_chunk)
        else:
            result.extend(raw_chunk)

        offset = chunk_end

    return bytes(result)


class ImportError(Exception):
    """Raised when character import fails."""

    pass


class CharacterImporter:
    """
    Imports ST Character Card V2 files into spindl.

    Handles validation, ID generation, and conflict resolution when importing
    character cards from the SillyTavern ecosystem.

    Usage:
        importer = CharacterImporter("./characters")

        # Import from file
        result = importer.import_file("./downloads/miku.json")
        print(result.character_id)  # "miku"

        # Import from raw JSON
        result = importer.import_json(json_string)

        # Import from dict
        result = importer.import_dict(card_data)
    """

    def __init__(self, characters_dir: str = "./characters"):
        """
        Initialize importer with target characters directory.

        Args:
            characters_dir: Path to directory where imported characters will be saved.
        """
        self.characters_dir = Path(characters_dir)
        self.loader = CharacterLoader(characters_dir)

    def import_file(
        self,
        file_path: str | Path,
        character_id: str | None = None,
        overwrite: bool = False,
    ) -> "ImportResult":
        """
        Import character from JSON or PNG file.

        PNG files are expected to contain a tEXt chunk with key 'chara'
        holding base64-encoded V2 JSON (SillyTavern/Chub standard).

        Args:
            file_path: Path to .json or .png character card file
            character_id: Optional ID override. If not provided, derives from card.
            overwrite: If True, overwrite existing character. Otherwise raise error.

        Returns:
            ImportResult with character_id and card

        Raises:
            ImportError: File not found, invalid format, or validation error
            FileExistsError: Character already exists and overwrite=False
        """
        path = Path(file_path)

        if not path.exists():
            raise ImportError(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext == ".png":
            return self.import_png(path, character_id=character_id, overwrite=overwrite)
        elif ext == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ImportError(f"Invalid JSON: {e}")
            return self.import_dict(data, character_id=character_id, overwrite=overwrite)
        else:
            raise ImportError(f"Unsupported file type: {ext}. Expected .json or .png")

    def import_png(
        self,
        file_path: str | Path,
        character_id: str | None = None,
        overwrite: bool = False,
    ) -> "ImportResult":
        """
        Import character from PNG file with embedded tEXt chunk.

        SillyTavern embeds character card JSON as a base64-encoded tEXt chunk
        with the keyword 'chara'. This is the standard format for character
        cards distributed on Chub.ai.

        Args:
            file_path: Path to PNG file
            character_id: Optional ID override
            overwrite: If True, overwrite existing character

        Returns:
            ImportResult with character_id, card, and avatar_data

        Raises:
            ImportError: No tEXt chunk found, invalid data, or validation error
            FileExistsError: Character already exists and overwrite=False
        """
        path = Path(file_path)

        if not path.exists():
            raise ImportError(f"File not found: {path}")

        png_bytes = path.read_bytes()

        # Extract character JSON from tEXt chunk
        json_str = extract_chara_from_png(png_bytes)
        if json_str is None:
            raise ImportError(
                "PNG file does not contain character data. "
                "Expected a tEXt chunk with key 'chara' (SillyTavern format)."
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ImportError(f"Invalid JSON in PNG tEXt chunk: {e}")

        # Import the card data
        result = self.import_dict(data, character_id=character_id, overwrite=overwrite)

        # Save the PNG as avatar.png in the character directory
        avatar_dest = self.characters_dir / result.character_id / "avatar.png"
        avatar_dest.write_bytes(png_bytes)
        result.has_avatar = True

        return result

    def import_json(
        self,
        json_string: str,
        character_id: str | None = None,
        overwrite: bool = False,
    ) -> "ImportResult":
        """
        Import character from JSON string.

        Args:
            json_string: ST Character Card V2 JSON as string
            character_id: Optional ID override
            overwrite: If True, overwrite existing character

        Returns:
            ImportResult with character_id and card

        Raises:
            ImportError: Invalid JSON or validation error
            FileExistsError: Character already exists and overwrite=False
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ImportError(f"Invalid JSON: {e}")

        return self.import_dict(data, character_id=character_id, overwrite=overwrite)

    def import_dict(
        self,
        data: dict[str, Any],
        character_id: str | None = None,
        overwrite: bool = False,
    ) -> "ImportResult":
        """
        Import character from dict.

        This is the core import method. Validates the card, resolves the
        character ID, checks for conflicts, and saves to disk.

        Args:
            data: ST Character Card V2 data as dict
            character_id: Optional ID override
            overwrite: If True, overwrite existing character

        Returns:
            ImportResult with character_id and card

        Raises:
            ImportError: Validation error or missing required fields
            FileExistsError: Character already exists and overwrite=False
        """
        # Validate and convert to CharacterCard
        card = self._validate_card(data)

        # Resolve character ID
        resolved_id = self._resolve_character_id(card, character_id)

        # Check for conflicts
        if self.loader.exists(resolved_id) and not overwrite:
            raise FileExistsError(
                f"Character '{resolved_id}' already exists. "
                f"Use overwrite=True to replace."
            )

        # Ensure spindl.id is set
        self._ensure_spindl_id(card, resolved_id)

        # Save the card
        self.loader.save(card, resolved_id)

        return ImportResult(
            character_id=resolved_id,
            card=card,
            was_overwrite=self.loader.exists(resolved_id) and overwrite,
        )

    def validate_only(self, data: dict[str, Any]) -> "ValidationResult":
        """
        Validate card without importing.

        Useful for pre-flight checks or import previews.

        Args:
            data: ST Character Card V2 data as dict

        Returns:
            ValidationResult with validity status and any errors
        """
        try:
            card = self._validate_card(data)
            return ValidationResult(
                valid=True,
                card=card,
                errors=[],
                warnings=self._check_warnings(card),
            )
        except ImportError as e:
            return ValidationResult(
                valid=False,
                card=None,
                errors=[str(e)],
                warnings=[],
            )

    def _validate_card(self, data: dict[str, Any]) -> CharacterCard:
        """
        Validate and parse card data.

        Handles both strict ST V2 format and common variations:
        - Missing spec/spec_version (adds defaults)
        - V1 cards (upgrades to V2)
        - Nested data structure variations

        Args:
            data: Raw card data

        Returns:
            Validated CharacterCard

        Raises:
            ImportError: Validation failed
        """
        # Handle V1 cards (no spec field, flat structure)
        if "spec" not in data:
            data = self._upgrade_v1_card(data)

        # Handle cards with wrong spec version
        if data.get("spec") != "chara_card_v2":
            if data.get("spec") == "chara_card_v1" or "name" in data:
                data = self._upgrade_v1_card(data)
            else:
                raise ImportError(
                    f"Unknown card spec: {data.get('spec')}. "
                    f"Expected 'chara_card_v2'."
                )

        # Ensure spec_version exists
        if "spec_version" not in data:
            data["spec_version"] = "2.0"

        # Ensure data block exists
        if "data" not in data:
            raise ImportError("Missing 'data' block in character card")

        # Ensure name exists
        if "name" not in data["data"]:
            raise ImportError("Missing required field: data.name")

        # Ensure extensions is a dict (per spec: MUST default to {})
        if "extensions" not in data["data"]:
            data["data"]["extensions"] = {}
        elif not isinstance(data["data"]["extensions"], dict):
            data["data"]["extensions"] = {}

        # Validate with Pydantic
        try:
            return CharacterCard.model_validate(data)
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise ImportError(f"Validation failed: {'; '.join(errors)}")

    def _upgrade_v1_card(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Upgrade V1 card to V2 format.

        V1 cards have a flat structure with fields like name, description, etc.
        directly at the root level instead of nested under 'data'.

        Args:
            data: V1 card data

        Returns:
            V2-formatted card data
        """
        # Check if this is already somewhat V2 structured
        if "data" in data and isinstance(data["data"], dict):
            # Already has data block, just ensure spec fields
            return {
                "spec": "chara_card_v2",
                "spec_version": "2.0",
                "data": data["data"],
            }

        # Pure V1 - flat structure
        v2_data: dict[str, Any] = {
            "name": data.get("name", "Unknown"),
            "description": data.get("description", ""),
            "personality": data.get("personality", ""),
            "scenario": data.get("scenario", ""),
            "first_mes": data.get("first_mes", ""),
            "mes_example": data.get("mes_example", ""),
            "extensions": {},
        }

        # V2 fields that might be present
        for field in [
            "creator_notes",
            "system_prompt",
            "post_history_instructions",
            "alternate_greetings",
            "tags",
            "creator",
            "character_version",
            "character_book",
        ]:
            if field in data:
                v2_data[field] = data[field]

        return {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": v2_data,
        }

    def _resolve_character_id(
        self, card: CharacterCard, override_id: str | None
    ) -> str:
        """
        Determine the character ID to use.

        Priority:
        1. Explicit override_id parameter
        2. spindl.id from extensions
        3. Derived from card.data.name

        Args:
            card: Validated CharacterCard
            override_id: Optional explicit ID

        Returns:
            Resolved character ID (lowercase, underscores for spaces)
        """
        if override_id:
            return self._sanitize_id(override_id)

        # Check spindl extension
        nano = card.data.spindl
        if nano and nano.id:
            return self._sanitize_id(nano.id)

        # Derive from name
        return self._sanitize_id(card.data.name)

    def _sanitize_id(self, raw_id: str) -> str:
        """
        Sanitize string into valid character ID.

        - Lowercase
        - Replace spaces with underscores
        - Remove non-alphanumeric except underscores and hyphens
        - Collapse multiple underscores

        Args:
            raw_id: Raw ID string

        Returns:
            Sanitized character ID
        """
        import re

        # Lowercase and replace spaces
        sanitized = raw_id.lower().replace(" ", "_")

        # Keep only alphanumeric, underscore, hyphen
        sanitized = re.sub(r"[^a-z0-9_-]", "", sanitized)

        # Collapse multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Strip leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Ensure not empty
        if not sanitized:
            sanitized = "imported_character"

        return sanitized

    def _ensure_spindl_id(self, card: CharacterCard, character_id: str) -> None:
        """
        Ensure spindl.id is set in the card's extensions.

        Modifies the card in place to ensure the ID is stored.

        Args:
            card: CharacterCard to modify
            character_id: ID to set
        """
        if "spindl" not in card.data.extensions:
            card.data.extensions["spindl"] = {}

        card.data.extensions["spindl"]["id"] = character_id

    def _check_warnings(self, card: CharacterCard) -> list[str]:
        """
        Check for non-fatal issues that might need attention.

        Args:
            card: Validated CharacterCard

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for missing personality
        if not card.data.personality and not card.data.system_prompt:
            warnings.append(
                "Character has no personality or system_prompt defined. "
                "Consider adding character definition."
            )

        # Check for very short description
        if len(card.data.description) < 10 and not card.data.personality:
            warnings.append(
                "Character has minimal description. "
                "Consider adding more detail for better responses."
            )

        # Check for empty character_book entries
        if card.data.character_book:
            empty_entries = sum(
                1
                for e in card.data.character_book.entries
                if not e.content.strip()
            )
            if empty_entries > 0:
                warnings.append(
                    f"Character book has {empty_entries} entries with empty content."
                )

        return warnings


class ImportResult:
    """Result of a successful character import."""

    def __init__(
        self,
        character_id: str,
        card: CharacterCard,
        was_overwrite: bool = False,
    ):
        self.character_id = character_id
        self.card = card
        self.was_overwrite = was_overwrite
        self.has_avatar = False

    def __repr__(self) -> str:
        return f"ImportResult(character_id='{self.character_id}', overwrite={self.was_overwrite}, avatar={self.has_avatar})"


class ValidationResult:
    """Result of card validation."""

    def __init__(
        self,
        valid: bool,
        card: CharacterCard | None,
        errors: list[str],
        warnings: list[str],
    ):
        self.valid = valid
        self.card = card
        self.errors = errors
        self.warnings = warnings

    def __repr__(self) -> str:
        return f"ValidationResult(valid={self.valid}, errors={len(self.errors)}, warnings={len(self.warnings)})"
