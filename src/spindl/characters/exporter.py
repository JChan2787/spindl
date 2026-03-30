"""
Character Card V2 exporter.

Exports spindl characters to ST-compatible JSON or PNG (Tavern Card) format,
enabling ecosystem interoperability with SillyTavern and other
applications that support the Character Card V2 spec.
"""

import json
from pathlib import Path
from typing import Any

from .importer import embed_chara_in_png
from .loader import CharacterLoader
from .models import CharacterCard


class ExportError(Exception):
    """Raised when character export fails."""

    pass


class CharacterExporter:
    """
    Exports characters to ST Character Card V2 format.

    Supports exporting with or without spindl specific extensions,
    allowing characters to be shared with the broader ST ecosystem.

    Usage:
        exporter = CharacterExporter("./characters")

        # Export to file
        exporter.export_to_file("spindle", "./exports/spindle.json")

        # Export to string
        json_str = exporter.export_to_json("spindle")

        # Export to dict
        data = exporter.export_to_dict("spindle")

        # Export without spindl extensions (pure ST)
        exporter.export_to_file("spindle", "./exports/spindle.json", include_spindl=False)
    """

    def __init__(self, characters_dir: str = "./characters"):
        """
        Initialize exporter with characters directory.

        Args:
            characters_dir: Path to directory containing character folders.
        """
        self.characters_dir = Path(characters_dir)
        self.loader = CharacterLoader(characters_dir)

    def export_to_file(
        self,
        character_id: str,
        output_path: str | Path,
        include_spindl: bool = True,
        include_codex: bool = True,
        pretty: bool = True,
    ) -> Path:
        """
        Export character to JSON file.

        Args:
            character_id: Character to export
            output_path: Path for output file
            include_spindl: Include spindl extensions (default True)
            include_codex: Include character_book entries (default True)
            pretty: Pretty-print JSON with indentation (default True)

        Returns:
            Path to created file

        Raises:
            ExportError: Character not found or export failed
        """
        data = self.export_to_dict(
            character_id,
            include_spindl=include_spindl,
            include_codex=include_codex,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=2 if pretty else None,
                    ensure_ascii=False,
                )
        except OSError as e:
            raise ExportError(f"Failed to write file: {e}")

        return output

    def export_to_png(
        self,
        character_id: str,
        include_spindl: bool = True,
        include_codex: bool = True,
    ) -> bytes:
        """
        Export character as a PNG (Tavern Card) with embedded JSON.

        Uses the character's avatar.png as the base image and embeds the
        V2 JSON as a base64-encoded tEXt chunk with key 'chara'.

        Args:
            character_id: Character to export
            include_spindl: Include spindl extensions
            include_codex: Include character_book entries

        Returns:
            PNG bytes with embedded character data

        Raises:
            ExportError: Character not found or has no avatar
        """
        # Get the export dict
        data = self.export_to_dict(
            character_id,
            include_spindl=include_spindl,
            include_codex=include_codex,
        )

        # Get avatar path
        avatar_path = self.loader.get_avatar_path(character_id)
        if avatar_path is None:
            raise ExportError(
                f"Character '{character_id}' has no avatar.png. "
                f"PNG export requires an avatar image."
            )

        png_bytes = avatar_path.read_bytes()
        json_str = json.dumps(data, ensure_ascii=False)

        return embed_chara_in_png(png_bytes, json_str)

    def export_to_png_file(
        self,
        character_id: str,
        output_path: str | Path,
        include_spindl: bool = True,
        include_codex: bool = True,
    ) -> Path:
        """
        Export character as a PNG file (Tavern Card) with embedded JSON.

        Args:
            character_id: Character to export
            output_path: Path for output PNG file
            include_spindl: Include spindl extensions
            include_codex: Include character_book entries

        Returns:
            Path to created file

        Raises:
            ExportError: Character not found, no avatar, or write failed
        """
        png_bytes = self.export_to_png(
            character_id,
            include_spindl=include_spindl,
            include_codex=include_codex,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            output.write_bytes(png_bytes)
        except OSError as e:
            raise ExportError(f"Failed to write PNG file: {e}")

        return output

    def export_to_json(
        self,
        character_id: str,
        include_spindl: bool = True,
        include_codex: bool = True,
        pretty: bool = True,
    ) -> str:
        """
        Export character to JSON string.

        Args:
            character_id: Character to export
            include_spindl: Include spindl extensions
            include_codex: Include character_book entries
            pretty: Pretty-print JSON with indentation

        Returns:
            JSON string

        Raises:
            ExportError: Character not found or export failed
        """
        data = self.export_to_dict(
            character_id,
            include_spindl=include_spindl,
            include_codex=include_codex,
        )

        return json.dumps(
            data,
            indent=2 if pretty else None,
            ensure_ascii=False,
        )

    def export_to_dict(
        self,
        character_id: str,
        include_spindl: bool = True,
        include_codex: bool = True,
    ) -> dict[str, Any]:
        """
        Export character to dict.

        This is the core export method. Loads the character, optionally
        filters out spindl extensions, and returns ST-compatible dict.

        Args:
            character_id: Character to export
            include_spindl: Include spindl extensions
            include_codex: Include character_book entries

        Returns:
            ST Character Card V2 compatible dict

        Raises:
            ExportError: Character not found
        """
        try:
            card = self.loader.load(character_id)
        except FileNotFoundError as e:
            raise ExportError(str(e))
        except ValueError as e:
            raise ExportError(f"Invalid character card: {e}")

        return self.card_to_dict(
            card,
            include_spindl=include_spindl,
            include_codex=include_codex,
        )

    def card_to_dict(
        self,
        card: CharacterCard,
        include_spindl: bool = True,
        include_codex: bool = True,
    ) -> dict[str, Any]:
        """
        Convert CharacterCard to export dict.

        Args:
            card: CharacterCard to export
            include_spindl: Include spindl extensions
            include_codex: Include character_book entries

        Returns:
            ST Character Card V2 compatible dict
        """
        # Get base dict from model
        data = card.model_dump(mode="json", exclude_none=True)

        # Handle extensions filtering
        if not include_spindl and "extensions" in data.get("data", {}):
            extensions = data["data"]["extensions"]
            if "spindl" in extensions:
                del extensions["spindl"]
            # Remove empty extensions dict
            if not extensions:
                del data["data"]["extensions"]

        # Handle codex filtering
        if not include_codex and "character_book" in data.get("data", {}):
            del data["data"]["character_book"]

        # Ensure ST compatibility: extensions must be {} not missing
        if "extensions" not in data.get("data", {}):
            data["data"]["extensions"] = {}

        return data

    def export_card_from_data(
        self,
        card_data: dict[str, Any],
        include_spindl: bool = True,
        include_codex: bool = True,
    ) -> dict[str, Any]:
        """
        Export from raw card data (useful for GUI which passes dicts).

        Args:
            card_data: Raw character card dict
            include_spindl: Include spindl extensions
            include_codex: Include character_book entries

        Returns:
            ST Character Card V2 compatible dict

        Raises:
            ExportError: Invalid card data
        """
        try:
            card = CharacterCard.model_validate(card_data)
        except Exception as e:
            raise ExportError(f"Invalid card data: {e}")

        return self.card_to_dict(
            card,
            include_spindl=include_spindl,
            include_codex=include_codex,
        )

    def get_export_filename(self, character_id: str, format: str = "json") -> str:
        """
        Generate suggested filename for export.

        Args:
            character_id: Character ID
            format: Export format — 'json' or 'png'

        Returns:
            Suggested filename (e.g., "spindle_v2.json" or "spindle_card.png")
        """
        if format == "png":
            return f"{character_id}_card.png"
        return f"{character_id}_v2.json"

    def batch_export(
        self,
        output_dir: str | Path,
        character_ids: list[str] | None = None,
        include_spindl: bool = True,
        include_codex: bool = True,
    ) -> "BatchExportResult":
        """
        Export multiple characters to a directory.

        Args:
            output_dir: Directory for exported files
            character_ids: List of IDs to export. If None, exports all.
            include_spindl: Include spindl extensions
            include_codex: Include character_book entries

        Returns:
            BatchExportResult with success/failure counts
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        if character_ids is None:
            character_ids = self.loader.list_characters()

        results = BatchExportResult()

        for char_id in character_ids:
            try:
                filename = self.get_export_filename(char_id)
                self.export_to_file(
                    char_id,
                    output / filename,
                    include_spindl=include_spindl,
                    include_codex=include_codex,
                )
                results.add_success(char_id, output / filename)
            except ExportError as e:
                results.add_failure(char_id, str(e))

        return results


class BatchExportResult:
    """Result of batch export operation."""

    def __init__(self):
        self.exported: list[tuple[str, Path]] = []
        self.failed: list[tuple[str, str]] = []

    def add_success(self, character_id: str, path: Path) -> None:
        self.exported.append((character_id, path))

    def add_failure(self, character_id: str, error: str) -> None:
        self.failed.append((character_id, error))

    @property
    def success_count(self) -> int:
        return len(self.exported)

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    @property
    def total(self) -> int:
        return self.success_count + self.failure_count

    def __repr__(self) -> str:
        return f"BatchExportResult(exported={self.success_count}, failed={self.failure_count})"
