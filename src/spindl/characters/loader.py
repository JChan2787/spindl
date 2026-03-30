"""
Character loader for ST V2 character cards.

Handles loading, saving, and managing character cards stored as JSON files.
Replaces the legacy PersonaLoader with full ST V2 compatibility.
"""

import json
from pathlib import Path
from typing import Any

from .models import CharacterCard, CharacterCardData
from spindl.utils.paths import resolve_relative_path


class CharacterLoader:
    """
    Loads and saves Character Card V2 files.

    Storage structure:
        characters/
        ├── spindle/
        │   ├── card.json      # ST V2 character card
        │   └── avatar.png     # Portrait (optional)
        └── _global/
            └── codex.json     # Global codex entries

    Usage:
        loader = CharacterLoader("./characters")
        card = loader.load("spindle")
        print(card.data.name)  # "Spindle"

        # Backward compatible dict access
        persona = card.to_persona_dict()
        print(persona["voice"])  # "af_bella"
    """

    CARD_FILENAME = "card.json"
    AVATAR_FILENAME = "avatar.png"
    GLOBAL_DIR = "_global"

    def __init__(self, characters_dir: str = "./characters"):
        """
        Initialize with characters directory path.

        Args:
            characters_dir: Path to directory containing character folders.
                           Each character has its own subdirectory with card.json.
        """
        self.characters_dir = Path(resolve_relative_path(characters_dir))

    def load(self, character_id: str) -> CharacterCard:
        """
        Load character by ID.

        Args:
            character_id: Character identifier (folder name)

        Returns:
            CharacterCard model instance

        Raises:
            FileNotFoundError: Character folder or card.json not found
            ValueError: Invalid card format or validation error
        """
        char_dir = self.characters_dir / character_id
        card_path = char_dir / self.CARD_FILENAME

        if not char_dir.exists():
            raise FileNotFoundError(
                f"Character '{character_id}' not found at {char_dir}"
            )

        if not card_path.exists():
            raise FileNotFoundError(
                f"Character card not found at {card_path}"
            )

        try:
            with open(card_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse character '{character_id}': {e}")

        try:
            return CharacterCard.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid character card '{character_id}': {e}")

    def load_as_dict(self, character_id: str) -> dict[str, Any]:
        """
        Load character and return as legacy persona dict.

        This method provides backward compatibility with code expecting
        the PersonaLoader dict format.

        Args:
            character_id: Character identifier (folder name)

        Returns:
            Dict matching PersonaLoader output format
        """
        card = self.load(character_id)
        return card.to_persona_dict()

    def save(self, card: CharacterCard, character_id: str | None = None) -> None:
        """
        Save character card to disk.

        Args:
            card: CharacterCard to save
            character_id: Optional ID override. If not provided, derives from
                         card.data.spindl.id or card.data.name.
        """
        if character_id is None:
            nano = card.data.spindl
            if nano and nano.id:
                character_id = nano.id
            else:
                character_id = card.data.name.lower().replace(" ", "_")

        char_dir = self.characters_dir / character_id
        char_dir.mkdir(parents=True, exist_ok=True)

        card_path = char_dir / self.CARD_FILENAME

        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(
                card.model_dump(mode="json", exclude_none=True),
                f,
                indent=2,
                ensure_ascii=False,
            )

    def delete(self, character_id: str) -> None:
        """
        Delete character folder.

        Args:
            character_id: Character identifier (folder name)

        Raises:
            FileNotFoundError: Character not found
        """
        import shutil

        char_dir = self.characters_dir / character_id

        if not char_dir.exists():
            raise FileNotFoundError(
                f"Character '{character_id}' not found at {char_dir}"
            )

        shutil.rmtree(char_dir)

    def list_characters(self) -> list[str]:
        """
        List available character IDs.

        Returns:
            List of character IDs (folder names excluding _global)
        """
        if not self.characters_dir.exists():
            return []

        return [
            d.name
            for d in self.characters_dir.iterdir()
            if d.is_dir()
            and d.name != self.GLOBAL_DIR
            and (d / self.CARD_FILENAME).exists()
        ]

    def exists(self, character_id: str) -> bool:
        """
        Check if character exists.

        Args:
            character_id: Character identifier

        Returns:
            True if character folder and card.json exist
        """
        char_dir = self.characters_dir / character_id
        card_path = char_dir / self.CARD_FILENAME
        return card_path.exists()

    def get_avatar_path(self, character_id: str) -> Path | None:
        """
        Get path to character avatar if it exists.

        Args:
            character_id: Character identifier

        Returns:
            Path to avatar.png if exists, None otherwise
        """
        avatar_path = self.characters_dir / character_id / self.AVATAR_FILENAME
        return avatar_path if avatar_path.exists() else None

    def get_vrm_path(self, character_id: str) -> Path | None:
        """
        Get absolute path to character's VRM avatar model if it exists.

        Reads the character card to find the avatar_vrm filename,
        then resolves it against the character directory.

        Args:
            character_id: Character identifier

        Returns:
            Absolute Path to .vrm file if field is set and file exists,
            None otherwise
        """
        try:
            card = self.load(character_id)
        except (FileNotFoundError, ValueError):
            return None
        nano = card.data.spindl
        if not nano or not nano.avatar_vrm:
            return None
        vrm_path = self.characters_dir / character_id / nano.avatar_vrm
        return vrm_path if vrm_path.exists() else None

    def get_avatar_expressions(self, character_id: str) -> dict[str, dict[str, float]] | None:
        """
        Get per-character expression composites from character card (NANO-098).

        Returns:
            Expression composites dict if set, None otherwise
        """
        try:
            card = self.load(character_id)
        except (FileNotFoundError, ValueError):
            return None
        nano = card.data.spindl
        if not nano or not nano.avatar_expressions:
            return None
        return nano.avatar_expressions

    def get_avatar_animations(self, character_id: str) -> dict | None:
        """
        Get per-character emotion-to-animation threshold map (NANO-098 Session 3).

        Returns:
            Animation config dict if set, None otherwise
        """
        try:
            card = self.load(character_id)
        except (FileNotFoundError, ValueError):
            return None
        nano = card.data.spindl
        if not nano or not nano.avatar_animations:
            return None
        return nano.avatar_animations

    def get_character_animations_dir(self, character_id: str) -> Path:
        """
        Get path to character's local animations directory (NANO-098 Session 3).

        Returns the path regardless of whether the directory exists — the
        renderer handles missing dirs gracefully via Tauri's list_directory.
        """
        return self.characters_dir / character_id / "animations"

    def has_structured_fields(self, card: CharacterCard) -> bool:
        """
        Check if character uses structured format.

        Backward compatibility helper matching PersonaLoader interface.

        Args:
            card: CharacterCard to check

        Returns:
            True if card has spindl structured fields
        """
        nano = card.data.spindl
        if not nano:
            # Check data.description even without nano extensions
            return bool(card.data.description or card.data.personality)
        return bool(card.data.description or nano.appearance or nano.rules) or bool(card.data.personality)

    def has_legacy_prompt(self, card: CharacterCard) -> bool:
        """
        Check if character has legacy system_prompt.

        Backward compatibility helper matching PersonaLoader interface.

        Args:
            card: CharacterCard to check

        Returns:
            True if card has system_prompt field
        """
        return bool(card.data.system_prompt)
