"""
CodexManager - orchestrates codex entry loading and activation.

Handles:
- Loading global codex from _global/codex.json
- Loading character codex from character_book field
- Merging entries with proper priority ordering
- Managing activation state per conversation
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from spindl.characters.models import (
    CharacterBook,
    CharacterBookEntry,
    CharacterCard,
    CodexVolume,
)
from spindl.codex.activation import activate_entries
from spindl.codex.models import ActivationResult, CodexState

DEFAULT_VOLUME_ID = "vol_default"

logger = logging.getLogger(__name__)


class CodexManager:
    """
    Manages codex entries and activation state for a conversation.

    The CodexManager:
    1. Loads entries from global codex and character-specific codex
    2. Merges entries with proper ordering
    3. Tracks activation state across conversation turns
    4. Provides methods to activate entries based on input text

    Usage:
        manager = CodexManager(characters_dir="./characters")
        manager.load_character("spindle")  # Loads character + global codex
        results = manager.activate("Tell me about coffee")
        manager.advance_turn()
    """

    def __init__(
        self,
        characters_dir: str | Path = "./characters",
        match_whole_words: bool = False,
        max_entries_per_turn: int | None = None,
    ):
        """
        Initialize the CodexManager.

        Args:
            characters_dir: Directory containing character folders and _global/
            match_whole_words: Default whole-word matching mode
            max_entries_per_turn: Max entries to activate per turn (None = unlimited)
        """
        self.characters_dir = Path(characters_dir)
        self.match_whole_words = match_whole_words
        self.max_entries_per_turn = max_entries_per_turn

        # Loaded entries
        self._global_entries: list[CharacterBookEntry] = []
        self._character_entries: list[CharacterBookEntry] = []
        self._merged_entries: list[CharacterBookEntry] = []

        # Volumes (NANO-128)
        self._volumes: list[CodexVolume] = []

        # State tracking
        self._state = CodexState()
        self._current_character_id: str | None = None

        # Config from character book
        self._scan_depth: int | None = None
        self._token_budget: int | None = None
        self._recursive_scanning: bool = False

    @property
    def state(self) -> CodexState:
        """Get current codex state."""
        return self._state

    @property
    def entries(self) -> list[CharacterBookEntry]:
        """Get all merged entries (global + character)."""
        return self._merged_entries

    @property
    def global_entries(self) -> list[CharacterBookEntry]:
        """Get global codex entries only."""
        return self._global_entries

    @property
    def character_entries(self) -> list[CharacterBookEntry]:
        """Get character codex entries only."""
        return self._character_entries

    def load_global_codex(self) -> None:
        """Load entries and volumes from _global/codex.json and merge into active entries."""
        global_path = self.characters_dir / "_global" / "codex.json"

        if not global_path.exists():
            logger.debug("No global codex found at %s", global_path)
            self._global_entries = []
            self._volumes = [self._make_default_volume()]
            self._merge_entries()
            return

        try:
            with open(global_path, encoding="utf-8") as f:
                data = json.load(f)

            book = CharacterBook.model_validate(data)
            self._global_entries = book.entries
            self._volumes = book.volumes
            logger.info("Loaded %d global codex entries", len(self._global_entries))

        except (json.JSONDecodeError, Exception) as e:
            logger.error("Failed to load global codex: %s", e)
            self._global_entries = []
            self._volumes = []

        self._ensure_default_volume()
        self._merge_entries()

    def load_character_codex(self, card: CharacterCard) -> None:
        """
        Load entries from a character card's character_book.

        Args:
            card: The CharacterCard to extract entries from
        """
        if card.data.character_book is None:
            logger.debug("Character has no embedded codex")
            self._character_entries = []
            return

        book = card.data.character_book
        self._character_entries = book.entries

        # Store config from character book
        self._scan_depth = book.scan_depth
        self._token_budget = book.token_budget
        self._recursive_scanning = book.recursive_scanning or False

        logger.info(
            "Loaded %d character codex entries from %s",
            len(self._character_entries),
            card.data.name,
        )

    def load_character(self, character_id: str) -> None:
        """
        Load both global and character-specific codex entries.

        Args:
            character_id: The character folder name
        """
        from spindl.characters.loader import CharacterLoader

        self._current_character_id = character_id

        # Load global codex
        self.load_global_codex()

        # Load character card and extract codex
        loader = CharacterLoader(self.characters_dir)
        if loader.exists(character_id):
            card = loader.load(character_id)
            self.load_character_codex(card)
        else:
            logger.warning("Character %s not found, skipping character codex", character_id)
            self._character_entries = []

        # Merge entries
        self._merge_entries()

        # Reset state for new character
        self._state.reset()

    def _merge_entries(self) -> None:
        """
        Merge global and character entries.

        Character entries take precedence (come first) when insertion_order is equal.
        Entries are sorted by insertion_order for prompt ordering.
        Global entries in disabled volumes are filtered out.
        Each entry gets a _runtime_id for namespaced state tracking (NANO-128 Layer 0).
        """
        disabled_vols = {v.id for v in self._volumes if not v.enabled}

        # Assign runtime IDs and filter disabled volumes for global entries
        filtered_global: list[CharacterBookEntry] = []
        for entry in self._global_entries:
            vol = entry.volume_id or DEFAULT_VOLUME_ID
            if vol in disabled_vols:
                continue
            entry._runtime_id = f"global_{entry.id}"  # type: ignore[attr-defined]
            filtered_global.append(entry)

        # Assign runtime IDs to character entries (no volume filtering)
        for entry in self._character_entries:
            entry._runtime_id = f"character_{entry.id}"  # type: ignore[attr-defined]

        # Combine with character entries first (for tiebreaking)
        all_entries = self._character_entries + filtered_global
        all_entries.sort(key=lambda e: e.insertion_order)

        self._merged_entries = all_entries
        logger.debug(
            "Merged %d total codex entries (%d character, %d global, %d filtered by volume)",
            len(self._merged_entries),
            len(self._character_entries),
            len(filtered_global),
            len(self._global_entries) - len(filtered_global),
        )

    def activate(self, text: str) -> list[ActivationResult]:
        """
        Check all entries against input text and return activated entries.

        Reloads codex files from disk before each activation to pick up
        any changes made via the GUI without requiring a backend restart.

        Args:
            text: The text to search for keywords (usually user input)

        Returns:
            List of ActivationResults for entries that activated
        """
        # Reload from disk to pick up GUI changes
        self.load_global_codex()

        results = activate_entries(
            text=text,
            entries=self._merged_entries,
            state=self._state,
            match_whole_words=self.match_whole_words,
            max_entries=self.max_entries_per_turn,
        )

        if results:
            logger.debug(
                "Activated %d codex entries: %s",
                len(results),
                [r.entry_name or f"entry_{r.entry_id}" for r in results],
            )

        return results

    def advance_turn(self) -> None:
        """Advance the conversation turn counter."""
        self._state.advance_turn()

    def reset_state(self) -> None:
        """Reset activation state (new conversation)."""
        self._state.reset()

    def get_activated_content(
        self,
        results: list[ActivationResult],
        position: Optional[str] = None,
    ) -> str:
        """
        Get concatenated content from activation results.

        Args:
            results: List of ActivationResults
            position: Filter by position ("before_char" or "after_char").
                     If None, returns all activated content (ignores position).

        Returns:
            Concatenated content string with newlines between entries.
            Entries are ordered by insertion_order then priority (from activate()).
        """
        if position is not None:
            filtered = [r for r in results if r.position == position]
        else:
            # Return all activated entries (position ignored)
            filtered = [r for r in results if r.activated]

        if not filtered:
            return ""

        return "\n\n".join(r.content for r in filtered)

    def save_global_codex(self, entries: list[CharacterBookEntry]) -> None:
        """
        Save entries and volumes to global codex file.

        Args:
            entries: List of entries to save
        """
        global_dir = self.characters_dir / "_global"
        global_dir.mkdir(parents=True, exist_ok=True)
        global_path = global_dir / "codex.json"

        self._ensure_default_volume()

        book = CharacterBook(
            name="Global Codex",
            description="Entries active across all characters",
            entries=entries,
            volumes=self._volumes,
        )

        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(book.model_dump(exclude_none=True), f, indent=2)

        self._global_entries = entries
        self._merge_entries()

        logger.info("Saved %d entries to global codex", len(entries))

    def add_global_entry(self, entry: CharacterBookEntry) -> None:
        """Add a single entry to global codex and save."""
        self._global_entries.append(entry)
        self.save_global_codex(self._global_entries)

    def remove_global_entry(self, entry_id: int) -> bool:
        """
        Remove an entry from global codex by ID.

        Args:
            entry_id: The entry ID to remove

        Returns:
            True if entry was found and removed
        """
        for i, entry in enumerate(self._global_entries):
            if entry.id == entry_id:
                self._global_entries.pop(i)
                self.save_global_codex(self._global_entries)
                return True
        return False

    def get_status(self) -> dict[str, Any]:
        """Get current codex status for debugging/GUI."""
        return {
            "character_id": self._current_character_id,
            "total_entries": len(self._merged_entries),
            "global_entries": len(self._global_entries),
            "character_entries": len(self._character_entries),
            "volumes": len(self._volumes),
            "current_turn": self._state.current_turn,
            "active_sticky": self._state.get_active_sticky_entries(),
            "on_cooldown": self._state.get_entries_on_cooldown(),
            "scan_depth": self._scan_depth,
            "token_budget": self._token_budget,
            "recursive_scanning": self._recursive_scanning,
        }

    # --- Volume helpers (NANO-128) ---

    @staticmethod
    def _make_default_volume() -> CodexVolume:
        return CodexVolume(
            id=DEFAULT_VOLUME_ID,
            name="Default",
            enabled=True,
            insertion_order=0,
            description="Default volume for unassigned entries",
        )

    def _ensure_default_volume(self) -> None:
        """Guarantee vol_default exists in self._volumes."""
        if not any(v.id == DEFAULT_VOLUME_ID for v in self._volumes):
            self._volumes.insert(0, self._make_default_volume())

    @property
    def volumes(self) -> list[CodexVolume]:
        return list(self._volumes)

    def _save_volumes(self) -> None:
        """Persist current volumes + entries to disk and re-merge."""
        self.save_global_codex(self._global_entries)

    @staticmethod
    def _slugify(name: str) -> str:
        slug = name.lower().strip().replace(" ", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        return slug or "volume"

    def create_volume(self, name: str, description: str | None = None) -> CodexVolume:
        slug = f"vol_{self._slugify(name)}"
        existing_ids = {v.id for v in self._volumes}
        if slug in existing_ids:
            counter = 2
            while f"{slug}_{counter}" in existing_ids:
                counter += 1
            slug = f"{slug}_{counter}"

        max_order = max((v.insertion_order for v in self._volumes), default=-1)
        volume = CodexVolume(
            id=slug,
            name=name,
            enabled=True,
            insertion_order=max_order + 1,
            description=description,
        )
        self._volumes.append(volume)
        self._save_volumes()
        logger.info("Created volume %s (%s)", volume.name, volume.id)
        return volume

    def update_volume(self, volume_id: str, **kwargs: Any) -> bool:
        for vol in self._volumes:
            if vol.id == volume_id:
                for key, value in kwargs.items():
                    if key != "id" and hasattr(vol, key):
                        setattr(vol, key, value)
                self._save_volumes()
                return True
        return False

    def delete_volume(self, volume_id: str) -> bool:
        if volume_id == DEFAULT_VOLUME_ID:
            logger.warning("Cannot delete the Default volume")
            return False

        found = False
        for i, vol in enumerate(self._volumes):
            if vol.id == volume_id:
                self._volumes.pop(i)
                found = True
                break

        if not found:
            return False

        for entry in self._global_entries:
            if entry.volume_id == volume_id:
                entry.volume_id = DEFAULT_VOLUME_ID

        self._save_volumes()
        logger.info("Deleted volume %s, reassigned entries to Default", volume_id)
        return True

    def set_volume_enabled(self, volume_id: str, enabled: bool) -> bool:
        for vol in self._volumes:
            if vol.id == volume_id:
                vol.enabled = enabled
                self._save_volumes()
                return True
        return False

    def assign_entry_to_volume(self, entry_id: int, volume_id: str) -> bool:
        vol_ids = {v.id for v in self._volumes}
        if volume_id not in vol_ids:
            return False
        for entry in self._global_entries:
            if entry.id == entry_id:
                entry.volume_id = volume_id
                self._save_volumes()
                return True
        return False
