"""
Character file utilities for E2E tests.

NANO-038: Provides helpers for modifying and restoring character card files
during test execution, enabling tests for hot-reload and codex features.
"""

import json
import yaml
from pathlib import Path
from typing import Any


# Cache of original file contents for restoration
_original_files: dict[str, str] = {}


def _get_project_root() -> Path:
    """Get the spindl-project root directory."""
    # helpers/ -> tests_e2e/ -> project root
    return Path(__file__).parent.parent.parent


def _get_characters_dir() -> Path:
    """
    Derive characters_dir from the active config.

    The harness copies the test config to config/spindl.yaml at startup.
    We read that file to find the characters_dir path.

    Returns:
        Path to the characters directory.
    """
    project_root = _get_project_root()
    config_path = project_root / "config" / "spindl.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check character block first, then legacy persona block
    if "character" in config:
        chars_dir = config["character"].get("directory", "./characters")
    elif "persona" in config:
        chars_dir = config["persona"].get("directory", "./personas")
    else:
        chars_dir = "./characters"

    # Resolve relative to project root
    chars_path = Path(chars_dir)
    if not chars_path.is_absolute():
        chars_path = project_root / chars_path

    return chars_path


def _get_card_path(character_id: str) -> Path:
    """Get the full path to a character's card.json."""
    return _get_characters_dir() / character_id / "card.json"


async def modify_character_file(
    harness,
    character_id: str,
    modifications: dict[str, Any],
) -> None:
    """
    Load a character card, apply modifications, and write back.

    Saves the original content for later restoration via restore_character_file().

    Modifications are applied as updates to the card's ``data`` block.
    Example: ``{"system_prompt": "New prompt"}`` updates ``data.system_prompt``.
    Nested: ``{"character_book": {"entries": [...]}}`` replaces ``data.character_book``.

    Args:
        harness: E2EHarness instance.
        character_id: Character folder name (e.g., "test_agent").
        modifications: Dict of fields to update in the card's data block.
    """
    card_path = _get_card_path(character_id)
    card_path_str = str(card_path)

    # Save original if not already saved
    if card_path_str not in _original_files:
        _original_files[card_path_str] = card_path.read_text(encoding="utf-8")

    # Load current card
    card_data = json.loads(card_path.read_text(encoding="utf-8"))

    # Apply modifications to data block
    for key, value in modifications.items():
        card_data["data"][key] = value

    # Write back
    card_path.write_text(
        json.dumps(card_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


async def restore_character_file(
    harness,
    character_id: str,
) -> None:
    """
    Restore a character file to its original state.

    Must be called after modify_character_file() to clean up.

    Args:
        harness: E2EHarness instance.
        character_id: Character folder name.
    """
    card_path = _get_card_path(character_id)
    card_path_str = str(card_path)

    if card_path_str in _original_files:
        card_path.write_text(_original_files[card_path_str], encoding="utf-8")
        del _original_files[card_path_str]


async def restore_all_character_files() -> None:
    """Restore all modified character files. Call in test teardown."""
    for path_str, original_content in list(_original_files.items()):
        Path(path_str).write_text(original_content, encoding="utf-8")
    _original_files.clear()


async def wait_for_state(
    harness,
    target_state: str,
    timeout: float = 10.0,
) -> dict:
    """
    Wait until the agent reaches a specific state.

    Listens for state_changed events and returns when the target state
    is reached.

    Args:
        harness: E2EHarness instance.
        target_state: State to wait for (e.g., "IDLE", "LISTENING", "PROCESSING").
        timeout: Maximum wait time in seconds.

    Returns:
        The state_changed event data.

    Raises:
        TimeoutError: If target state not reached within timeout.
    """
    return await harness.wait_for_event(
        "state_changed",
        timeout=timeout,
        predicate=lambda e: e.get("to") == target_state,
    )
