"""JSONL storage utilities for conversation history."""

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator


def generate_uuid() -> str:
    """Generate a UUID v4 string."""
    return str(uuid.uuid4())


def generate_session_filename(persona_id: str) -> str:
    """
    Generate a unique session filename.

    Args:
        persona_id: Persona identifier (e.g., "spindle")

    Returns:
        Filename like "spindle_20260119_114500.jsonl"
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{persona_id}_{timestamp}.jsonl"


def append_turn(filepath: Path, turn: dict) -> None:
    """
    Append a single turn to a JSONL file.

    Args:
        filepath: Path to the JSONL file
        turn: Turn dict with turn_id, uuid, role, content, timestamp, hidden

    Creates the file and parent directories if they don't exist.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn, ensure_ascii=False) + "\n")


def patch_last_turn(filepath: Path, updates: dict) -> None:
    """
    Patch metadata onto the last line of a JSONL file (NANO-094).

    Reads the last line, merges updates, rewrites it in place.
    Used to attach emotion metadata after classification (which runs
    after the pipeline's HistoryRecorder has already written the turn).
    """
    if not filepath.exists():
        return

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return

    last_turn = json.loads(lines[-1])
    last_turn.update(updates)
    lines[-1] = json.dumps(last_turn, ensure_ascii=False) + "\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)


def get_next_turn_id(filepath: Path) -> int:
    """
    Get the next sequential turn_id for a session file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        Next turn_id (1 if file doesn't exist or is empty)
    """
    if not filepath.exists():
        return 1
    turns = read_turns(filepath)
    if not turns:
        return 1
    return max(t.get("turn_id", 0) for t in turns) + 1


def read_turns(filepath: Path) -> list[dict]:
    """
    Read all turns from a JSONL file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of turn dicts in order

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    turns = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def read_visible_turns(filepath: Path) -> list[dict]:
    """
    Read only visible (non-hidden) turns from a JSONL file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of turn dicts where hidden == False
    """
    return [t for t in read_turns(filepath) if not t.get("hidden", False)]


def mark_turns_hidden(filepath: Path, up_to_turn_id: int) -> None:
    """
    Mark turns as hidden up to (and including) a given turn_id.

    This rewrites the entire file. Used when summarization occurs.

    Args:
        filepath: Path to the JSONL file
        up_to_turn_id: Mark turns with turn_id <= this value as hidden
    """
    turns = read_turns(filepath)
    for turn in turns:
        if turn.get("turn_id", 0) <= up_to_turn_id and turn.get("role") != "summary":
            turn["hidden"] = True

    # Rewrite file
    with open(filepath, "w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")


def _last_session_marker(conversations_dir: Path, persona_id: str) -> Path:
    """Return path to the .last_session marker file for a persona."""
    return conversations_dir / f".last_session_{persona_id}"


def save_last_session(conversations_dir: Path, persona_id: str, session_filename: str) -> None:
    """Persist the active session filename for a persona."""
    marker = _last_session_marker(conversations_dir, persona_id)
    marker.write_text(session_filename, encoding="utf-8")


def get_latest_session(conversations_dir: Path, persona_id: str) -> Path | None:
    """
    Get the most recent session file for a persona.

    Checks for a persisted .last_session marker first. Falls back to
    the most recent non-empty session file by filename.

    Args:
        conversations_dir: Directory containing JSONL files
        persona_id: Persona identifier

    Returns:
        Path to most recent session file, or None if no sessions exist
    """
    # Check persisted selection first
    marker = _last_session_marker(conversations_dir, persona_id)
    if marker.exists():
        saved_name = marker.read_text(encoding="utf-8").strip()
        if saved_name:
            saved_path = conversations_dir / saved_name
            if saved_path.exists() and saved_path.stat().st_size > 0:
                return saved_path

    # Fallback: most recent by filename
    pattern = f"{persona_id}_*.jsonl"
    sessions = sorted(
        (
            p for p in conversations_dir.glob(pattern)
            if ".snapshot." not in p.name and p.stat().st_size > 0
        ),
        reverse=True,
    )
    return sessions[0] if sessions else None


def append_summary(
    filepath: Path,
    summary_content: str,
    summarizes_up_to: int,
) -> dict:
    """
    Append a summary turn to a JSONL file.

    Creates a summary record and marks all previous turns (up to and including
    summarizes_up_to) as hidden.

    Args:
        filepath: Path to the JSONL file
        summary_content: The generated summary text
        summarizes_up_to: turn_id of the last turn covered by this summary

    Returns:
        The created summary turn dict

    Side Effects:
        - Appends summary turn to file
        - Rewrites file to mark covered turns as hidden
    """
    # Get next turn_id
    next_id = get_next_turn_id(filepath)

    # Create summary turn
    summary_turn = {
        "turn_id": next_id,
        "uuid": generate_uuid(),
        "role": "summary",
        "content": summary_content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hidden": False,
        "summarizes_up_to": summarizes_up_to,
    }

    # Append summary
    append_turn(filepath, summary_turn)

    # Mark old turns as hidden
    mark_turns_hidden(filepath, up_to_turn_id=summarizes_up_to)

    return summary_turn
