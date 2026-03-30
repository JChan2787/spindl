"""
Prompt snapshot JSONL sidecar store (NANO-076).

Persists prompt snapshots alongside session JSONL files as `.snapshot.jsonl` sidecars.
Each line is a JSON object representing one pipeline run's prompt state.

Hot path: read last line for hydration.
Cold path: read all lines for history/diffing.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _sidecar_path(session_file: Path) -> Path:
    """Derive the snapshot sidecar path from a session JSONL path."""
    return session_file.with_suffix(".snapshot.jsonl")


def append_snapshot(session_file: Path, snapshot: dict) -> None:
    """
    Append a prompt snapshot to the sidecar JSONL.

    Args:
        session_file: Path to the session's conversation JSONL.
        snapshot: Snapshot dict (messages, token_breakdown, block_contents, etc).

    Never raises — logs errors and moves on. A sidecar write failure
    must never block the live LLM response.
    """
    try:
        sidecar = _sidecar_path(session_file)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("Failed to write snapshot sidecar for %s", session_file, exc_info=True)


def read_latest_snapshot(session_file: Path) -> dict | None:
    """
    Read the most recent snapshot from the sidecar.

    Args:
        session_file: Path to the session's conversation JSONL.

    Returns:
        Parsed dict of the last snapshot line, or None if sidecar
        doesn't exist or is empty.
    """
    sidecar = _sidecar_path(session_file)
    if not sidecar.exists():
        return None

    try:
        last_line = None
        with open(sidecar, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if last_line:
            return json.loads(last_line)
    except Exception:
        logger.warning("Failed to read snapshot sidecar for %s", session_file, exc_info=True)

    return None


def read_snapshot_history(session_file: Path) -> list[dict]:
    """
    Read all snapshots from the sidecar.

    Args:
        session_file: Path to the session's conversation JSONL.

    Returns:
        List of snapshot dicts in chronological order.
        Empty list if sidecar doesn't exist.
    """
    sidecar = _sidecar_path(session_file)
    if not sidecar.exists():
        return []

    snapshots = []
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    snapshots.append(json.loads(stripped))
    except Exception:
        logger.warning("Failed to read snapshot history for %s", session_file, exc_info=True)

    return snapshots


def delete_sidecar(session_file: Path) -> None:
    """
    Delete the snapshot sidecar for a session.

    Called during session deletion cleanup. Silent if sidecar doesn't exist.
    """
    sidecar = _sidecar_path(session_file)
    try:
        if sidecar.exists():
            sidecar.unlink()
    except Exception:
        logger.warning("Failed to delete snapshot sidecar for %s", session_file, exc_info=True)
