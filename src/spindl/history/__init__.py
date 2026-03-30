"""
Conversation history storage utilities.

Provides JSONL-based storage for conversation turns with support
for hidden turns (for future summarization).
"""

from .jsonl_store import (
    generate_session_filename,
    generate_uuid,
    append_turn,
    append_summary,
    patch_last_turn,
    read_turns,
    read_visible_turns,
    mark_turns_hidden,
    get_latest_session,
    get_next_turn_id,
)
from .snapshot_store import (
    append_snapshot,
    read_latest_snapshot,
    read_snapshot_history,
    delete_sidecar,
)

__all__ = [
    "generate_session_filename",
    "generate_uuid",
    "append_turn",
    "append_summary",
    "patch_last_turn",
    "read_turns",
    "read_visible_turns",
    "mark_turns_hidden",
    "get_latest_session",
    "get_next_turn_id",
    "append_snapshot",
    "read_latest_snapshot",
    "read_snapshot_history",
    "delete_sidecar",
]
