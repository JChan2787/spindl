"""
Schema envelope validation for game-state bridge events (NANO-116).

Validates incoming TCP events against the vendored SPNDL-001 schema.
Envelope validation (required top-level fields) is done manually to
avoid a hard dependency on jsonschema. Per-payload validation is
intentionally loose — the conversational model is tolerant of
unexpected fields, and we'd rather log-and-pass than crash on a
minor schema drift.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = frozenset(
    {"protocol_version", "event_type", "event_source", "timestamp", "sequence", "payload"}
)

_SCHEMA_DIR = Path(__file__).parent / "schema"
_SCHEMA_FILE = _SCHEMA_DIR / "stimulus_event.schema.json"


def load_vendored_version() -> str | None:
    """Read the protocol_version from the vendored schema file."""
    try:
        with open(_SCHEMA_FILE, encoding="utf-8") as f:
            schema = json.load(f)
        return schema.get("protocol_version")
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load vendored schema from %s", _SCHEMA_FILE)
        return None


def validate_envelope(event: dict) -> tuple[bool, str]:
    """
    Check that an event dict has all required top-level fields.

    Returns (ok, reason). On success reason is empty.
    """
    missing = _REQUIRED_FIELDS - event.keys()
    if missing:
        return False, f"missing required fields: {sorted(missing)}"

    if not isinstance(event.get("sequence"), int):
        return False, "sequence must be an integer"

    if not isinstance(event.get("payload"), dict):
        return False, "payload must be an object"

    return True, ""


def check_protocol_version(
    event_version: str, schema_version: str
) -> tuple[bool, str]:
    """
    Compare an event's protocol_version against the vendored schema version.

    Returns (ok, level) where:
        - ok=True,  level="match" — exact match
        - ok=True,  level="patch" — patch drift, silent
        - ok=True,  level="minor" — minor drift, log warning
        - ok=False, level="major" — major mismatch, reject
        - ok=False, level="parse_error" — couldn't parse one/both versions
    """
    try:
        ev_parts = [int(x) for x in event_version.split(".")]
        sv_parts = [int(x) for x in schema_version.split(".")]
    except (ValueError, AttributeError):
        return False, "parse_error"

    if len(ev_parts) < 3 or len(sv_parts) < 3:
        return False, "parse_error"

    if ev_parts[0] != sv_parts[0]:
        return False, "major"

    if ev_parts[1] != sv_parts[1]:
        return True, "minor"

    if ev_parts[2] != sv_parts[2]:
        return True, "patch"

    return True, "match"
