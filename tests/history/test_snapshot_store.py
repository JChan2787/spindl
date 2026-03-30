"""Tests for snapshot_store.py (NANO-076)."""

import json
import pytest
from pathlib import Path

from spindl.history.snapshot_store import (
    _sidecar_path,
    append_snapshot,
    read_latest_snapshot,
    read_snapshot_history,
    delete_sidecar,
)


def _make_snapshot(turn_id: int = 1, estimated: bool = False) -> dict:
    """Create a minimal test snapshot."""
    return {
        "turn_id": turn_id,
        "messages": [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": f"Hello turn {turn_id}"},
        ],
        "token_breakdown": {
            "total": 100,
            "prompt": 80,
            "completion": 20,
            "system": 60,
            "user": 20,
            "sections": {"agent": 30, "context": 15, "rules": 10, "conversation": 5},
        },
        "block_contents": None,
        "input_modality": "TEXT",
        "state_trigger": None,
        "timestamp": 1709654400.0 + turn_id,
        "estimated": estimated,
    }


class TestSidecarPath:
    def test_derives_sidecar_from_session(self, tmp_path):
        session = tmp_path / "spindle_20260305_143022.jsonl"
        assert _sidecar_path(session) == tmp_path / "spindle_20260305_143022.snapshot.jsonl"


class TestAppendSnapshot:
    def test_creates_sidecar_file(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        append_snapshot(session, _make_snapshot())
        sidecar = _sidecar_path(session)
        assert sidecar.exists()

    def test_appends_json_line(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        append_snapshot(session, _make_snapshot(1))
        append_snapshot(session, _make_snapshot(2))
        sidecar = _sidecar_path(session)
        lines = sidecar.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["turn_id"] == 1
        assert json.loads(lines[1])["turn_id"] == 2

    def test_creates_parent_dirs(self, tmp_path):
        session = tmp_path / "deep" / "nested" / "test.jsonl"
        append_snapshot(session, _make_snapshot())
        assert _sidecar_path(session).exists()

    def test_never_raises_on_write_error(self, tmp_path, monkeypatch):
        """Sidecar write failure must never propagate."""
        session = tmp_path / "test.jsonl"
        session.touch()

        def _boom(*args, **kwargs):
            raise PermissionError("disk full")

        monkeypatch.setattr("builtins.open", _boom)
        # Should not raise
        append_snapshot(session, _make_snapshot())


class TestReadLatestSnapshot:
    def test_returns_none_for_missing_sidecar(self, tmp_path):
        session = tmp_path / "nonexistent.jsonl"
        assert read_latest_snapshot(session) is None

    def test_returns_none_for_empty_sidecar(self, tmp_path):
        session = tmp_path / "test.jsonl"
        _sidecar_path(session).write_text("", encoding="utf-8")
        assert read_latest_snapshot(session) is None

    def test_returns_last_snapshot(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        append_snapshot(session, _make_snapshot(1))
        append_snapshot(session, _make_snapshot(2))
        append_snapshot(session, _make_snapshot(3))
        result = read_latest_snapshot(session)
        assert result is not None
        assert result["turn_id"] == 3

    def test_handles_trailing_newlines(self, tmp_path):
        session = tmp_path / "test.jsonl"
        sidecar = _sidecar_path(session)
        sidecar.write_text(
            json.dumps(_make_snapshot(1)) + "\n\n\n",
            encoding="utf-8",
        )
        result = read_latest_snapshot(session)
        assert result is not None
        assert result["turn_id"] == 1


class TestReadSnapshotHistory:
    def test_returns_empty_for_missing_sidecar(self, tmp_path):
        session = tmp_path / "nonexistent.jsonl"
        assert read_snapshot_history(session) == []

    def test_returns_all_snapshots_in_order(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        for i in range(1, 6):
            append_snapshot(session, _make_snapshot(i))
        history = read_snapshot_history(session)
        assert len(history) == 5
        assert [s["turn_id"] for s in history] == [1, 2, 3, 4, 5]


class TestDeleteSidecar:
    def test_deletes_existing_sidecar(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        append_snapshot(session, _make_snapshot())
        sidecar = _sidecar_path(session)
        assert sidecar.exists()
        delete_sidecar(session)
        assert not sidecar.exists()

    def test_silent_on_missing_sidecar(self, tmp_path):
        session = tmp_path / "nonexistent.jsonl"
        # Should not raise
        delete_sidecar(session)

    def test_does_not_delete_session_file(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        append_snapshot(session, _make_snapshot())
        delete_sidecar(session)
        assert session.exists()
