"""
Tests for NANO-076 snapshot persistence in OrchestratorCallbacks.

Tests _persist_snapshot(), build_prompt_snapshot(), and set_session_file_getter().
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spindl.orchestrator.callbacks import OrchestratorCallbacks
from spindl.history.snapshot_store import _sidecar_path, read_latest_snapshot


def _make_callbacks(session_file=None, pipeline=None, persona=None):
    """Create a minimal OrchestratorCallbacks with optional session file getter."""
    cb = OrchestratorCallbacks.__new__(OrchestratorCallbacks)
    cb._pipeline = pipeline
    cb._persona = persona or {"id": "test", "system_prompt": "Test."}
    cb._event_bus = None
    cb._session_file_getter = (lambda: session_file) if session_file else None
    cb._total_turns = 0
    return cb


class TestPersistSnapshot:
    def test_persists_to_sidecar(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        cb = _make_callbacks(session_file=session)

        cb._persist_snapshot(
            messages=[{"role": "system", "content": "hi"}],
            token_breakdown={"total": 10, "prompt": 10, "completion": 0},
            input_modality="TEXT",
            state_trigger=None,
            block_contents=None,
        )

        snapshot = read_latest_snapshot(session)
        assert snapshot is not None
        assert snapshot["estimated"] is False
        assert snapshot["turn_id"] == 1
        assert snapshot["messages"] == [{"role": "system", "content": "hi"}]

    def test_skips_without_session_file_getter(self, tmp_path):
        cb = _make_callbacks(session_file=None)
        cb._session_file_getter = None
        # Should not raise
        cb._persist_snapshot(
            messages=[], token_breakdown={},
            input_modality="TEXT", state_trigger=None, block_contents=None,
        )

    def test_skips_when_getter_returns_none(self, tmp_path):
        cb = _make_callbacks()
        cb._session_file_getter = lambda: None
        # Should not raise
        cb._persist_snapshot(
            messages=[], token_breakdown={},
            input_modality="TEXT", state_trigger=None, block_contents=None,
        )

    def test_increments_turn_id(self, tmp_path):
        session = tmp_path / "test.jsonl"
        session.touch()
        cb = _make_callbacks(session_file=session)

        cb._total_turns = 3
        cb._persist_snapshot(
            messages=[], token_breakdown={},
            input_modality="TEXT", state_trigger=None, block_contents=None,
        )
        snapshot = read_latest_snapshot(session)
        assert snapshot["turn_id"] == 4


class TestBuildPromptSnapshot:
    def test_returns_none_without_pipeline(self):
        cb = _make_callbacks(pipeline=None)
        assert cb.build_prompt_snapshot() is None

    def test_returns_none_without_persona(self):
        cb = _make_callbacks(pipeline=MagicMock())
        cb._persona = None
        assert cb.build_prompt_snapshot() is None

    def test_delegates_to_pipeline(self):
        mock_pipeline = MagicMock()
        expected = {"messages": [], "estimated": True}
        mock_pipeline.build_snapshot.return_value = expected

        cb = _make_callbacks(
            pipeline=mock_pipeline,
            persona={"id": "test", "system_prompt": "Test."},
        )
        result = cb.build_prompt_snapshot()
        assert result == expected
        mock_pipeline.build_snapshot.assert_called_once_with(cb._persona)

    def test_returns_none_on_exception(self):
        mock_pipeline = MagicMock()
        mock_pipeline.build_snapshot.side_effect = RuntimeError("boom")

        cb = _make_callbacks(
            pipeline=mock_pipeline,
            persona={"id": "test"},
        )
        assert cb.build_prompt_snapshot() is None


class TestSetSessionFileGetter:
    def test_setter_wires_getter(self, tmp_path):
        cb = OrchestratorCallbacks.__new__(OrchestratorCallbacks)
        cb._session_file_getter = None
        cb._total_turns = 0

        session = tmp_path / "test.jsonl"
        session.touch()
        cb.set_session_file_getter(lambda: session)

        assert cb._session_file_getter() == session
