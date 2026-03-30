"""Tests for VTSDriver (NANO-060a). All pyvts calls mocked — no real WebSocket."""

import asyncio
import sys
import threading
import time
from pathlib import Path
from queue import SimpleQueue
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.config import VTubeStudioConfig
from spindl.vts.driver import VTSDriver


def _make_config(**overrides) -> VTubeStudioConfig:
    """Create a VTubeStudioConfig with test defaults."""
    defaults = {
        "enabled": True,
        "host": "localhost",
        "port": 8001,
        "token_path": "./test_token.txt",
        "plugin_name": "test-plugin",
        "developer": "test-dev",
        "expressions": {},
        "positions": {},
    }
    defaults.update(overrides)
    return VTubeStudioConfig(**defaults)


def _make_mock_vts(connected=True, authenticated=True):
    """Create a mock pyvts.vts instance.

    Uses MagicMock as base so sync methods (get_connection_status, etc.)
    return values directly. Async methods are explicitly set to AsyncMock.
    """
    mock_vts = MagicMock()

    # Sync status methods
    mock_vts.get_connection_status.return_value = 1 if connected else 0
    mock_vts.get_authentic_status.return_value = 2 if authenticated else 0

    # Async methods
    mock_vts.connect = AsyncMock()
    mock_vts.close = AsyncMock()
    mock_vts.request_authenticate_token = AsyncMock()
    mock_vts.request_authenticate = AsyncMock(return_value=authenticated)
    mock_vts.request = AsyncMock(return_value={
        "messageType": "APIResponse",
        "data": {},
    })

    # VTS request builder (sync object)
    mock_vts.vts_request = MagicMock()
    mock_vts.vts_request.requestTriggerHotKey.return_value = {
        "messageType": "HotkeyTriggerRequest",
        "data": {"hotkeyID": "test"},
    }
    mock_vts.vts_request.BaseRequest.return_value = {
        "messageType": "ExpressionActivationRequest",
        "data": {},
    }
    mock_vts.vts_request.requestHotKeyList.return_value = {
        "messageType": "HotkeysInCurrentModelRequest",
        "data": None,
    }
    mock_vts.vts_request.requestMoveModel.return_value = {
        "messageType": "MoveModelRequest",
        "data": {},
    }

    return mock_vts


class TestVTSDriverInit:
    """Tests for VTSDriver initialization."""

    def test_init_defaults(self):
        config = _make_config()
        driver = VTSDriver(config=config)
        assert driver.is_connected() is False
        assert driver._running is False

    def test_get_status_before_start(self):
        config = _make_config()
        driver = VTSDriver(config=config)
        status = driver.get_status()
        assert status["connected"] is False
        assert status["authenticated"] is False
        assert status["hotkeys"] == []
        assert status["expressions"] == []


class TestVTSDriverConnection:
    """Tests for connection and authentication."""

    @pytest.mark.asyncio
    async def test_connect_and_auth_happy_path(self):
        """Successful connection + auth sets connected and authenticated."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts(connected=True, authenticated=True)

        # Mock model info, hotkey/expression list responses
        mock_vts.request = AsyncMock(side_effect=[
            # First call: current model info
            {"messageType": "CurrentModelResponse", "data": {
                "modelLoaded": True, "modelName": "Hiyori",
            }},
            # Second call: list hotkeys
            {"messageType": "HotkeysInCurrentModelResponse", "data": {
                "availableHotkeys": [{"name": "Hotkey1"}, {"name": "Hotkey2"}],
            }},
            # Third call: list expressions
            {"messageType": "ExpressionStateResponse", "data": {
                "expressions": [
                    {"name": "Happy", "file": "happy.exp3.json", "active": False},
                ],
            }},
        ])

        driver._vts = mock_vts
        result = await driver._connect_and_auth()

        assert result is True
        assert driver._connected is True
        assert driver._authenticated is True
        assert driver._model_name == "Hiyori"
        assert driver._cached_hotkeys == ["Hotkey1", "Hotkey2"]
        assert len(driver._cached_expressions) == 1

    @pytest.mark.asyncio
    async def test_connect_failure_graceful(self):
        """Connection failure sets driver to inert state, no crash."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts(connected=False, authenticated=False)
        mock_vts.connect = AsyncMock(side_effect=Exception("Connection refused"))

        driver._vts = mock_vts
        result = await driver._connect_and_auth()

        assert result is False
        assert driver._connected is False
        assert driver._authenticated is False

    @pytest.mark.asyncio
    async def test_auth_recovery_on_stale_token(self):
        """Auth failure triggers force re-request, then retries."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts(connected=True, authenticated=False)

        # First auth fails, force token + second auth succeeds
        auth_call_count = 0
        async def auth_side_effect():
            nonlocal auth_call_count
            auth_call_count += 1
            if auth_call_count == 1:
                return False  # First attempt fails (stale token)
            return True  # Second attempt succeeds

        mock_vts.request_authenticate = AsyncMock(side_effect=auth_side_effect)
        # Suppress list queries after successful auth
        mock_vts.request = AsyncMock(return_value={"data": {}})

        driver._vts = mock_vts
        result = await driver._connect_and_auth()

        assert result is True
        assert driver._authenticated is True
        # force=True should have been called
        mock_vts.request_authenticate_token.assert_any_call(force=True)

    @pytest.mark.asyncio
    async def test_auth_recovery_both_fail(self):
        """Both auth attempts fail — driver stays unauthenticated."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts(connected=True, authenticated=False)
        mock_vts.request_authenticate = AsyncMock(return_value=False)

        driver._vts = mock_vts
        result = await driver._connect_and_auth()

        assert result is False
        assert driver._connected is True  # WebSocket connected
        assert driver._authenticated is False  # But not authenticated


class TestVTSDriverCommands:
    """Tests for command dispatch through queue."""

    @pytest.mark.asyncio
    async def test_trigger_hotkey_dispatch(self):
        """Hotkey trigger enqueues and dispatches correctly."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={"messageType": "OK", "data": {}})
        driver._vts = mock_vts
        driver._connected = True
        driver._authenticated = True

        await driver._do_trigger_hotkey("MyHotkey")

        mock_vts.vts_request.requestTriggerHotKey.assert_called_once_with("MyHotkey")
        mock_vts.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_expression_dispatch(self):
        """Expression activation dispatches BaseRequest correctly."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={"messageType": "OK", "data": {}})
        driver._vts = mock_vts
        driver._connected = True
        driver._authenticated = True

        await driver._do_trigger_expression("happy.exp3.json", True)

        mock_vts.vts_request.BaseRequest.assert_called_once_with(
            "ExpressionActivationRequest",
            {"expressionFile": "happy.exp3.json", "active": True},
        )

    @pytest.mark.asyncio
    async def test_move_model_default_preset(self):
        """Model movement uses default position presets."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={"messageType": "OK", "data": {}})
        driver._vts = mock_vts
        driver._connected = True
        driver._authenticated = True

        await driver._do_move_model("chat")

        mock_vts.vts_request.requestMoveModel.assert_called_once_with(
            0.4, -1.4, 0, -35, True, 0.5,
        )

    @pytest.mark.asyncio
    async def test_move_model_config_preset_overrides_default(self):
        """Config positions override default presets."""
        config = _make_config(positions={
            "chat": {"x": 0.1, "y": -0.5, "size": -20, "rotation": 10},
        })
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={"messageType": "OK", "data": {}})
        driver._vts = mock_vts
        driver._connected = True
        driver._authenticated = True

        await driver._do_move_model("chat")

        mock_vts.vts_request.requestMoveModel.assert_called_once_with(
            0.1, -0.5, 10, -20, True, 0.5,
        )

    @pytest.mark.asyncio
    async def test_move_model_unknown_preset_warns(self):
        """Unknown position preset logs warning, doesn't crash."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        driver._vts = mock_vts
        driver._connected = True
        driver._authenticated = True

        # Should not raise
        await driver._do_move_model("nonexistent_preset")
        mock_vts.vts_request.requestMoveModel.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_hotkeys(self):
        """Hotkey list query returns parsed names."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={
            "messageType": "HotkeysInCurrentModelResponse",
            "data": {
                "availableHotkeys": [
                    {"name": "Hotkey1"},
                    {"name": "Hotkey2"},
                    {"name": "Hotkey3"},
                ],
            },
        })
        driver._vts = mock_vts

        result = await driver._do_list_hotkeys()
        assert result == ["Hotkey1", "Hotkey2", "Hotkey3"]

    @pytest.mark.asyncio
    async def test_list_expressions(self):
        """Expression list query returns parsed dicts."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={
            "messageType": "ExpressionStateResponse",
            "data": {
                "expressions": [
                    {"name": "Happy", "file": "happy.exp3.json", "active": True},
                    {"name": "Sad", "file": "sad.exp3.json", "active": False},
                ],
            },
        })
        driver._vts = mock_vts

        result = await driver._do_list_expressions()
        assert len(result) == 2
        assert result[0]["name"] == "Happy"
        assert result[0]["file"] == "happy.exp3.json"
        assert result[0]["active"] is True
        assert result[1]["name"] == "Sad"


class TestVTSDriverQueueProcessing:
    """Tests for queue-based command dispatch."""

    @pytest.mark.asyncio
    async def test_process_queue_hotkey(self):
        """Queue processes hotkey command correctly."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={"messageType": "OK", "data": {}})
        driver._vts = mock_vts
        driver._connected = True
        driver._authenticated = True

        driver._queue.put(("trigger_hotkey", ("TestKey",), {}))
        await driver._process_queue()

        mock_vts.vts_request.requestTriggerHotKey.assert_called_once_with("TestKey")

    @pytest.mark.asyncio
    async def test_process_queue_drops_when_disconnected(self):
        """Commands dropped when not connected."""
        config = _make_config()
        driver = VTSDriver(config=config)
        driver._connected = False
        driver._authenticated = False

        driver._queue.put(("trigger_hotkey", ("TestKey",), {}))
        await driver._process_queue()
        # No crash, command silently dropped

    @pytest.mark.asyncio
    async def test_process_queue_with_callback(self):
        """List commands invoke callback with results."""
        config = _make_config()
        driver = VTSDriver(config=config)
        mock_vts = _make_mock_vts()
        mock_vts.request = AsyncMock(return_value={
            "data": {"availableHotkeys": [{"name": "HK1"}]},
        })
        driver._vts = mock_vts
        driver._connected = True
        driver._authenticated = True

        received = []
        driver._queue.put(("list_hotkeys", (), {"callback": lambda r: received.extend(r)}))
        await driver._process_queue()

        assert received == ["HK1"]
        assert driver._cached_hotkeys == ["HK1"]


class TestVTSDriverLifecycle:
    """Tests for start/stop lifecycle."""

    def test_stop_without_start(self):
        """Stop on unstarted driver is a no-op."""
        config = _make_config()
        driver = VTSDriver(config=config)
        driver.stop()  # Should not raise
        assert driver._running is False

    def test_start_sets_running(self):
        """Start sets _running and spawns thread."""
        config = _make_config()
        driver = VTSDriver(config=config)

        # Patch _run_loop to prevent actual connection
        with patch.object(driver, '_run_loop'):
            driver.start()
            assert driver._running is True
            assert driver._thread is not None
            assert driver._thread.daemon is True
            driver.stop()

    def test_double_start_noop(self):
        """Calling start() twice doesn't spawn a second thread."""
        config = _make_config()
        driver = VTSDriver(config=config)

        with patch.object(driver, '_run_loop'):
            driver.start()
            first_thread = driver._thread
            driver.start()  # Should be no-op
            assert driver._thread is first_thread
            driver.stop()

    def test_thread_safe_enqueue(self):
        """Commands can be enqueued from multiple threads."""
        config = _make_config()
        driver = VTSDriver(config=config)

        def enqueue_commands():
            for i in range(100):
                driver.trigger_hotkey(f"key_{i}")

        threads = [threading.Thread(target=enqueue_commands) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert driver._queue.qsize() == 400
