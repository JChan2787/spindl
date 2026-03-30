"""Tests for VTubeStudio socket handlers (NANO-060b).

Tests cover:
- set_vts_config: updates config, persists, emits response
- request_vts_status: returns correct state with/without driver
- request_vts_hotkeys: cached + refresh paths
- request_vts_expressions: cached + refresh paths
- send_vts_hotkey: dispatches to driver
- send_vts_expression: dispatches to driver
- send_vts_move: dispatches to driver
- Guards: handlers handle missing orchestrator/driver gracefully
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.gui.server import GUIServer
from spindl.orchestrator.config import VTubeStudioConfig


# ============================================================================
# Helpers
# ============================================================================


def _make_server_with_orchestrator(vts_enabled=True, vts_connected=True):
    """Create a GUIServer with mocked orchestrator + VTS driver."""
    server = GUIServer.__new__(GUIServer)
    server.sio = MagicMock()
    server.sio.event = lambda fn: fn  # passthrough decorator
    server.sio.emit = AsyncMock()
    server._orchestrator = MagicMock()
    server._config_path = "/tmp/test_config.yaml"
    server._event_loop = asyncio.new_event_loop()
    server._launch_in_progress = False

    # Config
    cfg = VTubeStudioConfig(
        enabled=vts_enabled,
        host="localhost",
        port=8001,
    )
    server._orchestrator._config = MagicMock()
    server._orchestrator._config.vtubestudio_config = cfg
    server._orchestrator._config.save_to_yaml = MagicMock()

    # VTS driver mock
    if vts_enabled:
        driver = MagicMock()
        driver.get_status.return_value = {
            "connected": vts_connected,
            "authenticated": vts_connected,
            "enabled": True,
            "model_name": "TestModel" if vts_connected else None,
            "hotkeys": ["wave", "dance"] if vts_connected else [],
            "expressions": [
                {"file": "happy.exp3.json", "name": "happy", "active": False}
            ] if vts_connected else [],
        }
        driver.trigger_hotkey = MagicMock()
        driver.trigger_expression = MagicMock()
        driver.move_model = MagicMock()
        driver.request_hotkey_list = MagicMock()
        driver.request_expression_list = MagicMock()
        server._orchestrator.vts_driver = driver
    else:
        server._orchestrator.vts_driver = None

    return server


# ============================================================================
# set_vts_config
# ============================================================================


class TestSetVTSConfig:
    """Tests for set_vts_config socket handler."""

    @pytest.mark.asyncio
    async def test_enable_toggle(self):
        """Should call update_vts_config and emit response."""
        server = _make_server_with_orchestrator(vts_enabled=False)

        # Register handlers to get references
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["set_vts_config"]
        await handler("test-sid", {"enabled": True})

        server._orchestrator.update_vts_config.assert_called_once_with(
            enabled=True, host=None, port=None,
        )
        server._orchestrator._config.save_to_yaml.assert_called_once()
        server.sio.emit.assert_called()
        call_args = server.sio.emit.call_args
        assert call_args[0][0] == "vts_config_updated"

    @pytest.mark.asyncio
    async def test_port_validation(self):
        """Should reject invalid port values."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["set_vts_config"]
        await handler("test-sid", {"port": -1})

        # Port should be None (invalid), not passed as -1
        server._orchestrator.update_vts_config.assert_called_once_with(
            enabled=None, host=None, port=None,
        )

    @pytest.mark.asyncio
    async def test_no_orchestrator(self):
        """Should not crash when orchestrator is None."""
        server = _make_server_with_orchestrator()
        server._orchestrator = None
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["set_vts_config"]
        # Should not raise
        await handler("test-sid", {"enabled": True})


# ============================================================================
# request_vts_status
# ============================================================================


class TestRequestVTSStatus:
    """Tests for request_vts_status socket handler."""

    @pytest.mark.asyncio
    async def test_with_connected_driver(self):
        """Should return full status from driver."""
        server = _make_server_with_orchestrator(vts_connected=True)
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_status"]
        await handler("test-sid", {})

        server.sio.emit.assert_called_once()
        call_args = server.sio.emit.call_args
        assert call_args[0][0] == "vts_status"
        data = call_args[0][1]
        assert data["connected"] is True
        assert data["model_name"] == "TestModel"
        assert len(data["hotkeys"]) == 2

    @pytest.mark.asyncio
    async def test_no_driver(self):
        """Should return disconnected status when driver is None."""
        server = _make_server_with_orchestrator(vts_enabled=False)
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_status"]
        await handler("test-sid", {})

        call_args = server.sio.emit.call_args
        data = call_args[0][1]
        assert data["connected"] is False
        assert data["hotkeys"] == []

    @pytest.mark.asyncio
    async def test_no_orchestrator(self):
        """Should return disconnected status when orchestrator is None."""
        server = _make_server_with_orchestrator()
        server._orchestrator = None
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_status"]
        await handler("test-sid", {})

        call_args = server.sio.emit.call_args
        data = call_args[0][1]
        assert data["connected"] is False
        assert data["enabled"] is False


# ============================================================================
# request_vts_hotkeys
# ============================================================================


class TestRequestVTSHotkeys:
    """Tests for request_vts_hotkeys socket handler."""

    @pytest.mark.asyncio
    async def test_cached_list(self):
        """Default should serve cached hotkeys from get_status()."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_hotkeys"]
        await handler("test-sid", {})

        call_args = server.sio.emit.call_args
        assert call_args[0][0] == "vts_hotkeys"
        assert call_args[0][1]["hotkeys"] == ["wave", "dance"]

    @pytest.mark.asyncio
    async def test_refresh_dispatches_live_query(self):
        """Refresh flag should call driver.request_hotkey_list with callback."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_hotkeys"]
        await handler("test-sid", {"refresh": True})

        driver = server._orchestrator.vts_driver
        driver.request_hotkey_list.assert_called_once()
        # Callback should be a callable
        call_kwargs = driver.request_hotkey_list.call_args
        assert call_kwargs[1]["callback"] is not None
        assert callable(call_kwargs[1]["callback"])

    @pytest.mark.asyncio
    async def test_no_driver(self):
        """Should return empty list when no driver."""
        server = _make_server_with_orchestrator(vts_enabled=False)
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_hotkeys"]
        await handler("test-sid", {})

        call_args = server.sio.emit.call_args
        assert call_args[0][1]["hotkeys"] == []


# ============================================================================
# request_vts_expressions
# ============================================================================


class TestRequestVTSExpressions:
    """Tests for request_vts_expressions socket handler."""

    @pytest.mark.asyncio
    async def test_cached_list(self):
        """Default should serve cached expressions from get_status()."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_expressions"]
        await handler("test-sid", {})

        call_args = server.sio.emit.call_args
        assert call_args[0][0] == "vts_expressions"
        assert len(call_args[0][1]["expressions"]) == 1

    @pytest.mark.asyncio
    async def test_refresh_dispatches_live_query(self):
        """Refresh flag should call driver.request_expression_list with callback."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["request_vts_expressions"]
        await handler("test-sid", {"refresh": True})

        driver = server._orchestrator.vts_driver
        driver.request_expression_list.assert_called_once()
        call_kwargs = driver.request_expression_list.call_args
        assert callable(call_kwargs[1]["callback"])


# ============================================================================
# send_vts_hotkey
# ============================================================================


class TestSendVTSHotkey:
    """Tests for send_vts_hotkey socket handler."""

    @pytest.mark.asyncio
    async def test_trigger(self):
        """Should dispatch to driver.trigger_hotkey."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_hotkey"]
        await handler("test-sid", {"name": "wave"})

        server._orchestrator.vts_driver.trigger_hotkey.assert_called_once_with("wave")
        call_args = server.sio.emit.call_args
        assert call_args[0][0] == "vts_hotkey_triggered"

    @pytest.mark.asyncio
    async def test_no_name(self):
        """Should not dispatch if name is missing."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_hotkey"]
        await handler("test-sid", {})

        server._orchestrator.vts_driver.trigger_hotkey.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_driver(self):
        """Should not crash when driver is None."""
        server = _make_server_with_orchestrator(vts_enabled=False)
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_hotkey"]
        await handler("test-sid", {"name": "wave"})
        # Should not raise


# ============================================================================
# send_vts_expression
# ============================================================================


class TestSendVTSExpression:
    """Tests for send_vts_expression socket handler."""

    @pytest.mark.asyncio
    async def test_activate(self):
        """Should dispatch activate to driver."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_expression"]
        await handler("test-sid", {"file": "happy.exp3.json", "active": True})

        server._orchestrator.vts_driver.trigger_expression.assert_called_once_with(
            "happy.exp3.json", True
        )

    @pytest.mark.asyncio
    async def test_deactivate(self):
        """Should dispatch deactivate to driver."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_expression"]
        await handler("test-sid", {"file": "happy.exp3.json", "active": False})

        server._orchestrator.vts_driver.trigger_expression.assert_called_once_with(
            "happy.exp3.json", False
        )

    @pytest.mark.asyncio
    async def test_default_active_true(self):
        """Should default to active=True when not specified."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_expression"]
        await handler("test-sid", {"file": "happy.exp3.json"})

        server._orchestrator.vts_driver.trigger_expression.assert_called_once_with(
            "happy.exp3.json", True
        )


# ============================================================================
# send_vts_move
# ============================================================================


class TestSendVTSMove:
    """Tests for send_vts_move socket handler."""

    @pytest.mark.asyncio
    async def test_move_preset(self):
        """Should dispatch to driver.move_model."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_move"]
        await handler("test-sid", {"preset": "chat"})

        server._orchestrator.vts_driver.move_model.assert_called_once_with("chat")
        call_args = server.sio.emit.call_args
        assert call_args[0][0] == "vts_move_triggered"

    @pytest.mark.asyncio
    async def test_no_preset(self):
        """Should not dispatch if preset is missing."""
        server = _make_server_with_orchestrator()
        handlers = {}
        server.sio.event = lambda fn: handlers.__setitem__(fn.__name__, fn) or fn
        server._register_handlers()

        handler = handlers["send_vts_move"]
        await handler("test-sid", {})

        server._orchestrator.vts_driver.move_model.assert_not_called()
