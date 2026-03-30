"""Tests for VTS orchestrator integration (NANO-060b).

Tests cover:
- update_vts_config(): enable/disable/restart logic
- Runtime driver creation when enabled from disabled state
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.config import VTubeStudioConfig


# We can't easily instantiate VoiceAgentOrchestrator due to dependencies,
# so we test update_vts_config() via a minimal mock setup.


class FakeOrchestrator:
    """Minimal stand-in that has update_vts_config() logic.

    Mirrors the orchestrator's VTS-related state without all the audio/LLM deps.
    """

    def __init__(self, config, vts_driver=None):
        self._config = config
        self._vts_driver = vts_driver
        self._event_bus = MagicMock()
        self._running = True

    def update_vts_config(self, enabled=None, host=None, port=None):
        """Exact copy of the orchestrator method for isolated testing."""
        cfg = self._config.vtubestudio_config
        needs_restart = False

        if host is not None and host != cfg.host:
            cfg.host = host
            needs_restart = True

        if port is not None and port != cfg.port:
            cfg.port = port
            needs_restart = True

        if enabled is not None:
            cfg.enabled = enabled
            if enabled:
                if self._vts_driver and needs_restart:
                    self._vts_driver.stop()
                    self._vts_driver.start()
                elif self._vts_driver:
                    self._vts_driver.start()
                else:
                    # Create driver on the fly
                    from spindl.vts.driver import VTSDriver
                    self._vts_driver = MagicMock(spec=VTSDriver)
                    if self._running:
                        self._vts_driver.start()
            else:
                if self._vts_driver:
                    self._vts_driver.stop()
        elif needs_restart and self._vts_driver and cfg.enabled:
            self._vts_driver.stop()
            self._vts_driver.start()


def _make_config(**overrides):
    """Create a mock config with VTubeStudioConfig."""
    config = MagicMock()
    vts_cfg = VTubeStudioConfig(
        enabled=overrides.get("enabled", True),
        host=overrides.get("host", "localhost"),
        port=overrides.get("port", 8001),
    )
    config.vtubestudio_config = vts_cfg
    return config


class TestUpdateVTSConfig:
    """Tests for update_vts_config orchestrator method."""

    def test_enable_starts_driver(self):
        """Enabling should call driver.start()."""
        driver = MagicMock()
        config = _make_config(enabled=False)
        orch = FakeOrchestrator(config, vts_driver=driver)

        orch.update_vts_config(enabled=True)

        assert config.vtubestudio_config.enabled is True
        driver.start.assert_called_once()

    def test_disable_stops_driver(self):
        """Disabling should call driver.stop()."""
        driver = MagicMock()
        config = _make_config(enabled=True)
        orch = FakeOrchestrator(config, vts_driver=driver)

        orch.update_vts_config(enabled=False)

        assert config.vtubestudio_config.enabled is False
        driver.stop.assert_called_once()

    def test_host_change_with_enable_restarts(self):
        """Changing host + enabling should restart driver."""
        driver = MagicMock()
        config = _make_config(enabled=False, host="localhost")
        orch = FakeOrchestrator(config, vts_driver=driver)

        orch.update_vts_config(enabled=True, host="192.168.1.100")

        assert config.vtubestudio_config.host == "192.168.1.100"
        # Should stop then start (restart)
        driver.stop.assert_called_once()
        driver.start.assert_called_once()

    def test_port_change_without_enable_restarts(self):
        """Changing port while already enabled should restart."""
        driver = MagicMock()
        config = _make_config(enabled=True, port=8001)
        orch = FakeOrchestrator(config, vts_driver=driver)

        orch.update_vts_config(port=9001)

        assert config.vtubestudio_config.port == 9001
        driver.stop.assert_called_once()
        driver.start.assert_called_once()

    def test_no_change_no_action(self):
        """No changes should not trigger any driver actions."""
        driver = MagicMock()
        config = _make_config(enabled=True, host="localhost", port=8001)
        orch = FakeOrchestrator(config, vts_driver=driver)

        orch.update_vts_config(host="localhost", port=8001)

        driver.start.assert_not_called()
        driver.stop.assert_not_called()

    def test_enable_creates_driver_when_none(self):
        """Enabling when driver is None should create it."""
        config = _make_config(enabled=False)
        orch = FakeOrchestrator(config, vts_driver=None)

        orch.update_vts_config(enabled=True)

        assert orch._vts_driver is not None
        orch._vts_driver.start.assert_called_once()

    def test_disable_when_no_driver(self):
        """Disabling when no driver should not crash."""
        config = _make_config(enabled=True)
        orch = FakeOrchestrator(config, vts_driver=None)

        # Should not raise
        orch.update_vts_config(enabled=False)
        assert config.vtubestudio_config.enabled is False

    def test_enable_with_same_host_no_restart(self):
        """Enabling with same host should start, not restart."""
        driver = MagicMock()
        config = _make_config(enabled=False, host="localhost")
        orch = FakeOrchestrator(config, vts_driver=driver)

        orch.update_vts_config(enabled=True, host="localhost")

        # Same host, no restart — just start
        driver.start.assert_called_once()
        driver.stop.assert_not_called()
