"""Tests for ServiceCapabilities registry (NANO-116 Phase B.4a)."""

import threading

import pytest

from spindl.orchestrator.service_capabilities import ServiceCapabilities, ServiceState


class TestServiceCapabilitiesInit:
    def test_all_services_default_disabled(self):
        caps = ServiceCapabilities()
        assert not caps.is_enabled("stt")
        assert not caps.is_enabled("tts")
        assert not caps.is_enabled("twitch")
        assert not caps.is_enabled("game_state")

    def test_from_config(self):
        caps = ServiceCapabilities.from_config(
            stt_enabled=True,
            tts_enabled=True,
            twitch_enabled=False,
            game_state_enabled=True,
        )
        assert caps.is_enabled("stt")
        assert caps.is_enabled("tts")
        assert not caps.is_enabled("twitch")
        assert caps.is_enabled("game_state")

    def test_from_config_defaults(self):
        caps = ServiceCapabilities.from_config()
        assert caps.is_enabled("stt")
        assert caps.is_enabled("tts")
        assert not caps.is_enabled("twitch")
        assert not caps.is_enabled("game_state")


class TestSetAndQuery:
    def test_set_enables_service(self):
        caps = ServiceCapabilities()
        caps.set("stt", True)
        assert caps.is_enabled("stt")

    def test_set_disables_service(self):
        caps = ServiceCapabilities()
        caps.set("stt", True)
        caps.set("stt", False)
        assert not caps.is_enabled("stt")

    def test_set_with_reason(self):
        caps = ServiceCapabilities()
        caps.set("tts", False, reason="disabled by launcher")
        state = caps.get_state("tts")
        assert state is not None
        assert not state.enabled
        assert state.reason == "disabled by launcher"

    def test_unknown_service_returns_false(self):
        caps = ServiceCapabilities()
        assert not caps.is_enabled("nonexistent")

    def test_unknown_service_set_is_noop(self):
        caps = ServiceCapabilities()
        caps.set("nonexistent", True)
        assert not caps.is_enabled("nonexistent")

    def test_get_state_unknown_returns_none(self):
        caps = ServiceCapabilities()
        assert caps.get_state("nonexistent") is None

    def test_get_state_returns_copy(self):
        caps = ServiceCapabilities()
        caps.set("stt", True, reason="config")
        state = caps.get_state("stt")
        assert state is not None
        state.enabled = False
        assert caps.is_enabled("stt"), "Mutating returned state must not affect registry"


class TestSnapshot:
    def test_snapshot_returns_all_services(self):
        caps = ServiceCapabilities.from_config(
            stt_enabled=True, tts_enabled=False,
        )
        snap = caps.snapshot()
        assert snap == {
            "stt": True,
            "tts": False,
            "twitch": False,
            "game_state": False,
        }

    def test_snapshot_is_isolated_copy(self):
        caps = ServiceCapabilities()
        caps.set("stt", True)
        snap = caps.snapshot()
        snap["stt"] = False
        assert caps.is_enabled("stt"), "Mutating snapshot must not affect registry"


class TestHealthStatus:
    def test_disabled_service_returns_disabled(self):
        caps = ServiceCapabilities()
        assert caps.health_status("stt") == "disabled"

    def test_enabled_service_returns_empty(self):
        caps = ServiceCapabilities()
        caps.set("stt", True)
        assert caps.health_status("stt") == ""

    def test_unknown_service_returns_disabled(self):
        caps = ServiceCapabilities()
        assert caps.health_status("nonexistent") == "disabled"


class TestThreadSafety:
    def test_concurrent_set_and_read(self):
        caps = ServiceCapabilities()
        errors = []

        def toggle_service(service: str, iterations: int):
            try:
                for i in range(iterations):
                    caps.set(service, i % 2 == 0)
                    caps.is_enabled(service)
                    caps.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=toggle_service, args=("stt", 1000)),
            threading.Thread(target=toggle_service, args=("tts", 1000)),
            threading.Thread(target=toggle_service, args=("twitch", 1000)),
            threading.Thread(target=toggle_service, args=("game_state", 1000)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety violation: {errors}"


class TestRepr:
    def test_repr_format(self):
        caps = ServiceCapabilities.from_config(stt_enabled=True, tts_enabled=False)
        r = repr(caps)
        assert "ServiceCapabilities(" in r
        assert "stt=on" in r
        assert "tts=off" in r
