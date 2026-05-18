"""
Service capabilities registry (NANO-116 Phase B.4a).

Single source of truth for which services are enabled/disabled at runtime.
Replaces scattered boolean flags across voice_agent.py and callbacks.py
with one atomic registry that the health check, UI, and internal guards
all read from.

Services tracked: STT, TTS, Twitch, Game State Bridge.
VTubeStudio is excluded — it has its own config/lifecycle path that doesn't
follow the stimuli toggle pattern.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ServiceState:
    """Enable/disable state for a single service."""

    enabled: bool = False
    reason: str = ""


class ServiceCapabilities:
    """
    Thread-safe registry of service enable/disable states.

    Built from config on startup. Updated atomically on runtime toggle.
    Read by health_check(), callbacks guards, and frontend via health_status
    WebSocket event.

    Usage:
        caps = ServiceCapabilities()
        caps.set("stt", True)
        caps.set("tts", False, reason="disabled by launcher config")
        if caps.is_enabled("stt"):
            ...
    """

    _KNOWN_SERVICES = frozenset({"stt", "tts", "twitch", "game_state"})

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, ServiceState] = {
            name: ServiceState() for name in self._KNOWN_SERVICES
        }

    def set(self, service: str, enabled: bool, reason: str = "") -> None:
        if service not in self._KNOWN_SERVICES:
            logger.warning("ServiceCapabilities.set: unknown service %r", service)
            return
        with self._lock:
            state = self._states[service]
            if state.enabled != enabled:
                logger.info(
                    "Service %s: %s → %s%s",
                    service,
                    "enabled" if state.enabled else "disabled",
                    "enabled" if enabled else "disabled",
                    f" ({reason})" if reason else "",
                )
            state.enabled = enabled
            state.reason = reason

    def is_enabled(self, service: str) -> bool:
        if service not in self._KNOWN_SERVICES:
            return False
        with self._lock:
            return self._states[service].enabled

    def get_state(self, service: str) -> Optional[ServiceState]:
        if service not in self._KNOWN_SERVICES:
            return None
        with self._lock:
            s = self._states[service]
            return ServiceState(enabled=s.enabled, reason=s.reason)

    def snapshot(self) -> dict[str, bool]:
        """Atomic snapshot of all service enabled states."""
        with self._lock:
            return {name: s.enabled for name, s in self._states.items()}

    def health_status(self, service: str) -> str:
        """Return 'disabled' if service is off, empty string otherwise.

        Used by health_check() to distinguish disabled from unhealthy.
        Caller is responsible for checking actual provider health when
        the service is enabled.
        """
        if not self.is_enabled(service):
            return "disabled"
        return ""

    @classmethod
    def from_config(
        cls,
        stt_enabled: bool = True,
        tts_enabled: bool = True,
        twitch_enabled: bool = False,
        game_state_enabled: bool = False,
    ) -> "ServiceCapabilities":
        caps = cls()
        caps.set("stt", stt_enabled, reason="config")
        caps.set("tts", tts_enabled, reason="config")
        caps.set("twitch", twitch_enabled, reason="config")
        caps.set("game_state", game_state_enabled, reason="config")
        return caps

    def __repr__(self) -> str:
        with self._lock:
            flags = ", ".join(
                f"{name}={'on' if s.enabled else 'off'}"
                for name, s in sorted(self._states.items())
            )
        return f"ServiceCapabilities({flags})"
