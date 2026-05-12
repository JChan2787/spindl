"""
PATIENCE idle timer module.

Fires a stimulus when the agent has been idle (no voice/text interaction)
for a configurable number of seconds. Lowest priority — only fires when
no other module has a pending stimulus.
"""

import time
from typing import Optional

from .base import StimulusModule
from .models import StimulusData, StimulusSource
from .weighted_rotator import WeightedRotator

import logging

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS = [
    "Continue the conversation naturally. "
    "You have been idle. Think of something interesting to say or ask."
]


class PatienceModule(StimulusModule):
    """
    Idle timer that fires when the agent hasn't interacted for N seconds.

    The timer resets whenever the StimuliEngine calls reset_activity()
    (triggered by RESPONSE_READY, TTS_COMPLETED, or user speech events).

    Supports pause/resume: when paused, elapsed reports 0 and the timer
    does not accumulate. On resume, the timer restarts from zero.

    Supports weighted prompt rotation (NANO-120) via WeightedRotator.

    Priority 0 — lowest. Only fires when no other module has a stimulus.
    """

    def __init__(
        self,
        timeout_seconds: float = 60.0,
        prompts: Optional[list[str]] = None,
        enabled: bool = True,
    ):
        self._timeout = timeout_seconds
        self._prompts = prompts if prompts else list(_DEFAULT_PROMPTS)
        self._rotator = WeightedRotator(self._prompts)
        self._enabled = enabled
        self._last_activity_time = time.monotonic()
        self._running = False
        self._paused = False

    @property
    def name(self) -> str:
        return "patience"

    @property
    def priority(self) -> int:
        return 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    @timeout_seconds.setter
    def timeout_seconds(self, value: float) -> None:
        self._timeout = max(0.0, value)

    @property
    def prompts(self) -> list[str]:
        return self._prompts

    @prompts.setter
    def prompts(self, value: list[str]) -> None:
        self._prompts = value if value else list(_DEFAULT_PROMPTS)
        self._rotator.items = self._prompts

    @property
    def paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self._running = True
        self._paused = False
        self._last_activity_time = time.monotonic()
        logger.info(
            "PATIENCE started (timeout=%.1fs, enabled=%s)",
            self._timeout,
            self._enabled,
        )

    def stop(self) -> None:
        self._running = False
        self._paused = False
        logger.info("PATIENCE stopped")

    def pause(self) -> None:
        """Pause the timer. Elapsed reports 0, timer does not accumulate."""
        self._paused = True

    def resume(self) -> None:
        """Resume the timer. Restarts from zero."""
        self._paused = False
        self._last_activity_time = time.monotonic()

    def reset_activity(self) -> None:
        """Reset the idle timer. Called by the engine on any interaction."""
        self._last_activity_time = time.monotonic()

    def has_stimulus(self) -> bool:
        if not self._enabled or not self._running or self._paused or self._timeout <= 0:
            return False
        return self._elapsed() >= self._timeout

    def get_stimulus(self) -> Optional[StimulusData]:
        if not self.has_stimulus():
            return None
        elapsed = self._elapsed()
        # Reset timer after firing (prevents re-fire next cycle)
        self._last_activity_time = time.monotonic()

        prompt = self._rotator.select() or self._prompts[0]

        return StimulusData(
            source=StimulusSource.PATIENCE,
            user_input=prompt,
            metadata={
                "elapsed_seconds": round(elapsed, 1),
                "timeout_seconds": self._timeout,
                "weight": 0.5,
            },
        )

    def health_check(self) -> bool:
        # PATIENCE has no external dependencies
        return True

    def get_progress(self) -> dict:
        """
        Return progress data for the dashboard PATIENCE bar.

        Returns:
            Dict with elapsed, total, and progress (0.0 to 1.0).
        """
        if self._paused or not self._running:
            return {
                "elapsed": 0.0,
                "total": self._timeout,
                "progress": 0.0,
            }
        elapsed = self._elapsed()
        progress = min(elapsed / self._timeout, 1.0) if self._timeout > 0 else 0.0
        return {
            "elapsed": round(elapsed, 1),
            "total": self._timeout,
            "progress": round(progress, 3),
        }

    def _elapsed(self) -> float:
        """Seconds since last activity."""
        return time.monotonic() - self._last_activity_time
