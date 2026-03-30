"""
Abstract base class for stimulus modules.

All stimulus sources (PATIENCE, Twitch, Custom Injection, etc.)
implement this interface. The StimuliEngine polls registered modules
and fires the highest-priority stimulus that has pending data.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .models import StimulusData


class StimulusModule(ABC):
    """
    Base class for stimulus modules.

    Modules are registered with the StimuliEngine and polled during
    each decision cycle. The module with the highest priority that
    returns True from has_stimulus() wins.

    Priority convention:
        0   = PATIENCE (lowest — only fires when nothing else wants to)
        50  = External integrations (Twitch, etc.)
        100 = Custom injection (dashboard-driven, high priority)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Module identifier for logging and GUI display."""

    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Module priority. Higher = checked first.

        When multiple modules have pending stimuli, the highest
        priority module wins.
        """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this module is currently active."""

    @abstractmethod
    def start(self) -> None:
        """Start the module (connect to services, begin listening)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the module (disconnect, cleanup)."""

    @abstractmethod
    def has_stimulus(self) -> bool:
        """Whether this module has a pending stimulus to fire."""

    @abstractmethod
    def get_stimulus(self) -> Optional[StimulusData]:
        """
        Get the pending stimulus data.

        Returns None if no stimulus is pending. After returning
        a stimulus, the module should clear its pending state
        (one-shot behavior).
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Whether the module's external dependencies are reachable."""
