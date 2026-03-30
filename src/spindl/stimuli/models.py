"""
Data models for the stimuli system.

Defines the core data structures used by the StimuliEngine
and StimulusModule implementations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StimulusSource(Enum):
    """Identifies what triggered a stimulus."""

    PATIENCE = "patience"
    TWITCH = "twitch"
    CUSTOM = "custom"
    MODULE = "module"


@dataclass
class StimulusData:
    """
    Payload for a stimulus ready to fire.

    Attributes:
        source: What type of stimulus this is.
        user_input: Text injected as the "user message" into process_text_input().
        metadata: Arbitrary data for logging, GUI display, or post-processing.
    """

    source: StimulusSource
    user_input: str
    metadata: dict[str, Any] = field(default_factory=dict)
