"""
Stimuli system for spindl.

Transforms the agent from purely reactive to proactive by monitoring
multiple stimulus sources and firing the highest-priority one when
the agent is idle.

Core components:
    - StimuliEngine: Daemon thread that polls modules and fires stimuli
    - StimulusModule: ABC for stimulus sources (PATIENCE, Custom, etc.)
    - PatienceModule: Idle timer that fires when no interaction for N seconds
"""

from .base import StimulusModule
from .engine import StimuliEngine
from .game_state import GameStateModule
from .models import StimulusData, StimulusSource
from .patience import PatienceModule
from .twitch import TwitchModule
from .weighted_rotator import WeightedRotator

__all__ = [
    "StimuliEngine",
    "StimulusModule",
    "StimulusData",
    "StimulusSource",
    "GameStateModule",
    "PatienceModule",
    "TwitchModule",
    "WeightedRotator",
]
