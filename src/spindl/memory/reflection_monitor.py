"""
ReflectionMonitor — PostProcessor that signals the ReflectionSystem.

Fires after each assistant response is recorded. Notifies the
background reflection thread to check if the message count threshold
has been reached. Non-blocking — just sets an event flag.

NANO-043 Phase 3.
"""

import logging

from ..llm.plugins.base import PipelineContext, PostProcessor
from .reflection import ReflectionSystem

logger = logging.getLogger(__name__)


class ReflectionMonitor(PostProcessor):
    """
    PostProcessor that notifies the ReflectionSystem after each turn.

    Must be registered AFTER HistoryRecorder so that the turn is
    already stored when the reflection system checks the count.
    """

    def __init__(self, reflection_system: ReflectionSystem):
        """
        Args:
            reflection_system: ReflectionSystem to notify.
        """
        self._reflection_system = reflection_system

    @property
    def name(self) -> str:
        return "reflection_monitor"

    def process(self, context: PipelineContext, response: str) -> str:
        """
        Signal reflection system to check message count.

        Returns response unchanged — notification only, no transformation.
        """
        self._reflection_system.notify()
        return response
