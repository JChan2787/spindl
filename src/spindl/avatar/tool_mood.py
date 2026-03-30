"""
Tool-mood mapping for avatar expressions (NANO-093).

Subscribes to TOOL_INVOKED events on the EventBus and re-emits them
as AVATAR_TOOL_MOOD events with a mapped visual category. This drives
brief expression flashes on the avatar when tools are used.
"""

import logging
from typing import Optional

from ..core.event_bus import EventBus
from ..core.events import EventType, ToolInvokedEvent, AvatarToolMoodEvent

logger = logging.getLogger(__name__)

TOOL_MOOD_MAP: dict[str, str] = {
    # Search tools
    "web_search": "search",
    "screen_describe": "search",
    # Execution tools
    "run_command": "execute",
    "python_exec": "execute",
    # Memory tools
    "memory_store": "memory",
    "memory_recall": "memory",
}


class AvatarToolMoodSubscriber:
    """
    Subscribes to TOOL_INVOKED events and re-emits AVATAR_TOOL_MOOD.

    Decoupled from the tool executor — listens via EventBus so no changes
    to tool dispatch code are needed.
    """

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._subscription_id: Optional[str] = None

    def start(self) -> None:
        """Subscribe to tool invocation events."""
        self._subscription_id = self._event_bus.subscribe(
            EventType.TOOL_INVOKED,
            self._on_tool_invoked,
        )
        logger.info("AvatarToolMoodSubscriber started")

    def stop(self) -> None:
        """Unsubscribe from tool invocation events."""
        if self._subscription_id:
            self._event_bus.unsubscribe(self._subscription_id)
            self._subscription_id = None
            logger.info("AvatarToolMoodSubscriber stopped")

    def _on_tool_invoked(self, event: ToolInvokedEvent) -> None:
        """Map tool name to visual category and re-emit."""
        tool_mood = TOOL_MOOD_MAP.get(event.tool_name)
        if tool_mood:
            self._event_bus.emit(AvatarToolMoodEvent(tool_mood=tool_mood))
