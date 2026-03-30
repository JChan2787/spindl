"""
Thread-safe event bus for publish/subscribe communication.

Provides:
- Subscribe/unsubscribe by event type
- Priority ordering (higher priority called first)
- Event consumption (stop propagation to lower priorities)
- One-shot subscriptions
- Thread-safe emit from any thread
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from .events import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """Represents a subscription to an event type."""

    callback: Callable[[Event], None]
    priority: int = 0  # Higher = called first
    once: bool = False  # Auto-unsubscribe after first call


class EventBus:
    """
    Simple thread-safe event bus with priority support.

    Features:
    - Subscribe/unsubscribe by event type
    - Priority ordering (higher priority called first)
    - One-shot subscriptions (auto-cleanup after first call)
    - Event consumption (stop propagation to lower-priority handlers)
    - Thread-safe (uses RLock, callbacks execute outside lock)

    Usage:
        bus = EventBus()

        def on_transcription(event: TranscriptionReadyEvent):
            print(f"User said: {event.text}")

        sub_id = bus.subscribe(EventType.TRANSCRIPTION_READY, on_transcription)

        # Later...
        bus.emit(TranscriptionReadyEvent(text="Hello"))

        # Cleanup
        bus.unsubscribe(sub_id)

    Thread Safety:
        - subscribe/unsubscribe/emit are all thread-safe
        - Callbacks execute outside the lock to prevent deadlocks
        - Safe to emit from audio threads, processing threads, etc.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._lock = threading.RLock()
        self._subscriptions: dict[EventType, dict[int, Subscription]] = defaultdict(
            dict
        )
        self._next_id = 0
        self._id_to_event_type: dict[int, EventType] = {}

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
        priority: int = 0,
        once: bool = False,
    ) -> int:
        """
        Subscribe to an event type.

        Args:
            event_type: Which event to listen for.
            callback: Function called with event instance when event fires.
            priority: Higher values are called first (default 0).
            once: If True, auto-unsubscribe after first call.

        Returns:
            Subscription ID for later unsubscribe.
        """
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1

            self._subscriptions[event_type][sub_id] = Subscription(
                callback=callback,
                priority=priority,
                once=once,
            )
            self._id_to_event_type[sub_id] = event_type

            return sub_id

    def unsubscribe(self, sub_id: int) -> bool:
        """
        Unsubscribe by subscription ID.

        Args:
            sub_id: ID returned from subscribe().

        Returns:
            True if subscription was found and removed, False otherwise.
        """
        with self._lock:
            event_type = self._id_to_event_type.pop(sub_id, None)
            if event_type is None:
                return False

            return self._subscriptions[event_type].pop(sub_id, None) is not None

    def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.

        Callbacks are called in priority order (highest first).
        If a callback calls event.consume(), lower-priority callbacks are skipped.

        Args:
            event: Event instance to emit.
        """
        # Get sorted callbacks under lock
        with self._lock:
            subs = self._subscriptions.get(event.event_type, {})
            # Sort by priority descending (higher = called first)
            sorted_subs = sorted(
                [(sub_id, sub) for sub_id, sub in subs.items()],
                key=lambda x: x[1].priority,
                reverse=True,
            )

        # Track one-shot subs to remove after iteration
        to_remove: list[int] = []

        # Call callbacks outside lock (prevents deadlock if callback emits)
        for sub_id, sub in sorted_subs:
            if event.consumed:
                break

            try:
                sub.callback(event)
            except Exception as e:
                logger.error(
                    f"EventBus: callback error for {event.event_type.name}: {e}"
                )

            if sub.once:
                to_remove.append(sub_id)

        # Clean up one-shot subscriptions
        for sub_id in to_remove:
            self.unsubscribe(sub_id)

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subscriptions.clear()
            self._id_to_event_type.clear()

    def subscriber_count(self, event_type: EventType) -> int:
        """
        Get number of subscribers for an event type.

        Args:
            event_type: Event type to check.

        Returns:
            Number of active subscriptions.
        """
        with self._lock:
            return len(self._subscriptions.get(event_type, {}))

    def has_subscribers(self, event_type: EventType) -> bool:
        """
        Check if an event type has any subscribers.

        Args:
            event_type: Event type to check.

        Returns:
            True if at least one subscriber exists.
        """
        return self.subscriber_count(event_type) > 0
