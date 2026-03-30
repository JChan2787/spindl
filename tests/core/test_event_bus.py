"""Tests for EventBus."""

import sys
import threading
import time
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.core.event_bus import EventBus
from spindl.core.events import (
    EventType,
    StateChangedEvent,
    TranscriptionReadyEvent,
)


class TestEventBusBasics:
    """Basic EventBus functionality tests."""

    def test_subscribe_and_emit(self):
        """Test basic subscribe and emit."""
        bus = EventBus()
        received = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append(e))
        bus.emit(StateChangedEvent(from_state="idle", to_state="listening", trigger="vad"))

        assert len(received) == 1
        assert received[0].from_state == "idle"
        assert received[0].to_state == "listening"

    def test_unsubscribe(self):
        """Test unsubscribe removes handler."""
        bus = EventBus()
        received = []

        sub_id = bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append(e))
        result = bus.unsubscribe(sub_id)

        assert result is True
        bus.emit(StateChangedEvent())
        assert len(received) == 0

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribe returns False for nonexistent ID."""
        bus = EventBus()
        result = bus.unsubscribe(999)
        assert result is False

    def test_multiple_subscribers(self):
        """Test multiple subscribers receive events."""
        bus = EventBus()
        received1 = []
        received2 = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: received1.append(e))
        bus.subscribe(EventType.STATE_CHANGED, lambda e: received2.append(e))
        bus.emit(StateChangedEvent())

        assert len(received1) == 1
        assert len(received2) == 1

    def test_different_event_types(self):
        """Test subscribers only receive matching event types."""
        bus = EventBus()
        state_events = []
        transcription_events = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: state_events.append(e))
        bus.subscribe(EventType.TRANSCRIPTION_READY, lambda e: transcription_events.append(e))

        bus.emit(StateChangedEvent())
        bus.emit(TranscriptionReadyEvent(text="hello"))

        assert len(state_events) == 1
        assert len(transcription_events) == 1
        assert transcription_events[0].text == "hello"


class TestEventBusPriority:
    """Priority ordering tests."""

    def test_priority_ordering(self):
        """Test higher priority handlers called first."""
        bus = EventBus()
        order = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: order.append("low"), priority=0)
        bus.subscribe(EventType.STATE_CHANGED, lambda e: order.append("high"), priority=10)
        bus.subscribe(EventType.STATE_CHANGED, lambda e: order.append("medium"), priority=5)

        bus.emit(StateChangedEvent())

        assert order == ["high", "medium", "low"]

    def test_same_priority_order(self):
        """Test handlers with same priority are both called."""
        bus = EventBus()
        received = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append("a"), priority=5)
        bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append("b"), priority=5)

        bus.emit(StateChangedEvent())

        assert len(received) == 2
        assert set(received) == {"a", "b"}


class TestEventConsumption:
    """Event consumption tests."""

    def test_event_consumption_stops_propagation(self):
        """Test consuming an event stops lower-priority handlers."""
        bus = EventBus()
        received = []

        def consumer(e):
            received.append("consumer")
            e.consume()

        bus.subscribe(EventType.STATE_CHANGED, consumer, priority=10)
        bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append("low"), priority=0)

        bus.emit(StateChangedEvent())

        assert received == ["consumer"]

    def test_event_consumed_flag(self):
        """Test consumed flag is set correctly."""
        event = StateChangedEvent()
        assert event.consumed is False

        event.consume()
        assert event.consumed is True


class TestOneShotSubscription:
    """One-shot subscription tests."""

    def test_once_subscription(self):
        """Test one-shot subscriptions auto-unsubscribe."""
        bus = EventBus()
        received = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append(1), once=True)

        bus.emit(StateChangedEvent())
        bus.emit(StateChangedEvent())
        bus.emit(StateChangedEvent())

        assert len(received) == 1

    def test_once_with_regular(self):
        """Test one-shot doesn't affect regular subscriptions."""
        bus = EventBus()
        once_received = []
        regular_received = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: once_received.append(1), once=True)
        bus.subscribe(EventType.STATE_CHANGED, lambda e: regular_received.append(1))

        bus.emit(StateChangedEvent())
        bus.emit(StateChangedEvent())

        assert len(once_received) == 1
        assert len(regular_received) == 2


class TestEventBusThreadSafety:
    """Thread safety tests."""

    def test_concurrent_emit(self):
        """Test concurrent emits don't corrupt state."""
        bus = EventBus()
        received = []
        lock = threading.Lock()

        def handler(e):
            with lock:
                received.append(e)

        bus.subscribe(EventType.STATE_CHANGED, handler)

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=lambda: bus.emit(StateChangedEvent(trigger=str(threading.current_thread().ident)))
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 10

    def test_subscribe_during_emit(self):
        """Test subscribing during emit doesn't cause issues."""
        bus = EventBus()
        received = []

        def handler(e):
            received.append(1)
            # Subscribe new handler during emit
            bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append(2))

        bus.subscribe(EventType.STATE_CHANGED, handler)
        bus.emit(StateChangedEvent())

        # First emit: only original handler
        assert 1 in received

        # Second emit: both handlers
        bus.emit(StateChangedEvent())
        assert received.count(2) >= 1


class TestEventBusUtilities:
    """Utility method tests."""

    def test_subscriber_count(self):
        """Test subscriber_count returns correct value."""
        bus = EventBus()

        assert bus.subscriber_count(EventType.STATE_CHANGED) == 0

        sub1 = bus.subscribe(EventType.STATE_CHANGED, lambda e: None)
        assert bus.subscriber_count(EventType.STATE_CHANGED) == 1

        sub2 = bus.subscribe(EventType.STATE_CHANGED, lambda e: None)
        assert bus.subscriber_count(EventType.STATE_CHANGED) == 2

        bus.unsubscribe(sub1)
        assert bus.subscriber_count(EventType.STATE_CHANGED) == 1

    def test_has_subscribers(self):
        """Test has_subscribers returns correct value."""
        bus = EventBus()

        assert bus.has_subscribers(EventType.STATE_CHANGED) is False

        sub_id = bus.subscribe(EventType.STATE_CHANGED, lambda e: None)
        assert bus.has_subscribers(EventType.STATE_CHANGED) is True

        bus.unsubscribe(sub_id)
        assert bus.has_subscribers(EventType.STATE_CHANGED) is False

    def test_clear(self):
        """Test clear removes all subscriptions."""
        bus = EventBus()
        received = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append(1))
        bus.subscribe(EventType.TRANSCRIPTION_READY, lambda e: received.append(2))

        bus.clear()

        bus.emit(StateChangedEvent())
        bus.emit(TranscriptionReadyEvent())

        assert len(received) == 0


class TestCallbackErrorHandling:
    """Callback error handling tests."""

    def test_callback_error_doesnt_break_others(self):
        """Test exception in one callback doesn't stop others."""
        bus = EventBus()
        received = []

        def bad_handler(e):
            raise ValueError("oops")

        def good_handler(e):
            received.append(1)

        # Bad handler has higher priority, called first
        bus.subscribe(EventType.STATE_CHANGED, bad_handler, priority=10)
        bus.subscribe(EventType.STATE_CHANGED, good_handler, priority=0)

        # Should not raise, should still call good_handler
        bus.emit(StateChangedEvent())

        assert len(received) == 1
