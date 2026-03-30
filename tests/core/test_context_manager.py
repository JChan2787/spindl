"""Tests for ContextManager."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.core.context_manager import ContextManager, AggregatedContext
from spindl.core.event_bus import EventBus
from spindl.core.events import EventType, ContextUpdatedEvent


class TestAggregatedContext:
    """Tests for AggregatedContext dataclass."""

    def test_basic_creation(self):
        """Test basic context creation."""
        ctx = AggregatedContext(primary_input="hello")
        assert ctx.primary_input == "hello"
        assert ctx.sources == {}
        assert ctx.contributing_sources == []

    def test_get_source(self):
        """Test get() returns source data."""
        ctx = AggregatedContext(
            primary_input="hello",
            sources={"lorebook": {"entry": "data"}},
        )
        assert ctx.get("lorebook") == {"entry": "data"}
        assert ctx.get("nonexistent") is None
        assert ctx.get("nonexistent", "default") == "default"

    def test_has_source(self):
        """Test has_source() checks contributing_sources."""
        ctx = AggregatedContext(
            primary_input="hello",
            contributing_sources=["lorebook", "vision"],
        )
        assert ctx.has_source("lorebook") is True
        assert ctx.has_source("vision") is True
        assert ctx.has_source("other") is False


class TestContextManagerBasics:
    """Basic ContextManager functionality tests."""

    def test_assemble_no_sources(self):
        """Test assemble with no registered sources."""
        cm = ContextManager()
        ctx = cm.assemble("hello world")

        assert ctx.primary_input == "hello world"
        assert ctx.sources == {}
        assert ctx.contributing_sources == []

    def test_register_and_assemble(self):
        """Test registering a source and assembling context."""
        cm = ContextManager()
        cm.register_source(
            name="test",
            provider=lambda: {"key": "value"},
        )

        ctx = cm.assemble("hello")

        assert ctx.primary_input == "hello"
        assert ctx.sources == {"test": {"key": "value"}}
        assert ctx.contributing_sources == ["test"]

    def test_source_returns_none(self):
        """Test source returning None is not included."""
        cm = ContextManager()
        cm.register_source(
            name="empty",
            provider=lambda: None,
        )

        ctx = cm.assemble("hello")

        assert "empty" not in ctx.sources
        assert "empty" not in ctx.contributing_sources

    def test_unregister_source(self):
        """Test unregistering a source."""
        cm = ContextManager()
        cm.register_source(name="test", provider=lambda: {"data": 1})

        assert cm.unregister_source("test") is True
        assert cm.unregister_source("test") is False  # Already removed

        ctx = cm.assemble("hello")
        assert "test" not in ctx.sources

    def test_registered_sources_property(self):
        """Test registered_sources returns source names."""
        cm = ContextManager()
        cm.register_source(name="a", provider=lambda: {})
        cm.register_source(name="b", provider=lambda: {})

        assert set(cm.registered_sources) == {"a", "b"}

    def test_source_count_property(self):
        """Test source_count returns correct count."""
        cm = ContextManager()
        assert cm.source_count == 0

        cm.register_source(name="a", provider=lambda: {})
        assert cm.source_count == 1

        cm.register_source(name="b", provider=lambda: {})
        assert cm.source_count == 2

        cm.unregister_source("a")
        assert cm.source_count == 1


class TestContextManagerPending:
    """Tests for pending_input functionality."""

    def test_pending_input_available_during_assemble(self):
        """Test pending_input is set during assemble."""
        cm = ContextManager()
        captured_input = None

        def capture_provider():
            nonlocal captured_input
            captured_input = cm.pending_input
            return {"captured": captured_input}

        cm.register_source(name="capture", provider=capture_provider)
        cm.assemble("test input")

        assert captured_input == "test input"

    def test_pending_input_none_outside_assemble(self):
        """Test pending_input is None when not assembling."""
        cm = ContextManager()
        assert cm.pending_input is None

        cm.assemble("hello")
        assert cm.pending_input is None  # Should be cleared after


class TestContextManagerPriority:
    """Tests for source priority."""

    def test_sources_called_in_priority_order(self):
        """Test sources are called in priority order."""
        cm = ContextManager()
        call_order = []

        cm.register_source(
            name="high",
            provider=lambda: (call_order.append("high"), {})[1],
            priority=10,
        )
        cm.register_source(
            name="low",
            provider=lambda: (call_order.append("low"), {})[1],
            priority=0,
        )
        cm.register_source(
            name="medium",
            provider=lambda: (call_order.append("medium"), {})[1],
            priority=5,
        )

        cm.assemble("test")

        # Lower priority called first (so higher priority can override)
        assert call_order == ["low", "medium", "high"]


class TestContextManagerReplacement:
    """Tests for source replacement."""

    def test_register_replaces_existing(self):
        """Test registering with same name replaces source."""
        cm = ContextManager()

        cm.register_source(name="test", provider=lambda: {"version": 1})
        ctx1 = cm.assemble("hello")
        assert ctx1.sources["test"] == {"version": 1}

        cm.register_source(name="test", provider=lambda: {"version": 2})
        ctx2 = cm.assemble("hello")
        assert ctx2.sources["test"] == {"version": 2}

        # Should only have one source
        assert cm.source_count == 1


class TestContextManagerErrorHandling:
    """Tests for error handling in sources."""

    def test_source_failure_isolated(self):
        """Test failing source doesn't break others."""
        cm = ContextManager()

        cm.register_source(
            name="bad",
            provider=lambda: 1 / 0,  # Will raise ZeroDivisionError
        )
        cm.register_source(
            name="good",
            provider=lambda: {"ok": True},
        )

        ctx = cm.assemble("test")

        assert "bad" not in ctx.sources
        assert "good" in ctx.sources
        assert ctx.sources["good"] == {"ok": True}

    def test_failing_source_not_in_contributing(self):
        """Test failing source not in contributing_sources."""
        cm = ContextManager()

        cm.register_source(name="bad", provider=lambda: 1 / 0)
        ctx = cm.assemble("test")

        assert "bad" not in ctx.contributing_sources


class TestContextManagerEvents:
    """Tests for event emission."""

    def test_emits_context_updated_event(self):
        """Test CONTEXT_UPDATED event is emitted."""
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.CONTEXT_UPDATED, handler)

        cm = ContextManager(event_bus=bus)
        cm.register_source(name="test", provider=lambda: {"data": 1})

        cm.assemble("hello")

        assert len(received) == 1
        assert isinstance(received[0], ContextUpdatedEvent)
        assert received[0].sources == ["test"]

    def test_event_has_correct_sources(self):
        """Test event contains correct contributing sources."""
        bus = EventBus()
        received = []

        bus.subscribe(EventType.CONTEXT_UPDATED, lambda e: received.append(e))

        cm = ContextManager(event_bus=bus)
        cm.register_source(name="a", provider=lambda: {"a": 1})
        cm.register_source(name="b", provider=lambda: None)  # Won't contribute
        cm.register_source(name="c", provider=lambda: {"c": 3})

        cm.assemble("hello")

        assert set(received[0].sources) == {"a", "c"}

    def test_no_event_without_bus(self):
        """Test no error when no event bus is configured."""
        cm = ContextManager()  # No event_bus
        cm.register_source(name="test", provider=lambda: {})

        # Should not raise
        ctx = cm.assemble("hello")
        assert ctx.primary_input == "hello"


class TestContextManagerMultipleSources:
    """Tests for multiple sources interaction."""

    def test_multiple_sources_all_contribute(self):
        """Test multiple sources all contribute to context."""
        cm = ContextManager()

        cm.register_source(name="lorebook", provider=lambda: {"entries": ["fact1"]})
        cm.register_source(name="vision", provider=lambda: {"scene": "office"})
        cm.register_source(name="memory", provider=lambda: {"recent": "chat"})

        ctx = cm.assemble("what do you see?")

        assert len(ctx.sources) == 3
        assert ctx.sources["lorebook"] == {"entries": ["fact1"]}
        assert ctx.sources["vision"] == {"scene": "office"}
        assert ctx.sources["memory"] == {"recent": "chat"}
        assert set(ctx.contributing_sources) == {"lorebook", "vision", "memory"}
