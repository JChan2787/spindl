"""Tests for MicLevelEvent and mic level pipeline (NANO-073b)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.core.event_bus import EventBus
from spindl.core.events import EventType, MicLevelEvent, AudioLevelEvent


class TestMicLevelEvent:
    """Tests for MicLevelEvent dataclass."""

    def test_mic_level_event_creation(self):
        """MicLevelEvent creates with correct defaults."""
        event = MicLevelEvent()

        assert event.event_type == EventType.MIC_LEVEL
        assert event.level == 0.0
        assert event.timestamp > 0

    def test_mic_level_event_with_level(self):
        """MicLevelEvent accepts level parameter."""
        event = MicLevelEvent(level=0.75)

        assert event.level == 0.75

    def test_mic_level_distinct_from_audio_level(self):
        """MicLevelEvent and AudioLevelEvent have different event types."""
        mic = MicLevelEvent(level=0.5)
        audio = AudioLevelEvent(level=0.5)

        assert mic.event_type == EventType.MIC_LEVEL
        assert audio.event_type == EventType.AUDIO_LEVEL
        assert mic.event_type != audio.event_type


class TestMicLevelEventBus:
    """Tests for MicLevelEvent through EventBus."""

    def test_mic_level_event_emits_and_receives(self):
        """MicLevelEvent can be emitted and received via EventBus."""
        bus = EventBus()
        received = []

        bus.subscribe(EventType.MIC_LEVEL, lambda e: received.append(e))
        bus.emit(MicLevelEvent(level=0.42))

        assert len(received) == 1
        assert received[0].level == 0.42

    def test_mic_level_does_not_trigger_audio_level_handler(self):
        """MIC_LEVEL events don't fire AUDIO_LEVEL handlers."""
        bus = EventBus()
        audio_received = []

        bus.subscribe(EventType.AUDIO_LEVEL, lambda e: audio_received.append(e))
        bus.emit(MicLevelEvent(level=0.5))

        assert len(audio_received) == 0

    def test_level_clamping_at_emission_site(self):
        """Level values above 1.0 should be clamped before emission (caller responsibility)."""
        # The clamping happens in the monitor thread, not in the event itself.
        # But the event should faithfully carry whatever value it's given.
        event = MicLevelEvent(level=1.5)
        assert event.level == 1.5  # Event doesn't clamp — caller does


class TestMicLevelBridge:
    """Tests for EventBridge mic level forwarding."""

    def test_bridge_forwards_mic_level(self):
        """EventBridge._on_mic_level forwards to GUI server."""
        import asyncio
        import threading
        from spindl.gui.bridge import EventBridge

        bus = EventBus()
        mock_server = MagicMock()
        mock_server.has_clients = True
        mock_server.emit_mic_level = AsyncMock()

        bridge = EventBridge(event_bus=bus, gui_server=mock_server)

        # Run event loop in background thread (mirrors real GUI server)
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        bridge.set_event_loop(loop)

        # Call handler directly
        event = MicLevelEvent(level=0.65)
        bridge._on_mic_level(event)

        # Give the loop time to process the scheduled coroutine
        import time
        time.sleep(0.05)

        mock_server.emit_mic_level.assert_called_once_with(level=0.65)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1)

    def test_bridge_skips_when_no_clients(self):
        """EventBridge._on_mic_level does nothing without connected clients."""
        from spindl.gui.bridge import EventBridge

        bus = EventBus()
        mock_server = MagicMock()
        mock_server.has_clients = False

        bridge = EventBridge(event_bus=bus, gui_server=mock_server)

        event = MicLevelEvent(level=0.5)
        bridge._on_mic_level(event)

        # emit_mic_level should NOT have been called
        mock_server.emit_mic_level.assert_not_called()
