"""
Smoke Tests - Basic harness functionality verification.

NANO-029 Phase 1: Verifies that the E2E harness can:
1. Start the GUI server
2. Connect a browser
3. Connect Socket.IO
4. Receive orchestrator_ready event
"""

import pytest

from .helpers import E2EHarness
from .helpers.assertions import assert_event_received, assert_non_empty


class TestHarnessSmoke:
    """Basic smoke tests for E2E harness functionality."""

    @pytest.mark.asyncio
    async def test_harness_starts_and_stops(self):
        """Verify harness lifecycle works without the full parametrized matrix."""
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_unified.yaml",
            headless=True,
        )

        # Start should not raise
        await harness.start()

        # Basic assertions
        assert harness.page is not None
        assert harness.frontend_url.startswith("http://")

        # Stop should not raise
        await harness.stop()

    @pytest.mark.asyncio
    async def test_server_responds_to_browser(self):
        """Verify browser can load the GUI page."""
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_unified.yaml",
            headless=True,
        )

        try:
            await harness.start()

            # Check page loaded (title or specific element)
            title = await harness.page.title()
            assert title is not None

        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_socket_receives_events(self):
        """Verify Socket.IO client receives events from server."""
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_unified.yaml",
            headless=True,
        )

        try:
            await harness.start()

            # Server should emit orchestrator_ready after startup
            event = await assert_event_received(
                harness,
                "orchestrator_ready",
                timeout=30.0,  # Orchestrator startup can be slow
                message="Expected orchestrator_ready event after server startup",
            )

            # Verify event has expected structure
            assert "persona" in event
            assert "has_orchestrator" in event
            assert event["has_orchestrator"] is True

        finally:
            await harness.stop()


class TestSendMessage:
    """Tests for the send_message text input mechanism (NANO-031)."""

    @pytest.mark.asyncio
    async def test_send_message_returns_success(self):
        """Verify send_message endpoint accepts text."""
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_unified.yaml",
            headless=True,
        )

        try:
            await harness.start()

            # Wait for orchestrator to be ready
            await assert_event_received(harness, "orchestrator_ready", timeout=30.0)

            # Send text message
            result = await harness.send_message("Hello")

            # Should return success
            assert result["success"] is True

        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_send_message_triggers_transcription_event(self):
        """Verify send_message triggers transcription event."""
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_unified.yaml",
            headless=True,
        )

        try:
            await harness.start()

            # Wait for orchestrator
            await assert_event_received(harness, "orchestrator_ready", timeout=30.0)

            # Clear any prior events
            harness.clear_events()

            # Send text message
            await harness.send_message("Test message")

            # Should receive transcription event
            event = await assert_event_received(
                harness,
                "transcription",
                timeout=5.0,
                message="Expected transcription event after send_message",
            )

            assert event["text"] == "Test message"

        finally:
            await harness.stop()


class TestParametrizedSmoke:
    """Smoke tests that run across all config variants."""

    @pytest.mark.asyncio
    async def test_harness_starts_with_config(self, e2e_harness):
        """Verify harness starts with each parametrized config."""
        # If we get here, harness started successfully
        assert e2e_harness.page is not None

    @pytest.mark.asyncio
    async def test_orchestrator_ready_all_configs(self, e2e_harness, response_timeout):
        """Verify orchestrator_ready event fires for all configs."""
        event = await assert_event_received(
            e2e_harness,
            "orchestrator_ready",
            timeout=response_timeout,
        )

        assert event["has_orchestrator"] is True
