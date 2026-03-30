"""
Service Launch Tests - GUI-first mode service startup verification.

NANO-029 Phase 3: Verifies that the launcher page correctly:
1. Loads a config from the matrix
2. Clicks the "Start Services" button
3. Receives orchestrator_ready event
4. All configured services report healthy
"""

import pytest

from .helpers import E2EHarness
from .helpers.assertions import assert_event_received


class TestServiceLaunchGUIFirst:
    """
    Test service launch in GUI-first mode.

    GUI-first mode:
    1. GUIServer starts without orchestrator
    2. User navigates to /launcher
    3. User clicks "Start Services"
    4. Backend saves config, starts services, initializes orchestrator
    5. orchestrator_ready event fires when complete
    """

    @pytest.mark.asyncio
    async def test_launch_services_cloud_cloud(self):
        """
        GUI-first: Load cloud_cloud config, click Start, verify orchestrator_ready.

        cloud_cloud uses DeepSeek + xAI APIs - no local GPU needed.
        """
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_cloud_cloud.yaml",
            headless=False,  # Show browser so we can see what's happening
            auto_start_services=False,  # We'll click the button manually
        )

        try:
            # Start harness - this starts frontend + backend but does NOT trigger service startup
            await harness.start()

            # Navigate to launcher page
            await harness.page.goto(f"{harness.frontend_url}/launcher")

            # Wait for page to load (config hydration)
            await harness.page.wait_for_selector('[data-testid="start-services-button"]', state="visible")

            # Clear any prior events
            harness.clear_events()

            # Click the Start Services button
            await harness.page.click('[data-testid="start-services-button"]')

            # Wait for orchestrator_ready event (60s timeout for service startup)
            try:
                event = await assert_event_received(
                    harness,
                    "orchestrator_ready",
                    timeout=60.0,
                    message="Expected orchestrator_ready after clicking Start Services",
                )
            except AssertionError:
                # Dump all received events for debugging
                harness.dump_events()
                raise

            # Verify orchestrator is ready
            assert event.get("has_orchestrator") is True, "Orchestrator should be attached"
            assert event.get("persona") is not None, "Persona should be set"

        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_launch_services_local_unified(self):
        """
        GUI-first: Load local_unified config, click Start, verify orchestrator_ready.

        local_unified uses local multimodal LLM - requires llama.cpp running.
        """
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_unified.yaml",
            headless=False,  # Show browser so we can see what's happening
            auto_start_services=False,
        )

        try:
            await harness.start()
            await harness.page.goto(f"{harness.frontend_url}/launcher")
            await harness.page.wait_for_selector('[data-testid="start-services-button"]', state="visible")
            harness.clear_events()

            await harness.page.click('[data-testid="start-services-button"]')

            event = await assert_event_received(
                harness,
                "orchestrator_ready",
                timeout=120.0,  # Local model loading can take >60s
                message="Expected orchestrator_ready after clicking Start Services",
            )

            assert event.get("has_orchestrator") is True
            assert event.get("persona") is not None

        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_launch_services_local_local(self):
        """
        GUI-first: Load local_local config, click Start, verify orchestrator_ready.

        local_local uses separate local LLM + local VLM - requires two llama.cpp instances.
        """
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_local.yaml",
            headless=False,
            auto_start_services=False,
        )

        try:
            await harness.start()
            await harness.page.goto(f"{harness.frontend_url}/launcher")
            await harness.page.wait_for_selector('[data-testid="start-services-button"]', state="visible")
            harness.clear_events()

            await harness.page.click('[data-testid="start-services-button"]')

            event = await assert_event_received(
                harness,
                "orchestrator_ready",
                timeout=60.0,
                message="Expected orchestrator_ready after clicking Start Services",
            )

            assert event.get("has_orchestrator") is True
            assert event.get("persona") is not None

        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_launch_services_local_cloud(self):
        """
        GUI-first: Load local_cloud config, click Start, verify orchestrator_ready.

        local_cloud uses local LLM + cloud VLM.
        """
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_local_cloud.yaml",
            headless=False,
            auto_start_services=False,
        )

        try:
            await harness.start()
            await harness.page.goto(f"{harness.frontend_url}/launcher")
            await harness.page.wait_for_selector('[data-testid="start-services-button"]', state="visible")
            harness.clear_events()

            await harness.page.click('[data-testid="start-services-button"]')

            event = await assert_event_received(
                harness,
                "orchestrator_ready",
                timeout=60.0,
                message="Expected orchestrator_ready after clicking Start Services",
            )

            assert event.get("has_orchestrator") is True
            assert event.get("persona") is not None

        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_launch_services_cloud_local(self):
        """
        GUI-first: Load cloud_local config, click Start, verify orchestrator_ready.

        cloud_local uses cloud LLM + local VLM.
        """
        harness = E2EHarness(
            config_path="fixtures/config/spindl_e2e_cloud_local.yaml",
            headless=False,
            auto_start_services=False,
        )

        try:
            await harness.start()
            await harness.page.goto(f"{harness.frontend_url}/launcher")
            await harness.page.wait_for_selector('[data-testid="start-services-button"]', state="visible")
            harness.clear_events()

            await harness.page.click('[data-testid="start-services-button"]')

            event = await assert_event_received(
                harness,
                "orchestrator_ready",
                timeout=60.0,
                message="Expected orchestrator_ready after clicking Start Services",
            )

            assert event.get("has_orchestrator") is True
            assert event.get("persona") is not None

        finally:
            await harness.stop()


class TestServiceLaunchParametrized:
    """
    Parametrized tests that run against all configs in the matrix.

    Uses the e2e_harness fixture from conftest.py which parametrizes
    across all 5 config files.
    """

    @pytest.mark.asyncio
    async def test_launch_services_all_configs(self, e2e_config_path: str):
        """
        GUI-first: Launch services for each config in the matrix.

        This test runs 5 times - once per config file.
        """
        harness = E2EHarness(
            config_path=e2e_config_path,
            headless=False,
            auto_start_services=False,
        )

        try:
            await harness.start()
            await harness.page.goto(f"{harness.frontend_url}/launcher")
            await harness.page.wait_for_selector('[data-testid="start-services-button"]', state="visible")
            harness.clear_events()

            await harness.page.click('[data-testid="start-services-button"]')

            event = await assert_event_received(
                harness,
                "orchestrator_ready",
                timeout=60.0,
                message=f"Expected orchestrator_ready for config: {e2e_config_path}",
            )

            assert event.get("has_orchestrator") is True, f"Config {e2e_config_path}: orchestrator not attached"
            assert event.get("persona") is not None, f"Config {e2e_config_path}: persona not set"

        finally:
            await harness.stop()
