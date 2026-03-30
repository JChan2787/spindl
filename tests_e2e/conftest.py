"""
E2E Test Configuration - Fixtures and parametrization.

NANO-029: Provides fixtures for the E2E test harness with
5-config LLM × VLM matrix parametrization.
"""

import pytest
import pytest_asyncio

from .helpers import E2EHarness


def pytest_addoption(parser):
    """Add custom CLI options for E2E tests."""
    # Only add if not already registered by pytest-playwright
    if not hasattr(parser, "_e2e_headed_added"):
        try:
            parser.addoption(
                "--e2e-headed",
                action="store_true",
                default=False,
                help="Run E2E tests with visible browser window",
            )
        except ValueError:
            # Option already exists
            pass
        parser._e2e_headed_added = True


@pytest.fixture(scope="session")
def e2e_headed(request) -> bool:
    """Get headed mode from CLI options, checking both our flag and pytest-playwright's."""
    # Check our custom flag first
    e2e_headed_flag = request.config.getoption("--e2e-headed", default=False)
    if e2e_headed_flag:
        return True
    # Fall back to pytest-playwright's --headed flag
    playwright_headed = request.config.getoption("--headed", default=False)
    return playwright_headed


# === LLM × VLM Test Matrix ===
# 5 configurations covering the key backend combinations

E2E_CONFIGS = {
    "local_unified": "fixtures/config/spindl_e2e_local_unified.yaml",
    "local_local": "fixtures/config/spindl_e2e_local_local.yaml",
    "local_cloud": "fixtures/config/spindl_e2e_local_cloud.yaml",
    "cloud_local": "fixtures/config/spindl_e2e_cloud_local.yaml",
    "cloud_cloud": "fixtures/config/spindl_e2e_cloud_cloud.yaml",
}


@pytest.fixture(params=list(E2E_CONFIGS.keys()))
def e2e_config_name(request) -> str:
    """
    Parametrized fixture that yields each config name.

    Tests using this fixture will run 5 times (once per config).
    """
    return request.param


@pytest.fixture
def e2e_config_path(e2e_config_name: str) -> str:
    """
    Get the config file path for the current parametrized config.
    """
    return E2E_CONFIGS[e2e_config_name]


@pytest_asyncio.fixture
async def e2e_harness(e2e_config_path: str, e2e_headed: bool):
    """
    Full E2E harness with server, browser, and Socket.IO client.

    Function-scoped: starts fresh for each test.
    Automatically starts and stops the harness.

    Args:
        e2e_config_path: Path to the test config YAML.
        e2e_headed: If True, run browser with visible window (--headed or --e2e-headed).
    """
    harness = E2EHarness(config_path=e2e_config_path, headless=not e2e_headed)
    try:
        await harness.start()
        yield harness
    finally:
        # Always stop, even if start() failed partway through
        # This ensures processes spawned before the failure get cleaned up
        await harness.stop()


# === Single-config fixtures for focused testing ===

@pytest_asyncio.fixture
async def e2e_harness_local_unified(e2e_headed: bool):
    """Harness with local unified (multimodal) config only."""
    harness = E2EHarness(config_path=E2E_CONFIGS["local_unified"], headless=not e2e_headed)
    try:
        await harness.start()
        yield harness
    finally:
        await harness.stop()


@pytest_asyncio.fixture
async def e2e_harness_cloud_cloud(e2e_headed: bool):
    """Harness with full cloud config only."""
    harness = E2EHarness(config_path=E2E_CONFIGS["cloud_cloud"], headless=not e2e_headed)
    try:
        await harness.start()
        yield harness
    finally:
        await harness.stop()


# === Test timeouts ===

@pytest.fixture
def response_timeout() -> float:
    """Default timeout for waiting for LLM responses."""
    return 30.0  # LLM can be slow, especially cloud


@pytest.fixture
def event_timeout() -> float:
    """Default timeout for waiting for Socket.IO events."""
    return 5.0


# === Markers ===

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "vision: marks tests that require VLM (vision) capabilities"
    )
    config.addinivalue_line(
        "markers",
        "cloud: marks tests that require cloud API credentials"
    )
