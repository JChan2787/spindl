"""
Tests for ServiceRunner -np 2 injection in unified vision mode (NANO-087).

Tests cover:
- _build_command() injects -np 2 when VLM provider is "llm" (unified mode)
- _build_command() skips -np injection when VLM provider is separate
- _build_command() respects user-provided -np / --parallel in command
"""

from unittest.mock import MagicMock, patch

import pytest

from spindl.launcher.config import (
    HealthCheckConfig,
    LLMProviderConfig,
    ServiceConfig,
    VisionProviderConfig,
)
from spindl.launcher.service_runner import ServiceRunner


def _make_llm_service_config(command: str = "/bin/llama-server -m model.gguf") -> ServiceConfig:
    """Create a minimal LLM ServiceConfig."""
    return ServiceConfig(
        name="llm",
        platform="native",
        command=command,
        health_check=HealthCheckConfig(type="none"),
    )


def _make_runner(
    vlm_provider: str = "llm",
    llm_provider: str = "llama",
    llm_context_size: int = 16384,
) -> ServiceRunner:
    """Create a ServiceRunner with mocked provider class lookups."""
    logger = MagicMock()
    logger.log_launcher = MagicMock()

    llm_config = LLMProviderConfig(
        provider=llm_provider,
        provider_config={
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
        },
    )

    vision_config = VisionProviderConfig(
        enabled=True,
        provider=vlm_provider,
        provider_config={},
    )

    runner = ServiceRunner(
        logger=logger,
        llm_provider_config=llm_config,
        vision_provider_config=vision_config,
        llm_context_size=llm_context_size,
    )

    # Mock the LLM provider class to avoid registry lookups
    mock_provider_class = MagicMock()
    mock_provider_class.is_cloud_provider.return_value = False
    mock_provider_class.get_server_command.return_value = None
    mock_provider_class.validate_config.return_value = []
    runner._llm_provider_class = mock_provider_class

    return runner


class TestUnifiedVisionNpInjection:
    """Tests for -np 2 injection in _build_command() (NANO-087)."""

    def test_unified_mode_injects_np_2(self):
        """When VLM provider is 'llm', -np 2 is appended to LLM command."""
        runner = _make_runner(vlm_provider="llm")
        config = _make_llm_service_config()

        cmd = runner._build_command(config)

        assert "-np 2" in cmd

    def test_separate_vlm_no_np_injection(self):
        """When VLM provider is 'llama' (separate), -np is NOT injected."""
        runner = _make_runner(vlm_provider="llama")
        config = _make_llm_service_config()

        cmd = runner._build_command(config)

        assert "-np" not in cmd

    def test_no_vision_config_no_np_injection(self):
        """When no vision config at all, -np is NOT injected."""
        logger = MagicMock()
        logger.log_launcher = MagicMock()

        llm_config = LLMProviderConfig(
            provider="llama",
            provider_config={
                "executable_path": "/bin/llama-server",
                "model_path": "/models/model.gguf",
            },
        )

        runner = ServiceRunner(
            logger=logger,
            llm_provider_config=llm_config,
            vision_provider_config=None,
            llm_context_size=16384,
        )

        mock_provider_class = MagicMock()
        mock_provider_class.is_cloud_provider.return_value = False
        mock_provider_class.get_server_command.return_value = None
        mock_provider_class.validate_config.return_value = []
        runner._llm_provider_class = mock_provider_class

        config = _make_llm_service_config()
        cmd = runner._build_command(config)

        assert "-np" not in cmd

    def test_user_np_override_not_clobbered(self):
        """User-provided -np in command is not overwritten by auto-injection."""
        runner = _make_runner(vlm_provider="llm")
        config = _make_llm_service_config(
            command="/bin/llama-server -m model.gguf -np 4"
        )

        cmd = runner._build_command(config)

        # Should keep user's -np 4, not inject a second -np 2
        assert "-np 4" in cmd
        assert "-np 2" not in cmd

    def test_user_parallel_flag_not_clobbered(self):
        """User-provided --parallel in command is not overwritten."""
        runner = _make_runner(vlm_provider="llm")
        config = _make_llm_service_config(
            command="/bin/llama-server -m model.gguf --parallel 3"
        )

        cmd = runner._build_command(config)

        assert "--parallel 3" in cmd
        assert "-np 2" not in cmd

    def test_np_injection_includes_context_split(self):
        """With -np 2, the -c value should still be present (context is split by server)."""
        runner = _make_runner(vlm_provider="llm", llm_context_size=16384)
        config = _make_llm_service_config()

        cmd = runner._build_command(config)

        assert "-c 16384" in cmd
        assert "-np 2" in cmd

    def test_non_llm_service_not_affected(self):
        """TTS service should never get -np injection."""
        runner = _make_runner(vlm_provider="llm")
        tts_config = ServiceConfig(
            name="tts",
            platform="native",
            command="/bin/kokoro-server",
            health_check=HealthCheckConfig(type="none"),
        )

        cmd = runner._build_command(tts_config)

        assert "-np" not in cmd
