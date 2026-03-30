"""Tests for VisionProvider (ContextProvider implementation)."""

import pytest
from unittest.mock import Mock, MagicMock

from spindl.vision.provider import VisionProvider, VisionStrategy
from spindl.vision.base import VLMProvider, VLMResponse, VLMProperties
from spindl.vision.capture import ScreenCapture
from spindl.llm.build_context import BuildContext, InputModality


class MockVLMProvider(VLMProvider):
    """Mock VLM provider for testing."""

    def __init__(self, healthy: bool = True, description: str = "Test description"):
        self._healthy = healthy
        self._description = description
        self._describe_calls = 0

    def initialize(self, config: dict) -> None:
        pass

    def get_properties(self) -> VLMProperties:
        return VLMProperties("mock-vlm", True, False)

    def describe(
        self,
        image_base64: str,
        prompt: str | None = None,
        max_tokens: int = 300,
        **kwargs,
    ) -> VLMResponse:
        self._describe_calls += 1
        return VLMResponse(
            description=self._description,
            input_tokens=1000,
            output_tokens=20,
            latency_ms=500.0,
        )

    def health_check(self) -> bool:
        return self._healthy

    def shutdown(self) -> None:
        pass

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        return []


class MockScreenCapture:
    """Mock screen capture for testing."""

    def __init__(self, base64_data: str = "mockbase64image"):
        self._base64_data = base64_data
        self._capture_calls = 0

    def capture_base64(self) -> str:
        self._capture_calls += 1
        return self._base64_data


class TestVisionStrategy:
    """Tests for VisionStrategy enum."""

    def test_never_value(self) -> None:
        """NEVER strategy has correct value."""
        assert VisionStrategy.NEVER.value == "never"

    def test_always_value(self) -> None:
        """ALWAYS strategy has correct value."""
        assert VisionStrategy.ALWAYS.value == "always"

    def test_on_demand_value(self) -> None:
        """ON_DEMAND strategy has correct value."""
        assert VisionStrategy.ON_DEMAND.value == "on_demand"


class TestVisionProvider:
    """Tests for VisionProvider."""

    def test_placeholder_is_vision_analysis(self) -> None:
        """Placeholder is [VISION_ANALYSIS]."""
        vlm = MockVLMProvider()
        capture = MockScreenCapture()
        provider = VisionProvider(capture, vlm)

        assert provider.placeholder == "[VISION_ANALYSIS]"

    def test_provide_strategy_never_returns_empty(self) -> None:
        """Strategy NEVER returns empty string without capture."""
        vlm = MockVLMProvider()
        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.NEVER,
        )

        context = BuildContext(input_content="test")
        result = provider.provide(context)

        assert result == ""
        assert capture._capture_calls == 0
        assert vlm._describe_calls == 0

    def test_provide_strategy_always_captures_and_describes(self) -> None:
        """Strategy ALWAYS captures screen and gets description."""
        vlm = MockVLMProvider(description="A code editor with Python")
        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ALWAYS,
        )

        context = BuildContext(input_content="test")
        result = provider.provide(context)

        assert capture._capture_calls == 1
        assert vlm._describe_calls == 1
        assert "A code editor with Python" in result
        assert "[What you currently see on screen:" in result

    def test_provide_vlm_unhealthy_returns_fallback(self) -> None:
        """Unhealthy VLM returns fallback text."""
        vlm = MockVLMProvider(healthy=False)
        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ALWAYS,
            fallback_text="Vision unavailable",
        )

        context = BuildContext(input_content="test")
        result = provider.provide(context)

        assert result == "Vision unavailable"
        assert capture._capture_calls == 0  # No capture if unhealthy

    def test_provide_capture_error_returns_fallback(self) -> None:
        """Capture error returns fallback text."""
        vlm = MockVLMProvider()

        # Create capture that raises error
        capture = MockScreenCapture()
        capture.capture_base64 = Mock(side_effect=RuntimeError("Monitor not found"))

        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ALWAYS,
            fallback_text="Capture failed",
        )

        context = BuildContext(input_content="test")
        result = provider.provide(context)

        assert result == "Capture failed"

    def test_provide_vlm_error_returns_fallback(self) -> None:
        """VLM error returns fallback text."""
        vlm = MockVLMProvider()
        vlm.describe = Mock(side_effect=ConnectionError("VLM unreachable"))

        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ALWAYS,
            fallback_text="VLM error",
        )

        context = BuildContext(input_content="test")
        result = provider.provide(context)

        assert result == "VLM error"

    def test_provide_empty_description_returns_fallback(self) -> None:
        """Empty description returns fallback text."""
        vlm = MockVLMProvider(description="   ")  # Whitespace only
        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ALWAYS,
            fallback_text="No description",
        )

        context = BuildContext(input_content="test")
        result = provider.provide(context)

        assert result == "No description"

    def test_set_strategy_changes_behavior(self) -> None:
        """set_strategy changes provider behavior."""
        vlm = MockVLMProvider()
        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ALWAYS,
        )

        context = BuildContext(input_content="test")

        # Initially ALWAYS - should capture
        provider.provide(context)
        assert capture._capture_calls == 1

        # Change to NEVER
        provider.set_strategy(VisionStrategy.NEVER)
        assert provider.get_strategy() == VisionStrategy.NEVER

        # Now should not capture
        provider.provide(context)
        assert capture._capture_calls == 1  # Still 1

    def test_force_capture_ignores_strategy(self) -> None:
        """force_capture works regardless of strategy."""
        vlm = MockVLMProvider(description="Forced capture result")
        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.NEVER,
        )

        result = provider.force_capture()

        assert capture._capture_calls == 1
        assert "Forced capture result" in result

    def test_custom_description_prompt_passed_to_vlm(self) -> None:
        """Custom description prompt is passed to VLM."""
        vlm = MockVLMProvider()
        vlm.describe = Mock(return_value=VLMResponse(
            description="Custom prompt response",
            input_tokens=100,
            output_tokens=10,
            latency_ms=100.0,
        ))
        capture = MockScreenCapture()

        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ALWAYS,
            description_prompt="Describe the fan art in detail.",
        )

        context = BuildContext(input_content="test")
        provider.provide(context)

        vlm.describe.assert_called_once()
        call_kwargs = vlm.describe.call_args
        assert call_kwargs.kwargs["prompt"] == "Describe the fan art in detail."

    def test_strategy_on_demand_returns_empty_for_now(self) -> None:
        """ON_DEMAND strategy returns empty (not yet implemented)."""
        vlm = MockVLMProvider()
        capture = MockScreenCapture()
        provider = VisionProvider(
            capture, vlm,
            strategy=VisionStrategy.ON_DEMAND,
        )

        context = BuildContext(input_content="test")
        result = provider.provide(context)

        assert result == ""
        assert capture._capture_calls == 0
