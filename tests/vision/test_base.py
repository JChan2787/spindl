"""Tests for VLM provider base classes and abstractions."""

import pytest
from typing import Optional

from spindl.vision.base import (
    VLMProperties,
    VLMProvider,
    VLMResponse,
)


class TestVLMProperties:
    """Tests for VLMProperties dataclass."""

    def test_creates_with_all_fields(self) -> None:
        """VLMProperties stores all capability fields."""
        props = VLMProperties(
            name="X-Ray_Alpha",
            is_local=True,
            supports_streaming=False,
            max_image_size=2048,
        )

        assert props.name == "X-Ray_Alpha"
        assert props.is_local is True
        assert props.supports_streaming is False
        assert props.max_image_size == 2048

    def test_cloud_provider_properties(self) -> None:
        """VLMProperties can represent cloud providers."""
        props = VLMProperties(
            name="gpt-4o",
            is_local=False,
            supports_streaming=False,
        )

        assert props.name == "gpt-4o"
        assert props.is_local is False
        assert props.max_image_size is None


class TestVLMResponse:
    """Tests for VLMResponse dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """VLMResponse requires description and usage stats."""
        response = VLMResponse(
            description="A screenshot showing a code editor with Python code.",
            input_tokens=1500,
            output_tokens=42,
            latency_ms=1234.5,
        )

        assert response.description == "A screenshot showing a code editor with Python code."
        assert response.input_tokens == 1500
        assert response.output_tokens == 42
        assert response.latency_ms == 1234.5

    def test_empty_description(self) -> None:
        """VLMResponse can hold empty description (edge case)."""
        response = VLMResponse(
            description="",
            input_tokens=1000,
            output_tokens=0,
            latency_ms=500.0,
        )

        assert response.description == ""
        assert response.output_tokens == 0


class TestVLMProviderInterface:
    """Tests for VLMProvider abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """VLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VLMProvider()  # type: ignore

    def test_concrete_implementation_works(self) -> None:
        """A concrete VLMProvider implementation can be instantiated."""

        class MockVLMProvider(VLMProvider):
            def __init__(self) -> None:
                self._initialized = False

            def initialize(self, config: dict) -> None:
                self._initialized = True

            def get_properties(self) -> VLMProperties:
                return VLMProperties(
                    name="mock-vlm",
                    is_local=True,
                    supports_streaming=False,
                )

            def describe(
                self,
                image_base64: str,
                prompt: Optional[str] = None,
                max_tokens: int = 300,
                **kwargs,
            ) -> VLMResponse:
                return VLMResponse(
                    description=f"Description of image ({len(image_base64)} bytes)",
                    input_tokens=len(image_base64) // 4,
                    output_tokens=10,
                    latency_ms=100.0,
                )

            def health_check(self) -> bool:
                return self._initialized

            def shutdown(self) -> None:
                self._initialized = False

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                errors = []
                if "model_path" not in config:
                    errors.append("Missing required field: model_path")
                return errors

        provider = MockVLMProvider()

        # Before initialization
        assert provider.health_check() is False

        # Initialize
        provider.initialize({"model_path": "/path/to/model.gguf"})
        assert provider.health_check() is True

        # Check properties
        props = provider.get_properties()
        assert props.name == "mock-vlm"
        assert props.is_local is True

        # Describe image
        response = provider.describe("base64encodedimage")
        assert isinstance(response, VLMResponse)
        assert "Description" in response.description

        # Shutdown
        provider.shutdown()
        assert provider.health_check() is False

    def test_validate_config_returns_errors(self) -> None:
        """validate_config returns list of error messages."""

        class StrictProvider(VLMProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> VLMProperties:
                return VLMProperties("test", True, False)

            def describe(
                self,
                image_base64: str,
                prompt: Optional[str] = None,
                max_tokens: int = 300,
                **kwargs,
            ) -> VLMResponse:
                return VLMResponse("", 0, 0, 0.0)

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                errors = []
                if "model_path" not in config:
                    errors.append("Missing required field: model_path")
                if "port" not in config:
                    errors.append("Missing required field: port")
                return errors

        # Valid config
        errors = StrictProvider.validate_config({
            "model_path": "/path/model.gguf",
            "port": 5558,
        })
        assert errors == []

        # Missing fields
        errors = StrictProvider.validate_config({})
        assert len(errors) == 2
        assert "model_path" in errors[0]
        assert "port" in errors[1]

    def test_is_cloud_provider_default_false(self) -> None:
        """Default is_cloud_provider returns False."""

        class LocalProvider(VLMProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> VLMProperties:
                return VLMProperties("local", True, False)

            def describe(
                self,
                image_base64: str,
                prompt: Optional[str] = None,
                max_tokens: int = 300,
                **kwargs,
            ) -> VLMResponse:
                return VLMResponse("", 0, 0, 0.0)

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                return []

        assert LocalProvider.is_cloud_provider() is False

    def test_get_server_command_default_none(self) -> None:
        """Default get_server_command returns None."""

        class SimpleProvider(VLMProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> VLMProperties:
                return VLMProperties("simple", True, False)

            def describe(
                self,
                image_base64: str,
                prompt: Optional[str] = None,
                max_tokens: int = 300,
                **kwargs,
            ) -> VLMResponse:
                return VLMResponse("", 0, 0, 0.0)

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                return []

        assert SimpleProvider.get_server_command({}) is None
