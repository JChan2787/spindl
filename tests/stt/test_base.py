"""Tests for STT provider base classes and abstractions."""

import pytest
from typing import Optional

import numpy as np

from spindl.stt.base import (
    STTProperties,
    STTProvider,
)


class TestSTTProperties:
    """Tests for STTProperties dataclass."""

    def test_creates_with_all_fields(self) -> None:
        """STTProperties stores all capability fields."""
        props = STTProperties(
            sample_rate=16000,
            audio_format="pcm_f32le",
            supports_streaming=False,
        )

        assert props.sample_rate == 16000
        assert props.audio_format == "pcm_f32le"
        assert props.supports_streaming is False

    def test_streaming_provider_properties(self) -> None:
        """STTProperties can represent streaming providers."""
        props = STTProperties(
            sample_rate=16000,
            audio_format="pcm_s16le",
            supports_streaming=True,
        )

        assert props.supports_streaming is True
        assert props.audio_format == "pcm_s16le"

    def test_different_sample_rates(self) -> None:
        """STTProperties supports various sample rates."""
        for rate in [8000, 16000, 22050, 44100, 48000]:
            props = STTProperties(
                sample_rate=rate,
                audio_format="pcm_f32le",
                supports_streaming=False,
            )
            assert props.sample_rate == rate


class TestSTTProviderABC:
    """Tests for STTProvider abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """STTProvider cannot be instantiated without implementing all methods."""
        with pytest.raises(TypeError):
            STTProvider()

    def test_concrete_implementation_works(self) -> None:
        """A complete implementation of STTProvider can be instantiated."""

        class MockSTTProvider(STTProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> STTProperties:
                return STTProperties(
                    sample_rate=16000,
                    audio_format="pcm_f32le",
                    supports_streaming=False,
                )

            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                return "hello world"

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                return []

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                return None

        provider = MockSTTProvider()
        assert provider.transcribe(np.zeros(16000, dtype=np.float32)) == "hello world"
        assert provider.health_check() is True
        assert provider.get_properties().sample_rate == 16000

    def test_partial_implementation_fails(self) -> None:
        """Missing even one abstract method prevents instantiation."""

        class IncompleteSTTProvider(STTProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> STTProperties:
                return STTProperties(16000, "pcm_f32le", False)

            # Missing: transcribe, health_check, shutdown, validate_config, get_server_command

        with pytest.raises(TypeError):
            IncompleteSTTProvider()

    def test_validate_config_is_classmethod(self) -> None:
        """validate_config can be called on the class without instantiation."""

        class TestProvider(STTProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> STTProperties:
                return STTProperties(16000, "pcm_f32le", False)

            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                return ""

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                errors = []
                if "host" not in config:
                    errors.append("Missing host")
                return errors

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                return "python server.py"

        # Call on class, not instance
        errors = TestProvider.validate_config({})
        assert errors == ["Missing host"]

        errors = TestProvider.validate_config({"host": "localhost"})
        assert errors == []

    def test_get_server_command_is_classmethod(self) -> None:
        """get_server_command can be called on the class without instantiation."""

        class TestProvider(STTProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> STTProperties:
                return STTProperties(16000, "pcm_f32le", False)

            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                return ""

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                return []

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                port = config.get("port", 5555)
                return f"python server.py --port {port}"

        cmd = TestProvider.get_server_command({"port": 9999})
        assert cmd == "python server.py --port 9999"

    def test_in_process_provider_returns_none_for_server_command(self) -> None:
        """In-process providers return None from get_server_command."""

        class InProcessProvider(STTProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> STTProperties:
                return STTProperties(16000, "pcm_f32le", False)

            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                return ""

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                return []

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                return None

        assert InProcessProvider.get_server_command({}) is None

    def test_transcribe_signature_accepts_audio_and_sample_rate(self) -> None:
        """transcribe accepts numpy array and sample rate."""

        class TestProvider(STTProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> STTProperties:
                return STTProperties(16000, "pcm_f32le", False)

            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                assert isinstance(audio, np.ndarray)
                assert sample_rate == 16000
                return "test transcription"

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                return []

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                return None

        provider = TestProvider()
        audio = np.zeros(16000, dtype=np.float32)
        result = provider.transcribe(audio, sample_rate=16000)
        assert result == "test transcription"
