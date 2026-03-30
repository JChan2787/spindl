"""Tests for TTS provider base classes and abstractions."""

import pytest
from typing import Iterator, Optional

from spindl.tts.base import (
    AudioResult,
    TTSProperties,
    TTSProvider,
)


class TestTTSProperties:
    """Tests for TTSProperties dataclass."""

    def test_creates_with_all_fields(self) -> None:
        """TTSProperties stores all capability fields."""
        props = TTSProperties(
            sample_rate=24000,
            audio_format="pcm_f32le",
            channels=1,
            supports_streaming=False,
        )

        assert props.sample_rate == 24000
        assert props.audio_format == "pcm_f32le"
        assert props.channels == 1
        assert props.supports_streaming is False

    def test_streaming_provider_properties(self) -> None:
        """TTSProperties can represent streaming providers."""
        props = TTSProperties(
            sample_rate=12000,
            audio_format="pcm_s16le",
            channels=2,
            supports_streaming=True,
        )

        assert props.sample_rate == 12000
        assert props.supports_streaming is True
        assert props.channels == 2


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """AudioResult requires data, sample_rate, and format."""
        audio_bytes = b"\x00\x00\x00\x00"
        result = AudioResult(
            data=audio_bytes,
            sample_rate=24000,
            format="pcm_f32le",
        )

        assert result.data == audio_bytes
        assert result.sample_rate == 24000
        assert result.format == "pcm_f32le"
        assert result.is_final is True  # Default

    def test_streaming_chunk_not_final(self) -> None:
        """AudioResult can mark chunks as non-final for streaming."""
        result = AudioResult(
            data=b"\x00\x00",
            sample_rate=12000,
            format="pcm_s16le",
            is_final=False,
        )

        assert result.is_final is False

    def test_empty_audio_data(self) -> None:
        """AudioResult can hold empty audio (edge case)."""
        result = AudioResult(
            data=b"",
            sample_rate=24000,
            format="pcm_f32le",
        )

        assert result.data == b""
        assert len(result.data) == 0


class TestTTSProviderInterface:
    """Tests for TTSProvider abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """TTSProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TTSProvider()  # type: ignore

    def test_concrete_implementation_works(self) -> None:
        """A concrete TTSProvider implementation can be instantiated."""

        class MockTTSProvider(TTSProvider):
            def __init__(self) -> None:
                self._initialized = False
                self._voices = ["voice_a", "voice_b"]

            def initialize(self, config: dict) -> None:
                self._initialized = True

            def get_properties(self) -> TTSProperties:
                return TTSProperties(
                    sample_rate=24000,
                    audio_format="pcm_f32le",
                    channels=1,
                    supports_streaming=False,
                )

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                return AudioResult(
                    data=b"\x00" * len(text),
                    sample_rate=24000,
                    format="pcm_f32le",
                )

            def list_voices(self) -> list[str]:
                return self._voices

            def health_check(self) -> bool:
                return self._initialized

            def shutdown(self) -> None:
                self._initialized = False

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                errors = []
                if "host" not in config:
                    errors.append("Missing required field: host")
                return errors

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                return None  # In-process provider

        provider = MockTTSProvider()

        # Before initialization
        assert provider.health_check() is False

        # Initialize
        provider.initialize({"host": "localhost"})
        assert provider.health_check() is True

        # Check properties
        props = provider.get_properties()
        assert props.sample_rate == 24000
        assert props.supports_streaming is False

        # Synthesize
        result = provider.synthesize("Hello")
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000

        # Voices
        voices = provider.list_voices()
        assert "voice_a" in voices
        assert "voice_b" in voices

        # Shutdown
        provider.shutdown()
        assert provider.health_check() is False

    def test_default_synthesize_stream_yields_single_result(self) -> None:
        """Default synthesize_stream yields single result from synthesize."""

        class SimpleTTSProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(24000, "pcm_f32le", 1, False)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                return AudioResult(
                    data=f"audio:{text}".encode(),
                    sample_rate=24000,
                    format="pcm_f32le",
                )

            def list_voices(self) -> list[str]:
                return ["default"]

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

        provider = SimpleTTSProvider()

        # Default stream implementation
        chunks = list(provider.synthesize_stream("test"))
        assert len(chunks) == 1
        assert chunks[0].data == b"audio:test"

    def test_streaming_provider_can_yield_multiple_chunks(self) -> None:
        """A streaming provider can override synthesize_stream to yield multiple chunks."""

        class StreamingTTSProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(12000, "pcm_s16le", 1, True)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                # Full synthesis still works
                return AudioResult(
                    data=text.encode(),
                    sample_rate=12000,
                    format="pcm_s16le",
                )

            def synthesize_stream(self, text: str, voice: Optional[str] = None, **kwargs) -> Iterator[AudioResult]:
                # Yield word by word
                words = text.split()
                for i, word in enumerate(words):
                    is_last = i == len(words) - 1
                    yield AudioResult(
                        data=word.encode(),
                        sample_rate=12000,
                        format="pcm_s16le",
                        is_final=is_last,
                    )

            def list_voices(self) -> list[str]:
                return ["stream_voice"]

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

        provider = StreamingTTSProvider()

        # Stream yields multiple chunks
        chunks = list(provider.synthesize_stream("hello world today"))
        assert len(chunks) == 3
        assert chunks[0].data == b"hello"
        assert chunks[0].is_final is False
        assert chunks[1].data == b"world"
        assert chunks[1].is_final is False
        assert chunks[2].data == b"today"
        assert chunks[2].is_final is True

    def test_validate_config_returns_errors(self) -> None:
        """validate_config returns list of error messages."""

        class StrictProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(24000, "pcm_f32le", 1, False)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                return AudioResult(b"", 24000, "pcm_f32le")

            def list_voices(self) -> list[str]:
                return []

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                errors = []
                if "host" not in config:
                    errors.append("Missing required field: host")
                if "port" not in config:
                    errors.append("Missing required field: port")
                if "port" in config and not isinstance(config["port"], int):
                    errors.append("Field 'port' must be an integer")
                return errors

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                return f"python server.py --port {config.get('port', 5000)}"

        # Valid config
        errors = StrictProvider.validate_config({"host": "localhost", "port": 5556})
        assert errors == []

        # Missing fields
        errors = StrictProvider.validate_config({})
        assert len(errors) == 2
        assert "host" in errors[0]
        assert "port" in errors[1]

        # Invalid type
        errors = StrictProvider.validate_config({"host": "localhost", "port": "5556"})
        assert len(errors) == 1
        assert "integer" in errors[0]

    def test_get_server_command_for_server_based_provider(self) -> None:
        """Server-based providers return startup command."""

        class ServerBasedProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(24000, "pcm_f32le", 1, False)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                return AudioResult(b"", 24000, "pcm_f32le")

            def list_voices(self) -> list[str]:
                return []

            def health_check(self) -> bool:
                return True

            def shutdown(self) -> None:
                pass

            @classmethod
            def validate_config(cls, config: dict) -> list[str]:
                return []

            @classmethod
            def get_server_command(cls, config: dict) -> Optional[str]:
                port = config.get("port", 5556)
                return f"conda run -n tts python server.py --port {port}"

        cmd = ServerBasedProvider.get_server_command({"port": 6000})
        assert cmd == "conda run -n tts python server.py --port 6000"

    def test_get_server_command_none_for_in_process_provider(self) -> None:
        """In-process providers return None for server command."""

        class InProcessProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(24000, "pcm_f32le", 1, False)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                return AudioResult(b"", 24000, "pcm_f32le")

            def list_voices(self) -> list[str]:
                return []

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

        cmd = InProcessProvider.get_server_command({})
        assert cmd is None


class TestVoiceResolution:
    """Tests for the _resolve_voice helper method."""

    def test_resolve_voice_returns_requested_if_available(self) -> None:
        """_resolve_voice returns requested voice when available."""

        class VoiceProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(24000, "pcm_f32le", 1, False)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                resolved = self._resolve_voice(voice, "default_voice")
                return AudioResult(resolved.encode(), 24000, "pcm_f32le")

            def list_voices(self) -> list[str]:
                return ["voice_a", "voice_b", "default_voice"]

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

        provider = VoiceProvider()

        # Available voice
        result = provider.synthesize("test", voice="voice_a")
        assert result.data == b"voice_a"

    def test_resolve_voice_returns_default_if_none(self) -> None:
        """_resolve_voice returns default when voice is None."""

        class VoiceProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(24000, "pcm_f32le", 1, False)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                resolved = self._resolve_voice(voice, "default_voice")
                return AudioResult(resolved.encode(), 24000, "pcm_f32le")

            def list_voices(self) -> list[str]:
                return ["voice_a", "default_voice"]

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

        provider = VoiceProvider()

        # None voice
        result = provider.synthesize("test", voice=None)
        assert result.data == b"default_voice"

    def test_resolve_voice_falls_back_with_warning(self, caplog) -> None:
        """_resolve_voice falls back to default with warning for unknown voice."""

        class VoiceProvider(TTSProvider):
            def initialize(self, config: dict) -> None:
                pass

            def get_properties(self) -> TTSProperties:
                return TTSProperties(24000, "pcm_f32le", 1, False)

            def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
                resolved = self._resolve_voice(voice, "fallback")
                return AudioResult(resolved.encode(), 24000, "pcm_f32le")

            def list_voices(self) -> list[str]:
                return ["voice_a", "fallback"]

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

        provider = VoiceProvider()

        # Unknown voice triggers fallback
        import logging
        with caplog.at_level(logging.WARNING):
            result = provider.synthesize("test", voice="nonexistent_voice")

        assert result.data == b"fallback"
        assert "nonexistent_voice" in caplog.text
        assert "not found" in caplog.text
