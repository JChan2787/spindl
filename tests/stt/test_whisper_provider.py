"""Tests for Whisper.cpp STT provider wrapper."""

import pytest
from unittest.mock import patch, MagicMock

import numpy as np

from spindl.stt.base import STTProperties, STTProvider
from spindl.stt.builtin.whisper.provider import WhisperSTTProvider
from spindl.stt.builtin.whisper.client import WhisperSTT


class TestWhisperSTTProviderInit:
    """Tests for WhisperSTTProvider initialization."""

    def test_is_stt_provider_subclass(self) -> None:
        """WhisperSTTProvider inherits from STTProvider."""
        assert issubclass(WhisperSTTProvider, STTProvider)

    def test_instantiation_is_lightweight(self) -> None:
        """Instantiation creates no connections."""
        provider = WhisperSTTProvider()
        assert provider._client is None
        assert provider._initialized is False

    def test_initialize_creates_client(self) -> None:
        """initialize() creates a WhisperSTT client and checks server."""
        config = {"host": "127.0.0.1", "port": 8080, "timeout": 30.0}

        with patch.object(WhisperSTT, "is_server_available", return_value=True):
            provider = WhisperSTTProvider()
            provider.initialize(config)

            assert provider._client is not None
            assert provider._initialized is True
            assert provider._client.host == "127.0.0.1"
            assert provider._client.port == 8080
            assert provider._client.timeout == 30.0

    def test_initialize_with_defaults(self) -> None:
        """initialize() uses defaults when config values are missing."""
        config = {}

        with patch.object(WhisperSTT, "is_server_available", return_value=True):
            provider = WhisperSTTProvider()
            provider.initialize(config)

            assert provider._client.host == "127.0.0.1"
            assert provider._client.port == 8080
            assert provider._client.timeout == 30.0
            assert provider._client.language == "en"
            assert provider._client.response_format == "json"
            assert provider._client.inference_path == "/inference"

    def test_initialize_with_full_config(self) -> None:
        """initialize() passes all config fields to client."""
        config = {
            "host": "10.0.0.1",
            "port": 9090,
            "timeout": 60.0,
            "language": "fr",
            "response_format": "verbose_json",
            "inference_path": "/transcribe",
        }

        with patch.object(WhisperSTT, "is_server_available", return_value=True):
            provider = WhisperSTTProvider()
            provider.initialize(config)

            assert provider._client.host == "10.0.0.1"
            assert provider._client.port == 9090
            assert provider._client.timeout == 60.0
            assert provider._client.language == "fr"
            assert provider._client.response_format == "verbose_json"
            assert provider._client.inference_path == "/transcribe"

    def test_initialize_raises_when_server_unavailable(self) -> None:
        """initialize() raises ConnectionError if server is down."""
        config = {"host": "127.0.0.1", "port": 8080}

        with patch.object(WhisperSTT, "is_server_available", return_value=False):
            provider = WhisperSTTProvider()

            with pytest.raises(ConnectionError, match="not available"):
                provider.initialize(config)


class TestWhisperSTTProviderProperties:
    """Tests for WhisperSTTProvider.get_properties()."""

    def test_returns_correct_properties(self) -> None:
        """get_properties() returns whisper.cpp's fixed properties."""
        provider = WhisperSTTProvider()
        props = provider.get_properties()

        assert isinstance(props, STTProperties)
        assert props.sample_rate == 16000
        assert props.audio_format == "pcm_s16le"
        assert props.supports_streaming is False

    def test_properties_match_class_constants(self) -> None:
        """Properties match the class-level constants."""
        provider = WhisperSTTProvider()
        props = provider.get_properties()

        assert props.sample_rate == WhisperSTTProvider.SAMPLE_RATE
        assert props.audio_format == WhisperSTTProvider.AUDIO_FORMAT


class TestWhisperSTTProviderTranscribe:
    """Tests for WhisperSTTProvider.transcribe()."""

    def _make_initialized_provider(self) -> WhisperSTTProvider:
        """Create an initialized provider with mocked server."""
        provider = WhisperSTTProvider()
        with patch.object(WhisperSTT, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 8080})
        return provider

    def test_transcribe_delegates_to_client(self) -> None:
        """transcribe() delegates to the underlying WhisperSTT client."""
        provider = self._make_initialized_provider()

        audio = np.zeros(16000, dtype=np.float32)
        with patch.object(provider._client, "transcribe", return_value="hello world") as mock:
            result = provider.transcribe(audio, sample_rate=16000)

            assert result == "hello world"
            mock.assert_called_once_with(audio, 16000)

    def test_transcribe_raises_when_not_initialized(self) -> None:
        """transcribe() raises RuntimeError if called before initialize()."""
        provider = WhisperSTTProvider()

        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not initialized"):
            provider.transcribe(audio)

    def test_transcribe_passes_sample_rate(self) -> None:
        """transcribe() passes sample_rate to client."""
        provider = self._make_initialized_provider()

        audio = np.zeros(16000, dtype=np.float32)
        with patch.object(provider._client, "transcribe", return_value="test") as mock:
            provider.transcribe(audio, sample_rate=16000)
            mock.assert_called_once_with(audio, 16000)

    def test_transcribe_default_sample_rate(self) -> None:
        """transcribe() defaults to 16000 sample rate."""
        provider = self._make_initialized_provider()

        audio = np.zeros(16000, dtype=np.float32)
        with patch.object(provider._client, "transcribe", return_value="test") as mock:
            provider.transcribe(audio)
            mock.assert_called_once_with(audio, 16000)


class TestWhisperSTTProviderHealthCheck:
    """Tests for WhisperSTTProvider.health_check()."""

    def test_health_check_delegates_to_client(self) -> None:
        """health_check() delegates to client.is_server_available()."""
        provider = WhisperSTTProvider()
        with patch.object(WhisperSTT, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 8080})

        with patch.object(provider._client, "is_server_available", return_value=True):
            assert provider.health_check() is True

        with patch.object(provider._client, "is_server_available", return_value=False):
            assert provider.health_check() is False

    def test_health_check_returns_false_when_no_client(self) -> None:
        """health_check() returns False if provider was never initialized."""
        provider = WhisperSTTProvider()
        assert provider.health_check() is False


class TestWhisperSTTProviderShutdown:
    """Tests for WhisperSTTProvider.shutdown()."""

    def test_shutdown_clears_client(self) -> None:
        """shutdown() clears client reference and resets state."""
        provider = WhisperSTTProvider()
        with patch.object(WhisperSTT, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 8080})

        assert provider._initialized is True
        assert provider._client is not None

        provider.shutdown()

        assert provider._client is None
        assert provider._initialized is False

    def test_shutdown_when_never_initialized(self) -> None:
        """shutdown() is safe to call even if never initialized."""
        provider = WhisperSTTProvider()
        provider.shutdown()  # Should not raise
        assert provider._client is None


class TestWhisperSTTProviderValidateConfig:
    """Tests for WhisperSTTProvider.validate_config()."""

    def test_valid_config(self) -> None:
        """Valid config returns no errors."""
        config = {"host": "127.0.0.1", "port": 8080, "timeout": 30.0}
        errors = WhisperSTTProvider.validate_config(config)
        assert errors == []

    def test_missing_host(self) -> None:
        """Missing host produces error."""
        config = {"port": 8080}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("host" in e for e in errors)

    def test_missing_port(self) -> None:
        """Missing port produces error."""
        config = {"host": "127.0.0.1"}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("port" in e for e in errors)

    def test_invalid_host_type(self) -> None:
        """Non-string host produces error."""
        config = {"host": 12345, "port": 8080}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("host" in e and "string" in e for e in errors)

    def test_invalid_port_type(self) -> None:
        """Non-integer port produces error."""
        config = {"host": "127.0.0.1", "port": "8080"}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("port" in e and "integer" in e for e in errors)

    def test_port_out_of_range(self) -> None:
        """Port outside 1-65535 produces error."""
        config = {"host": "127.0.0.1", "port": 99999}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("port" in e for e in errors)

    def test_zero_port(self) -> None:
        """Port 0 is invalid."""
        config = {"host": "127.0.0.1", "port": 0}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("port" in e for e in errors)

    def test_negative_timeout(self) -> None:
        """Negative timeout produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "timeout": -1.0}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_zero_timeout(self) -> None:
        """Zero timeout produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "timeout": 0}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_invalid_timeout_type(self) -> None:
        """Non-numeric timeout produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "timeout": "fast"}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_valid_response_formats(self) -> None:
        """All valid response_format values accepted."""
        for fmt in ("json", "verbose_json", "text"):
            config = {"host": "127.0.0.1", "port": 8080, "response_format": fmt}
            errors = WhisperSTTProvider.validate_config(config)
            assert errors == [], f"Format '{fmt}' should be valid"

    def test_invalid_response_format(self) -> None:
        """Invalid response_format produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "response_format": "xml"}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("response_format" in e for e in errors)

    def test_invalid_inference_path(self) -> None:
        """Inference path not starting with / produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "inference_path": "inference"}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("inference_path" in e for e in errors)

    def test_valid_inference_path(self) -> None:
        """Inference path starting with / is accepted."""
        config = {"host": "127.0.0.1", "port": 8080, "inference_path": "/transcribe"}
        errors = WhisperSTTProvider.validate_config(config)
        assert errors == []

    def test_negative_threads(self) -> None:
        """Negative threads produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "threads": -1}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("threads" in e for e in errors)

    def test_zero_threads(self) -> None:
        """Zero threads produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "threads": 0}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("threads" in e for e in errors)

    def test_invalid_no_gpu_type(self) -> None:
        """Non-boolean no_gpu produces error."""
        config = {"host": "127.0.0.1", "port": 8080, "no_gpu": "yes"}
        errors = WhisperSTTProvider.validate_config(config)
        assert any("no_gpu" in e for e in errors)

    def test_optional_fields_accepted(self) -> None:
        """All optional fields accepted when valid."""
        config = {
            "host": "127.0.0.1",
            "port": 8080,
            "timeout": 30.0,
            "language": "en",
            "response_format": "json",
            "inference_path": "/inference",
            "model_path": "models/ggml-base.en.bin",
            "binary_path": "whisper-server",
            "threads": 4,
            "no_gpu": False,
        }
        errors = WhisperSTTProvider.validate_config(config)
        assert errors == []

    def test_is_classmethod(self) -> None:
        """validate_config is callable on the class."""
        errors = WhisperSTTProvider.validate_config({"host": "localhost", "port": 8080})
        assert isinstance(errors, list)


class TestWhisperSTTProviderServerCommand:
    """Tests for WhisperSTTProvider.get_server_command()."""

    def test_returns_none_without_model_path(self) -> None:
        """No model_path means we can't generate a command."""
        config = {"host": "127.0.0.1", "port": 8080}
        cmd = WhisperSTTProvider.get_server_command(config)
        assert cmd is None

    def test_basic_command_with_model(self) -> None:
        """Basic command includes binary and model path."""
        config = {"model_path": "models/ggml-base.en.bin"}
        cmd = WhisperSTTProvider.get_server_command(config)

        assert cmd is not None
        assert "whisper-server" in cmd
        assert "-m models/ggml-base.en.bin" in cmd

    def test_custom_host_and_port(self) -> None:
        """Custom host and port appear in command."""
        config = {
            "model_path": "model.bin",
            "host": "0.0.0.0",
            "port": 9090,
        }
        cmd = WhisperSTTProvider.get_server_command(config)

        assert "--host 0.0.0.0" in cmd
        assert "--port 9090" in cmd

    def test_language_flag(self) -> None:
        """Language appears in command."""
        config = {"model_path": "model.bin", "language": "fr"}
        cmd = WhisperSTTProvider.get_server_command(config)

        assert "-l fr" in cmd

    def test_threads_flag(self) -> None:
        """Thread count appears in command."""
        config = {"model_path": "model.bin", "threads": 8}
        cmd = WhisperSTTProvider.get_server_command(config)

        assert "-t 8" in cmd

    def test_no_gpu_flag(self) -> None:
        """--no-gpu appears when no_gpu is True."""
        config = {"model_path": "model.bin", "no_gpu": True}
        cmd = WhisperSTTProvider.get_server_command(config)

        assert "--no-gpu" in cmd

    def test_no_gpu_flag_absent_when_false(self) -> None:
        """--no-gpu absent when no_gpu is False."""
        config = {"model_path": "model.bin", "no_gpu": False}
        cmd = WhisperSTTProvider.get_server_command(config)

        assert "--no-gpu" not in cmd

    def test_custom_binary_path(self) -> None:
        """Custom binary path in command."""
        config = {
            "model_path": "model.bin",
            "binary_path": "/opt/whisper/whisper-server",
        }
        cmd = WhisperSTTProvider.get_server_command(config)

        assert "/opt/whisper/whisper-server" in cmd
        assert cmd.startswith("/opt/whisper/whisper-server")

    def test_is_classmethod(self) -> None:
        """get_server_command is callable on the class."""
        cmd = WhisperSTTProvider.get_server_command({"model_path": "m.bin"})
        assert isinstance(cmd, str)
