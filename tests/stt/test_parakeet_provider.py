"""Tests for Parakeet STT provider wrapper."""

import pytest
from unittest.mock import patch, MagicMock
from typing import Optional

import numpy as np

from spindl.stt.base import STTProperties, STTProvider
from spindl.stt.builtin.parakeet.provider import ParakeetSTTProvider
from spindl.stt.builtin.parakeet.client import ParakeetSTT


class TestParakeetSTTProviderInit:
    """Tests for ParakeetSTTProvider initialization."""

    def test_is_stt_provider_subclass(self) -> None:
        """ParakeetSTTProvider inherits from STTProvider."""
        assert issubclass(ParakeetSTTProvider, STTProvider)

    def test_instantiation_is_lightweight(self) -> None:
        """Instantiation creates no connections."""
        provider = ParakeetSTTProvider()
        assert provider._client is None
        assert provider._initialized is False

    def test_initialize_creates_client(self) -> None:
        """initialize() creates a ParakeetSTT client and checks server."""
        config = {"host": "127.0.0.1", "port": 5555, "timeout": 30.0}

        with patch.object(ParakeetSTT, "is_server_available", return_value=True):
            provider = ParakeetSTTProvider()
            provider.initialize(config)

            assert provider._client is not None
            assert provider._initialized is True
            assert provider._client.host == "127.0.0.1"
            assert provider._client.port == 5555
            assert provider._client.timeout == 30.0

    def test_initialize_with_defaults(self) -> None:
        """initialize() uses defaults when config values are missing."""
        config = {}

        with patch.object(ParakeetSTT, "is_server_available", return_value=True):
            provider = ParakeetSTTProvider()
            provider.initialize(config)

            assert provider._client.host == "127.0.0.1"
            assert provider._client.port == 5555
            assert provider._client.timeout == 30.0

    def test_initialize_with_retry_config(self) -> None:
        """initialize() passes retry settings to client."""
        config = {
            "host": "127.0.0.1",
            "port": 5555,
            "max_retries": 5,
            "retry_delay": 2.0,
        }

        with patch.object(ParakeetSTT, "is_server_available", return_value=True):
            provider = ParakeetSTTProvider()
            provider.initialize(config)

            assert provider._client.max_retries == 5
            assert provider._client.retry_delay == 2.0

    def test_initialize_raises_when_server_unavailable(self) -> None:
        """initialize() raises ConnectionError if server is down."""
        config = {"host": "127.0.0.1", "port": 5555}

        with patch.object(ParakeetSTT, "is_server_available", return_value=False):
            provider = ParakeetSTTProvider()

            with pytest.raises(ConnectionError, match="not available"):
                provider.initialize(config)


class TestParakeetSTTProviderProperties:
    """Tests for ParakeetSTTProvider.get_properties()."""

    def test_returns_correct_properties(self) -> None:
        """get_properties() returns Parakeet's fixed properties."""
        provider = ParakeetSTTProvider()
        props = provider.get_properties()

        assert isinstance(props, STTProperties)
        assert props.sample_rate == 16000
        assert props.audio_format == "pcm_f32le"
        assert props.supports_streaming is False

    def test_properties_match_class_constants(self) -> None:
        """Properties match the class-level constants."""
        provider = ParakeetSTTProvider()
        props = provider.get_properties()

        assert props.sample_rate == ParakeetSTTProvider.SAMPLE_RATE
        assert props.audio_format == ParakeetSTTProvider.AUDIO_FORMAT


class TestParakeetSTTProviderTranscribe:
    """Tests for ParakeetSTTProvider.transcribe()."""

    def _make_initialized_provider(self) -> ParakeetSTTProvider:
        """Create an initialized provider with mocked server."""
        provider = ParakeetSTTProvider()
        with patch.object(ParakeetSTT, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5555})
        return provider

    def test_transcribe_delegates_to_client(self) -> None:
        """transcribe() delegates to the underlying ParakeetSTT client."""
        provider = self._make_initialized_provider()

        audio = np.zeros(16000, dtype=np.float32)
        with patch.object(provider._client, "transcribe", return_value="hello world") as mock:
            result = provider.transcribe(audio, sample_rate=16000)

            assert result == "hello world"
            mock.assert_called_once_with(audio, 16000)

    def test_transcribe_raises_when_not_initialized(self) -> None:
        """transcribe() raises RuntimeError if called before initialize()."""
        provider = ParakeetSTTProvider()

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


class TestParakeetSTTProviderHealthCheck:
    """Tests for ParakeetSTTProvider.health_check()."""

    def test_health_check_delegates_to_client(self) -> None:
        """health_check() delegates to client.is_server_available()."""
        provider = ParakeetSTTProvider()
        with patch.object(ParakeetSTT, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5555})

        with patch.object(provider._client, "is_server_available", return_value=True):
            assert provider.health_check() is True

        with patch.object(provider._client, "is_server_available", return_value=False):
            assert provider.health_check() is False

    def test_health_check_returns_false_when_no_client(self) -> None:
        """health_check() returns False if provider was never initialized."""
        provider = ParakeetSTTProvider()
        assert provider.health_check() is False


class TestParakeetSTTProviderShutdown:
    """Tests for ParakeetSTTProvider.shutdown()."""

    def test_shutdown_clears_client(self) -> None:
        """shutdown() clears client reference and resets state."""
        provider = ParakeetSTTProvider()
        with patch.object(ParakeetSTT, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5555})

        assert provider._initialized is True
        assert provider._client is not None

        provider.shutdown()

        assert provider._client is None
        assert provider._initialized is False

    def test_shutdown_when_never_initialized(self) -> None:
        """shutdown() is safe to call even if never initialized."""
        provider = ParakeetSTTProvider()
        provider.shutdown()  # Should not raise
        assert provider._client is None


class TestParakeetSTTProviderValidateConfig:
    """Tests for ParakeetSTTProvider.validate_config()."""

    def test_valid_config(self) -> None:
        """Valid config returns no errors."""
        config = {"host": "127.0.0.1", "port": 5555, "timeout": 30.0}
        errors = ParakeetSTTProvider.validate_config(config)
        assert errors == []

    def test_missing_host(self) -> None:
        """Missing host produces error."""
        config = {"port": 5555}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("host" in e for e in errors)

    def test_missing_port(self) -> None:
        """Missing port produces error."""
        config = {"host": "127.0.0.1"}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("port" in e for e in errors)

    def test_invalid_host_type(self) -> None:
        """Non-string host produces error."""
        config = {"host": 12345, "port": 5555}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("host" in e and "string" in e for e in errors)

    def test_invalid_port_type(self) -> None:
        """Non-integer port produces error."""
        config = {"host": "127.0.0.1", "port": "5555"}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("port" in e and "integer" in e for e in errors)

    def test_port_out_of_range(self) -> None:
        """Port outside 1-65535 produces error."""
        config = {"host": "127.0.0.1", "port": 99999}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("port" in e for e in errors)

    def test_zero_port(self) -> None:
        """Port 0 is invalid."""
        config = {"host": "127.0.0.1", "port": 0}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("port" in e for e in errors)

    def test_negative_timeout(self) -> None:
        """Negative timeout produces error."""
        config = {"host": "127.0.0.1", "port": 5555, "timeout": -1.0}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_zero_timeout(self) -> None:
        """Zero timeout produces error."""
        config = {"host": "127.0.0.1", "port": 5555, "timeout": 0}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_invalid_timeout_type(self) -> None:
        """Non-numeric timeout produces error."""
        config = {"host": "127.0.0.1", "port": 5555, "timeout": "fast"}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_optional_fields_accepted(self) -> None:
        """Optional fields don't cause errors when present and valid."""
        config = {
            "host": "127.0.0.1",
            "port": 5555,
            "timeout": 30.0,
            "max_retries": 5,
            "retry_delay": 2.0,
        }
        errors = ParakeetSTTProvider.validate_config(config)
        assert errors == []

    def test_negative_max_retries(self) -> None:
        """Negative max_retries produces error."""
        config = {"host": "127.0.0.1", "port": 5555, "max_retries": -1}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("max_retries" in e for e in errors)

    def test_negative_retry_delay(self) -> None:
        """Negative retry_delay produces error."""
        config = {"host": "127.0.0.1", "port": 5555, "retry_delay": -0.5}
        errors = ParakeetSTTProvider.validate_config(config)
        assert any("retry_delay" in e for e in errors)

    def test_is_classmethod(self) -> None:
        """validate_config is callable on the class."""
        # Should not require instantiation
        errors = ParakeetSTTProvider.validate_config({"host": "localhost", "port": 5555})
        assert isinstance(errors, list)


class TestParakeetSTTProviderServerCommand:
    """Tests for ParakeetSTTProvider.get_server_command()."""

    def test_default_command(self) -> None:
        """Default command uses default port and script path."""
        config = {}
        cmd = ParakeetSTTProvider.get_server_command(config)

        assert cmd is not None
        assert "python" in cmd
        assert "nemo_server.py" in cmd
        assert "--port 5555" in cmd

    def test_custom_port(self) -> None:
        """Custom port appears in command."""
        config = {"port": 9999}
        cmd = ParakeetSTTProvider.get_server_command(config)

        assert "--port 9999" in cmd

    def test_conda_env(self) -> None:
        """Conda env wraps the command."""
        config = {"conda_env": "nemo"}
        cmd = ParakeetSTTProvider.get_server_command(config)

        assert "conda run -n nemo" in cmd

    def test_no_conda_env(self) -> None:
        """Without conda_env, command runs directly."""
        config = {"port": 5555}
        cmd = ParakeetSTTProvider.get_server_command(config)

        assert "conda" not in cmd

    def test_custom_server_script(self) -> None:
        """Custom server script path in command (absolute passthrough)."""
        from pathlib import Path
        abs_script = str(Path("/opt/stt/my_server.py").resolve())
        config = {"server_script": abs_script}
        cmd = ParakeetSTTProvider.get_server_command(config)

        assert abs_script in cmd

    def test_is_classmethod(self) -> None:
        """get_server_command is callable on the class."""
        cmd = ParakeetSTTProvider.get_server_command({})
        assert isinstance(cmd, str)
