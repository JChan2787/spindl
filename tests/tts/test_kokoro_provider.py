"""Tests for KokoroTTSProvider implementation."""

import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spindl.tts.base import AudioResult, TTSProperties
from spindl.tts.builtin.kokoro import KokoroTTS, KokoroTTSProvider


class TestKokoroTTSProviderInit:
    """Tests for KokoroTTSProvider initialization."""

    def test_creates_uninitialized(self) -> None:
        """Provider starts in uninitialized state."""
        provider = KokoroTTSProvider()

        assert provider._initialized is False
        assert provider._client is None

    def test_has_default_voice(self) -> None:
        """Provider has default voice configured."""
        provider = KokoroTTSProvider()

        assert provider._default_voice == "af_bella"

    def test_has_default_language(self) -> None:
        """Provider has default language configured."""
        provider = KokoroTTSProvider()

        assert provider._default_language == "a"


class TestKokoroTTSProviderInitialize:
    """Tests for initialize() method."""

    def test_initialize_with_valid_config(self) -> None:
        """initialize() succeeds when server is available."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({
                "host": "127.0.0.1",
                "port": 5556,
            })

        assert provider._initialized is True
        assert provider._client is not None

    def test_initialize_raises_when_server_unavailable(self) -> None:
        """initialize() raises ConnectionError when server not reachable."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=False):
            with pytest.raises(ConnectionError) as exc_info:
                provider.initialize({
                    "host": "127.0.0.1",
                    "port": 5556,
                })

            assert "not available" in str(exc_info.value)

    def test_initialize_uses_config_values(self) -> None:
        """initialize() uses config values for client creation."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({
                "host": "192.168.1.100",
                "port": 9999,
                "timeout": 60.0,
                "voice": "am_michael",
                "language": "b",
            })

        assert provider._client is not None
        assert provider._client.host == "192.168.1.100"
        assert provider._client.port == 9999
        assert provider._client.timeout == 60.0
        assert provider._default_voice == "am_michael"
        assert provider._default_language == "b"

    def test_initialize_sets_models_dir(self) -> None:
        """initialize() configures models directory from config."""
        provider = KokoroTTSProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(KokoroTTS, "is_server_available", return_value=True):
                provider.initialize({
                    "host": "127.0.0.1",
                    "port": 5556,
                    "models_dir": tmpdir,
                })

            assert provider._models_dir == Path(tmpdir)


class TestKokoroTTSProviderGetProperties:
    """Tests for get_properties() method."""

    def test_returns_correct_sample_rate(self) -> None:
        """get_properties() returns 24000 Hz sample rate."""
        provider = KokoroTTSProvider()
        props = provider.get_properties()

        assert props.sample_rate == 24000

    def test_returns_correct_format(self) -> None:
        """get_properties() returns pcm_f32le format."""
        provider = KokoroTTSProvider()
        props = provider.get_properties()

        assert props.audio_format == "pcm_f32le"

    def test_returns_mono(self) -> None:
        """get_properties() returns 1 channel (mono)."""
        provider = KokoroTTSProvider()
        props = provider.get_properties()

        assert props.channels == 1

    def test_reports_no_streaming(self) -> None:
        """get_properties() reports non-streaming."""
        provider = KokoroTTSProvider()
        props = provider.get_properties()

        assert props.supports_streaming is False

    def test_returns_tts_properties_instance(self) -> None:
        """get_properties() returns TTSProperties dataclass."""
        provider = KokoroTTSProvider()
        props = provider.get_properties()

        assert isinstance(props, TTSProperties)


class TestKokoroTTSProviderSynthesize:
    """Tests for synthesize() method."""

    def test_raises_when_not_initialized(self) -> None:
        """synthesize() raises RuntimeError when not initialized."""
        provider = KokoroTTSProvider()

        with pytest.raises(RuntimeError) as exc_info:
            provider.synthesize("hello")

        assert "not initialized" in str(exc_info.value)

    def test_delegates_to_client(self) -> None:
        """synthesize() delegates to KokoroTTS client."""
        provider = KokoroTTSProvider()

        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        with patch.object(provider._client, "synthesize", return_value=mock_audio) as mock_synth:
            result = provider.synthesize("hello world")

            mock_synth.assert_called_once_with(
                text="hello world",
                voice="af_bella",
                lang="a",
            )

    def test_returns_audio_result(self) -> None:
        """synthesize() returns AudioResult with correct format."""
        provider = KokoroTTSProvider()

        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        with patch.object(provider._client, "synthesize", return_value=mock_audio):
            result = provider.synthesize("hello world")

            assert isinstance(result, AudioResult)
            assert result.sample_rate == 24000
            assert result.format == "pcm_f32le"
            assert result.is_final is True
            assert result.data == mock_audio.tobytes()

    def test_uses_provided_voice(self) -> None:
        """synthesize() uses provided voice parameter."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        # Mock list_voices to return the voice we're requesting
        with patch.object(provider, "list_voices", return_value=["af_nicole", "af_bella"]):
            with patch.object(provider._client, "synthesize", return_value=np.array([0.1])) as mock_synth:
                provider.synthesize("hello", voice="af_nicole")

                mock_synth.assert_called_once_with(
                    text="hello",
                    voice="af_nicole",
                    lang="a",
                )

    def test_uses_provided_language(self) -> None:
        """synthesize() uses language from kwargs."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        with patch.object(provider._client, "synthesize", return_value=np.array([0.1])) as mock_synth:
            provider.synthesize("cheerio", language="b")

            mock_synth.assert_called_once_with(
                text="cheerio",
                voice="af_bella",
                lang="b",
            )


class TestKokoroTTSProviderSynthesizeStream:
    """Tests for synthesize_stream() method."""

    def test_yields_single_result(self) -> None:
        """synthesize_stream() yields exactly one AudioResult."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        with patch.object(provider._client, "synthesize", return_value=np.array([0.1])):
            results = list(provider.synthesize_stream("test"))

            assert len(results) == 1
            assert isinstance(results[0], AudioResult)

    def test_result_is_final(self) -> None:
        """synthesize_stream() yields result with is_final=True."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        with patch.object(provider._client, "synthesize", return_value=np.array([0.1])):
            results = list(provider.synthesize_stream("test"))

            assert results[0].is_final is True


class TestKokoroTTSProviderListVoices:
    """Tests for list_voices() method."""

    def test_returns_default_list_when_no_models_dir(self) -> None:
        """list_voices() returns default list when models_dir not configured."""
        provider = KokoroTTSProvider()

        voices = provider.list_voices()

        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "af_bella" in voices

    def test_scans_voices_directory(self) -> None:
        """list_voices() scans models/voices/*.pt files."""
        provider = KokoroTTSProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create voices directory with mock .pt files
            voices_dir = Path(tmpdir) / "voices"
            voices_dir.mkdir()
            (voices_dir / "af_test.pt").touch()
            (voices_dir / "am_other.pt").touch()
            (voices_dir / "not_a_voice.txt").touch()  # Should be ignored

            with patch.object(KokoroTTS, "is_server_available", return_value=True):
                provider.initialize({
                    "host": "127.0.0.1",
                    "port": 5556,
                    "models_dir": tmpdir,
                })

            voices = provider.list_voices()

            assert "af_test" in voices
            assert "am_other" in voices
            assert "not_a_voice" not in voices

    def test_returns_sorted_list(self) -> None:
        """list_voices() returns sorted voice list."""
        provider = KokoroTTSProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir) / "voices"
            voices_dir.mkdir()
            (voices_dir / "z_voice.pt").touch()
            (voices_dir / "a_voice.pt").touch()
            (voices_dir / "m_voice.pt").touch()

            with patch.object(KokoroTTS, "is_server_available", return_value=True):
                provider.initialize({
                    "host": "127.0.0.1",
                    "port": 5556,
                    "models_dir": tmpdir,
                })

            voices = provider.list_voices()

            assert voices == sorted(voices)


class TestKokoroTTSProviderHealthCheck:
    """Tests for health_check() method."""

    def test_returns_false_when_not_initialized(self) -> None:
        """health_check() returns False when client not created."""
        provider = KokoroTTSProvider()

        assert provider.health_check() is False

    def test_delegates_to_client(self) -> None:
        """health_check() delegates to client.is_server_available()."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        with patch.object(provider._client, "is_server_available", return_value=True) as mock_check:
            result = provider.health_check()

            mock_check.assert_called_once()
            assert result is True

    def test_returns_false_when_server_down(self) -> None:
        """health_check() returns False when server not responding."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        with patch.object(provider._client, "is_server_available", return_value=False):
            assert provider.health_check() is False


class TestKokoroTTSProviderShutdown:
    """Tests for shutdown() method."""

    def test_clears_client(self) -> None:
        """shutdown() sets client to None."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        assert provider._client is not None

        provider.shutdown()

        assert provider._client is None

    def test_clears_initialized_flag(self) -> None:
        """shutdown() sets _initialized to False."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        assert provider._initialized is True

        provider.shutdown()

        assert provider._initialized is False

    def test_safe_to_call_when_not_initialized(self) -> None:
        """shutdown() doesn't raise when called on uninitialized provider."""
        provider = KokoroTTSProvider()

        # Should not raise
        provider.shutdown()

        assert provider._client is None


class TestKokoroTTSProviderValidateConfig:
    """Tests for validate_config() classmethod."""

    def test_valid_config_returns_empty_list(self) -> None:
        """validate_config() returns empty list for valid config."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "port": 5556,
            "conda_env": "kokoro",
        })

        assert errors == []

    def test_missing_host_returns_error(self) -> None:
        """validate_config() returns error when host missing."""
        errors = KokoroTTSProvider.validate_config({
            "port": 5556,
            "conda_env": "kokoro",
        })

        assert len(errors) == 1
        assert "host" in errors[0]

    def test_missing_port_returns_error(self) -> None:
        """validate_config() returns error when port missing."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "conda_env": "kokoro",
        })

        assert len(errors) == 1
        assert "port" in errors[0]

    def test_missing_conda_env_is_valid(self) -> None:
        """validate_config() allows missing conda_env (NANO-032: optional for non-conda users)."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "port": 5556,
        })

        assert len(errors) == 0

    def test_invalid_port_type_returns_error(self) -> None:
        """validate_config() returns error when port is not int."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "port": "5556",  # String, not int
            "conda_env": "kokoro",
        })

        assert len(errors) == 1
        assert "port" in errors[0]
        assert "integer" in errors[0]

    def test_port_out_of_range_returns_error(self) -> None:
        """validate_config() returns error when port out of range."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "port": 99999,
            "conda_env": "kokoro",
        })

        assert len(errors) == 1
        assert "port" in errors[0]
        assert "65535" in errors[0]

    def test_invalid_language_returns_error(self) -> None:
        """validate_config() returns error for invalid language."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "port": 5556,
            "conda_env": "kokoro",
            "language": "x",
        })

        assert len(errors) == 1
        assert "language" in errors[0]

    def test_negative_timeout_returns_error(self) -> None:
        """validate_config() returns error for negative timeout."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "port": 5556,
            "conda_env": "kokoro",
            "timeout": -1,
        })

        assert len(errors) == 1
        assert "timeout" in errors[0]

    def test_nonexistent_models_dir_returns_error(self) -> None:
        """validate_config() returns error for non-existent models_dir."""
        errors = KokoroTTSProvider.validate_config({
            "host": "127.0.0.1",
            "port": 5556,
            "conda_env": "kokoro",
            "models_dir": "/nonexistent/path/that/does/not/exist",
        })

        assert len(errors) == 1
        assert "models_dir" in errors[0]

    def test_multiple_errors_returned(self) -> None:
        """validate_config() returns multiple errors when config has multiple issues."""
        errors = KokoroTTSProvider.validate_config({
            # Missing host, conda_env, port is wrong type, language invalid
            "port": "bad",
            "language": "z",
        })

        assert len(errors) >= 2


class TestKokoroTTSProviderGetServerCommand:
    """Tests for get_server_command() classmethod."""

    def test_returns_conda_command(self) -> None:
        """get_server_command() returns conda run command with configured env."""
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
            "models_dir": "/abs/tts/models",
        })

        assert command is not None
        assert "conda run -n kokoro" in command

    def test_missing_conda_env_runs_directly(self) -> None:
        """get_server_command() runs directly without conda wrapper when conda_env omitted (NANO-032)."""
        command = KokoroTTSProvider.get_server_command({
            "port": 5556,
            "models_dir": "/abs/tts/models",
        })

        assert command is not None
        assert "conda run" not in command
        assert "python" in command

    def test_includes_port(self) -> None:
        """get_server_command() includes port in command."""
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 9999,
            "models_dir": "/abs/tts/models",
        })

        assert "--port 9999" in command

    def test_includes_models_dir(self) -> None:
        """get_server_command() includes models directory (absolute passthrough)."""
        from pathlib import Path
        abs_models = str(Path("/path/to/models").resolve())
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
            "models_dir": abs_models,
        })

        assert f"--models-dir {abs_models}" in command

    def test_uses_default_models_dir(self) -> None:
        """get_server_command() resolves default models_dir to absolute path."""
        from pathlib import Path
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
        })

        assert "--models-dir" in command
        # Extract the models_dir value from the command (may be quoted, followed by other flags)
        parts = command.split("--models-dir ")
        assert len(parts) == 2
        models_path = parts[1].split(" --")[0].strip().strip('"')
        assert Path(models_path).is_absolute()
        assert models_path.replace("/", os.sep).endswith("tts" + os.sep + "models")

    def test_uses_custom_server_script(self) -> None:
        """get_server_command() resolves custom server script path (absolute passthrough)."""
        from pathlib import Path
        abs_script = str(Path("/abs/custom/path/server.py").resolve())
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
            "server_script": abs_script,
        })

        assert abs_script in command

    def test_default_server_script_is_absolute(self) -> None:
        """get_server_command() default server_script resolves to co-located server.py."""
        from pathlib import Path
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
        })

        assert "server.py" in command
        # Extract the script path and verify it's absolute (may be quoted)
        script_path = command.split("python ")[-1].split(" --port")[0].strip().strip('"')
        assert Path(script_path).is_absolute()


class TestKokoroTTSProviderDeviceSelection:
    """Tests for NANO-085: TTS device selector in get_server_command()."""

    def test_default_device_is_cuda(self) -> None:
        """get_server_command() defaults to --device cuda when no device specified."""
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
            "models_dir": "/abs/tts/models",
        })

        assert "--device cuda" in command

    def test_device_cpu(self) -> None:
        """get_server_command() passes --device cpu when configured."""
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
            "models_dir": "/abs/tts/models",
            "device": "cpu",
        })

        assert "--device cpu" in command

    def test_device_cuda_explicit(self) -> None:
        """get_server_command() passes --device cuda:0 for specific GPU."""
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
            "models_dir": "/abs/tts/models",
            "device": "cuda:0",
        })

        assert "--device cuda:0" in command

    def test_device_cuda_1(self) -> None:
        """get_server_command() passes --device cuda:1 for second GPU."""
        command = KokoroTTSProvider.get_server_command({
            "conda_env": "kokoro",
            "port": 5556,
            "models_dir": "/abs/tts/models",
            "device": "cuda:1",
        })

        assert "--device cuda:1" in command

    def test_device_without_conda(self) -> None:
        """get_server_command() passes --device flag even without conda wrapper."""
        command = KokoroTTSProvider.get_server_command({
            "port": 5556,
            "models_dir": "/abs/tts/models",
            "device": "cpu",
        })

        assert "conda run" not in command
        assert "--device cpu" in command


class TestKokoroTTSProviderVoiceResolution:
    """Tests for voice fallback behavior."""

    def test_falls_back_to_default_for_unknown_voice(self) -> None:
        """Provider falls back to default voice when requested voice unknown."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({"host": "127.0.0.1", "port": 5556})

        # Mock list_voices to not include the requested voice
        with patch.object(provider, "list_voices", return_value=["af_bella", "af_nicole"]):
            with patch.object(provider._client, "synthesize", return_value=np.array([0.1])) as mock_synth:
                provider.synthesize("hello", voice="nonexistent_voice")

                # Should use default voice, not the nonexistent one
                mock_synth.assert_called_once_with(
                    text="hello",
                    voice="af_bella",  # Default fallback
                    lang="a",
                )

    def test_uses_default_when_voice_is_none(self) -> None:
        """Provider uses default voice when voice parameter is None."""
        provider = KokoroTTSProvider()

        with patch.object(KokoroTTS, "is_server_available", return_value=True):
            provider.initialize({
                "host": "127.0.0.1",
                "port": 5556,
                "voice": "am_michael",
            })

        with patch.object(provider._client, "synthesize", return_value=np.array([0.1])) as mock_synth:
            provider.synthesize("hello", voice=None)

            mock_synth.assert_called_once_with(
                text="hello",
                voice="am_michael",  # Configured default
                lang="a",
            )


class TestKokoroTTSProviderRegistryIntegration:
    """Tests for registry discovery of KokoroTTSProvider."""

    def test_registry_discovers_kokoro(self) -> None:
        """Registry discovers kokoro as built-in provider."""
        from spindl.tts.registry import TTSProviderRegistry

        registry = TTSProviderRegistry()
        available = registry.list_available()

        assert "kokoro" in available

    def test_registry_returns_kokoro_class(self) -> None:
        """Registry returns KokoroTTSProvider class for 'kokoro'."""
        from spindl.tts.registry import TTSProviderRegistry

        registry = TTSProviderRegistry()
        provider_class = registry.get_provider_class("kokoro")

        assert provider_class is KokoroTTSProvider
