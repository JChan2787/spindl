"""Tests for AudioPlayback dynamic sample rate configuration."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from spindl.audio.playback import AudioPlayback, FullDuplexStream


class TestAudioPlaybackConfiguration:
    """Tests for AudioPlayback.configure() and dynamic sample rate."""

    def test_default_sample_rate_is_16khz(self) -> None:
        """Unconfigured AudioPlayback defaults to 16kHz."""
        playback = AudioPlayback()

        assert playback.sample_rate == 16000
        assert playback.channels == 1
        assert playback.is_configured is False

    def test_configure_sets_sample_rate(self) -> None:
        """configure() sets custom sample rate."""
        playback = AudioPlayback()
        playback.configure(sample_rate=24000)

        assert playback.sample_rate == 24000
        assert playback.is_configured is True

    def test_configure_sets_channels(self) -> None:
        """configure() sets custom channel count."""
        playback = AudioPlayback()
        playback.configure(sample_rate=24000, channels=2)

        assert playback.channels == 2

    def test_configure_24khz_for_kokoro(self) -> None:
        """configure() accepts Kokoro's 24kHz sample rate."""
        playback = AudioPlayback()
        playback.configure(sample_rate=24000, channels=1)

        assert playback.sample_rate == 24000
        assert playback.channels == 1

    def test_configure_12khz_for_qwen3(self) -> None:
        """configure() accepts Qwen3's 12kHz sample rate."""
        playback = AudioPlayback()
        playback.configure(sample_rate=12000, channels=1)

        assert playback.sample_rate == 12000

    def test_configure_rejects_zero_sample_rate(self) -> None:
        """configure() rejects sample_rate <= 0."""
        playback = AudioPlayback()

        with pytest.raises(ValueError, match="sample_rate must be positive"):
            playback.configure(sample_rate=0)

    def test_configure_rejects_negative_sample_rate(self) -> None:
        """configure() rejects negative sample rate."""
        playback = AudioPlayback()

        with pytest.raises(ValueError, match="sample_rate must be positive"):
            playback.configure(sample_rate=-1)

    def test_configure_rejects_invalid_channels(self) -> None:
        """configure() rejects channels not in [1, 2]."""
        playback = AudioPlayback()

        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            playback.configure(sample_rate=24000, channels=0)

        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            playback.configure(sample_rate=24000, channels=3)

    def test_configure_can_be_called_multiple_times(self) -> None:
        """configure() can update sample rate after initial configuration."""
        playback = AudioPlayback()

        playback.configure(sample_rate=24000)
        assert playback.sample_rate == 24000

        playback.configure(sample_rate=12000)
        assert playback.sample_rate == 12000


class TestAudioPlaybackDuration:
    """Tests for duration calculations with dynamic sample rate."""

    def test_duration_uses_default_rate_when_unconfigured(self) -> None:
        """Duration calculation uses 16kHz when not configured."""
        playback = AudioPlayback()

        # Manually set audio data (bypassing play() which needs sounddevice)
        playback._audio_data = np.zeros(16000, dtype=np.float32)

        # 16000 samples at 16000 Hz = 1.0 second
        assert playback.duration == 1.0

    def test_duration_uses_configured_rate(self) -> None:
        """Duration calculation uses configured sample rate."""
        playback = AudioPlayback()
        playback.configure(sample_rate=24000)

        # Manually set audio data
        playback._audio_data = np.zeros(24000, dtype=np.float32)

        # 24000 samples at 24000 Hz = 1.0 second
        assert playback.duration == 1.0

    def test_duration_24khz_audio_at_16khz_rate_is_wrong(self) -> None:
        """Demonstrates why matching sample rates matters."""
        playback = AudioPlayback()
        # Don't configure - use default 16kHz

        # 24000 samples at default 16kHz rate = 1.5 seconds (WRONG!)
        playback._audio_data = np.zeros(24000, dtype=np.float32)
        assert playback.duration == 1.5  # This would be incorrect playback speed

        # Configure correctly
        playback.configure(sample_rate=24000)
        assert playback.duration == 1.0  # Now correct

    def test_position_uses_configured_rate(self) -> None:
        """Playback position uses configured sample rate."""
        playback = AudioPlayback()
        playback.configure(sample_rate=24000)

        # Simulate playback progress
        playback._playback_position = 12000  # Half of 24000

        # 12000 samples at 24000 Hz = 0.5 seconds
        assert playback.position == 0.5


class TestAudioPlaybackPlay:
    """Tests for play() with dynamic sample rate."""

    @patch("spindl.audio.playback.sd")
    def test_play_uses_configured_sample_rate(self, mock_sd: MagicMock) -> None:
        """play() creates stream with configured sample rate."""
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

        playback = AudioPlayback()
        playback.configure(sample_rate=24000, channels=1)

        audio = np.zeros(1000, dtype=np.float32)
        playback.play(audio, blocking=False)

        # Verify OutputStream was created with correct sample rate
        mock_sd.OutputStream.assert_called_once()
        call_kwargs = mock_sd.OutputStream.call_args.kwargs
        assert call_kwargs["samplerate"] == 24000
        assert call_kwargs["channels"] == 1

    @patch("spindl.audio.playback.sd")
    def test_play_uses_default_rate_when_unconfigured(self, mock_sd: MagicMock) -> None:
        """play() uses 16kHz default when not configured."""
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

        playback = AudioPlayback()
        # Don't call configure()

        audio = np.zeros(1000, dtype=np.float32)
        playback.play(audio, blocking=False)

        call_kwargs = mock_sd.OutputStream.call_args.kwargs
        assert call_kwargs["samplerate"] == 16000
        assert call_kwargs["channels"] == 1

    @patch("spindl.audio.playback.sd")
    def test_play_stereo_audio(self, mock_sd: MagicMock) -> None:
        """play() supports stereo configuration."""
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

        playback = AudioPlayback()
        playback.configure(sample_rate=44100, channels=2)

        audio = np.zeros(1000, dtype=np.float32)
        playback.play(audio, blocking=False)

        call_kwargs = mock_sd.OutputStream.call_args.kwargs
        assert call_kwargs["samplerate"] == 44100
        assert call_kwargs["channels"] == 2


class TestFullDuplexStreamConfiguration:
    """Tests for FullDuplexStream.configure_playback()."""

    @patch("spindl.audio.playback.sd")
    @patch("spindl.audio.capture.sd")
    def test_configure_playback_passes_to_internal_playback(
        self, mock_capture_sd: MagicMock, mock_playback_sd: MagicMock
    ) -> None:
        """configure_playback() configures internal AudioPlayback."""
        stream = FullDuplexStream()
        stream.configure_playback(sample_rate=24000, channels=1)

        assert stream._playback.sample_rate == 24000
        assert stream._playback.channels == 1

    @patch("spindl.audio.playback.sd")
    @patch("spindl.audio.capture.sd")
    def test_input_sample_rate_constant_is_16khz(
        self, mock_capture_sd: MagicMock, mock_playback_sd: MagicMock
    ) -> None:
        """Input sample rate remains fixed at 16kHz (VAD requirement)."""
        assert FullDuplexStream.INPUT_SAMPLE_RATE == 16000
