"""Tests for Whisper.cpp STT HTTP client."""

import io
import wave

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import requests

from spindl.stt.builtin.whisper.client import WhisperSTT


class TestWhisperSTTInit:
    """Tests for WhisperSTT initialization."""

    def test_default_config(self) -> None:
        """Default config matches whisper.cpp server defaults."""
        client = WhisperSTT()

        assert client.host == "127.0.0.1"
        assert client.port == 8080
        assert client.timeout == 30.0
        assert client.language == "en"
        assert client.response_format == "json"
        assert client.inference_path == "/inference"

    def test_custom_config(self) -> None:
        """All parameters pass through to attributes."""
        client = WhisperSTT(
            host="192.168.1.100",
            port=9090,
            timeout=60.0,
            language="fr",
            response_format="verbose_json",
            inference_path="/transcribe",
        )

        assert client.host == "192.168.1.100"
        assert client.port == 9090
        assert client.timeout == 60.0
        assert client.language == "fr"
        assert client.response_format == "verbose_json"
        assert client.inference_path == "/transcribe"

    def test_base_url_construction(self) -> None:
        """Base URL is constructed from host and port."""
        client = WhisperSTT(host="10.0.0.1", port=7777)
        assert client._base_url == "http://10.0.0.1:7777"


class TestWhisperSTTTranscribe:
    """Tests for WhisperSTT.transcribe()."""

    def _make_mock_response(
        self, json_data: dict, status_code: int = 200
    ) -> MagicMock:
        """Create a mock requests.Response."""
        mock = MagicMock()
        mock.status_code = status_code
        mock.json.return_value = json_data
        mock.text = str(json_data)
        return mock

    def test_transcribe_sends_multipart_post(self) -> None:
        """transcribe() sends multipart POST to correct URL."""
        client = WhisperSTT(host="127.0.0.1", port=8080)
        audio = np.zeros(16000, dtype=np.float32)

        mock_response = self._make_mock_response({"text": "hello"})

        with patch("spindl.stt.builtin.whisper.client.requests.post", return_value=mock_response) as mock_post:
            client.transcribe(audio)

            mock_post.assert_called_once()
            call_args = mock_post.call_args

            assert call_args.args[0] == "http://127.0.0.1:8080/inference"
            assert "files" in call_args.kwargs
            assert "data" in call_args.kwargs
            assert call_args.kwargs["timeout"] == 30.0

            files = call_args.kwargs["files"]
            assert "file" in files
            filename, content, content_type = files["file"]
            assert filename == "audio.wav"
            assert content_type == "audio/wav"

            data = call_args.kwargs["data"]
            assert data["response_format"] == "json"
            assert data["language"] == "en"
            assert data["temperature"] == "0.0"

    def test_transcribe_returns_text(self) -> None:
        """transcribe() returns text from JSON response."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float32)

        mock_response = self._make_mock_response({"text": "hello world"})

        with patch("spindl.stt.builtin.whisper.client.requests.post", return_value=mock_response):
            result = client.transcribe(audio)
            assert result == "hello world"

    def test_transcribe_strips_whitespace(self) -> None:
        """transcribe() strips leading/trailing whitespace from result."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float32)

        mock_response = self._make_mock_response({"text": "  hello world  "})

        with patch("spindl.stt.builtin.whisper.client.requests.post", return_value=mock_response):
            result = client.transcribe(audio)
            assert result == "hello world"

    def test_transcribe_validates_numpy_type(self) -> None:
        """transcribe() rejects non-ndarray input."""
        client = WhisperSTT()

        with pytest.raises(ValueError, match="numpy array"):
            client.transcribe([1.0, 2.0, 3.0])

    def test_transcribe_validates_empty_audio(self) -> None:
        """transcribe() rejects empty audio array."""
        client = WhisperSTT()
        audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            client.transcribe(audio)

    def test_transcribe_validates_sample_rate(self) -> None:
        """transcribe() rejects non-16000 sample rate."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float32)

        with pytest.raises(ValueError, match="16000"):
            client.transcribe(audio, sample_rate=44100)

    def test_transcribe_converts_dtype(self) -> None:
        """transcribe() converts non-float32 input to float32."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float64)

        mock_response = self._make_mock_response({"text": "test"})

        with patch("spindl.stt.builtin.whisper.client.requests.post", return_value=mock_response):
            result = client.transcribe(audio)
            assert result == "test"

    def test_transcribe_connection_error(self) -> None:
        """transcribe() raises ConnectionError on connection failure."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float32)

        with patch(
            "spindl.stt.builtin.whisper.client.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                client.transcribe(audio)

    def test_transcribe_timeout(self) -> None:
        """transcribe() raises TimeoutError on timeout."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float32)

        with patch(
            "spindl.stt.builtin.whisper.client.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            with pytest.raises(TimeoutError, match="did not respond"):
                client.transcribe(audio)

    def test_transcribe_server_error(self) -> None:
        """transcribe() raises RuntimeError on HTTP 500."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float32)

        mock_response = self._make_mock_response(
            {"error": "model not loaded"}, status_code=500
        )

        with patch("spindl.stt.builtin.whisper.client.requests.post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="model not loaded"):
                client.transcribe(audio)

    def test_transcribe_uses_custom_inference_path(self) -> None:
        """transcribe() uses configured inference path."""
        client = WhisperSTT(inference_path="/transcribe")
        audio = np.zeros(16000, dtype=np.float32)

        mock_response = self._make_mock_response({"text": "test"})

        with patch("spindl.stt.builtin.whisper.client.requests.post", return_value=mock_response) as mock_post:
            client.transcribe(audio)
            assert mock_post.call_args.args[0] == "http://127.0.0.1:8080/transcribe"


class TestWhisperSTTAudioConversion:
    """Tests for WhisperSTT._audio_to_wav_bytes()."""

    def test_wav_bytes_valid_header(self) -> None:
        """Output is valid WAV with correct parameters."""
        client = WhisperSTT()
        audio = np.zeros(16000, dtype=np.float32)

        wav_bytes = client._audio_to_wav_bytes(audio, 16000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 16000

    def test_wav_bytes_correct_length(self) -> None:
        """Number of frames matches input array length."""
        client = WhisperSTT()
        audio = np.zeros(8000, dtype=np.float32)

        wav_bytes = client._audio_to_wav_bytes(audio, 16000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnframes() == 8000

    def test_wav_bytes_clipping(self) -> None:
        """Values beyond [-1.0, 1.0] are clipped to int16 range."""
        client = WhisperSTT()
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)

        wav_bytes = client._audio_to_wav_bytes(audio, 16000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = np.frombuffer(wf.readframes(3), dtype=np.int16)
            assert frames[0] == 32767   # clipped max
            assert frames[1] == -32768  # clipped min
            assert frames[2] == 16383   # 0.5 * 32767 ≈ 16383


class TestWhisperSTTHealthCheck:
    """Tests for WhisperSTT.is_server_available()."""

    def test_healthy_server(self) -> None:
        """Returns True for healthy server."""
        client = WhisperSTT()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        with patch("spindl.stt.builtin.whisper.client.requests.get", return_value=mock_response):
            assert client.is_server_available() is True

    def test_loading_server(self) -> None:
        """Returns False when server is still loading model."""
        client = WhisperSTT()

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.return_value = {"status": "loading model"}

        with patch("spindl.stt.builtin.whisper.client.requests.get", return_value=mock_response):
            assert client.is_server_available() is False

    def test_unreachable_server(self) -> None:
        """Returns False when server is unreachable."""
        client = WhisperSTT()

        with patch(
            "spindl.stt.builtin.whisper.client.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            assert client.is_server_available() is False

    def test_timeout(self) -> None:
        """Returns False on connection timeout."""
        client = WhisperSTT()

        with patch(
            "spindl.stt.builtin.whisper.client.requests.get",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            assert client.is_server_available() is False
