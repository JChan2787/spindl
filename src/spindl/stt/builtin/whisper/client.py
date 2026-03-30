"""
Whisper.cpp STT client - HTTP client for whisper.cpp server.

Communicates with whisper-server via HTTP REST API.
Endpoint: POST /inference (multipart/form-data with WAV audio).
Health: GET /health.
"""

import io
import logging
import wave
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class WhisperSTT:
    """
    Speech-to-text client that communicates with whisper.cpp HTTP server.

    Uses HTTP POST /inference for transcription and GET /health for status.
    Audio is converted from float32 numpy arrays to 16-bit PCM WAV in-memory
    before uploading as multipart form data.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        timeout: float = 30.0,
        language: str = "en",
        response_format: str = "json",
        inference_path: str = "/inference",
    ):
        """
        Initialize the whisper.cpp HTTP client.

        Args:
            host: Server hostname (default: localhost)
            port: Server port (default: 8080, whisper.cpp default)
            timeout: HTTP request timeout in seconds (default: 30s)
            language: Language code for transcription (default: "en")
            response_format: Response format - json, verbose_json, text (default: "json")
            inference_path: Server endpoint path (default: "/inference")
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.language = language
        self.response_format = response_format
        self.inference_path = inference_path
        self._base_url = f"http://{host}:{port}"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Send audio to whisper.cpp server for transcription.

        Args:
            audio: float32 numpy array (mono, normalized -1.0 to 1.0)
            sample_rate: Sample rate in Hz (must be 16000)

        Returns:
            Transcribed text string

        Raises:
            ConnectionError: Server unreachable
            TimeoutError: Server didn't respond in time
            RuntimeError: Server returned an error
            ValueError: Invalid audio format
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16000, got {sample_rate}")

        wav_bytes = self._audio_to_wav_bytes(audio, sample_rate)

        url = f"{self._base_url}{self.inference_path}"
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "response_format": self.response_format,
            "language": self.language,
            "temperature": "0.0",
        }

        try:
            response = requests.post(
                url, files=files, data=data, timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to whisper.cpp server at {self.host}:{self.port}: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"whisper.cpp server did not respond within {self.timeout}s"
            ) from e

        if response.status_code >= 500:
            try:
                error_msg = response.json().get("error", "Unknown server error")
            except Exception:
                error_msg = response.text or f"HTTP {response.status_code}"
            raise RuntimeError(f"whisper.cpp server error: {error_msg}")

        result = response.json()
        return result.get("text", "").strip()

    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert float32 numpy audio to 16-bit PCM WAV bytes.

        Args:
            audio: float32 array normalized to [-1.0, 1.0]
            sample_rate: Sample rate in Hz

        Returns:
            WAV file bytes (header + PCM data)
        """
        pcm_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16.tobytes())

        return buffer.getvalue()

    def is_server_available(self) -> bool:
        """
        Check if the whisper.cpp server is reachable and ready.

        Returns:
            True if server returns healthy status, False otherwise
        """
        try:
            response = requests.get(
                f"{self._base_url}/health", timeout=2.0
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok"
            return False
        except (requests.exceptions.RequestException, ValueError):
            return False
