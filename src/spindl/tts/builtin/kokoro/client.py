"""
Kokoro TTS client - TCP client for Kokoro TTS server.

Communicates with kokoro_server.py running on Windows over localhost TCP.
"""

import json
import socket
import time
from typing import Optional

import numpy as np


class KokoroTTS:
    """
    Text-to-speech client that communicates with Kokoro TTS server.

    Uses TCP socket connection to send text and receive synthesized audio.
    Includes automatic retry logic for handling server restarts.
    """

    # Server's output sample rate
    SAMPLE_RATE = 24000

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the TTS client.

        Args:
            host: Server hostname (default: localhost)
            port: Server port (default: 5556)
            timeout: Socket timeout in seconds (default: 30s for long text)
            max_retries: Number of connection attempts before failing
            retry_delay: Seconds to wait between retry attempts
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def synthesize(
        self,
        text: str,
        voice: str = "af_bella",
        lang: str = "a",
    ) -> np.ndarray:
        """
        Send text to server for speech synthesis.

        Args:
            text: Text to synthesize
            voice: Voice ID (default: af_bella)
            lang: Language code - 'a' for American, 'b' for British (default: a)

        Returns:
            float32 numpy array of audio samples at 24000 Hz

        Raises:
            ConnectionError: Server unreachable after all retries
            TimeoutError: Server didn't respond in time
            RuntimeError: Server returned an error
            ValueError: Invalid parameters
        """
        # Validate input
        if not text:
            raise ValueError("Text cannot be empty")

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if lang not in ("a", "b"):
            raise ValueError(f"Language must be 'a' or 'b', got '{lang}'")

        # Build request
        request = {
            "action": "synthesize",
            "text": text,
            "voice": voice,
            "lang": lang,
        }
        request_json = json.dumps(request) + "\n"

        # Try to send with retries
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self._send_request(request_json)
                break
            except (ConnectionError, socket.error) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
        else:
            raise ConnectionError(
                f"Failed to connect to TTS server at {self.host}:{self.port} "
                f"after {self.max_retries} attempts: {last_error}"
            )

        # Parse response
        if response.get("status") == "error":
            raise RuntimeError(f"TTS server error: {response.get('error')}")

        # Decode audio from hex
        audio_hex = response.get("audio", "")
        if not audio_hex:
            return np.array([], dtype=np.float32)

        audio_bytes = bytes.fromhex(audio_hex)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        return audio

    def _send_request(self, request_json: str) -> dict:
        """
        Send a request to the server and return the response.

        Args:
            request_json: JSON string with trailing newline

        Returns:
            Parsed response dict

        Raises:
            ConnectionError: Cannot connect
            TimeoutError: Read timeout
            ValueError: Invalid response
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect((self.host, self.port))
            sock.sendall(request_json.encode("utf-8"))

            # Read response until newline (may be large due to audio data)
            buffer = b""
            while True:
                try:
                    chunk = sock.recv(1048576)  # 1MB chunks for audio data
                except socket.timeout:
                    raise TimeoutError(
                        f"Server did not respond within {self.timeout}s"
                    )

                if not chunk:
                    break
                buffer += chunk
                if b"\n" in buffer:
                    break

            if not buffer:
                raise ValueError("Server returned empty response")

            response_str = buffer.decode("utf-8").strip()
            return json.loads(response_str)

        finally:
            sock.close()

    def is_server_available(self) -> bool:
        """
        Check if the TTS server is reachable.

        Returns:
            True if server accepts connections, False otherwise
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)  # Short timeout for availability check

        try:
            sock.connect((self.host, self.port))
            sock.close()
            return True
        except (ConnectionError, socket.error, socket.timeout):
            return False

    def get_audio_duration(self, audio: np.ndarray) -> float:
        """
        Calculate duration of audio in seconds.

        Args:
            audio: Audio array from synthesize()

        Returns:
            Duration in seconds
        """
        return len(audio) / self.SAMPLE_RATE
