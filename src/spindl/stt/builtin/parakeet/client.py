"""
Parakeet STT client - Windows-side TCP client for WSL NeMo server.

Communicates with nemo_server.py running in WSL over localhost TCP.
"""

import json
import socket
import time
from typing import Optional

import numpy as np


class ParakeetSTT:
    """
    Speech-to-text client that communicates with NeMo server in WSL.

    Uses TCP socket connection to send audio and receive transcriptions.
    Includes automatic retry logic for handling server cold starts.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the STT client.

        Args:
            host: Server hostname (default: localhost)
            port: Server port (default: 5555)
            timeout: Socket timeout in seconds (default: 30s for long utterances)
            max_retries: Number of connection attempts before failing
            retry_delay: Seconds to wait between retry attempts
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Send audio to WSL server for transcription.

        Args:
            audio: float32 numpy array (mono, normalized -1.0 to 1.0)
            sample_rate: Sample rate in Hz (must be 16000)

        Returns:
            Transcribed text string

        Raises:
            ConnectionError: Server unreachable after all retries
            TimeoutError: Server didn't respond in time
            RuntimeError: Server returned an error
            ValueError: Invalid audio format
        """
        # Validate input
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16000, got {sample_rate}")

        # Encode audio as hex
        audio_hex = audio.tobytes().hex()

        # Build request
        request = {
            "action": "transcribe",
            "audio": audio_hex,
            "sample_rate": sample_rate,
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
                f"Failed to connect to STT server at {self.host}:{self.port} "
                f"after {self.max_retries} attempts: {last_error}"
            )

        # Parse response
        if response.get("status") == "error":
            raise RuntimeError(f"STT server error: {response.get('error')}")

        return response.get("text", "")

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

            # Read response until newline
            buffer = b""
            while True:
                try:
                    chunk = sock.recv(65536)
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
        Check if the WSL server is reachable.

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
