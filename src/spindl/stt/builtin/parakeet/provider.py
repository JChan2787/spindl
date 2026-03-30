"""
Parakeet STT Provider - STTProvider implementation for NVIDIA NeMo Parakeet.

Wraps the ParakeetSTT TCP client with the standardized STTProvider interface,
enabling swappable STT backends via configuration.
"""

import logging
from typing import Optional

import numpy as np

from ...base import STTProperties, STTProvider
from .client import ParakeetSTT
from spindl.utils.paths import resolve_relative_path

logger = logging.getLogger(__name__)


class ParakeetSTTProvider(STTProvider):
    """
    STTProvider implementation for NVIDIA NeMo Parakeet server.

    Parakeet is a server-based provider: the model runs in a separate process
    (WSL with CUDA, managed by the launcher) and this provider communicates
    via TCP using JSON-over-TCP protocol.

    Properties:
        - Sample rate: 16000 Hz (required)
        - Format: pcm_f32le (32-bit float, little-endian)
        - Streaming: No (full-utterance transcription)

    Config schema (stt.providers.parakeet):
        host: str       - Server hostname (default: "127.0.0.1")
        port: int       - Server port (default: 5555)
        timeout: float  - Socket timeout in seconds (default: 30.0)
        max_retries: int - Connection retry attempts (default: 3)
        retry_delay: float - Seconds between retries (default: 1.0)
    """

    SAMPLE_RATE = 16000
    AUDIO_FORMAT = "pcm_f32le"

    def __init__(self):
        """Initialize provider (no heavy work - that's in initialize())."""
        self._client: Optional[ParakeetSTT] = None
        self._config: dict = {}
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        """
        Establish connection to Parakeet STT server.

        Args:
            config: Provider config from stt.providers.parakeet section

        Raises:
            ConnectionError: Server not reachable
        """
        self._config = config

        host = config.get("host", "127.0.0.1")
        port = config.get("port", 5555)
        timeout = config.get("timeout", 30.0)
        max_retries = config.get("max_retries", 3)
        retry_delay = config.get("retry_delay", 1.0)

        self._client = ParakeetSTT(
            host=host,
            port=port,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        if not self._client.is_server_available():
            raise ConnectionError(
                f"Parakeet STT server not available at {host}:{port}. "
                "Ensure the NeMo server is running in WSL (launcher should start it)."
            )

        self._initialized = True
        logger.info(f"ParakeetSTTProvider initialized: {host}:{port}")

    def get_properties(self) -> STTProperties:
        """Return Parakeet's input format requirements."""
        return STTProperties(
            sample_rate=self.SAMPLE_RATE,
            audio_format=self.AUDIO_FORMAT,
            supports_streaming=False,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio via Parakeet server.

        Args:
            audio: float32 numpy array (mono, normalized -1.0 to 1.0)
            sample_rate: Sample rate in Hz (must be 16000)

        Returns:
            Transcribed text string

        Raises:
            RuntimeError: Provider not initialized or server error
            ConnectionError: Server connection lost
            ValueError: Invalid audio format
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("ParakeetSTTProvider not initialized. Call initialize() first.")

        return self._client.transcribe(audio, sample_rate)

    def health_check(self) -> bool:
        """
        Check if Parakeet server is reachable.

        Returns:
            True if server accepts connections, False otherwise
        """
        if self._client is None:
            return False
        return self._client.is_server_available()

    def shutdown(self) -> None:
        """
        Cleanup provider resources.

        Note: Server process is managed by launcher, not this provider.
        We just clean up the client reference.
        """
        self._client = None
        self._initialized = False
        logger.debug("ParakeetSTTProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate Parakeet provider config.

        Required fields: host, port
        Optional fields: timeout, max_retries, retry_delay

        Args:
            config: Provider config dict

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        host = config.get("host")
        if host is None:
            errors.append("Missing required field: host")
        elif not isinstance(host, str):
            errors.append(f"host must be a string, got {type(host).__name__}")

        port = config.get("port")
        if port is None:
            errors.append("Missing required field: port")
        elif not isinstance(port, int):
            errors.append(f"port must be an integer, got {type(port).__name__}")
        elif not (1 <= port <= 65535):
            errors.append(f"port must be between 1 and 65535, got {port}")

        timeout = config.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                errors.append(f"timeout must be a number, got {type(timeout).__name__}")
            elif timeout <= 0:
                errors.append(f"timeout must be positive, got {timeout}")

        max_retries = config.get("max_retries")
        if max_retries is not None:
            if not isinstance(max_retries, int):
                errors.append(f"max_retries must be an integer, got {type(max_retries).__name__}")
            elif max_retries < 0:
                errors.append(f"max_retries must be non-negative, got {max_retries}")

        retry_delay = config.get("retry_delay")
        if retry_delay is not None:
            if not isinstance(retry_delay, (int, float)):
                errors.append(f"retry_delay must be a number, got {type(retry_delay).__name__}")
            elif retry_delay < 0:
                errors.append(f"retry_delay must be non-negative, got {retry_delay}")

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        Return command to start Parakeet NeMo server in WSL.

        Args:
            config: Provider config from stt.providers.parakeet section

        Returns:
            Shell command string to start the NeMo server
        """
        port = config.get("port", 5555)
        # Resolve server_script against project root if relative
        server_script = config.get("server_script")
        if server_script is None:
            server_script = resolve_relative_path("stt/server/nemo_server.py")
        else:
            server_script = resolve_relative_path(server_script)
        conda_env = config.get("conda_env")

        # Quote path if it contains spaces
        script_q = f'"{server_script}"' if " " in server_script else server_script

        if conda_env:
            command = (
                f"conda run -n {conda_env} --no-capture-output "
                f"python {script_q} "
                f"--port {port}"
            )
        else:
            command = (
                f"python {script_q} "
                f"--port {port}"
            )

        return command
