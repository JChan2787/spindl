"""
Whisper.cpp STT Provider - STTProvider implementation for whisper.cpp server.

Wraps the WhisperSTT HTTP client with the standardized STTProvider interface,
enabling swappable STT backends via configuration.
"""

import logging
from typing import Optional

import numpy as np

from ...base import STTProperties, STTProvider
from .client import WhisperSTT

logger = logging.getLogger(__name__)


class WhisperSTTProvider(STTProvider):
    """
    STTProvider implementation for whisper.cpp HTTP server.

    whisper.cpp is a server-based provider: the model runs in a separate process
    (native binary on any platform) and this provider communicates via HTTP
    using multipart form uploads to POST /inference.

    Properties:
        - Sample rate: 16000 Hz (required)
        - Format: pcm_s16le (16-bit signed int, converted internally from float32)
        - Streaming: No (full-utterance transcription)

    Config schema (stt.providers.whisper):
        host: str            - Server hostname (default: "127.0.0.1")
        port: int            - Server port (default: 8080)
        timeout: float       - HTTP timeout in seconds (default: 30.0)
        language: str        - Language code (default: "en")
        response_format: str - json/verbose_json/text (default: "json")
        inference_path: str  - Custom endpoint path (default: "/inference")
        model_path: str      - Path to GGML model file (for server command)
        binary_path: str     - Server executable (default: "whisper-server")
        threads: int         - Server thread count (default: 4)
        no_gpu: bool         - Disable GPU acceleration (default: false)
    """

    SAMPLE_RATE = 16000
    AUDIO_FORMAT = "pcm_s16le"

    def __init__(self):
        """Initialize provider (no heavy work - that's in initialize())."""
        self._client: Optional[WhisperSTT] = None
        self._config: dict = {}
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        """
        Establish connection to whisper.cpp HTTP server.

        Args:
            config: Provider config from stt.providers.whisper section

        Raises:
            ConnectionError: Server not reachable
        """
        self._config = config

        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8080)
        timeout = config.get("timeout", 30.0)
        language = config.get("language", "en")
        response_format = config.get("response_format", "json")
        inference_path = config.get("inference_path", "/inference")

        self._client = WhisperSTT(
            host=host,
            port=port,
            timeout=timeout,
            language=language,
            response_format=response_format,
            inference_path=inference_path,
        )

        if not self._client.is_server_available():
            raise ConnectionError(
                f"whisper.cpp server not available at {host}:{port}. "
                "Ensure whisper-server is running (launcher should start it)."
            )

        self._initialized = True
        logger.info(f"WhisperSTTProvider initialized: {host}:{port}")

    def get_properties(self) -> STTProperties:
        """Return whisper.cpp input format requirements."""
        return STTProperties(
            sample_rate=self.SAMPLE_RATE,
            audio_format=self.AUDIO_FORMAT,
            supports_streaming=False,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio via whisper.cpp server.

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
            raise RuntimeError("WhisperSTTProvider not initialized. Call initialize() first.")

        return self._client.transcribe(audio, sample_rate)

    def health_check(self) -> bool:
        """
        Check if whisper.cpp server is reachable and healthy.

        Returns:
            True if server returns ok status, False otherwise
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
        logger.debug("WhisperSTTProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate whisper.cpp provider config.

        Required fields: host, port
        Optional fields: timeout, language, response_format, inference_path,
                         model_path, binary_path, threads, no_gpu

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

        language = config.get("language")
        if language is not None:
            if not isinstance(language, str):
                errors.append(f"language must be a string, got {type(language).__name__}")

        response_format = config.get("response_format")
        if response_format is not None:
            valid_formats = ("json", "verbose_json", "text")
            if not isinstance(response_format, str):
                errors.append(f"response_format must be a string, got {type(response_format).__name__}")
            elif response_format not in valid_formats:
                errors.append(
                    f"response_format must be one of {valid_formats}, got '{response_format}'"
                )

        inference_path = config.get("inference_path")
        if inference_path is not None:
            if not isinstance(inference_path, str):
                errors.append(f"inference_path must be a string, got {type(inference_path).__name__}")
            elif not inference_path.startswith("/"):
                errors.append(f"inference_path must start with '/', got '{inference_path}'")

        model_path = config.get("model_path")
        if model_path is not None:
            if not isinstance(model_path, str):
                errors.append(f"model_path must be a string, got {type(model_path).__name__}")

        binary_path = config.get("binary_path")
        if binary_path is not None:
            if not isinstance(binary_path, str):
                errors.append(f"binary_path must be a string, got {type(binary_path).__name__}")

        threads = config.get("threads")
        if threads is not None:
            if not isinstance(threads, int):
                errors.append(f"threads must be an integer, got {type(threads).__name__}")
            elif threads <= 0:
                errors.append(f"threads must be positive, got {threads}")

        no_gpu = config.get("no_gpu")
        if no_gpu is not None:
            if not isinstance(no_gpu, bool):
                errors.append(f"no_gpu must be a boolean, got {type(no_gpu).__name__}")

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        Return command to start whisper.cpp server.

        Args:
            config: Provider config from stt.providers.whisper section

        Returns:
            Shell command string, or None if model_path not configured
        """
        model_path = config.get("model_path")
        if not model_path:
            return None

        binary_path = config.get("binary_path", "whisper-server")
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8080)
        language = config.get("language", "en")
        threads = config.get("threads", 4)
        no_gpu = config.get("no_gpu", False)

        command = (
            f"{binary_path} "
            f"-m {model_path} "
            f"--host {host} "
            f"--port {port} "
            f"-l {language} "
            f"-t {threads}"
        )

        if no_gpu:
            command += " --no-gpu"

        return command
