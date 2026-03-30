"""
Kokoro TTS Provider - TTSProvider implementation for Kokoro-82M.

Wraps the KokoroTTS TCP client with the standardized TTSProvider interface,
enabling swappable TTS backends via configuration.
"""

import logging
from pathlib import Path
from typing import Iterator, Optional

from ...base import AudioResult, TTSProperties, TTSProvider
from .client import KokoroTTS
from spindl.utils.paths import resolve_relative_path

logger = logging.getLogger(__name__)


class KokoroTTSProvider(TTSProvider):
    """
    TTSProvider implementation for Kokoro-82M TTS server.

    Kokoro is a server-based provider: the model runs in a separate process
    (managed by the launcher) and this provider communicates via TCP.

    Properties:
        - Sample rate: 24000 Hz
        - Format: pcm_f32le (32-bit float, little-endian)
        - Channels: 1 (mono)
        - Streaming: No (yields single complete result)

    Config schema (tts.providers.kokoro):
        host: str       - Server hostname (default: "127.0.0.1")
        port: int       - Server port (default: 5556)
        voice: str      - Default voice ID (default: "af_bella")
        language: str   - Language code 'a' or 'b' (default: "a")
        models_dir: str - Path to models directory (for voice listing)
        device: str     - PyTorch device for model (default: "cuda")
        timeout: float  - Socket timeout in seconds (default: 30.0)
    """

    # Kokoro's fixed output properties
    SAMPLE_RATE = 24000
    AUDIO_FORMAT = "pcm_f32le"
    CHANNELS = 1

    def __init__(self):
        """Initialize provider (no heavy work - that's in initialize())."""
        self._client: Optional[KokoroTTS] = None
        self._config: dict = {}
        self._default_voice: str = "af_bella"
        self._default_language: str = "a"
        self._models_dir: Optional[Path] = None
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        """
        Establish connection to Kokoro TTS server.

        Args:
            config: Provider config from tts.providers.kokoro section

        Raises:
            ConnectionError: Server not reachable
            RuntimeError: Initialization failed
        """
        self._config = config

        # Extract config values with defaults
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 5556)
        timeout = config.get("timeout", 30.0)

        self._default_voice = config.get("voice", "af_bella")
        self._default_language = config.get("language", "a")

        # Models directory for voice listing (resolve relative paths)
        models_dir = config.get("models_dir")
        if models_dir:
            self._models_dir = Path(resolve_relative_path(models_dir))

        # Create client
        self._client = KokoroTTS(
            host=host,
            port=port,
            timeout=timeout,
        )

        # Verify connection
        if not self._client.is_server_available():
            raise ConnectionError(
                f"Kokoro TTS server not available at {host}:{port}. "
                "Ensure the server is running (launcher should start it)."
            )

        self._initialized = True
        logger.info(f"KokoroTTSProvider initialized: {host}:{port}")

    def get_properties(self) -> TTSProperties:
        """Return Kokoro's output format properties."""
        return TTSProperties(
            sample_rate=self.SAMPLE_RATE,
            audio_format=self.AUDIO_FORMAT,
            channels=self.CHANNELS,
            supports_streaming=False,
        )

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
        """
        Synthesize text to audio via Kokoro server.

        Args:
            text: Text to synthesize
            voice: Voice ID (falls back to provider default if not found)
            **kwargs: Additional options
                - language: 'a' (American) or 'b' (British)

        Returns:
            AudioResult with complete synthesized audio

        Raises:
            RuntimeError: Synthesis failed or provider not initialized
            ConnectionError: Server connection lost
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("KokoroTTSProvider not initialized. Call initialize() first.")

        # Resolve voice with fallback
        resolved_voice = self._resolve_voice(voice, self._default_voice)

        # Get language from kwargs or use default
        language = kwargs.get("language", self._default_language)

        # Delegate to client
        audio_array = self._client.synthesize(
            text=text,
            voice=resolved_voice,
            lang=language,
        )

        # Wrap in AudioResult
        return AudioResult(
            data=audio_array.tobytes(),
            sample_rate=self.SAMPLE_RATE,
            format=self.AUDIO_FORMAT,
            is_final=True,
        )

    def synthesize_stream(self, text: str, voice: Optional[str] = None, **kwargs) -> Iterator[AudioResult]:
        """
        Kokoro doesn't support streaming - yields single complete result.

        Args:
            text: Text to synthesize
            voice: Voice ID
            **kwargs: Additional options

        Yields:
            Single AudioResult with complete audio
        """
        yield self.synthesize(text, voice, **kwargs)

    def list_voices(self) -> list[str]:
        """
        List available Kokoro voices by scanning the voices directory.

        Returns:
            List of voice IDs (without .pt extension)
        """
        if self._models_dir is None:
            # Fallback: return common Kokoro voices
            logger.warning(
                "models_dir not configured for Kokoro provider. "
                "Returning default voice list."
            )
            return [
                "af_bella", "af_nicole", "af_sarah", "af_sky",
                "am_adam", "am_michael",
                "bf_emma", "bf_isabella",
                "bm_george", "bm_lewis",
            ]

        voices_dir = self._models_dir / "voices"
        if not voices_dir.exists():
            logger.warning(f"Voices directory not found: {voices_dir}")
            return [self._default_voice]

        # Scan for .pt files
        voices = []
        for voice_file in voices_dir.glob("*.pt"):
            voice_id = voice_file.stem  # filename without extension
            voices.append(voice_id)

        return sorted(voices)

    def health_check(self) -> bool:
        """
        Check if Kokoro server is reachable.

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
        We just clean up the client connection state.
        """
        self._client = None
        self._initialized = False
        logger.debug("KokoroTTSProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate Kokoro provider config.

        Required fields: host, port
        Optional fields: conda_env, voice, language, models_dir, timeout, server_script

        Args:
            config: Provider config dict

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Host validation
        host = config.get("host")
        if host is None:
            errors.append("Missing required field: host")
        elif not isinstance(host, str):
            errors.append(f"host must be a string, got {type(host).__name__}")

        # Conda environment validation (optional - if provided, must be string)
        conda_env = config.get("conda_env")
        if conda_env is not None and not isinstance(conda_env, str):
            errors.append(f"conda_env must be a string, got {type(conda_env).__name__}")

        # Port validation
        port = config.get("port")
        if port is None:
            errors.append("Missing required field: port")
        elif not isinstance(port, int):
            errors.append(f"port must be an integer, got {type(port).__name__}")
        elif not (1 <= port <= 65535):
            errors.append(f"port must be between 1 and 65535, got {port}")

        # Language validation (optional but must be valid if provided)
        language = config.get("language")
        if language is not None and language not in ("a", "b"):
            errors.append(f"language must be 'a' or 'b', got '{language}'")

        # Timeout validation (optional but must be positive if provided)
        timeout = config.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                errors.append(f"timeout must be a number, got {type(timeout).__name__}")
            elif timeout <= 0:
                errors.append(f"timeout must be positive, got {timeout}")

        # Models dir validation (optional but must exist if provided)
        models_dir = config.get("models_dir")
        if models_dir is not None:
            models_path = Path(resolve_relative_path(models_dir))
            if not models_path.exists():
                errors.append(f"models_dir does not exist: {models_path}")

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        Return command to start Kokoro TTS server.

        Builds command from provider config values:
        - conda_env: Conda environment name (optional - if provided, wraps with conda run)
        - port: Server port (default: 5556)
        - models_dir: Path to models directory (default: ./tts/models)
        - server_script: Path to server script (default: builtin location)

        Args:
            config: Provider config from tts.providers.kokoro section

        Returns:
            Shell command string to start the server
        """
        # Optional: conda environment (if omitted, runs directly)
        conda_env = config.get("conda_env")

        # Optional with defaults (resolve relative paths against project root)
        port = config.get("port", 5556)
        models_dir = resolve_relative_path(config.get("models_dir", "tts/models"))
        server_script = config.get("server_script")
        if server_script is None:
            server_script = str(Path(__file__).resolve().parent / "server.py")
        else:
            server_script = resolve_relative_path(server_script)

        # Quote paths that may contain spaces (e.g. "VS CODE Projects")
        script_q = f'"{server_script}"' if " " in server_script else server_script
        models_q = f'"{models_dir}"' if " " in models_dir else models_dir

        # Device selection (default: cuda — backward compatible)
        device = config.get("device", "cuda")

        # Build command - with or without conda wrapper
        if conda_env:
            command = (
                f"conda run -n {conda_env} --no-capture-output "
                f"python {script_q} "
                f"--port {port} "
                f"--models-dir {models_q} "
                f"--device {device}"
            )
        else:
            command = (
                f"python {script_q} "
                f"--port {port} "
                f"--models-dir {models_q} "
                f"--device {device}"
            )

        return command
