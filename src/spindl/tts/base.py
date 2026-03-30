"""
TTS Provider Base Classes - Abstract base for all TTS implementations.

This module defines the protocol that all TTS providers must implement,
enabling swappable TTS backends via configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class TTSProperties:
    """
    Provider self-describes its capabilities.

    Used by the orchestrator to configure the audio output pipeline
    without hardcoding assumptions about any specific provider.
    """
    sample_rate: int
    """Output sample rate in Hz (e.g., 24000 for Kokoro, 12000 for Qwen3)."""

    audio_format: str
    """Audio format string (e.g., "pcm_f32le", "pcm_s16le")."""

    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""

    supports_streaming: bool
    """True if synthesize_stream() yields multiple chunks."""


@dataclass
class AudioResult:
    """
    Chunk of synthesized audio returned by TTS providers.

    Contains raw audio data along with metadata about the format.
    For non-streaming providers, a single AudioResult contains the complete audio.
    For streaming providers, multiple AudioResults are yielded with is_final=False
    until the last chunk.
    """
    data: bytes
    """Raw audio bytes in the format specified by `format`."""

    sample_rate: int
    """Sample rate of this audio chunk in Hz."""

    format: str
    """Audio format (e.g., "pcm_f32le", "pcm_s16le")."""

    is_final: bool = True
    """False if more chunks are coming, True for the last/only chunk."""


class TTSProvider(ABC):
    """
    Protocol all TTS plugins must implement.

    This abstract base class defines the contract between the orchestrator
    and TTS backends. Providers can be server-based (like Kokoro) or
    in-process (like future Qwen3 integration).

    Lifecycle:
        1. Instantiation (no heavy work here)
        2. initialize(config) - establish connections, load resources
        3. get_properties() - orchestrator queries capabilities
        4. synthesize() / synthesize_stream() - actual TTS work
        5. shutdown() - cleanup resources

    The launcher manages server processes separately via get_server_command().
    """

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Called once at startup with provider-specific config.

        For server-based providers: establish connection, verify server ready.
        For in-process providers: load model, allocate resources.

        Args:
            config: Provider-specific configuration dict from spindl.yaml
                   (the section under tts.providers.<provider_name>)

        Raises:
            ConnectionError: For server-based providers that can't connect
            RuntimeError: For initialization failures
        """
        pass

    @abstractmethod
    def get_properties(self) -> TTSProperties:
        """
        Return provider capabilities.

        Called after initialize(). The orchestrator uses this to configure
        the audio output pipeline (sample rate, format, etc.) without
        hardcoding assumptions.

        Returns:
            TTSProperties describing this provider's output format
        """
        pass

    @abstractmethod
    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
        """
        Convert text to audio. Blocks until complete.

        Args:
            text: Text to synthesize
            voice: Requested voice ID from persona config.
                   Provider uses its default if None or not found.
            **kwargs: Provider-specific options (language, speed, etc.)

        Returns:
            Complete audio as AudioResult

        Raises:
            RuntimeError: Synthesis failed
            ConnectionError: Server-based provider lost connection
        """
        pass

    def synthesize_stream(self, text: str, voice: Optional[str] = None, **kwargs) -> Iterator[AudioResult]:
        """
        Convert text to audio, yielding chunks as available.

        Default implementation: yield single complete result.
        Streaming providers override to yield multiple chunks with
        is_final=False until the last chunk.

        Args:
            text: Text to synthesize
            voice: Requested voice ID from persona config
            **kwargs: Provider-specific options

        Yields:
            AudioResult chunks (is_final=False until last chunk)
        """
        yield self.synthesize(text, voice, **kwargs)

    @abstractmethod
    def list_voices(self) -> list[str]:
        """
        Return list of available voice IDs.

        Used for:
        - Validation before synthesis
        - Voice fallback logic
        - User-facing voice selection UI (future)

        Returns:
            List of valid voice identifiers for this provider
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if provider is operational.

        For server-based: verify TCP/HTTP connection
        For in-process: verify model loaded

        Returns:
            True if ready to synthesize, False otherwise
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Cleanup resources.

        For server-based: close connections (server process managed by launcher)
        For in-process: unload model, free VRAM
        """
        pass

    @classmethod
    @abstractmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate provider-specific config before instantiation.

        Called by the launcher/orchestrator to catch config errors early,
        before attempting to start servers or initialize providers.

        Args:
            config: Provider-specific configuration dict

        Returns:
            List of error messages. Empty list = valid config.
        """
        pass

    @classmethod
    @abstractmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        Return command to start provider's server, if applicable.

        Launcher calls this to start external server processes.
        In-process providers return None.

        Args:
            config: Provider-specific config section

        Returns:
            Shell command string, or None if no server needed
        """
        pass

    def _resolve_voice(self, requested: Optional[str], default_voice: str) -> str:
        """
        Resolve requested voice to an available voice.

        Helper method for providers to implement graceful voice fallback.

        Args:
            requested: Voice ID from persona config, or None
            default_voice: Provider's default voice to fall back to

        Returns:
            Resolved voice ID (either requested if available, or default)
        """
        if requested is None:
            return default_voice

        available = self.list_voices()
        if requested in available:
            return requested

        # Import here to avoid circular dependency
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f'Voice "{requested}" not found in {self.__class__.__name__}. '
            f'Using default voice "{default_voice}".'
        )
        return default_voice
