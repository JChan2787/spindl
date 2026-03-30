"""
STT Provider Base Classes - Abstract base for all STT implementations.

This module defines the protocol that all STT providers must implement,
enabling swappable STT backends via configuration.

Mirrors the TTS provider architecture (NANO-015) for consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class STTProperties:
    """
    Provider self-describes its capabilities.

    Used by the orchestrator to understand provider requirements
    without hardcoding assumptions about any specific provider.
    """
    sample_rate: int
    """Expected input sample rate in Hz (e.g., 16000 for Parakeet/Whisper)."""

    audio_format: str
    """Expected audio format string (e.g., "pcm_f32le", "pcm_s16le")."""

    supports_streaming: bool
    """True if provider supports partial transcription during speech."""


class STTProvider(ABC):
    """
    Protocol all STT plugins must implement.

    This abstract base class defines the contract between the orchestrator
    and STT backends. Providers can be server-based (like Parakeet over TCP,
    or Whisper.cpp over HTTP) or in-process.

    Lifecycle:
        1. Instantiation (no heavy work here)
        2. initialize(config) - establish connections, load resources
        3. get_properties() - orchestrator queries capabilities
        4. transcribe() - actual STT work
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
                   (the section under stt.providers.<provider_name>)

        Raises:
            ConnectionError: For server-based providers that can't connect
            RuntimeError: For initialization failures
        """
        pass

    @abstractmethod
    def get_properties(self) -> STTProperties:
        """
        Return provider capabilities.

        Called after initialize(). The orchestrator uses this to understand
        expected input format (sample rate, audio format) without
        hardcoding assumptions.

        Returns:
            STTProperties describing this provider's requirements
        """
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Convert audio to text. Blocks until complete.

        Args:
            audio: float32 numpy array (mono, normalized -1.0 to 1.0)
            sample_rate: Sample rate of the audio in Hz

        Returns:
            Transcribed text string

        Raises:
            RuntimeError: Transcription failed
            ConnectionError: Server-based provider lost connection
            ValueError: Invalid audio format
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if provider is operational.

        For server-based: verify TCP/HTTP connection
        For in-process: verify model loaded

        Returns:
            True if ready to transcribe, False otherwise
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
