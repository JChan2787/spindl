"""
VLM Provider Base Classes - Abstract base for all Vision Language Model implementations.

This module defines the protocol that all VLM providers must implement,
enabling swappable VLM backends via configuration. VLM providers handle
image-to-text description, used by VisionProvider to inject visual context
into the LLM prompt.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMProperties:
    """
    Provider self-describes its capabilities.

    Used by the orchestrator to understand provider features
    without hardcoding assumptions about any specific provider.
    """

    name: str
    """Name/ID of the model being used (e.g., "X-Ray_Alpha", "gpt-4o")."""

    is_local: bool
    """True if this is a local server provider, False for cloud APIs."""

    supports_streaming: bool
    """True if describe_stream() yields multiple chunks."""

    max_image_size: Optional[int] = None
    """Maximum image dimension in pixels, if known."""


@dataclass
class VLMResponse:
    """
    Response from a VLM description request.

    Contains both the generated description and usage statistics.
    """

    description: str
    """The generated text description of the image."""

    input_tokens: int
    """Number of tokens in the prompt (including image tokens)."""

    output_tokens: int
    """Number of tokens in the response."""

    latency_ms: float
    """Time taken for the request in milliseconds."""


class VLMProvider(ABC):
    """
    Protocol all VLM plugins must implement.

    This abstract base class defines the contract between the VisionProvider
    and VLM backends. Providers can be server-based (like llama-server with
    multimodal models) or cloud APIs (like OpenAI GPT-4o).

    Lifecycle:
        1. Instantiation (no heavy work here)
        2. initialize(config) - establish connections, set up auth
        3. get_properties() - orchestrator queries capabilities
        4. describe() - actual VLM work (image to text)
        5. shutdown() - cleanup resources
    """

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Called once at startup with provider-specific config.

        For server-based providers: establish connection, verify server ready.
        For cloud APIs: validate API key, set up headers.

        Args:
            config: Provider-specific configuration dict from spindl.yaml
                   (the section under vision.providers.<provider_name>)

        Raises:
            ConnectionError: For server-based providers that can't connect
            ValueError: For invalid configuration (missing API key, etc.)
            RuntimeError: For initialization failures
        """
        pass

    @abstractmethod
    def get_properties(self) -> VLMProperties:
        """
        Return provider capabilities.

        Called after initialize(). The orchestrator uses this to understand
        provider features without hardcoding assumptions.

        Returns:
            VLMProperties describing this provider's capabilities
        """
        pass

    @abstractmethod
    def describe(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        max_tokens: int = 300,
        **kwargs,
    ) -> VLMResponse:
        """
        Generate a text description of an image. Blocks until complete.

        Args:
            image_base64: Base64-encoded JPEG image data
            prompt: Optional custom prompt. If None, use provider's default.
                   Example: "Describe what you see in this image."
            max_tokens: Maximum response length
            **kwargs: Provider-specific options (temperature, etc.)

        Returns:
            VLMResponse containing description and usage stats

        Raises:
            RuntimeError: Description generation failed
            ConnectionError: Server/API unreachable
            TimeoutError: Request timed out
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if provider is operational.

        For server-based: verify TCP/HTTP connection
        For cloud APIs: verify API is reachable (lightweight check)

        Returns:
            True if ready to describe, False otherwise
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Cleanup resources.

        For server-based: close connections (server process managed by launcher)
        For cloud APIs: cleanup any persistent connections
        """
        pass

    @classmethod
    @abstractmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate provider-specific config before instantiation.

        Called by the launcher/orchestrator to catch config errors early,
        before attempting to initialize providers.

        Args:
            config: Provider-specific configuration dict

        Returns:
            List of error messages. Empty list = valid config.
        """
        pass

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        Return command to start provider's server, if applicable.

        Launcher calls this to start external server processes.
        Cloud API providers return None.

        Args:
            config: Provider-specific config section

        Returns:
            Shell command string, or None if no server needed
        """
        return None

    @classmethod
    def get_health_url(cls, config: dict) -> Optional[str]:
        """
        Return health check URL for server-based providers.

        Launcher uses this for health polling during startup.
        Cloud API providers return None.

        Args:
            config: Provider-specific config section

        Returns:
            Health check URL string, or None if not applicable
        """
        return None

    @classmethod
    def is_cloud_provider(cls) -> bool:
        """
        Return True if this is a cloud/API provider that needs no local server.

        Used by the launcher to decide whether to skip service startup entirely.
        Cloud providers (OpenAI, Together.ai) return True.
        Local providers (llama-server) return False.

        Default is False (assumes local server needed).
        """
        return False
