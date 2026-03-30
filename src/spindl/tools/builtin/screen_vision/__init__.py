"""
Screen Vision Tool - Capture and describe what's currently on screen.

This tool wraps the VLM infrastructure to provide on-demand screen
analysis when the LLM needs to know what the user is looking at.

This is the ONLY way to invoke vision capabilities (NANO-024).
The LLM calls this tool when it decides visual context would be helpful.
"""

import asyncio
import logging
import threading
import time
from typing import Optional

from ....tools.base import Tool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class ScreenVisionTool(Tool):
    """
    Tool for capturing and describing the current screen.

    Integrates with the VLM infrastructure (ScreenCapture + VLM provider)
    and exposes it as an on-demand tool the LLM can invoke.

    Config (spindl.yaml):
        tools:
          tools:
            screen_vision:
              enabled: true
              vlm_provider: "llama"  # References vlm.providers.<name>
              monitor: 1             # Which monitor to capture
              description_prompt: null  # Custom prompt (optional)
    """

    def __init__(self):
        self._capture = None
        self._vlm_provider = None
        self._description_prompt: Optional[str] = None
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def name(self) -> str:
        return "screen_vision"

    @property
    def description(self) -> str:
        return (
            "Capture a screenshot of the user's current screen and describe what's visible. "
            "Use this when the user asks about what's on their screen, references something "
            "they're looking at, or when visual context would help answer their question."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        # No parameters needed - captures current screen
        return []

    def initialize(self, config: dict) -> None:
        """
        Initialize the screen vision tool.

        Lazily imports vision components to avoid circular dependencies
        and sets up the capture + VLM pipeline.

        Args:
            config: Tool config from spindl.yaml tools.tools.screen_vision section
        """
        # Import here to avoid circular dependencies
        from ....vision.capture import ScreenCapture
        from ....vision.registry import VLMProviderRegistry

        # Get config values with defaults
        monitor = config.get("monitor", 1)
        vlm_provider_name = config.get("vlm_provider", "llama")
        vlm_plugin_paths = config.get("vlm_plugin_paths", [])
        self._description_prompt = config.get("description_prompt")

        # Initialize screen capture
        self._capture = ScreenCapture(monitor=monitor)
        logger.info(f"ScreenVisionTool: Initialized capture for monitor {monitor}")

        # Initialize VLM provider
        vlm_registry = VLMProviderRegistry(plugin_paths=vlm_plugin_paths)

        # Get VLM provider config (might be in parent config or nested)
        vlm_config = config.get("vlm_config", {})

        try:
            vlm_class = vlm_registry.get_provider_class(vlm_provider_name)
            self._vlm_provider = vlm_class()
            self._vlm_provider.initialize(vlm_config)
            logger.info(f"ScreenVisionTool: Initialized VLM provider '{vlm_provider_name}'")
        except Exception as e:
            logger.error(f"ScreenVisionTool: Failed to initialize VLM: {e}")
            raise RuntimeError(f"Failed to initialize VLM provider: {e}")

        self._initialized = True

    def swap_vlm_provider(
        self,
        provider_name: str,
        vlm_config: dict,
        vlm_plugin_paths: list[str],
    ) -> None:
        """Hot-swap the VLM provider at runtime (NANO-065c).

        Shuts down the current provider, instantiates and initializes the new
        one, and assigns it.  In-flight ``execute()`` calls hold the old
        provider reference locally and complete safely (GIL-atomic assignment).

        Args:
            provider_name: Registry name (e.g., "llama", "openai", "llm").
            vlm_config: Provider-specific config dict.
            vlm_plugin_paths: Plugin search paths for external VLM providers.
        """
        from ....vision.registry import VLMProviderRegistry

        # Shutdown current provider
        if self._vlm_provider is not None:
            try:
                self._vlm_provider.shutdown()
            except Exception:
                pass
            self._vlm_provider = None

        # Initialize new provider
        vlm_registry = VLMProviderRegistry(plugin_paths=vlm_plugin_paths)
        vlm_class = vlm_registry.get_provider_class(provider_name)
        new_provider = vlm_class()
        new_provider.initialize(vlm_config)
        self._vlm_provider = new_provider

        logger.info(
            "ScreenVisionTool: VLM provider swapped to '%s'", provider_name
        )

    async def execute(self, **kwargs) -> ToolResult:
        """
        Capture the screen and return a description.

        Thread-safe: Uses non-blocking lock to prevent concurrent captures.
        If a capture is already in progress, returns a helpful message.

        Returns:
            ToolResult with screen description or error
        """
        if not self._initialized:
            return ToolResult(
                success=False,
                output="",
                error="Screen vision tool not initialized",
            )

        # Non-blocking lock: if another capture is in-flight, don't wait
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return ToolResult(
                success=True,
                output="[Screen capture already in progress, please wait a moment]",
                metadata={"skipped": True},
            )

        start_time = time.time()

        try:
            # Health check
            if not self._vlm_provider.health_check():
                return ToolResult(
                    success=False,
                    output="",
                    error="Vision system unavailable",
                )

            # Capture screen
            try:
                image_base64 = self._capture.capture_base64()
            except Exception as e:
                logger.error(f"ScreenVisionTool: Capture failed: {e}")
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Screen capture failed: {e}",
                )

            # Run VLM in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._vlm_provider.describe(
                    image_base64=image_base64,
                    prompt=self._description_prompt,
                )
            )

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"ScreenVisionTool: Got description "
                f"({response.output_tokens} tokens, {elapsed_ms:.0f}ms)"
            )

            # Log for orchestrator stream
            vlm_type = type(self._vlm_provider).__name__
            vlm_cloud = self._vlm_provider.__class__.is_cloud_provider()
            vlm_label = f"{vlm_type} ({'cloud' if vlm_cloud else 'local'})"
            print(
                f"[TOOL:screen_vision] [{vlm_label}] ({response.output_tokens} tokens, "
                f"{response.latency_ms:.0f}ms): {response.description}",
                flush=True,
            )

            return ToolResult(
                success=True,
                output=response.description,
                metadata={
                    "tokens": response.output_tokens,
                    "vlm_latency_ms": response.latency_ms,
                    "total_latency_ms": elapsed_ms,
                },
            )

        except TimeoutError as e:
            logger.warning(f"ScreenVisionTool: VLM timeout: {e}")
            return ToolResult(
                success=False,
                output="",
                error="Vision analysis timed out",
            )
        except ConnectionError as e:
            logger.warning(f"ScreenVisionTool: VLM connection error: {e}")
            return ToolResult(
                success=False,
                output="",
                error="Vision system connection failed",
            )
        except Exception as e:
            logger.error(f"ScreenVisionTool: Error: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )
        finally:
            self._lock.release()

    def health_check(self) -> bool:
        """Check if the tool is operational."""
        if not self._initialized:
            return False
        if self._vlm_provider is None:
            return False
        return self._vlm_provider.health_check()

    def shutdown(self) -> None:
        """Cleanup resources."""
        if self._vlm_provider is not None:
            try:
                self._vlm_provider.shutdown()
            except Exception as e:
                logger.warning(f"ScreenVisionTool: Error during shutdown: {e}")

        self._vlm_provider = None
        self._capture = None
        self._initialized = False

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """Validate tool configuration."""
        errors = []

        # VLM provider is required (but has sensible default)
        vlm_provider = config.get("vlm_provider", "llama")
        if not isinstance(vlm_provider, str) or not vlm_provider:
            errors.append("vlm_provider must be a non-empty string")

        # Monitor must be positive int if specified
        monitor = config.get("monitor", 1)
        if not isinstance(monitor, int) or monitor < 1:
            errors.append("monitor must be a positive integer")

        return errors
