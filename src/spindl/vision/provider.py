"""
VisionProvider - ContextProvider that injects VLM-generated image descriptions.

This is the integration point between the vision system and the prompt builder.
It captures the screen, sends it to a VLM for description, and injects that
description into the prompt template via the [VISION_ANALYSIS] placeholder.
"""

import logging
import threading
from enum import Enum
from typing import Optional

from ..llm.build_context import BuildContext
from ..llm.context_provider import ContextProvider
from .base import VLMProvider
from .capture import ScreenCapture

logger = logging.getLogger(__name__)


class VisionStrategy(Enum):
    """
    When to capture and process vision.

    Controls whether the VisionProvider actually captures on each call.
    """

    NEVER = "never"
    """Vision disabled - always return empty string."""

    ALWAYS = "always"
    """Capture screen on every prompt build."""

    ON_DEMAND = "on_demand"
    """Capture only when triggered by voice command (future)."""


class VisionProvider(ContextProvider):
    """
    ContextProvider that injects VLM-generated screen descriptions.

    Captures the screen, sends to VLM for description, and returns
    formatted text for injection into the prompt template.

    Placeholder: [VISION_ANALYSIS]

    Error Handling:
        Vision is enhancement, not requirement. All errors result in
        graceful fallback (empty string or fallback_text), never exceptions.
        The conversation loop must never be blocked by vision failures.

    Usage:
        capture = ScreenCapture(monitor=1)
        vlm_provider = LlamaVLMProvider()
        vlm_provider.initialize(config)

        vision_provider = VisionProvider(
            capture=capture,
            vlm_provider=vlm_provider,
            strategy=VisionStrategy.ALWAYS,
        )

        # Register with context registry
        context_registry.register("vision", vision_provider)
    """

    def __init__(
        self,
        capture: ScreenCapture,
        vlm_provider: VLMProvider,
        strategy: VisionStrategy = VisionStrategy.ALWAYS,
        fallback_text: str = "",
        description_prompt: Optional[str] = None,
    ):
        """
        Initialize VisionProvider.

        Args:
            capture: ScreenCapture instance for grabbing screen
            vlm_provider: Initialized VLMProvider for image description
            strategy: When to capture (NEVER, ALWAYS, ON_DEMAND)
            fallback_text: Text to inject if VLM fails (default: empty)
            description_prompt: Custom prompt for VLM (uses VLM default if None)
        """
        self._capture = capture
        self._vlm_provider = vlm_provider
        self._strategy = strategy
        self._fallback_text = fallback_text
        self._description_prompt = description_prompt

        # Mutex to prevent concurrent VLM calls (race condition protection)
        # If a second request comes in while one is in-flight, it gets fallback
        self._lock = threading.Lock()

    @property
    def placeholder(self) -> str:
        """The placeholder this provider fills."""
        return "[VISION_ANALYSIS]"

    def provide(self, context: BuildContext) -> Optional[str]:
        """
        Capture screen, get VLM description, return formatted text.

        Args:
            context: BuildContext (not currently used, but required by interface)

        Returns:
            Formatted vision description, or empty string on failure/disabled
        """
        # Check strategy
        if self._strategy == VisionStrategy.NEVER:
            return ""

        if self._strategy == VisionStrategy.ON_DEMAND:
            # Future: Check context.input_metadata for trigger
            # For now, treat as disabled
            logger.debug("VisionProvider: ON_DEMAND not yet implemented, skipping")
            return ""

        # Strategy is ALWAYS - proceed with capture
        return self._capture_and_describe()

    def _capture_and_describe(self) -> str:
        """
        Perform the actual capture and VLM description.

        Thread-safe: Uses non-blocking lock acquisition. If another thread
        is already processing a VLM request, returns fallback immediately
        rather than waiting (prevents queue buildup during rapid interactions).

        Returns:
            Formatted description or fallback text
        """
        # Non-blocking lock: if another request is in-flight, skip this one
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            logger.info("VisionProvider: VLM request already in-flight, using fallback")
            print("[VLM] (skipped - request in-flight)", flush=True)
            return self._fallback_text

        try:
            # Health check first (fast fail)
            if not self._vlm_provider.health_check():
                logger.warning("VisionProvider: VLM not healthy, using fallback")
                return self._fallback_text

            # Capture screen
            try:
                image_base64 = self._capture.capture_base64()
            except Exception as e:
                logger.error(f"VisionProvider: Screen capture failed: {e}")
                return self._fallback_text

            # Get VLM description
            try:
                response = self._vlm_provider.describe(
                    image_base64=image_base64,
                    prompt=self._description_prompt,
                )
                description = response.description

                logger.debug(
                    f"VisionProvider: Got description ({response.output_tokens} tokens, "
                    f"{response.latency_ms:.0f}ms)"
                )

                # Log full VLM output for [NANO-ORCHESTRATOR] stream (no truncation)
                print(
                    f"[VLM] ({response.output_tokens} tokens, {response.latency_ms:.0f}ms): {description}",
                    flush=True,
                )

                return self._format_description(description)

            except TimeoutError as e:
                logger.warning(f"VisionProvider: VLM timeout: {e}")
                return self._fallback_text
            except ConnectionError as e:
                logger.warning(f"VisionProvider: VLM connection error: {e}")
                return self._fallback_text
            except Exception as e:
                logger.error(f"VisionProvider: VLM error: {e}")
                return self._fallback_text

        finally:
            self._lock.release()

    def _format_description(self, description: str) -> str:
        """
        Format the VLM description for prompt injection.

        Wraps the description in a clear marker so the LLM knows
        this is visual context from the screen.

        Args:
            description: Raw description from VLM

        Returns:
            Formatted description string
        """
        if not description or not description.strip():
            return self._fallback_text

        return f"[What you currently see on screen: {description.strip()}]"

    def set_strategy(self, strategy: VisionStrategy) -> None:
        """
        Change the vision strategy at runtime.

        Useful for enabling/disabling vision without recreating the provider.

        Args:
            strategy: New VisionStrategy to use
        """
        old_strategy = self._strategy
        self._strategy = strategy
        logger.info(f"VisionProvider: Strategy changed {old_strategy} -> {strategy}")

    def get_strategy(self) -> VisionStrategy:
        """Get the current vision strategy."""
        return self._strategy

    def force_capture(self) -> str:
        """
        Force a capture regardless of strategy.

        Useful for testing or manual triggers.

        Returns:
            Formatted description or fallback text
        """
        return self._capture_and_describe()
