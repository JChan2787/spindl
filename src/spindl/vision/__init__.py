"""
Vision Module - Screen capture and VLM integration for spindl.

This module provides vision/multimodal capability via a VisionProvider
(ContextProvider pattern). The system captures screen content, sends it
to any OpenAI-compatible VLM endpoint for description, and injects that
description into the LLM prompt context.

Main Components:
    - ScreenCapture: MSS-based screen capture with preprocessing
    - VLMProvider: Abstract base for VLM backends (local/cloud)
    - VLMProviderRegistry: Provider discovery and instantiation
    - VisionProvider: ContextProvider that ties it all together

Provider Types:
    - llama: Local VLM via llama-server (Gemma3, LLaVA, etc.)
    - openai: Cloud VLM via OpenAI-compatible API

Usage:
    from spindl.vision import (
        ScreenCapture,
        VisionProvider,
        VisionStrategy,
        VLMProviderRegistry,
    )

    # Create screen capture
    capture = ScreenCapture(monitor=1, target_width=1920, target_height=1080)

    # Create VLM provider via registry
    registry = VLMProviderRegistry()
    vlm = registry.create_provider("llama", config["vision"]["providers"]["llama"])

    # Create vision context provider
    vision = VisionProvider(
        capture=capture,
        vlm_provider=vlm,
        strategy=VisionStrategy.ALWAYS,
    )

    # Register with prompt builder
    context_registry.register("vision", vision)
"""

import logging
from typing import Optional

from .base import VLMProperties, VLMProvider, VLMResponse
from .capture import CaptureResult, ScreenCapture
from .provider import VisionProvider, VisionStrategy
from .registry import VLMProviderRegistry

logger = logging.getLogger(__name__)


def create_vision_provider(config: dict) -> Optional[VisionProvider]:
    """
    Factory function to create a fully configured VisionProvider.

    Reads the vision section of the config and creates all necessary
    components (ScreenCapture, VLMProvider, VisionProvider).

    Args:
        config: Full spindl.yaml config dict (must contain 'vision' section)

    Returns:
        Configured VisionProvider, or None if vision is disabled

    Example:
        config = load_config("./config/spindl.yaml")
        vision_provider = create_vision_provider(config)
        if vision_provider:
            registry.register(vision_provider)
    """
    vision_config = config.get("vision", {})

    # Check if vision is enabled
    if not vision_config.get("enabled", False):
        logger.info("Vision disabled in config")
        return None

    # Parse strategy
    strategy_str = vision_config.get("strategy", "always")
    try:
        strategy = VisionStrategy(strategy_str)
    except ValueError:
        logger.warning(
            f"Invalid vision strategy '{strategy_str}', defaulting to ALWAYS"
        )
        strategy = VisionStrategy.ALWAYS

    # Create screen capture
    capture_config = vision_config.get("capture", {})
    capture = ScreenCapture(
        monitor=capture_config.get("monitor", 1),
        target_width=capture_config.get("width", 1920),
        target_height=capture_config.get("height", 1080),
        jpeg_quality=capture_config.get("jpeg_quality", 95),
    )

    # Create VLM provider
    provider_name = vision_config.get("provider", "llama")
    providers_config = vision_config.get("providers", {})

    if provider_name not in providers_config:
        logger.error(
            f"Vision provider '{provider_name}' not found in config. "
            f"Available: {list(providers_config.keys())}"
        )
        return None

    provider_config = providers_config[provider_name]

    # Get plugin paths for registry
    plugin_paths = vision_config.get("plugin_paths", [])
    registry = VLMProviderRegistry(plugin_paths=plugin_paths)

    try:
        vlm_provider = registry.create_provider(provider_name, provider_config)
    except Exception as e:
        logger.error(f"Failed to create VLM provider '{provider_name}': {e}")
        return None

    # Create VisionProvider
    fallback_text = vision_config.get("fallback_text", "")
    description_prompt = provider_config.get("prompt")

    vision_provider = VisionProvider(
        capture=capture,
        vlm_provider=vlm_provider,
        strategy=strategy,
        fallback_text=fallback_text,
        description_prompt=description_prompt,
    )

    logger.info(
        f"Vision provider created: {provider_name} with strategy {strategy.value}"
    )

    return vision_provider


__all__ = [
    # Base classes
    "VLMProvider",
    "VLMProperties",
    "VLMResponse",
    # Screen capture
    "ScreenCapture",
    "CaptureResult",
    # Context provider
    "VisionProvider",
    "VisionStrategy",
    # Registry
    "VLMProviderRegistry",
    # Factory
    "create_vision_provider",
]
