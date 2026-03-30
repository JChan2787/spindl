"""
VLMProviderRegistry - Discovery and instantiation of VLM providers.

Follows the same pattern as LLMProviderRegistry and TTSProviderRegistry.
Discovers built-in providers and allows plugin registration.
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Optional, Type

from .base import VLMProvider
from spindl.utils.paths import resolve_relative_path

logger = logging.getLogger(__name__)

# Built-in provider locations
BUILTIN_PROVIDERS = {
    "llama": "spindl.vision.builtin.llama.provider",
    "openai": "spindl.vision.builtin.openai.provider",
    "llm": "spindl.vision.builtin.llm.provider",
}


class VLMProviderRegistry:
    """
    Registry for VLM provider discovery and instantiation.

    Discovers providers from:
    1. Built-in providers (llama, openai)
    2. Plugin paths (future extensibility)

    Usage:
        registry = VLMProviderRegistry()
        provider_class = registry.get_provider_class("llama")
        provider = provider_class()
        provider.initialize(config)
    """

    def __init__(self, plugin_paths: Optional[list[str]] = None):
        """
        Initialize registry with optional plugin paths.

        Args:
            plugin_paths: Additional paths to search for VLM provider plugins
        """
        self._plugin_paths = plugin_paths or []
        self._provider_cache: Dict[str, Type[VLMProvider]] = {}
        self._discovered = False

    def _discover_builtin(self) -> None:
        """Load built-in providers into cache."""
        for name, module_path in BUILTIN_PROVIDERS.items():
            try:
                module = importlib.import_module(module_path)
                # Look for provider class (naming convention: *VLMProvider)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, VLMProvider)
                        and attr is not VLMProvider
                    ):
                        self._provider_cache[name] = attr
                        logger.debug(f"Registered built-in VLM provider: {name}")
                        break
            except ImportError as e:
                logger.warning(f"Failed to load built-in VLM provider '{name}': {e}")

    def _discover_plugins(self) -> None:
        """
        Discover providers from plugin paths.

        Plugin structure expected:
            plugin_path/
                provider_name/
                    __init__.py  (must export VLMProvider subclass)
        """
        for plugin_path in self._plugin_paths:
            path = Path(resolve_relative_path(plugin_path))
            if not path.exists():
                logger.debug(f"Plugin path does not exist: {plugin_path}")
                continue

            for provider_dir in path.iterdir():
                if not provider_dir.is_dir():
                    continue
                if not (provider_dir / "__init__.py").exists():
                    continue

                provider_name = provider_dir.name
                if provider_name in self._provider_cache:
                    logger.debug(
                        f"Skipping plugin '{provider_name}' (already registered)"
                    )
                    continue

                try:
                    # Add plugin path to sys.path temporarily
                    import sys
                    sys.path.insert(0, str(path))
                    try:
                        module = importlib.import_module(provider_name)
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, VLMProvider)
                                and attr is not VLMProvider
                            ):
                                self._provider_cache[provider_name] = attr
                                logger.info(
                                    f"Registered VLM plugin provider: {provider_name}"
                                )
                                break
                    finally:
                        sys.path.remove(str(path))
                except Exception as e:
                    logger.warning(
                        f"Failed to load VLM plugin '{provider_name}': {e}"
                    )

    def _ensure_discovered(self) -> None:
        """Ensure providers have been discovered."""
        if not self._discovered:
            self._discover_builtin()
            self._discover_plugins()
            self._discovered = True

    def get_provider_class(self, name: str) -> Type[VLMProvider]:
        """
        Get a provider class by name.

        Args:
            name: Provider name (e.g., "llama", "openai")

        Returns:
            VLMProvider subclass

        Raises:
            ValueError: If provider not found
        """
        self._ensure_discovered()

        if name not in self._provider_cache:
            available = ", ".join(sorted(self._provider_cache.keys()))
            raise ValueError(
                f"Unknown VLM provider: '{name}'. "
                f"Available: {available}"
            )

        return self._provider_cache[name]

    def create_provider(self, name: str, config: dict) -> VLMProvider:
        """
        Create and initialize a provider instance.

        Convenience method that combines get_provider_class() and initialize().

        Args:
            name: Provider name
            config: Provider-specific configuration

        Returns:
            Initialized VLMProvider instance
        """
        provider_class = self.get_provider_class(name)
        provider = provider_class()
        provider.initialize(config)
        return provider

    def list_providers(self) -> list[str]:
        """
        List all available provider names.

        Returns:
            Sorted list of provider names
        """
        self._ensure_discovered()
        return sorted(self._provider_cache.keys())

    def validate_provider_config(self, name: str, config: dict) -> list[str]:
        """
        Validate configuration for a specific provider.

        Args:
            name: Provider name
            config: Provider configuration to validate

        Returns:
            List of error messages (empty if valid)
        """
        try:
            provider_class = self.get_provider_class(name)
            return provider_class.validate_config(config)
        except ValueError as e:
            return [str(e)]
