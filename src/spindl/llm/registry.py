"""
LLM Provider Registry - Discovery and instantiation of LLM providers.

The registry discovers built-in providers and external plugins,
allowing the orchestrator to instantiate providers by name from config.
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Optional

from .base import LLMProvider
from spindl.utils.paths import resolve_relative_path

logger = logging.getLogger(__name__)


class ProviderNotFoundError(Exception):
    """
    Raised when a requested LLM provider cannot be found.

    Includes list of available providers to help users fix config.
    """

    def __init__(self, provider_name: str, available: list[str]):
        self.provider_name = provider_name
        self.available = available
        available_str = ", ".join(available) if available else "(none)"
        super().__init__(
            f'LLM provider "{provider_name}" not found. '
            f"Available providers: {available_str}"
        )


class LLMProviderRegistry:
    """
    Discovers and instantiates LLM providers.

    Search order for providers:
    1. Built-in providers from spindl.llm.builtin.*
    2. External plugins from configured plugin_paths

    Usage:
        registry = LLMProviderRegistry(plugin_paths=["./plugins/llm"])
        provider_class = registry.get_provider_class("llama")
        provider = provider_class()
        provider.initialize(config)
    """

    def __init__(self, plugin_paths: Optional[list[str]] = None):
        """
        Initialize the registry.

        Args:
            plugin_paths: Optional list of paths to scan for external providers.
                         Each path should contain subdirectories, one per provider.
        """
        self._builtin_providers: dict[str, type[LLMProvider]] = {}
        self._plugin_paths = [Path(resolve_relative_path(p)) for p in (plugin_paths or [])]
        self._plugin_cache: dict[str, type[LLMProvider]] = {}
        self._discover_builtin()

    def _discover_builtin(self) -> None:
        """
        Register built-in providers from spindl.llm.builtin.*.

        Scans the builtin package for submodules that export an LLMProvider
        subclass. Each provider folder should have an __init__.py that
        exports the provider class.
        """
        try:
            # Import the builtin package to check if it exists
            builtin_spec = importlib.util.find_spec("spindl.llm.builtin")
            if builtin_spec is None or builtin_spec.submodule_search_locations is None:
                logger.debug("No builtin LLM providers package found yet")
                return

            # Scan for provider subpackages
            for location in builtin_spec.submodule_search_locations:
                builtin_path = Path(location)
                if not builtin_path.exists():
                    continue

                for item in builtin_path.iterdir():
                    if item.is_dir() and (item / "__init__.py").exists():
                        provider_name = item.name
                        self._try_register_builtin(provider_name)

        except Exception as e:
            logger.warning(f"Error discovering built-in LLM providers: {e}")

    def _try_register_builtin(self, provider_name: str) -> None:
        """
        Attempt to register a built-in provider by name.

        Args:
            provider_name: Name of the provider subpackage (e.g., "llama")
        """
        try:
            module = importlib.import_module(
                f"spindl.llm.builtin.{provider_name}"
            )

            # Look for a class that inherits from LLMProvider
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, LLMProvider)
                    and attr is not LLMProvider
                ):
                    self._builtin_providers[provider_name] = attr
                    logger.debug(f"Registered built-in LLM provider: {provider_name}")
                    return

            logger.warning(
                f"Built-in provider '{provider_name}' has no LLMProvider subclass"
            )

        except Exception as e:
            logger.warning(f"Failed to load built-in provider '{provider_name}': {e}")

    def _discover_plugin(self, name: str) -> Optional[type[LLMProvider]]:
        """
        Attempt to discover an external plugin provider.

        Args:
            name: Provider name to look for in plugin paths

        Returns:
            Provider class if found, None otherwise
        """
        for plugin_path in self._plugin_paths:
            provider_dir = plugin_path / name
            init_file = provider_dir / "__init__.py"

            if not init_file.exists():
                continue

            try:
                # Load the module from file path
                spec = importlib.util.spec_from_file_location(
                    f"llm_plugin_{name}",
                    init_file
                )
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for LLMProvider subclass
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, LLMProvider)
                        and attr is not LLMProvider
                    ):
                        logger.info(f"Loaded external LLM plugin: {name} from {provider_dir}")
                        return attr

            except Exception as e:
                logger.warning(f"Failed to load plugin '{name}' from {provider_dir}: {e}")

        return None

    def get_provider_class(self, name: str) -> type[LLMProvider]:
        """
        Get provider class by name.

        Search order:
        1. Built-in providers (spindl.llm.builtin.*)
        2. Cached external plugins
        3. Plugin paths (scans for matching folder name)

        Args:
            name: Provider name (e.g., "llama", "deepseek")

        Returns:
            Provider class (not instantiated)

        Raises:
            ProviderNotFoundError: Provider not found, with list of available providers
        """
        # Check built-in first
        if name in self._builtin_providers:
            return self._builtin_providers[name]

        # Check plugin cache
        if name in self._plugin_cache:
            return self._plugin_cache[name]

        # Try to discover from plugin paths
        provider_class = self._discover_plugin(name)
        if provider_class is not None:
            self._plugin_cache[name] = provider_class
            return provider_class

        # Not found
        raise ProviderNotFoundError(name, self.list_available())

    def list_available(self) -> list[str]:
        """
        Return names of all discoverable providers.

        Returns:
            Sorted list of provider names (built-in + cached plugins)
        """
        all_providers = set(self._builtin_providers.keys())
        all_providers.update(self._plugin_cache.keys())

        # Also scan plugin paths for not-yet-loaded providers
        for plugin_path in self._plugin_paths:
            if plugin_path.exists():
                for item in plugin_path.iterdir():
                    if item.is_dir() and (item / "__init__.py").exists():
                        all_providers.add(item.name)

        return sorted(all_providers)

    def refresh(self) -> None:
        """
        Re-scan for providers.

        Clears plugin cache and re-discovers built-in providers.
        Useful if plugins are added at runtime.
        """
        self._builtin_providers.clear()
        self._plugin_cache.clear()
        self._discover_builtin()


def create_default_registry(plugin_paths: Optional[list[str]] = None) -> LLMProviderRegistry:
    """
    Create an LLM provider registry with default configuration.

    Args:
        plugin_paths: Optional list of external plugin directories

    Returns:
        Configured LLMProviderRegistry
    """
    return LLMProviderRegistry(plugin_paths=plugin_paths)
