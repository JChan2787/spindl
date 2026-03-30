"""
STT Provider Registry - Discovery and instantiation of STT providers.

The registry discovers built-in providers and external plugins,
allowing the orchestrator to instantiate providers by name from config.

Mirrors the TTS provider registry (NANO-015) for consistency.
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Optional

from spindl.utils.paths import resolve_relative_path

from .base import STTProvider

logger = logging.getLogger(__name__)


class STTProviderNotFoundError(Exception):
    """
    Raised when a requested STT provider cannot be found.

    Includes list of available providers to help users fix config.
    """

    def __init__(self, provider_name: str, available: list[str]):
        self.provider_name = provider_name
        self.available = available
        available_str = ", ".join(available) if available else "(none)"
        super().__init__(
            f'STT provider "{provider_name}" not found. '
            f"Available providers: {available_str}"
        )


class STTProviderRegistry:
    """
    Discovers and instantiates STT providers.

    Search order for providers:
    1. Built-in providers from spindl.stt.builtin.*
    2. External plugins from configured plugin_paths

    Usage:
        registry = STTProviderRegistry(plugin_paths=["./plugins/stt"])
        provider_class = registry.get_provider_class("parakeet")
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
        self._builtin_providers: dict[str, type[STTProvider]] = {}
        self._plugin_paths = [Path(resolve_relative_path(p)) for p in (plugin_paths or [])]
        self._plugin_cache: dict[str, type[STTProvider]] = {}
        self._discover_builtin()

    def _discover_builtin(self) -> None:
        """
        Register built-in providers from spindl.stt.builtin.*.

        Scans the builtin package for submodules that export an STTProvider
        subclass. Each provider folder should have an __init__.py that
        exports the provider class.
        """
        try:
            builtin_spec = importlib.util.find_spec("spindl.stt.builtin")
            if builtin_spec is None or builtin_spec.submodule_search_locations is None:
                logger.debug("No builtin STT providers package found yet")
                return

            for location in builtin_spec.submodule_search_locations:
                builtin_path = Path(location)
                if not builtin_path.exists():
                    continue

                for item in builtin_path.iterdir():
                    if item.is_dir() and (item / "__init__.py").exists():
                        provider_name = item.name
                        self._try_register_builtin(provider_name)

        except Exception as e:
            logger.warning(f"Error discovering built-in STT providers: {e}")

    def _try_register_builtin(self, provider_name: str) -> None:
        """
        Attempt to register a built-in provider by name.

        Args:
            provider_name: Name of the provider subpackage (e.g., "parakeet")
        """
        try:
            module = importlib.import_module(
                f"spindl.stt.builtin.{provider_name}"
            )

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, STTProvider)
                    and attr is not STTProvider
                ):
                    self._builtin_providers[provider_name] = attr
                    logger.debug(f"Registered built-in STT provider: {provider_name}")
                    return

            logger.warning(
                f"Built-in provider '{provider_name}' has no STTProvider subclass"
            )

        except Exception as e:
            logger.warning(f"Failed to load built-in provider '{provider_name}': {e}")

    def _discover_plugin(self, name: str) -> Optional[type[STTProvider]]:
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
                spec = importlib.util.spec_from_file_location(
                    f"stt_plugin_{name}",
                    init_file
                )
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, STTProvider)
                        and attr is not STTProvider
                    ):
                        logger.info(f"Loaded external STT plugin: {name} from {provider_dir}")
                        return attr

            except Exception as e:
                logger.warning(f"Failed to load plugin '{name}' from {provider_dir}: {e}")

        return None

    def get_provider_class(self, name: str) -> type[STTProvider]:
        """
        Get provider class by name.

        Search order:
        1. Built-in providers (spindl.stt.builtin.*)
        2. Cached external plugins
        3. Plugin paths (scans for matching folder name)

        Args:
            name: Provider name (e.g., "parakeet", "whisper")

        Returns:
            Provider class (not instantiated)

        Raises:
            STTProviderNotFoundError: Provider not found, with list of available providers
        """
        if name in self._builtin_providers:
            return self._builtin_providers[name]

        if name in self._plugin_cache:
            return self._plugin_cache[name]

        provider_class = self._discover_plugin(name)
        if provider_class is not None:
            self._plugin_cache[name] = provider_class
            return provider_class

        raise STTProviderNotFoundError(name, self.list_available())

    def list_available(self) -> list[str]:
        """
        Return names of all discoverable providers.

        Returns:
            Sorted list of provider names (built-in + cached plugins)
        """
        all_providers = set(self._builtin_providers.keys())
        all_providers.update(self._plugin_cache.keys())

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


def create_default_registry(plugin_paths: Optional[list[str]] = None) -> STTProviderRegistry:
    """
    Create an STT provider registry with default configuration.

    Args:
        plugin_paths: Optional list of external plugin directories

    Returns:
        Configured STTProviderRegistry
    """
    return STTProviderRegistry(plugin_paths=plugin_paths)
