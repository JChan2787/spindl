"""
Tool Registry - Discovery, instantiation, and enable/disable management of tools.

The registry discovers built-in tools and external plugins, allowing the
orchestrator to instantiate and manage tools based on configuration.

Follows the same pattern as LLMProviderRegistry and VLMProviderRegistry.
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Optional

from spindl.utils.paths import resolve_relative_path

from .base import Tool

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    """
    Raised when a requested tool cannot be found.

    Includes list of available tools to help users fix config.
    """

    def __init__(self, tool_name: str, available: list[str]):
        self.tool_name = tool_name
        self.available = available
        available_str = ", ".join(available) if available else "(none)"
        super().__init__(
            f'Tool "{tool_name}" not found. '
            f"Available tools: {available_str}"
        )


class ToolRegistry:
    """
    Discovers, instantiates, and manages tools.

    Search order for tools:
    1. Built-in tools from spindl.tools.builtin.*
    2. External plugins from configured plugin_paths

    Features:
    - Per-tool enable/disable via configuration
    - Lazy loading (tools instantiated only when needed)
    - Runtime refresh for hot-loading new tools

    Usage:
        registry = ToolRegistry(plugin_paths=["./plugins/tools"])

        # Get all available tool names
        available = registry.list_available()

        # Initialize specific tools from config
        registry.initialize_tools(tools_config)

        # Get enabled tools for prompt injection
        enabled = registry.get_enabled_tools()

        # Execute a tool
        result = await registry.execute_tool("screen_vision", **kwargs)
    """

    def __init__(self, plugin_paths: Optional[list[str]] = None):
        """
        Initialize the registry.

        Args:
            plugin_paths: Optional list of paths to scan for external tools.
                         Each path should contain subdirectories, one per tool.
        """
        self._builtin_tools: dict[str, type[Tool]] = {}
        self._plugin_paths = [Path(resolve_relative_path(p)) for p in (plugin_paths or [])]
        self._plugin_cache: dict[str, type[Tool]] = {}

        # Instantiated and initialized tools
        self._tools: dict[str, Tool] = {}

        # Enable/disable state per tool
        self._enabled: dict[str, bool] = {}

        self._discover_builtin()

    def _discover_builtin(self) -> None:
        """
        Register built-in tools from spindl.tools.builtin.*.

        Scans the builtin package for submodules that export a Tool
        subclass. Each tool folder should have an __init__.py that
        exports the tool class.
        """
        try:
            builtin_spec = importlib.util.find_spec("spindl.tools.builtin")
            if builtin_spec is None or builtin_spec.submodule_search_locations is None:
                logger.debug("No builtin tools package found yet")
                return

            for location in builtin_spec.submodule_search_locations:
                builtin_path = Path(location)
                if not builtin_path.exists():
                    continue

                for item in builtin_path.iterdir():
                    if item.is_dir() and (item / "__init__.py").exists():
                        tool_name = item.name
                        self._try_register_builtin(tool_name)

        except Exception as e:
            logger.warning(f"Error discovering built-in tools: {e}")

    def _try_register_builtin(self, tool_name: str) -> None:
        """
        Attempt to register a built-in tool by name.

        Args:
            tool_name: Name of the tool subpackage (e.g., "screen_vision")
        """
        try:
            module = importlib.import_module(
                f"spindl.tools.builtin.{tool_name}"
            )

            # Look for a class that inherits from Tool
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Tool)
                    and attr is not Tool
                ):
                    self._builtin_tools[tool_name] = attr
                    logger.debug(f"Registered built-in tool: {tool_name}")
                    return

            logger.warning(
                f"Built-in tool '{tool_name}' has no Tool subclass"
            )

        except Exception as e:
            logger.warning(f"Failed to load built-in tool '{tool_name}': {e}")

    def _discover_plugin(self, name: str) -> Optional[type[Tool]]:
        """
        Attempt to discover an external plugin tool.

        Args:
            name: Tool name to look for in plugin paths

        Returns:
            Tool class if found, None otherwise
        """
        for plugin_path in self._plugin_paths:
            tool_dir = plugin_path / name
            init_file = tool_dir / "__init__.py"

            if not init_file.exists():
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"tool_plugin_{name}",
                    init_file
                )
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for Tool subclass
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, Tool)
                        and attr is not Tool
                    ):
                        logger.info(f"Loaded external tool plugin: {name} from {tool_dir}")
                        return attr

            except Exception as e:
                logger.warning(f"Failed to load tool plugin '{name}' from {tool_dir}: {e}")

        return None

    def get_tool_class(self, name: str) -> type[Tool]:
        """
        Get tool class by name.

        Search order:
        1. Built-in tools (spindl.tools.builtin.*)
        2. Cached external plugins
        3. Plugin paths (scans for matching folder name)

        Args:
            name: Tool name (e.g., "screen_vision", "web_search")

        Returns:
            Tool class (not instantiated)

        Raises:
            ToolNotFoundError: Tool not found, with list of available tools
        """
        # Check built-in first
        if name in self._builtin_tools:
            return self._builtin_tools[name]

        # Check plugin cache
        if name in self._plugin_cache:
            return self._plugin_cache[name]

        # Try to discover from plugin paths
        tool_class = self._discover_plugin(name)
        if tool_class is not None:
            self._plugin_cache[name] = tool_class
            return tool_class

        # Not found
        raise ToolNotFoundError(name, self.list_available())

    def list_available(self) -> list[str]:
        """
        Return names of all discoverable tools.

        Returns:
            Sorted list of tool names (built-in + cached plugins + scannable)
        """
        all_tools = set(self._builtin_tools.keys())
        all_tools.update(self._plugin_cache.keys())

        # Also scan plugin paths for not-yet-loaded tools
        for plugin_path in self._plugin_paths:
            if plugin_path.exists():
                for item in plugin_path.iterdir():
                    if item.is_dir() and (item / "__init__.py").exists():
                        all_tools.add(item.name)

        return sorted(all_tools)

    def initialize_tools(self, tools_config: dict) -> None:
        """
        Initialize tools based on configuration.

        Instantiates and initializes each configured tool, setting
        enable/disable state as specified.

        Args:
            tools_config: The "tools" section from spindl.yaml, e.g.:
                {
                    "tools": {
                        "screen_vision": {"enabled": true, ...},
                        "web_search": {"enabled": false, ...}
                    }
                }
        """
        tools_section = tools_config.get("tools", {})

        for tool_name, tool_config in tools_section.items():
            enabled = tool_config.get("enabled", True)
            self._enabled[tool_name] = enabled

            if not enabled:
                logger.info(f"Tool '{tool_name}' is disabled in config")
                continue

            try:
                tool_class = self.get_tool_class(tool_name)

                # Validate config before instantiation
                errors = tool_class.validate_config(tool_config)
                if errors:
                    logger.error(
                        f"Tool '{tool_name}' config errors: {', '.join(errors)}"
                    )
                    self._enabled[tool_name] = False
                    continue

                # Instantiate and initialize
                tool = tool_class()
                tool.initialize(tool_config)

                self._tools[tool_name] = tool
                logger.info(f"Initialized tool: {tool_name}")

            except ToolNotFoundError:
                logger.warning(f"Tool '{tool_name}' not found, skipping")
                self._enabled[tool_name] = False
            except Exception as e:
                logger.error(f"Failed to initialize tool '{tool_name}': {e}")
                self._enabled[tool_name] = False

    def get_enabled_tools(self) -> list[Tool]:
        """
        Get all currently enabled and initialized tools.

        Returns:
            List of Tool instances that are enabled
        """
        return [
            tool for name, tool in self._tools.items()
            if self._enabled.get(name, False)
        ]

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a specific initialized tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance if initialized and enabled, None otherwise
        """
        if not self._enabled.get(name, False):
            return None
        return self._tools.get(name)

    def is_enabled(self, name: str) -> bool:
        """
        Check if a tool is enabled.

        Args:
            name: Tool name

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled.get(name, False)

    def set_enabled(self, name: str, enabled: bool) -> None:
        """
        Enable or disable a tool at runtime.

        Note: The tool must already be initialized. This only
        controls whether it's included in get_enabled_tools().

        Args:
            name: Tool name
            enabled: New enabled state
        """
        if name not in self._tools:
            logger.warning(f"Cannot set enabled state for uninitialized tool: {name}")
            return

        old_state = self._enabled.get(name, False)
        self._enabled[name] = enabled
        logger.info(f"Tool '{name}' enabled: {old_state} -> {enabled}")

    async def execute_tool(self, name: str, **kwargs) -> Optional[dict]:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Parameters to pass to the tool

        Returns:
            ToolResult as dict, or None if tool not found/disabled
        """
        tool = self.get_tool(name)
        if tool is None:
            logger.warning(f"Cannot execute tool '{name}': not found or disabled")
            return None

        try:
            result = await tool.execute(**kwargs)
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "metadata": result.metadata,
            }
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "metadata": {},
            }

    def get_function_definitions(self) -> list[dict]:
        """
        Get function definitions for all enabled tools.

        Returns list suitable for OpenAI-style function calling.

        Returns:
            List of function definition dicts
        """
        return [
            tool.get_function_definition()
            for tool in self.get_enabled_tools()
        ]

    def shutdown(self) -> None:
        """
        Shutdown all initialized tools.

        Calls shutdown() on each tool to release resources.
        """
        for name, tool in self._tools.items():
            try:
                tool.shutdown()
                logger.debug(f"Shut down tool: {name}")
            except Exception as e:
                logger.warning(f"Error shutting down tool '{name}': {e}")

        self._tools.clear()
        self._enabled.clear()

    def refresh(self) -> None:
        """
        Re-scan for tools.

        Clears plugin cache and re-discovers built-in tools.
        Does NOT re-initialize already-initialized tools.
        """
        self._builtin_tools.clear()
        self._plugin_cache.clear()
        self._discover_builtin()


def create_tool_registry(plugin_paths: Optional[list[str]] = None) -> ToolRegistry:
    """
    Create a tool registry with default configuration.

    Args:
        plugin_paths: Optional list of external plugin directories

    Returns:
        Configured ToolRegistry
    """
    return ToolRegistry(plugin_paths=plugin_paths)
