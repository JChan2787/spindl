"""
Tools Module - Scalable, configurable tool system for spindl.

Tools are invoked on-demand by the LLM via function calling, rather than
being injected into every prompt. This gives the LLM agency over when to
use each capability and prevents context pollution.

Architecture:
    - base.py: Tool ABC defining the interface all tools must implement
    - registry.py: Discovery, instantiation, and enable/disable management
    - builtin/: Built-in tools (screen_vision, etc.)
    - External plugins loaded from configured plugin_paths

Usage:
    from spindl.tools import create_tool_registry, ToolRegistry

    # Create registry with optional plugin paths
    registry = create_tool_registry(plugin_paths=["./plugins/tools"])

    # Initialize tools from config
    registry.initialize_tools(config["tools"])

    # Get function definitions for LLM
    functions = registry.get_function_definitions()

    # Execute a tool
    result = await registry.execute_tool("screen_vision")

Configuration (spindl.yaml):
    tools:
      enabled: true
      plugin_paths: ["./plugins/tools"]
      tools:
        screen_vision:
          enabled: true
        web_search:
          enabled: false
          api_key: "${SEARCH_API_KEY}"
"""

from .base import Tool, ToolParameter, ToolResult
from .registry import ToolRegistry, ToolNotFoundError, create_tool_registry
from .executor import ToolExecutor, ToolExecutionResult, create_tool_executor

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "ToolNotFoundError",
    "create_tool_registry",
    "ToolExecutor",
    "ToolExecutionResult",
    "create_tool_executor",
]
