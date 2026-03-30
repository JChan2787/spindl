"""
Tool Base Classes - Abstract base for all tool implementations.

This module defines the protocol that all tools must implement,
enabling a scalable, configurable tool system for the LLM pipeline.

Tools are invoked on-demand by the LLM via function calling, rather than
being injected into every prompt. This prevents context pollution and
gives the LLM agency over when to use each capability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """
    Result from a tool execution.

    Contains the output and metadata about the execution.
    """

    success: bool
    """Whether the tool executed successfully."""

    output: str
    """The tool's output text (to be shown to LLM)."""

    error: Optional[str] = None
    """Error message if success is False."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata (latency, tokens, etc.)."""


@dataclass
class ToolParameter:
    """
    Definition of a single tool parameter.

    Used to build the JSON schema for the tool's parameters.
    """

    name: str
    """Parameter name."""

    param_type: str
    """JSON schema type: "string", "number", "boolean", "array", "object"."""

    description: str
    """Description of what this parameter does."""

    required: bool = True
    """Whether this parameter is required."""

    default: Any = None
    """Default value if not provided."""

    enum: Optional[list[str]] = None
    """If set, parameter must be one of these values."""


class Tool(ABC):
    """
    Protocol all tools must implement.

    This abstract base class defines the contract between the tool system
    and individual tool implementations. Tools can be built-in or loaded
    from external plugin directories.

    Lifecycle:
        1. Instantiation (lightweight, no heavy work)
        2. initialize(config) - set up resources, connections
        3. LLM calls execute() when it wants to use the tool
        4. shutdown() - cleanup resources

    Example:
        class ScreenVisionTool(Tool):
            @property
            def name(self) -> str:
                return "screen_vision"

            @property
            def description(self) -> str:
                return "Capture and describe what's currently on screen"

            @property
            def parameters(self) -> list[ToolParameter]:
                return []  # No parameters needed

            async def execute(self, **kwargs) -> ToolResult:
                description = await self._capture_screen()
                return ToolResult(success=True, output=description)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this tool.

        Used in config to enable/disable, and by LLM to invoke.
        Should be snake_case (e.g., "screen_vision", "web_search").
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this tool does.

        This is shown to the LLM so it knows when to use the tool.
        Should be clear and concise (1-2 sentences).
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """
        List of parameters this tool accepts.

        Return empty list for tools that need no input.
        Used to build JSON schema for function calling.
        """
        pass

    def get_schema(self) -> dict[str, Any]:
        """
        Generate JSON schema for this tool's parameters.

        Returns OpenAI-compatible function schema format.
        Override if you need custom schema generation.

        Returns:
            Dict with "type": "object", "properties", "required"
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def get_function_definition(self) -> dict[str, Any]:
        """
        Get the complete function definition for LLM tool calling.

        Returns OpenAI-compatible function definition format.

        Returns:
            Dict with "name", "description", "parameters"
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_schema(),
        }

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Called once at startup with tool-specific config.

        Set up any resources needed (connections, models, etc.).

        Args:
            config: Tool-specific configuration dict from spindl.yaml
                   (the section under tools.tools.<tool_name>)

        Raises:
            ValueError: For invalid configuration
            RuntimeError: For initialization failures
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.

        This is called when the LLM invokes the tool. Parameters
        are passed as keyword arguments matching the tool's schema.

        Args:
            **kwargs: Parameters as defined in self.parameters

        Returns:
            ToolResult with success status and output
        """
        pass

    def health_check(self) -> bool:
        """
        Check if tool is operational.

        Default returns True. Override for tools that depend on
        external services.

        Returns:
            True if ready to execute, False otherwise
        """
        return True

    def shutdown(self) -> None:
        """
        Cleanup resources.

        Called on orchestrator shutdown. Override if you need
        to close connections, release resources, etc.
        """
        pass

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate tool-specific config before instantiation.

        Called by the orchestrator to catch config errors early.

        Args:
            config: Tool-specific configuration dict

        Returns:
            List of error messages. Empty list = valid config.
        """
        return []
