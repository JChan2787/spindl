"""
LLM Provider Base Classes - Abstract base for all LLM implementations.

This module defines the protocol that all LLM providers must implement,
enabling swappable LLM backends via configuration.

Includes support for native function/tool calling for providers that support it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


# =============================================================================
# Tool Calling Data Classes
# =============================================================================


@dataclass
class ToolCall:
    """
    Represents a tool call requested by the LLM.

    When the LLM decides to use a tool, it returns one or more ToolCall
    objects indicating which tool to invoke and with what arguments.
    """

    id: str
    """Unique identifier for this tool call (used to match results)."""

    name: str
    """Name of the tool to invoke (e.g., "screen_vision")."""

    arguments: dict[str, Any]
    """Arguments to pass to the tool, parsed from JSON."""


@dataclass
class ToolResult:
    """
    Result of executing a tool, to be fed back to the LLM.

    After executing a tool, wrap the result in this class and include
    it in the next LLM call so the model can incorporate the tool output.
    """

    tool_call_id: str
    """ID of the ToolCall this result corresponds to."""

    content: str
    """The tool's output (string representation)."""

    is_error: bool = False
    """True if the tool execution failed."""


# =============================================================================
# LLM Response Data Classes
# =============================================================================


@dataclass
class LLMProperties:
    """
    Provider self-describes its capabilities.

    Used by the orchestrator to understand provider features
    without hardcoding assumptions about any specific provider.
    """

    model_name: str
    """Name/ID of the model being used (e.g., "deepseek-chat", "llama-3.2-3b")."""

    supports_streaming: bool
    """True if generate_stream() yields multiple chunks."""

    context_length: Optional[int] = None
    """Maximum context window in tokens, if known."""

    supports_tools: bool = False
    """True if provider supports native function/tool calling."""

    supports_tool_role: bool = True
    """True if model's chat template handles role: 'tool' messages.
    False for models with strict user/assistant alternation (e.g., gemma3).
    When False, tool results are sent as role: 'user' instead."""


@dataclass
class LLMResponse:
    """
    Response from a synchronous LLM generation request.

    Contains both the generated content and token usage statistics.
    May also contain tool calls if the LLM decided to invoke tools.
    """

    content: str
    """The generated text response (may be empty if tool_calls present)."""

    input_tokens: int
    """Number of tokens in the prompt."""

    output_tokens: int
    """Number of tokens in the response."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """Tool calls requested by the LLM (empty if no tools invoked)."""

    finish_reason: Optional[str] = None
    """Why generation stopped: "stop", "tool_calls", "length", etc."""

    reasoning: Optional[str] = None
    """Thinking/reasoning content from models like Qwen3, DeepSeek R1 (NANO-042)."""

    reasoning_tokens: Optional[int] = None
    """Token count for reasoning/thinking content, if reported by provider (NANO-042)."""


@dataclass
class StreamChunk:
    """
    A single chunk from a streaming LLM response.

    For streaming providers, multiple StreamChunks are yielded with
    is_final=False until the last chunk.
    """

    content: str
    """Text content in this chunk (may be empty for metadata-only chunks)."""

    reasoning: Optional[str] = None
    """Reasoning/thinking content for models like deepseek-reasoner."""

    is_final: bool = False
    """True for the last chunk in the stream."""

    input_tokens: Optional[int] = None
    """Token count for prompt (typically only in final chunk)."""

    output_tokens: Optional[int] = None
    """Token count for response (typically only in final chunk)."""

    # Cache metrics (DeepSeek KV cache)
    cache_hit_tokens: Optional[int] = None
    """Tokens retrieved from cache (DeepSeek: prompt_cache_hit_tokens)."""

    cache_miss_tokens: Optional[int] = None
    """Tokens not in cache (DeepSeek: prompt_cache_miss_tokens)."""

    reasoning_tokens: Optional[int] = None
    """Token count for reasoning/thinking content (deepseek-reasoner)."""

    # Tool calling support
    tool_calls: list[ToolCall] = field(default_factory=list)
    """Tool calls in this chunk (accumulated across chunks, final in last)."""

    finish_reason: Optional[str] = None
    """Why generation stopped (typically in final chunk)."""


class LLMProvider(ABC):
    """
    Protocol all LLM plugins must implement.

    This abstract base class defines the contract between the orchestrator
    and LLM backends. Providers can be server-based (like llama.cpp) or
    cloud APIs (like DeepSeek).

    Lifecycle:
        1. Instantiation (no heavy work here)
        2. initialize(config) - establish connections, set up auth
        3. get_properties() - orchestrator queries capabilities
        4. generate() / generate_stream() - actual LLM work
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
                   (the section under llm.providers.<provider_name>)

        Raises:
            ConnectionError: For server-based providers that can't connect
            ValueError: For invalid configuration (missing API key, etc.)
            RuntimeError: For initialization failures
        """
        pass

    @abstractmethod
    def get_properties(self) -> LLMProperties:
        """
        Return provider capabilities.

        Called after initialize(). The orchestrator uses this to understand
        provider features without hardcoding assumptions.

        Returns:
            LLMProperties describing this provider's capabilities
        """
        pass

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response synchronously. Blocks until complete.

        Args:
            messages: OpenAI-style message list
                      [{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length
            tools: Optional list of tool definitions (OpenAI format):
                   [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
                   If provided and LLM decides to use a tool, response.tool_calls will be populated.
            **kwargs: Provider-specific options (top_p, stop, tool_choice, etc.)

        Returns:
            Complete response as LLMResponse (check tool_calls if finish_reason == "tool_calls")

        Raises:
            RuntimeError: Generation failed
            ConnectionError: Server/API unreachable
            TimeoutError: Request timed out
        """
        pass

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """
        Generate a response with streaming. Yields chunks as available.

        Default implementation: yield single complete result.
        Streaming providers override to yield multiple chunks with
        is_final=False until the last chunk.

        Args:
            messages: OpenAI-style message list
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length
            tools: Optional list of tool definitions (OpenAI format)
            **kwargs: Provider-specific options

        Yields:
            StreamChunk objects (is_final=False until last chunk)
        """
        response = self.generate(messages, temperature, max_tokens, tools=tools, **kwargs)
        yield StreamChunk(
            content=response.content,
            reasoning=response.reasoning,
            is_final=True,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
        )

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text for budget enforcement.

        Used by plugins like BudgetEnforcer and SummarizationTrigger to
        measure conversation context usage against limits.

        Implementation varies by provider:
        - Server-based (llama.cpp): HTTP call to /tokenize endpoint
        - Cloud APIs (DeepSeek): Local tokenizer with bundled vocab files

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens in the text

        Raises:
            RuntimeError: Provider not initialized or tokenization failed
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if provider is operational.

        For server-based: verify TCP/HTTP connection
        For cloud APIs: verify API is reachable (lightweight check)

        Returns:
            True if ready to generate, False otherwise
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
    def is_cloud_provider(cls) -> bool:
        """
        Return True if this is a cloud/API provider that needs no local server.

        Used by the launcher to decide whether to skip service startup entirely.
        Cloud providers (DeepSeek, OpenAI, Anthropic) return True.
        Local providers (llama.cpp, Ollama) return False.

        Default is False (assumes local server needed).
        """
        return False
