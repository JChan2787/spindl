"""
OpenRouter LLM Provider - Cloud API with SSE streaming support.

Implements the LLMProvider interface for OpenRouter's OpenAI-compatible API,
providing access to 200+ models from multiple providers through a single key.

Features:
- SSE streaming with generate_stream()
- Reasoning token support (unified `reasoning` field across providers)
- Tool/function calling passthrough
- Configurable base URL (for custom deployments)
- Char-ratio token estimation (no bundled tokenizer — model-agnostic)

API Reference: https://openrouter.ai/docs
"""

import json
import logging
import os
import re
from typing import Iterator, Optional

import requests

from ...base import LLMProperties, LLMProvider, LLMResponse, StreamChunk, ToolCall

logger = logging.getLogger(__name__)

# Default OpenRouter API endpoint
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_ENDPOINT = "/chat/completions"

# Environment variable pattern: ${VAR_NAME}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_env_vars(value: str) -> str:
    """
    Replace ${VAR_NAME} patterns with environment variable values.

    Args:
        value: String potentially containing ${VAR_NAME} patterns

    Returns:
        String with patterns replaced by env var values

    Raises:
        ValueError: If referenced env var is not set
    """
    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(
                f"Environment variable '{var_name}' not set. "
                f"Required for OpenRouter API key configuration."
            )
        return env_value

    return ENV_VAR_PATTERN.sub(replace_match, value)


class OpenRouterProvider(LLMProvider):
    """
    LLMProvider implementation for OpenRouter cloud API.

    OpenRouter is an OpenAI-compatible router that provides access to 200+
    models from multiple providers (OpenAI, Anthropic, Google, Meta, etc.)
    through a single API key.

    Properties:
        - Models: 200+ (provider-prefixed, e.g., "openai/gpt-4o")
        - Streaming: Yes (SSE)
        - Context: Model-dependent
        - Tools: Model-dependent

    Config schema (llm.providers.openrouter):
        url: str           - API base URL (default: "https://openrouter.ai/api/v1")
        api_key: str       - API key or ${ENV_VAR} reference (required)
        model: str         - Model ID, provider-prefixed (required)
        timeout: float     - Request timeout in seconds (default: 120.0)
        temperature: float - Default temperature (default: 0.7)
        max_tokens: int    - Default max tokens (default: 256)
        stream: bool       - Enable streaming by default (default: true)
    """

    def __init__(self):
        """Initialize provider (no heavy work - that's in initialize())."""
        self._api_key: str = ""
        self._model: str = ""
        self._base_url: str = DEFAULT_BASE_URL
        self._timeout: float = 120.0
        self._default_temperature: float = 0.7
        self._default_max_tokens: int = 256
        self._stream_by_default: bool = True
        self._context_size: Optional[int] = None
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        """
        Initialize OpenRouter API client.

        Args:
            config: Provider config from llm.providers.openrouter section

        Raises:
            ValueError: Missing or invalid API key
            ConnectionError: API not reachable
        """
        # Base URL (optional, overridable)
        self._base_url = config.get("url", DEFAULT_BASE_URL).rstrip("/")

        # API key (required) - supports ${ENV_VAR} substitution
        api_key_raw = config.get("api_key")
        if not api_key_raw:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set llm.providers.openrouter.api_key in config or use ${OPENROUTER_API_KEY}."
            )

        try:
            self._api_key = resolve_env_vars(str(api_key_raw))
        except ValueError as e:
            raise ValueError(str(e)) from e

        if not self._api_key:
            raise ValueError("OpenRouter API key resolved to empty string.")

        # Model selection (required — no sensible default for 200+ models)
        self._model = config.get("model", "")
        if not self._model:
            raise ValueError(
                "OpenRouter model not configured. "
                "Set llm.providers.openrouter.model (e.g., 'google/gemini-2.5-pro')."
            )

        # Timeouts and defaults
        self._timeout = config.get("timeout", 120.0)
        self._default_temperature = config.get("temperature", 0.7)
        self._default_max_tokens = config.get("max_tokens", 256)
        self._stream_by_default = config.get("stream", True)

        # Context size from user config (NANO-096: user is authoritative for cloud)
        self._context_size = config.get("context_size")
        if self._context_size is None:
            logger.warning(
                "No context_size configured for OpenRouter, defaulting to 8192. "
                "Set via Dashboard or Launcher for accurate budget enforcement."
            )

        # Verify API connectivity with a lightweight check
        if not self._health_check_internal():
            raise ConnectionError(
                "OpenRouter API not reachable. Check your internet connection and API key."
            )

        self._initialized = True
        logger.info(f"OpenRouterProvider initialized: model={self._model}, url={self._base_url}")

    def get_properties(self) -> LLMProperties:
        """Return OpenRouter API properties."""
        # NANO-096: context_length from user config, fallback 8192
        return LLMProperties(
            model_name=self._model,
            supports_streaming=True,
            context_length=self._context_size or 8192,
            supports_tools=True,  # Most models support it; silently dropped if not
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response synchronously (non-streaming).

        Args:
            messages: OpenAI-style message list
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length
            tools: Optional list of tool definitions (OpenAI format)
            **kwargs: Additional options (top_p, stop, response_format, tool_choice)

        Returns:
            LLMResponse with content and token usage

        Raises:
            RuntimeError: Generation failed
            ConnectionError: API unreachable
            TimeoutError: Request timed out
        """
        if not self._initialized:
            raise RuntimeError("OpenRouterProvider not initialized. Call initialize() first.")

        # Use defaults if not overridden
        if temperature == 0.7:
            temperature = self._default_temperature
        if max_tokens == 256:
            max_tokens = self._default_max_tokens

        # Build request payload
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # Tool calling support
        if tools:
            payload["tools"] = tools
            tool_choice = kwargs.pop("tool_choice", "auto")
            payload["tool_choice"] = tool_choice

        # Optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        # Make request
        try:
            response = requests.post(
                f"{self._base_url}{CHAT_ENDPOINT}",
                headers=self._build_headers(),
                json=payload,
                timeout=self._timeout,
            )
            self._check_response_errors(response)
            data = response.json()

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request timed out after {self._timeout}s") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("OpenRouter API unreachable") from e

        # Parse response
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            # OpenRouter uses "reasoning" field (not "reasoning_content" like DeepSeek)
            reasoning = message.get("reasoning")
            finish_reason = choice.get("finish_reason")

            usage = data.get("usage", {})

            # Parse tool calls if present
            tool_calls = self._parse_tool_calls(message.get("tool_calls", []))

            response_obj = LLMResponse(
                content=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning=reasoning,
                reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            )

            if reasoning:
                logger.debug(f"Reasoning content: {reasoning[:200]}...")

            if tool_calls:
                logger.info(f"LLM requested {len(tool_calls)} tool call(s): {[tc.name for tc in tool_calls]}")

            return response_obj

        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format: {data}") from e

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """
        Generate response with SSE streaming.

        Yields chunks as they arrive from the OpenRouter API.
        Handles reasoning tokens (via `reasoning` field in delta).

        Args:
            messages: OpenAI-style message list
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length
            tools: Optional list of tool definitions (OpenAI format)
            **kwargs: Additional options (top_p, stop, response_format, tool_choice)

        Yields:
            StreamChunk objects with content/reasoning as available
        """
        if not self._initialized:
            raise RuntimeError("OpenRouterProvider not initialized. Call initialize() first.")

        # Use defaults if not overridden
        if temperature == 0.7:
            temperature = self._default_temperature
        if max_tokens == 256:
            max_tokens = self._default_max_tokens

        # Build request payload with streaming enabled
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Tool calling support
        if tools:
            payload["tools"] = tools
            tool_choice = kwargs.pop("tool_choice", "auto")
            payload["tool_choice"] = tool_choice

        # Optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        # Make streaming request
        try:
            response = requests.post(
                f"{self._base_url}{CHAT_ENDPOINT}",
                headers=self._build_headers(),
                json=payload,
                timeout=self._timeout,
                stream=True,
            )
            self._check_response_errors(response)

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request timed out after {self._timeout}s") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("OpenRouter API unreachable") from e

        # Parse SSE stream
        yield from self._parse_sse_stream(response)

    def _parse_sse_stream(self, response: requests.Response) -> Iterator[StreamChunk]:
        """
        Parse Server-Sent Events stream from OpenRouter API.

        SSE Format:
            data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
            data: {"id":"...","choices":[{"delta":{"reasoning":"..."}}]}
            data: [DONE]

        OpenRouter may also send keep-alive comments:
            : OPENROUTER PROCESSING

        Args:
            response: Streaming requests.Response object

        Yields:
            StreamChunk objects parsed from SSE events
        """
        # Accumulate tool calls across chunks (they come in pieces)
        accumulated_tool_calls: dict[int, dict] = {}

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # Skip SSE comments (keep-alive signals like ": OPENROUTER PROCESSING")
            if line_str.startswith(":"):
                continue

            # Must be a data line
            if not line_str.startswith("data: "):
                continue

            json_str = line_str[6:]  # Strip "data: " prefix

            # End of stream marker
            if json_str == "[DONE]":
                break

            try:
                chunk_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse SSE chunk: {e}")
                continue

            # Extract delta content
            choices = chunk_data.get("choices", [])
            if not choices:
                # Might be a usage-only chunk at the end
                usage = chunk_data.get("usage")
                if usage:
                    final_tool_calls = self._build_tool_calls_from_accumulated(accumulated_tool_calls)
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                        tool_calls=final_tool_calls,
                    )
                continue

            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason")

            content = delta.get("content", "")
            # OpenRouter uses "reasoning" in delta (not "reasoning_content")
            reasoning = delta.get("reasoning")

            # Handle tool calls in delta (they come incrementally)
            delta_tool_calls = delta.get("tool_calls", [])
            for tc in delta_tool_calls:
                idx = tc.get("index", 0)
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": "",
                        "arguments": "",
                    }

                if tc.get("id"):
                    accumulated_tool_calls[idx]["id"] = tc["id"]

                if "function" in tc:
                    func = tc["function"]
                    if func.get("name"):
                        accumulated_tool_calls[idx]["name"] = func["name"]
                    if func.get("arguments"):
                        accumulated_tool_calls[idx]["arguments"] += func["arguments"]

            # Check for usage in chunk
            usage = chunk_data.get("usage")

            # Determine if this is the final chunk
            is_final = finish_reason is not None

            # Build tool calls for final chunk
            tool_calls = []
            if is_final and accumulated_tool_calls:
                tool_calls = self._build_tool_calls_from_accumulated(accumulated_tool_calls)

            # Build chunk
            chunk = StreamChunk(
                content=content or "",
                reasoning=reasoning,
                is_final=is_final,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )

            # Add usage if present (typically final chunk)
            if usage:
                chunk = StreamChunk(
                    content=content or "",
                    reasoning=reasoning,
                    is_final=True,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )

            # Only yield if there's actual content, tool calls, or it's final
            if content or reasoning or delta_tool_calls or is_final:
                yield chunk

    def _parse_tool_calls(self, raw_tool_calls: list) -> list[ToolCall]:
        """Parse tool calls from a non-streaming response message."""
        tool_calls = []
        for tc in raw_tool_calls:
            try:
                args_str = tc.get("function", {}).get("arguments", "{}")
                args = json.loads(args_str) if args_str else {}

                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=args,
                ))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call arguments: {e}")
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments={},
                ))
        return tool_calls

    def _build_tool_calls_from_accumulated(self, accumulated: dict[int, dict]) -> list[ToolCall]:
        """Build ToolCall objects from accumulated streaming data."""
        tool_calls = []
        for idx in sorted(accumulated.keys()):
            tc_data = accumulated[idx]
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse accumulated tool arguments: {tc_data['arguments']}")
                args = {}

            tool_calls.append(ToolCall(
                id=tc_data["id"],
                name=tc_data["name"],
                arguments=args,
            ))
        return tool_calls

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count using character ratio.

        OpenRouter serves 200+ models from different providers — no single
        bundled tokenizer is accurate. We use a 4-chars-per-token estimate,
        which is reasonable for English text across most tokenizers.

        This is used by BudgetEnforcer for FIFO eviction — directional
        accuracy is sufficient, precision is not critical.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        if not self._initialized:
            raise RuntimeError("OpenRouterProvider not initialized. Call initialize() first.")

        # 4 characters per token is a reasonable average across tokenizers
        return max(1, len(text) // 4)

    def health_check(self) -> bool:
        """
        Check if OpenRouter API is reachable.

        Returns:
            True if API accepts requests, False otherwise
        """
        return self._health_check_internal()

    def _health_check_internal(self) -> bool:
        """Internal health check that works before full initialization."""
        try:
            response = requests.get(
                f"{self._base_url}/models",
                headers=self._build_headers(),
                timeout=10.0,
            )
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def _build_headers(self) -> dict:
        """Build HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/spindl-ai/spindl",
            "X-Title": "SpindL",
            "Content-Type": "application/json",
        }

    def _check_response_errors(self, response: requests.Response) -> None:
        """
        Check response for API errors and raise appropriate exceptions.

        Args:
            response: requests.Response object

        Raises:
            ValueError: Authentication or parameter errors (4xx)
            RuntimeError: Server errors (5xx)
        """
        if response.status_code == 200:
            return

        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if response.status_code == 401:
            raise ValueError(f"Invalid API key: {error_msg}")
        elif response.status_code == 402:
            raise ValueError(f"Insufficient credits: {error_msg}")
        elif response.status_code == 408:
            raise TimeoutError(f"Upstream timeout: {error_msg}")
        elif response.status_code == 422:
            raise ValueError(f"Invalid parameters: {error_msg}")
        elif response.status_code == 429:
            raise RuntimeError(f"Rate limited: {error_msg}")
        elif response.status_code == 502:
            raise RuntimeError(f"Upstream provider error: {error_msg}")
        elif response.status_code >= 500:
            raise RuntimeError(f"OpenRouter server error ({response.status_code}): {error_msg}")
        else:
            raise RuntimeError(f"API error ({response.status_code}): {error_msg}")

    def shutdown(self) -> None:
        """Cleanup provider resources."""
        self._initialized = False
        self._api_key = ""  # Clear sensitive data
        logger.debug("OpenRouterProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate OpenRouter provider config.

        Required fields: api_key, model
        Optional fields: url, timeout, temperature, max_tokens, stream

        Args:
            config: Provider config dict

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # API key validation (required)
        api_key = config.get("api_key")
        if not api_key:
            errors.append("Missing required field: api_key")
        elif not isinstance(api_key, str):
            errors.append(f"api_key must be a string, got {type(api_key).__name__}")

        # Model validation (required — no sensible default for 200+ models)
        model = config.get("model")
        if not model:
            errors.append("Missing required field: model (e.g., 'google/gemini-2.5-pro')")
        elif not isinstance(model, str):
            errors.append(f"model must be a string, got {type(model).__name__}")

        # URL validation (optional)
        url = config.get("url")
        if url is not None and not isinstance(url, str):
            errors.append(f"url must be a string, got {type(url).__name__}")

        # Timeout validation
        timeout = config.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                errors.append(f"timeout must be a number, got {type(timeout).__name__}")
            elif timeout <= 0:
                errors.append(f"timeout must be positive, got {timeout}")

        # Temperature validation
        temperature = config.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                errors.append(f"temperature must be a number, got {type(temperature).__name__}")
            elif not (0.0 <= temperature <= 2.0):
                errors.append(f"temperature must be between 0.0 and 2.0, got {temperature}")

        # max_tokens validation
        max_tokens = config.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                errors.append(f"max_tokens must be an integer, got {type(max_tokens).__name__}")
            elif max_tokens < 1:
                errors.append(f"max_tokens must be at least 1, got {max_tokens}")

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        OpenRouter is a cloud API - no server to start.

        Returns:
            None (cloud API, no local server)
        """
        return None

    @classmethod
    def is_cloud_provider(cls) -> bool:
        """OpenRouter is a cloud API provider."""
        return True
