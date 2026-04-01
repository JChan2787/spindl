"""
DeepSeek LLM Provider - Cloud API with SSE streaming support.

Implements the LLMProvider interface for DeepSeek's OpenAI-compatible API,
with proper Server-Sent Events (SSE) streaming for real-time response chunks.

Features:
- SSE streaming with generate_stream()
- Support for deepseek-reasoner thinking mode (reasoning_content)
- KV cache metrics (prompt_cache_hit_tokens, prompt_cache_miss_tokens)
- Environment variable substitution for API keys (${VAR_NAME} syntax)

API Reference: https://api-docs.deepseek.com/
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterator, Optional

import requests

from ...base import LLMProperties, LLMProvider, LLMResponse, StreamChunk, ToolCall

# Lazy-loaded tokenizer (transformers can be slow to import)
_tokenizer = None

logger = logging.getLogger(__name__)

# DeepSeek API endpoints
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_ENDPOINT = "/chat/completions"

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
                f"Required for DeepSeek API key configuration."
            )
        return env_value

    return ENV_VAR_PATTERN.sub(replace_match, value)


class DeepSeekProvider(LLMProvider):
    """
    LLMProvider implementation for DeepSeek cloud API.

    DeepSeek is a cloud-based provider with an OpenAI-compatible API.
    Supports SSE streaming and the deepseek-reasoner model with
    chain-of-thought reasoning.

    Properties:
        - Models: deepseek-chat, deepseek-reasoner
        - Streaming: Yes (SSE)
        - Context: 128K tokens
        - Max output: 8K (chat) / 64K (reasoner)

    Config schema (llm.providers.deepseek):
        api_key: str       - API key or ${ENV_VAR} reference (required)
        model: str         - Model ID (default: "deepseek-chat")
        timeout: float     - Request timeout in seconds (default: 60.0)
        temperature: float - Default temperature (default: 0.7)
        max_tokens: int    - Default max tokens (default: 256)
        stream: bool       - Enable streaming by default (default: true)
    """

    def __init__(self):
        """Initialize provider (no heavy work - that's in initialize())."""
        self._api_key: str = ""
        self._model: str = "deepseek-chat"
        self._timeout: float = 60.0
        self._default_temperature: float = 0.7
        self._default_max_tokens: int = 256
        self._stream_by_default: bool = True
        self._context_size: Optional[int] = None
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        """
        Initialize DeepSeek API client.

        Args:
            config: Provider config from llm.providers.deepseek section

        Raises:
            ValueError: Missing or invalid API key
            ConnectionError: API not reachable
        """
        # API key (required) - supports ${ENV_VAR} substitution
        api_key_raw = config.get("api_key")
        if not api_key_raw:
            raise ValueError(
                "DeepSeek API key not configured. "
                "Set llm.providers.deepseek.api_key in config or use ${DEEPSEEK_API_KEY}."
            )

        try:
            self._api_key = resolve_env_vars(str(api_key_raw))
        except ValueError as e:
            raise ValueError(str(e)) from e

        if not self._api_key:
            raise ValueError("DeepSeek API key resolved to empty string.")

        # Model selection
        self._model = config.get("model", "deepseek-chat")

        # Timeouts and defaults
        self._timeout = config.get("timeout", 60.0)
        self._default_temperature = config.get("temperature", 0.7)
        self._default_max_tokens = config.get("max_tokens", 256)
        self._stream_by_default = config.get("stream", True)

        # Context size from user config (NANO-096: user is authoritative for cloud)
        self._context_size = config.get("context_size")
        if self._context_size is None:
            logger.info("No context_size configured for DeepSeek, defaulting to 128000")

        # Verify API connectivity with a lightweight check
        if not self._health_check_internal():
            raise ConnectionError(
                "DeepSeek API not reachable. Check your internet connection and API key."
            )

        self._initialized = True
        logger.info(f"DeepSeekProvider initialized: model={self._model}")

    def get_properties(self) -> LLMProperties:
        """Return DeepSeek API properties."""
        # NANO-096: context_length from user config, fallback 128K (all current models)
        return LLMProperties(
            model_name=self._model,
            supports_streaming=True,
            context_length=self._context_size or 128000,
            supports_tools=self._model == "deepseek-chat",  # reasoner doesn't support tools
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
            **kwargs: Additional options
                - top_p: Nucleus sampling threshold
                - stop: Stop sequences
                - response_format: {"type": "json_object"} for JSON mode
                - tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}

        Returns:
            LLMResponse with content and token usage (and tool_calls if applicable)

        Raises:
            RuntimeError: Generation failed
            ConnectionError: API unreachable
            TimeoutError: Request timed out
        """
        if not self._initialized:
            raise RuntimeError("DeepSeekProvider not initialized. Call initialize() first.")

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
            # Default to auto tool choice unless specified
            tool_choice = kwargs.pop("tool_choice", "auto")
            payload["tool_choice"] = tool_choice

        # Optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]
        if "frequency_penalty" in kwargs and kwargs["frequency_penalty"] != 0.0:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs and kwargs["presence_penalty"] != 0.0:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        # Make request
        try:
            response = requests.post(
                f"{DEEPSEEK_BASE_URL}{DEEPSEEK_CHAT_ENDPOINT}",
                headers=self._build_headers(),
                json=payload,
                timeout=self._timeout,
            )
            self._check_response_errors(response)
            data = response.json()

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request timed out after {self._timeout}s") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("DeepSeek API unreachable") from e

        # Parse response
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content")
            finish_reason = choice.get("finish_reason")

            usage = data.get("usage", {})

            # Parse tool calls if present
            tool_calls = []
            raw_tool_calls = message.get("tool_calls", [])
            for tc in raw_tool_calls:
                try:
                    # Parse the arguments JSON string
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    args = json.loads(args_str) if args_str else {}

                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=args,
                    ))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool call arguments: {e}")
                    # Still include the tool call with empty args
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments={},
                    ))

            response_obj = LLMResponse(
                content=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning=reasoning,
                reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            )

            # Log reasoning if present (deepseek-reasoner)
            if reasoning:
                logger.debug(f"Reasoning content: {reasoning[:200]}...")

            # Log tool calls if present
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

        Yields chunks as they arrive from the DeepSeek API.
        Handles both content and reasoning_content (for deepseek-reasoner).

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
            raise RuntimeError("DeepSeekProvider not initialized. Call initialize() first.")

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
            "stream_options": {"include_usage": True},  # Get token counts in stream
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
                f"{DEEPSEEK_BASE_URL}{DEEPSEEK_CHAT_ENDPOINT}",
                headers=self._build_headers(),
                json=payload,
                timeout=self._timeout,
                stream=True,
            )
            self._check_response_errors(response)

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request timed out after {self._timeout}s") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("DeepSeek API unreachable") from e

        # Parse SSE stream
        yield from self._parse_sse_stream(response)

    def _parse_sse_stream(self, response: requests.Response) -> Iterator[StreamChunk]:
        """
        Parse Server-Sent Events stream from DeepSeek API.

        SSE Format:
            data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
            data: {"id":"...","choices":[{"delta":{"reasoning_content":"..."}}]}
            data: {"id":"...","choices":[{"delta":{"tool_calls":[...]}}]}
            data: [DONE]

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

            # Skip SSE comments (keep-alive signals)
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
                    # Build final tool calls from accumulated data
                    final_tool_calls = self._build_tool_calls_from_accumulated(accumulated_tool_calls)
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
                        cache_miss_tokens=usage.get("prompt_cache_miss_tokens"),
                        reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                        tool_calls=final_tool_calls,
                    )
                continue

            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason")

            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content")

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

                # Accumulate ID (usually in first chunk)
                if tc.get("id"):
                    accumulated_tool_calls[idx]["id"] = tc["id"]

                # Accumulate function name and arguments
                if "function" in tc:
                    func = tc["function"]
                    if func.get("name"):
                        accumulated_tool_calls[idx]["name"] = func["name"]
                    if func.get("arguments"):
                        accumulated_tool_calls[idx]["arguments"] += func["arguments"]

            # Check for usage in chunk (with stream_options.include_usage)
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
                    cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
                    cache_miss_tokens=usage.get("prompt_cache_miss_tokens"),
                    reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )

            # Only yield if there's actual content, tool calls, or it's final
            if content or reasoning or delta_tool_calls or is_final:
                yield chunk

    def _build_tool_calls_from_accumulated(self, accumulated: dict[int, dict]) -> list[ToolCall]:
        """
        Build ToolCall objects from accumulated streaming data.

        Args:
            accumulated: Dict mapping index -> {id, name, arguments}

        Returns:
            List of ToolCall objects
        """
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
        Count tokens using local DeepSeek V3 tokenizer.

        Uses bundled tokenizer files (tokenizer.json, tokenizer_config.json)
        from the tokenizer/ subdirectory. Tokenizer is lazy-loaded on first use.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens

        Raises:
            RuntimeError: Provider not initialized or tokenizer load failed
        """
        if not self._initialized:
            raise RuntimeError("DeepSeekProvider not initialized. Call initialize() first.")

        global _tokenizer
        if _tokenizer is None:
            try:
                from transformers import AutoTokenizer
                tokenizer_path = Path(__file__).parent / "tokenizer"
                _tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path), trust_remote_code=True
                )
                logger.debug(f"Loaded DeepSeek tokenizer from {tokenizer_path}")
            except ImportError as e:
                raise RuntimeError(
                    "transformers package required for DeepSeek tokenization. "
                    "Install with: pip install transformers"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load DeepSeek tokenizer: {e}") from e

        return len(_tokenizer.encode(text))

    def health_check(self) -> bool:
        """
        Check if DeepSeek API is reachable.

        Returns:
            True if API accepts requests, False otherwise
        """
        return self._health_check_internal()

    def _health_check_internal(self) -> bool:
        """Internal health check that works before full initialization."""
        try:
            # Use models endpoint as lightweight check
            response = requests.get(
                f"{DEEPSEEK_BASE_URL}/models",
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
            raise ValueError(f"Insufficient balance: {error_msg}")
        elif response.status_code == 422:
            raise ValueError(f"Invalid parameters: {error_msg}")
        elif response.status_code == 429:
            raise RuntimeError(f"Rate limited: {error_msg}")
        elif response.status_code >= 500:
            raise RuntimeError(f"DeepSeek server error ({response.status_code}): {error_msg}")
        else:
            raise RuntimeError(f"API error ({response.status_code}): {error_msg}")

    def shutdown(self) -> None:
        """Cleanup provider resources."""
        self._initialized = False
        self._api_key = ""  # Clear sensitive data
        logger.debug("DeepSeekProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate DeepSeek provider config.

        Required fields: api_key
        Optional fields: model, timeout, temperature, max_tokens, stream

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

        # Model validation
        model = config.get("model")
        if model is not None:
            valid_models = ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]
            if model not in valid_models:
                errors.append(
                    f"Unknown model '{model}'. Valid models: {', '.join(valid_models)}"
                )

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
        DeepSeek is a cloud API - no server to start.

        Returns:
            None (cloud API, no local server)
        """
        return None

    @classmethod
    def is_cloud_provider(cls) -> bool:
        """DeepSeek is a cloud API provider."""
        return True
