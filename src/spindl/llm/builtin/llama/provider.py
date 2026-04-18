"""
Llama LLM Provider - LLMProvider implementation for llama.cpp server.

Wraps the existing HTTP client logic with the standardized LLMProvider interface,
enabling swappable LLM backends via configuration.

Features:
- OpenAI-compatible chat completions API
- SSE streaming with cancel-on-disconnect (NANO-111)
- Native tool/function calling support (llama.cpp >= b2000)
- Token counting via /tokenize endpoint
"""

import json
import logging
import time
from typing import Iterator, Optional

import requests

from ...base import LLMProperties, LLMProvider, LLMResponse, StreamChunk, ToolCall

logger = logging.getLogger(__name__)


class LlamaProvider(LLMProvider):
    """
    LLMProvider implementation for llama.cpp server.

    llama.cpp is a server-based provider: the model runs in a separate process
    and this provider communicates via HTTP (OpenAI-compatible API).

    Properties:
        - Model: Depends on what's loaded in llama.cpp
        - Streaming: SSE via /v1/chat/completions with stream=true (NANO-111)

    Config schema (llm.providers.llama):
        url: str          - Server URL (default: "http://127.0.0.1:5557")
        timeout: float    - Request timeout in seconds (default: 30.0)
        max_retries: int  - Connection retry attempts (default: 3)
        retry_delay: float - Seconds between retries (default: 1.0)
        temperature: float - Default temperature (default: 0.7)
        max_tokens: int   - Default max tokens (default: 256)
        top_p: float      - Default top_p (default: 0.95)
        mmproj_path: str  - Multimodal projector GGUF path for unified vision (optional)
    """

    def __init__(self):
        """Initialize provider (no heavy work - that's in initialize())."""
        self._base_url: str = "http://127.0.0.1:5557"
        self._timeout: float = 30.0
        self._max_retries: int = 3
        self._retry_delay: float = 1.0
        self._default_temperature: float = 0.7
        self._default_max_tokens: int = 256
        self._default_top_p: float = 0.95
        self._default_top_k: int = 40
        self._default_min_p: float = 0.05
        self._default_repeat_penalty: float = 1.1
        self._default_repeat_last_n: int = 64
        self._default_frequency_penalty: float = 0.0
        self._default_presence_penalty: float = 0.0
        self._model_name: str = "llama.cpp"
        self._context_length: Optional[int] = None
        self._unified_vision: bool = False  # NANO-087: unified vision slot pinning
        self._model_architecture: Optional[str] = None  # NANO-087e: from /props
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        """
        Initialize connection to llama.cpp server.

        Args:
            config: Provider config from llm.providers.llama section

        Raises:
            ConnectionError: Server not reachable
            RuntimeError: Initialization failed
        """
        # Extract config values with defaults
        self._base_url = config.get("url", "http://127.0.0.1:5557").rstrip("/")
        self._timeout = config.get("timeout", 30.0)
        self._max_retries = config.get("max_retries", 3)
        self._retry_delay = config.get("retry_delay", 1.0)
        self._default_temperature = config.get("temperature", 0.7)
        self._default_max_tokens = config.get("max_tokens", 256)
        self._default_top_p = config.get("top_p", 0.95)
        self._default_top_k = config.get("top_k", 40)
        self._default_min_p = config.get("min_p", 0.05)
        self._default_repeat_penalty = config.get("repeat_penalty", 1.1)
        self._default_repeat_last_n = config.get("repeat_last_n", 64)
        self._default_frequency_penalty = config.get("frequency_penalty", 0.0)
        self._default_presence_penalty = config.get("presence_penalty", 0.0)
        self._unified_vision = config.get("unified_vision", False)  # NANO-087

        # Verify connection and get model info
        if not self._health_check_internal():
            raise ConnectionError(
                f"llama.cpp server not available at {self._base_url}. "
                "Ensure the server is running."
            )

        # Try to get context length and model info from server
        try:
            props = self._get_model_info()
            self._context_length = props.get("default_generation_settings", {}).get("n_ctx")
            # Try to get model name
            model_path = props.get("default_generation_settings", {}).get("model")
            if model_path:
                # Extract just the filename
                self._model_name = model_path.split("/")[-1].split("\\")[-1]

            # NANO-087e: detect model architecture for capability flags
            self._model_architecture = self._detect_architecture(
                self._model_name, props
            )
            if self._model_architecture:
                logger.info(f"Detected model architecture: {self._model_architecture}")
        except Exception as e:
            logger.warning(f"Could not get model info from llama.cpp: {e}")

        self._initialized = True
        logger.info(f"LlamaProvider initialized: {self._base_url} (model: {self._model_name})")

    # NANO-087e: architectures whose chat templates reject role: "tool"
    _STRICT_ALTERNATION_ARCHITECTURES = frozenset({
        "gemma3",
    })

    @staticmethod
    def _detect_architecture(model_name: str, props: dict) -> Optional[str]:
        """
        Detect model architecture from /props response.

        Checks the chat_template field for architecture markers first,
        then falls back to model filename pattern matching.

        Args:
            model_name: Extracted model filename
            props: Full /props response dict

        Returns:
            Architecture string (e.g., "gemma3") or None if unknown
        """
        # Check chat template for known markers
        chat_template = props.get("chat_template", "")
        if "<start_of_turn>" in chat_template:
            return "gemma3"

        # Fallback: pattern match on model filename
        name_lower = model_name.lower()
        if "gemma-3" in name_lower or "gemma3" in name_lower:
            return "gemma3"

        return None

    def get_properties(self) -> LLMProperties:
        """Return llama.cpp server properties."""
        supports_tool_role = (
            self._model_architecture not in self._STRICT_ALTERNATION_ARCHITECTURES
            if self._model_architecture
            else False  # Unknown architecture — safe default
        )

        return LLMProperties(
            model_name=self._model_name,
            supports_streaming=True,  # NANO-111: SSE streaming
            context_length=self._context_length,
            supports_tools=True,  # llama.cpp supports OpenAI-compatible tool calling
            supports_tool_role=supports_tool_role,
            supports_role_history=True,  # NANO-114: llama.cpp with --jinja wraps role-array history in real template delimiters
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
        Generate response via llama.cpp server.

        Args:
            messages: OpenAI-style message list
            temperature: Sampling temperature (uses default if not specified)
            max_tokens: Maximum response length (uses default if not specified)
            tools: Optional list of tool definitions (OpenAI format)
            **kwargs: Additional options
                - top_p: Nucleus sampling threshold
                - stop: Stop sequences
                - tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}

        Returns:
            LLMResponse with content and token usage (and tool_calls if applicable)

        Raises:
            RuntimeError: Generation failed
            ConnectionError: Server unreachable
            TimeoutError: Request timed out
        """
        if not self._initialized:
            raise RuntimeError("LlamaProvider not initialized. Call initialize() first.")

        endpoint = f"{self._base_url}/v1/chat/completions"

        # Use defaults if not specified
        if temperature == 0.7:  # default arg
            temperature = self._default_temperature
        if max_tokens == 256:  # default arg
            max_tokens = self._default_max_tokens

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", self._default_top_p),
            "top_k": kwargs.get("top_k", self._default_top_k),
            "min_p": kwargs.get("min_p", self._default_min_p),
            "repeat_penalty": kwargs.get("repeat_penalty", self._default_repeat_penalty),
            "repeat_last_n": kwargs.get("repeat_last_n", self._default_repeat_last_n),
            "frequency_penalty": kwargs.get("frequency_penalty", self._default_frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self._default_presence_penalty),
        }

        # Tool calling support
        if tools:
            payload["tools"] = tools
            # Default to auto tool choice unless specified
            tool_choice = kwargs.pop("tool_choice", "auto")
            payload["tool_choice"] = tool_choice

        stop = kwargs.get("stop")
        if stop:
            payload["stop"] = stop

        # NANO-087: pin chat to slot 0 when sharing server with vision (slot 1)
        if self._unified_vision:
            payload["id_slot"] = 0

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                break

            except requests.exceptions.Timeout as e:
                raise TimeoutError(
                    f"Request timed out after {self._timeout}s"
                ) from e

            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
                continue

            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"Server returned error: {e.response.status_code} - {e.response.text}"
                ) from e

        else:
            raise ConnectionError(
                f"Failed to connect to LLM server at {self._base_url} "
                f"after {self._max_retries} attempts: {last_error}"
            )

        # Parse response
        data = response.json()

        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content")  # NANO-042: llama.cpp --reasoning-format deepseek
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

            # Log tool calls if present
            if tool_calls:
                logger.info(f"LLM requested {len(tool_calls)} tool call(s): {[tc.name for tc in tool_calls]}")

            return LLMResponse(
                content=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning=reasoning,
                reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            )
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected response format: {data}"
            ) from e

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """
        Generate response with SSE streaming (NANO-111).

        Yields StreamChunk objects as tokens arrive from llama.cpp.
        Closing the response (e.g., on barge-in) cancels server-side
        generation — llama.cpp honors client disconnect since PR #11418.

        Args:
            messages: OpenAI-style message list
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length
            tools: Optional list of tool definitions (OpenAI format)
            **kwargs: Additional options (top_p, stop, tool_choice)

        Yields:
            StreamChunk objects (is_final=False until last chunk)
        """
        if not self._initialized:
            raise RuntimeError("LlamaProvider not initialized. Call initialize() first.")

        endpoint = f"{self._base_url}/v1/chat/completions"

        # Use defaults if not specified
        if temperature == 0.7:
            temperature = self._default_temperature
        if max_tokens == 256:
            max_tokens = self._default_max_tokens

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", self._default_top_p),
            "top_k": kwargs.get("top_k", self._default_top_k),
            "min_p": kwargs.get("min_p", self._default_min_p),
            "repeat_penalty": kwargs.get("repeat_penalty", self._default_repeat_penalty),
            "repeat_last_n": kwargs.get("repeat_last_n", self._default_repeat_last_n),
            "frequency_penalty": kwargs.get("frequency_penalty", self._default_frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self._default_presence_penalty),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Tool calling support
        if tools:
            payload["tools"] = tools
            tool_choice = kwargs.pop("tool_choice", "auto")
            payload["tool_choice"] = tool_choice

        stop = kwargs.get("stop")
        if stop:
            payload["stop"] = stop

        # NANO-087: pin chat to slot 0 when sharing server with vision (slot 1)
        if self._unified_vision:
            payload["id_slot"] = 0

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self._timeout,
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request timed out after {self._timeout}s") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to LLM server at {self._base_url}: {e}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Server returned error: {e.response.status_code} - {e.response.text}"
            ) from e

        # Parse SSE stream — closing `response` cancels server-side generation
        yield from self._parse_sse_stream(response)

    def _parse_sse_stream(self, response: requests.Response) -> Iterator[StreamChunk]:
        """
        Parse Server-Sent Events stream from llama.cpp.

        SSE format (OpenAI-compatible):
            data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
            data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}
            data: [DONE]

        Args:
            response: Streaming requests.Response object

        Yields:
            StreamChunk objects parsed from SSE events
        """
        accumulated_tool_calls: dict[int, dict] = {}

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # Skip SSE comments (keep-alive)
            if line_str.startswith(":"):
                continue

            if not line_str.startswith("data: "):
                continue

            json_str = line_str[6:]  # Strip "data: " prefix

            # End of stream
            if json_str == "[DONE]":
                break

            try:
                chunk_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse SSE chunk: {e}")
                continue

            choices = chunk_data.get("choices", [])
            usage = chunk_data.get("usage")

            if not choices:
                # Usage-only chunk at stream end
                if usage:
                    final_tool_calls = self._build_tool_calls(accumulated_tool_calls)
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
            reasoning = delta.get("reasoning_content")  # llama.cpp --reasoning-format deepseek

            # Accumulate tool calls across chunks
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.get("id"):
                    accumulated_tool_calls[idx]["id"] = tc["id"]
                if "function" in tc:
                    func = tc["function"]
                    if func.get("name"):
                        accumulated_tool_calls[idx]["name"] = func["name"]
                    if func.get("arguments"):
                        accumulated_tool_calls[idx]["arguments"] += func["arguments"]

            is_final = finish_reason is not None

            tool_calls = []
            if is_final and accumulated_tool_calls:
                tool_calls = self._build_tool_calls(accumulated_tool_calls)

            chunk = StreamChunk(
                content=content or "",
                reasoning=reasoning,
                is_final=is_final,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )

            # Attach usage if present (typically final chunk)
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

            if content or reasoning or delta.get("tool_calls") or is_final:
                yield chunk

    @staticmethod
    def _build_tool_calls(accumulated: dict[int, dict]) -> list[ToolCall]:
        """Build ToolCall list from accumulated streaming tool call fragments."""
        tool_calls = []
        for _idx, tc in sorted(accumulated.items()):
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))
        return tool_calls

    def count_tokens(self, text: str) -> int:
        """
        Count tokens via llama.cpp /tokenize endpoint.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens

        Raises:
            RuntimeError: Provider not initialized or tokenization failed
        """
        if not self._initialized:
            raise RuntimeError("LlamaProvider not initialized. Call initialize() first.")

        try:
            response = requests.post(
                f"{self._base_url}/tokenize",
                json={"content": text},
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
            # llama.cpp returns {"tokens": [list of token IDs]}
            return len(data.get("tokens", []))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Tokenization failed: {e}") from e

    def health_check(self) -> bool:
        """
        Check if llama.cpp server is reachable.

        Returns:
            True if server accepts connections, False otherwise
        """
        return self._health_check_internal()

    def _health_check_internal(self) -> bool:
        """Internal health check that works before initialization."""
        try:
            response = requests.get(
                f"{self._base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def _get_model_info(self) -> dict:
        """Get model info from /props endpoint."""
        try:
            response = requests.get(
                f"{self._base_url}/props",
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}

    def shutdown(self) -> None:
        """
        Cleanup provider resources.

        Note: Server process is managed externally, not by this provider.
        """
        self._initialized = False
        logger.debug("LlamaProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate Llama provider config.

        Supports two modes:
        1. Local mode: executable_path + model_path (generates server command)
        2. Cloud/external mode: url only (server managed externally)

        Args:
            config: Provider config dict

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Detect mode based on config fields
        has_executable = config.get("executable_path") is not None
        has_model = config.get("model_path") is not None
        has_url = config.get("url") is not None

        # Local mode validation
        if has_executable or has_model:
            if not has_executable:
                errors.append("executable_path is required when model_path is specified")
            elif not isinstance(config.get("executable_path"), str):
                errors.append(f"executable_path must be a string, got {type(config.get('executable_path')).__name__}")

            if not has_model:
                errors.append("model_path is required when executable_path is specified")
            elif not isinstance(config.get("model_path"), str):
                errors.append(f"model_path must be a string, got {type(config.get('model_path')).__name__}")

            # GPU selection validation
            device = config.get("device")
            tensor_split = config.get("tensor_split")

            if device is not None and not isinstance(device, str):
                errors.append(f"device must be a string, got {type(device).__name__}")

            if tensor_split is not None:
                if not isinstance(tensor_split, list):
                    errors.append(f"tensor_split must be a list, got {type(tensor_split).__name__}")
                elif len(tensor_split) < 2:
                    errors.append("tensor_split must have at least 2 values for multi-GPU")
                elif not all(isinstance(x, (int, float)) for x in tensor_split):
                    errors.append("tensor_split values must be numbers")

        # Cloud/external mode validation
        elif has_url:
            url = config.get("url")
            if not isinstance(url, str):
                errors.append(f"url must be a string, got {type(url).__name__}")
            elif not (url.startswith("http://") or url.startswith("https://")):
                errors.append(f"url must start with http:// or https://, got '{url}'")

        # Neither mode - invalid
        else:
            errors.append("Either url (external server) or executable_path + model_path (local server) is required")

        # Port validation (local mode)
        port = config.get("port")
        if port is not None:
            if not isinstance(port, int):
                errors.append(f"port must be an integer, got {type(port).__name__}")
            elif not (1 <= port <= 65535):
                errors.append(f"port must be between 1 and 65535, got {port}")

        # gpu_layers validation
        gpu_layers = config.get("gpu_layers")
        if gpu_layers is not None:
            if not isinstance(gpu_layers, int):
                errors.append(f"gpu_layers must be an integer, got {type(gpu_layers).__name__}")
            elif gpu_layers < 0:
                errors.append(f"gpu_layers must be non-negative, got {gpu_layers}")

        # extra_args validation (coerce YAML booleans like on/off/yes/no to strings)
        # YAML 1.1 parses bare on/off/yes/no as Python True/False.
        # Map them back to "on"/"off" so CLI flags like `-fa on` survive the round-trip.
        extra_args = config.get("extra_args")
        if extra_args is not None:
            if not isinstance(extra_args, list):
                errors.append(f"extra_args must be a list, got {type(extra_args).__name__}")
            else:
                _BOOL_MAP = {
                    True: "on", False: "off",
                    "True": "on", "False": "off",
                    "Yes": "on", "No": "off",
                }
                coerced = [
                    _BOOL_MAP.get(x, x) if isinstance(x, bool)
                    else _BOOL_MAP.get(x, x) if isinstance(x, str) and x in _BOOL_MAP
                    else str(x) if not isinstance(x, str)
                    else x
                    for x in extra_args
                ]
                if coerced != extra_args:
                    logger.warning(
                        "extra_args contained non-string values (YAML boolean gotcha?) "
                        "— coerced to strings: %s", coerced
                    )
                    config["extra_args"] = coerced

        # Timeout validation
        timeout = config.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                errors.append(f"timeout must be a number, got {type(timeout).__name__}")
            elif timeout <= 0:
                errors.append(f"timeout must be positive, got {timeout}")

        # max_retries validation
        max_retries = config.get("max_retries")
        if max_retries is not None:
            if not isinstance(max_retries, int):
                errors.append(f"max_retries must be an integer, got {type(max_retries).__name__}")
            elif max_retries < 1:
                errors.append(f"max_retries must be at least 1, got {max_retries}")

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

        # reasoning_format validation (NANO-042)
        reasoning_format = config.get("reasoning_format")
        if reasoning_format is not None:
            valid_formats = ("deepseek", "none")
            if reasoning_format not in valid_formats:
                errors.append(f"reasoning_format must be one of {valid_formats}, got '{reasoning_format}'")

        # reasoning_budget validation (NANO-042)
        reasoning_budget = config.get("reasoning_budget")
        if reasoning_budget is not None:
            if not isinstance(reasoning_budget, int):
                errors.append(f"reasoning_budget must be an integer, got {type(reasoning_budget).__name__}")
            elif reasoning_budget < -1:
                errors.append(f"reasoning_budget must be -1 (unlimited) or >= 0, got {reasoning_budget}")

        # mmproj_path validation — file must exist if specified (Session 606 bug)
        mmproj_path = config.get("mmproj_path")
        if mmproj_path:
            if not isinstance(mmproj_path, str):
                errors.append(f"mmproj_path must be a string, got {type(mmproj_path).__name__}")
            else:
                import os
                if not os.path.isfile(mmproj_path):
                    errors.append(
                        f"mmproj_path does not exist: {mmproj_path} — "
                        "stale path from a previous model? Clear it if VLM is disabled"
                    )

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        Generate llama-server launch command from config.

        Args:
            config: Provider config with executable_path, model_path, etc.

        Returns:
            Shell command string to start llama-server, or None if not configured
            for local mode (falls back to manual command or cloud mode).
        """
        executable = config.get("executable_path")
        model = config.get("model_path")

        # Not local mode - server managed externally
        if not executable or not model:
            return None

        host = config.get("host", "127.0.0.1")
        port = config.get("port", 5557)
        gpu_layers = config.get("gpu_layers", 99)

        cmd_parts = [
            executable,
            "-m", model,
            "--host", host,
            "--port", str(port),
            "-ngl", str(gpu_layers),
        ]

        # Multimodal projector for unified vision mode (NANO-030)
        mmproj_path = config.get("mmproj_path")
        if mmproj_path:
            cmd_parts.extend(["--mmproj", mmproj_path])

        # GPU selection (tensor_split takes precedence over device)
        tensor_split = config.get("tensor_split")
        device = config.get("device")

        if tensor_split:
            # Multi-GPU: distribute across GPUs with specified ratios
            split_str = ",".join(str(x) for x in tensor_split)
            cmd_parts.extend(["--tensor-split", split_str])
        elif device:
            # Single GPU: use specified device
            cmd_parts.extend(["--device", device])

        # Context size can be injected by launcher (NANO-021)
        context_size = config.get("context_size")
        if context_size:
            cmd_parts.extend(["-c", str(context_size)])

        # Jinja is required for proper chat template rendering on all modern models
        # (Gemma 3, Qwen3, etc.). Always inject unconditionally.
        cmd_parts.append("--jinja")

        # Reasoning model support (NANO-042)
        reasoning_format = config.get("reasoning_format")
        if reasoning_format:
            cmd_parts.extend(["--reasoning-format", reasoning_format])
        reasoning_budget = config.get("reasoning_budget", -1)
        cmd_parts.extend(["--reasoning-budget", str(reasoning_budget)])

        # Extra arguments (after all structured flags)
        extra_args = config.get("extra_args", [])
        if extra_args:
            cmd_parts.extend(extra_args)

        return " ".join(cmd_parts)

    @classmethod
    def get_health_url(cls, config: dict) -> Optional[str]:
        """
        Get health check URL for launcher.

        Args:
            config: Provider config

        Returns:
            Health check URL string
        """
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 5557)
        return f"http://{host}:{port}/health"
