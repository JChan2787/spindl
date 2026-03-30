"""
LlamaClient - HTTP client for llama.cpp server.

Communicates with llama.cpp's OpenAI-compatible API endpoint.

INTERNAL: This class is used internally by LlamaProvider. External code
should use the LLMProvider abstraction via LLMProviderRegistry instead.
This class is intentionally NOT exported from the llm package (NANO-019).
"""

import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class TokenUsage:
    """
    Token usage statistics from an LLM response.

    Matches the OpenAI-compatible 'usage' field in chat completion responses.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, data: dict) -> "TokenUsage":
        """
        Create TokenUsage from API response dict.

        Args:
            data: The 'usage' dict from a chat completion response.
                  Missing keys default to 0.

        Returns:
            TokenUsage instance
        """
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )

    @classmethod
    def empty(cls) -> "TokenUsage":
        """Return a zero-value TokenUsage for cases where usage is unavailable."""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)


@dataclass
class ChatResponse:
    """
    Response from a chat completion request.

    Contains both the generated content and token usage statistics.
    """

    content: str
    usage: TokenUsage


class LlamaClient:
    """
    HTTP client for llama.cpp server's OpenAI-compatible API.

    Sends chat completion requests and handles retries/timeouts.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5557",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: Server URL (default: http://127.0.0.1:5557)
            timeout: Request timeout in seconds (default: 30s)
            max_retries: Number of connection attempts before failing
            retry_delay: Seconds to wait between retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.95,
        stop: Optional[list[str]] = None,
    ) -> ChatResponse:
        """
        Send chat completion request to llama.cpp server.

        Args:
            messages: OpenAI-style message list
                      [{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length
            top_p: Nucleus sampling threshold
            stop: Stop sequences (optional)

        Returns:
            ChatResponse with content and token usage statistics

        Raises:
            ConnectionError: Server unreachable after retries
            TimeoutError: Request timed out
            RuntimeError: Generation failed
        """
        endpoint = f"{self.base_url}/v1/chat/completions"

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        if stop:
            payload["stop"] = stop

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break

            except requests.exceptions.Timeout as e:
                raise TimeoutError(
                    f"Request timed out after {self.timeout}s"
                ) from e

            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"Server returned error: {e.response.status_code} - {e.response.text}"
                ) from e

        else:
            raise ConnectionError(
                f"Failed to connect to LLM server at {self.base_url} "
                f"after {self.max_retries} attempts: {last_error}"
            )

        # Parse response
        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
            usage = TokenUsage.from_dict(data.get("usage", {}))
            return ChatResponse(content=content, usage=usage)
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected response format: {data}"
            ) from e

    def health_check(self) -> bool:
        """
        Check if llama.cpp server is running and responsive.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def get_model_info(self) -> dict:
        """
        Get loaded model metadata from /props endpoint.

        Returns:
            Dict with model properties (varies by llama.cpp version)

        Raises:
            ConnectionError: Server unreachable
            RuntimeError: Request failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/props",
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot reach server at {self.base_url}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Failed to get model info: {e.response.status_code}"
            ) from e

    def tokenize(self, content: str, add_special: bool = False) -> list[int]:
        """
        Tokenize text using the model's tokenizer.

        Args:
            content: Text to tokenize
            add_special: Whether to add special tokens (BOS, etc.)

        Returns:
            List of token IDs

        Raises:
            ConnectionError: Server unreachable
            RuntimeError: Tokenization failed
        """
        endpoint = f"{self.base_url}/tokenize"

        payload = {
            "content": content,
            "add_special": add_special,
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("tokens", [])

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot reach server at {self.base_url}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Tokenization failed: {e.response.status_code} - {e.response.text}"
            ) from e

    def get_context_length(self) -> int:
        """
        Get the model's context window size from /props endpoint.

        Returns:
            Context length (n_ctx) in tokens

        Raises:
            ConnectionError: Server unreachable
            RuntimeError: Failed to get props or parse n_ctx

        Note:
            Result should be cached by caller; this makes a network request.
        """
        props = self.get_model_info()

        try:
            n_ctx = props["default_generation_settings"]["n_ctx"]
            return int(n_ctx)
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Could not parse n_ctx from props: {props}"
            ) from e

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Convenience wrapper around tokenize().

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        tokens = self.tokenize(text, add_special=False)
        return len(tokens)
