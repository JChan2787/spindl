"""
OpenAIVLMProvider - Cloud VLM via OpenAI-compatible API.

This provider connects to cloud VLM endpoints that implement the
OpenAI vision API format. Works with:
- OpenAI (GPT-4o, GPT-4-vision)
- Together.ai (LLaMA Vision, etc.)
- Fireworks.ai
- Any OpenAI-compatible VLM endpoint
"""

import logging
import os
import re
import time
from typing import Optional

import requests

from ...base import VLMProperties, VLMProvider, VLMResponse

logger = logging.getLogger(__name__)


class OpenAIVLMProvider(VLMProvider):
    """
    Cloud VLM via OpenAI-compatible API.

    Sends images to cloud endpoints for description. Supports any
    provider with OpenAI-compatible multimodal API.

    Environment variable substitution: API keys can use ${VAR} syntax.
    Example: api_key: "${OPENAI_API_KEY}"
    """

    def __init__(self):
        """Initialize provider (lightweight, no connections yet)."""
        self._config: Optional[dict] = None
        self._api_key: Optional[str] = None
        self._base_url: str = "https://api.openai.com"
        self._model: str = "gpt-4o"
        self._timeout: float = 30.0
        self._max_tokens: int = 300
        self._prompt: str = "Describe what you see in this image."
        self._initialized: bool = False

    def _resolve_env_vars(self, value: str) -> str:
        """
        Resolve environment variable references in a string.

        Supports ${VAR_NAME} syntax for env var substitution.

        Args:
            value: String potentially containing ${VAR} references

        Returns:
            String with env vars resolved
        """
        pattern = r"\$\{([^}]+)\}"

        def replace_env(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace_env, value)

    def initialize(self, config: dict) -> None:
        """
        Initialize connection to cloud VLM API.

        Args:
            config: Provider config from spindl.yaml vision.providers.openai section
                   Required: api_key (can use ${ENV_VAR} syntax)
                   Optional: base_url, model, timeout, max_tokens, prompt

        Raises:
            ValueError: Invalid or missing API key
        """
        self._config = config

        # Resolve API key (supports ${ENV_VAR} syntax)
        api_key_raw = config.get("api_key", "")
        self._api_key = self._resolve_env_vars(api_key_raw)

        if not self._api_key or self._api_key.startswith("${"):
            raise ValueError(
                "api_key is required. Use ${ENV_VAR} syntax for environment variables. "
                f"Got: {api_key_raw}"
            )

        self._base_url = config.get("base_url", "https://api.openai.com").rstrip("/")
        self._model = config.get("model", "gpt-4o")
        self._timeout = config.get("timeout", 30.0)
        self._max_tokens = config.get("max_tokens", 300)

        # OpenAI-native endpoints require max_completion_tokens instead of max_tokens.
        # Detect by checking if the base URL is OpenAI's own API.
        self._use_completion_tokens = "api.openai.com" in self._base_url
        self._prompt = config.get("prompt", "Describe what you see in this image.")

        self._initialized = True
        logger.info(
            f"OpenAIVLMProvider initialized: {self._model} at {self._base_url}"
        )

    def get_properties(self) -> VLMProperties:
        """Return provider capabilities."""
        return VLMProperties(
            name=self._model,
            is_local=False,
            supports_streaming=False,
            max_image_size=None,  # Depends on model/provider
        )

    def describe(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        max_tokens: int = 300,
        **kwargs,
    ) -> VLMResponse:
        """
        Send image to cloud VLM and get description.

        Args:
            image_base64: Base64-encoded JPEG image
            prompt: Description prompt (uses default if None)
            max_tokens: Max response tokens
            **kwargs: Additional options (temperature, etc.)

        Returns:
            VLMResponse with description and usage stats

        Raises:
            RuntimeError: Request failed
            ConnectionError: API unreachable
            TimeoutError: Request timed out
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        effective_prompt = prompt or self._prompt
        effective_max_tokens = max_tokens or self._max_tokens

        # Build OpenAI multimodal message
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": effective_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        }

        # OpenAI-native endpoints have stricter parameter requirements:
        # - max_completion_tokens instead of max_tokens
        # - temperature only supports 1 (default) on newer models
        if self._use_completion_tokens:
            payload["max_completion_tokens"] = effective_max_tokens
        else:
            payload["max_tokens"] = effective_max_tokens
            payload["temperature"] = kwargs.get("temperature", 0.7)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"VLM request timed out after {self._timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to API at {self._base_url}: {e}"
            )
        except requests.exceptions.HTTPError as e:
            # Extract error message from response if available
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except Exception:
                pass
            raise RuntimeError(
                f"VLM request failed: {e}. {error_detail}"
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse response
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("VLM returned empty response")

        content = choices[0].get("message", {}).get("content", "")

        # Extract token counts
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        logger.debug(
            f"VLM describe: {output_tokens} tokens in {elapsed_ms:.0f}ms"
        )

        return VLMResponse(
            description=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        """
        Check if API is reachable.

        For cloud providers, we do a lightweight check without
        sending an actual image (just verify the endpoint responds).
        """
        if not self._api_key:
            return False

        try:
            # Just check if the API endpoint is reachable
            # We don't send a full request to avoid costs
            response = requests.get(
                f"{self._base_url}/v1/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=5.0,
            )
            # 200 or 401 means the server is up (401 = bad key but server works)
            return response.status_code in (200, 401)
        except Exception:
            return False

    def shutdown(self) -> None:
        """Cleanup (nothing to do for cloud provider)."""
        self._initialized = False
        logger.info("OpenAIVLMProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate provider configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not config.get("api_key"):
            errors.append(
                "api_key is required. Use ${ENV_VAR} syntax for environment variables."
            )

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """Cloud provider - no server to start."""
        return None

    @classmethod
    def get_health_url(cls, config: dict) -> Optional[str]:
        """Cloud provider - no local health URL."""
        return None

    @classmethod
    def is_cloud_provider(cls) -> bool:
        """This is a cloud provider."""
        return True
