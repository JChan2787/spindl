"""
LLMVisionProvider - VLM bridge through the existing LLM endpoint.

Routes image description requests through the user's already-configured
LLM service instead of a separate VLM service. This saves GPU VRAM when
the LLM is a multimodal model that can handle both text and vision tasks.

Uses the standard OpenAI-compatible multimodal message format:
    content: [{"type": "text", ...}, {"type": "image_url", ...}]

Works with both local (llama-server) and cloud (OpenAI, DeepSeek, etc.)
LLM endpoints that accept image content blocks.
"""

import logging
import os
import re
import time
from typing import Optional

import requests

from ...base import VLMProperties, VLMProvider, VLMResponse

logger = logging.getLogger(__name__)


class LLMVisionProvider(VLMProvider):
    """
    VLM provider that bridges to the existing LLM endpoint.

    Instead of running a separate VLM service, this provider sends
    image description requests to the same LLM endpoint used for
    text generation. The LLM must be a multimodal model that accepts
    image content blocks (e.g., Gemma 3 with --mmproj, GPT-4o).

    From the launcher's perspective this is a "cloud" provider — no
    VLM server process to spawn. The LLM service is already running.

    Config schema (vlm.providers.llm):
        url: str          - LLM endpoint URL (default: "http://127.0.0.1:5557")
        api_key: str      - API key for cloud endpoints (optional, supports ${ENV_VAR})
        model: str        - Model name (default: "local-llm", informational)
        timeout: float    - Request timeout in seconds (default: 30.0)
        max_tokens: int   - Max response tokens (default: 300)
        prompt: str       - Default description prompt
    """

    def __init__(self):
        """Initialize provider (lightweight, no connections yet)."""
        self._config: Optional[dict] = None
        self._url: str = "http://127.0.0.1:5557"
        self._api_key: Optional[str] = None
        self._model: str = "local-llm"
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

    def _build_completions_url(self) -> str:
        """
        Build the chat completions endpoint URL.

        Handles both styles:
            - "http://127.0.0.1:5557" → "http://127.0.0.1:5557/v1/chat/completions"
            - "https://api.deepseek.com/v1" → "https://api.deepseek.com/v1/chat/completions"

        Returns:
            Full URL to the chat completions endpoint
        """
        if self._url.rstrip("/").endswith("/v1"):
            return f"{self._url.rstrip('/')}/chat/completions"
        return f"{self._url.rstrip('/')}/v1/chat/completions"

    def initialize(self, config: dict) -> None:
        """
        Initialize connection to LLM endpoint.

        Args:
            config: Provider config from spindl.yaml vlm.providers.llm section
                   Required: url
                   Optional: api_key (supports ${ENV_VAR}), model, timeout,
                            max_tokens, prompt

        Raises:
            ValueError: Invalid configuration
        """
        self._config = config

        self._url = config.get("url", "http://127.0.0.1:5557").rstrip("/")

        # Resolve API key (supports ${ENV_VAR} syntax)
        api_key_raw = config.get("api_key", "")
        if api_key_raw:
            resolved = self._resolve_env_vars(api_key_raw)
            # Only set if actually resolved (not still a ${VAR} reference)
            if resolved and not resolved.startswith("${"):
                self._api_key = resolved
            else:
                self._api_key = None
        else:
            self._api_key = None

        self._model = config.get("model", "local-llm")
        self._timeout = config.get("timeout", 30.0)
        self._max_tokens = config.get("max_tokens", 300)
        self._prompt = config.get("prompt", "Describe what you see in this image.")

        # Warn if LLM endpoint not reachable (don't crash — server may start later)
        if not self.health_check():
            logger.warning(
                f"LLMVisionProvider: LLM endpoint at {self._url} not responding. "
                f"Ensure the LLM service is running with a multimodal model."
            )

        self._initialized = True
        logger.info(
            f"LLMVisionProvider initialized: {self._model} at {self._url}"
        )

    def get_properties(self) -> VLMProperties:
        """Return provider capabilities."""
        return VLMProperties(
            name=self._model,
            is_local=not bool(self._api_key),
            supports_streaming=False,
            max_image_size=None,
        )

    def describe(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        max_tokens: int = 300,
        **kwargs,
    ) -> VLMResponse:
        """
        Send image to LLM endpoint and get description.

        Uses OpenAI-compatible multimodal format with base64 image.

        Args:
            image_base64: Base64-encoded JPEG image
            prompt: Description prompt (uses default if None)
            max_tokens: Max response tokens
            **kwargs: Additional options (temperature, etc.)

        Returns:
            VLMResponse with description and usage stats

        Raises:
            RuntimeError: Request failed or provider not initialized
            ConnectionError: LLM endpoint unreachable
            TimeoutError: Request timed out
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        effective_prompt = prompt or self._prompt
        effective_max_tokens = max_tokens or self._max_tokens

        # Build OpenAI-compatible multimodal message
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
            "max_tokens": effective_max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "cache_prompt": False,
            "id_slot": 1,  # NANO-087: pin to slot 1 (chat uses slot 0)
        }

        # Build headers — auth only for cloud endpoints
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        start_time = time.perf_counter()

        try:
            response = requests.post(
                self._build_completions_url(),
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
                f"Cannot connect to LLM endpoint at {self._url}: {e}"
            )
        except requests.exceptions.HTTPError as e:
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
            f"LLM vision describe: {output_tokens} tokens in {elapsed_ms:.0f}ms"
        )

        return VLMResponse(
            description=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        """
        Check if LLM endpoint is reachable.

        Uses different strategies for local vs cloud:
        - Local (no api_key): GET /health (llama-server pattern)
        - Cloud (has api_key): GET /v1/models with auth header
        """
        try:
            if self._api_key:
                # Cloud: check models endpoint with auth
                # Handle both URL styles (with and without /v1 suffix)
                if self._url.endswith("/v1"):
                    models_url = f"{self._url}/models"
                else:
                    models_url = f"{self._url}/v1/models"
                response = requests.get(
                    models_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=5.0,
                )
                return response.status_code in (200, 401)
            else:
                # Local: check /health endpoint (llama-server)
                response = requests.get(
                    f"{self._url}/health",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception:
            return False

    def shutdown(self) -> None:
        """Cleanup (LLM service managed externally)."""
        self._initialized = False
        logger.info("LLMVisionProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate provider configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # url is required
        url = config.get("url")
        if not url:
            errors.append(
                "url is required (LLM endpoint URL, e.g., 'http://127.0.0.1:5557')"
            )
        elif not isinstance(url, str):
            errors.append(f"url must be a string, got {type(url).__name__}")
        elif not (url.startswith("http://") or url.startswith("https://")):
            errors.append(
                f"url must start with http:// or https://, got '{url}'"
            )

        # Optional field validation
        api_key = config.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            errors.append(f"api_key must be a string, got {type(api_key).__name__}")

        timeout = config.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                errors.append(f"timeout must be a number, got {type(timeout).__name__}")
            elif timeout <= 0:
                errors.append(f"timeout must be positive, got {timeout}")

        max_tokens = config.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                errors.append(f"max_tokens must be an integer, got {type(max_tokens).__name__}")
            elif max_tokens < 1:
                errors.append(f"max_tokens must be at least 1, got {max_tokens}")

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """LLM service managed externally — no server to start."""
        return None

    @classmethod
    def get_health_url(cls, config: dict) -> Optional[str]:
        """Health check is dynamic (local vs cloud) — no static URL."""
        return None

    @classmethod
    def is_cloud_provider(cls) -> bool:
        """No VLM process to launch — LLM service already running."""
        return True
