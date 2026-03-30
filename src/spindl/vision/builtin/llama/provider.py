"""
LlamaVLMProvider - Local VLM via llama-server with multimodal models.

This provider connects to llama-server running a vision-capable model
(Gemma3, Qwen2-VL, LLaVA, etc.) and sends images for description.

Uses the model config system to generate architecture-specific launch commands.
"""

import logging
import time
from typing import Optional

import requests

from ...base import VLMProperties, VLMProvider, VLMResponse
from .models import get_model_config

logger = logging.getLogger(__name__)


class LlamaVLMProvider(VLMProvider):
    """
    Local VLM via llama-server.

    Connects to a llama-server instance running a multimodal model.
    Uses OpenAI-compatible /v1/chat/completions endpoint with base64 images.

    The model config system generates the correct launch command based on
    the model architecture (Gemma3 needs --mmproj, Qwen2-VL doesn't, etc.).
    """

    def __init__(self):
        """Initialize provider (lightweight, no connections yet)."""
        self._config: Optional[dict] = None
        self._url: Optional[str] = None
        self._timeout: float = 30.0
        self._max_tokens: int = 300
        self._prompt: str = "Describe what you see in this image."
        self._model_type: str = "gemma3"
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        """
        Initialize connection to llama-server.

        Args:
            config: Provider config from spindl.yaml vision.providers.llama section
                   Required: model_type, model_path, port
                   Optional: mmproj_path (required for some architectures),
                            timeout, max_tokens, prompt, context_size, gpu_layers

        Raises:
            ValueError: Invalid configuration
            ConnectionError: Cannot connect to server
        """
        self._config = config
        self._model_type = config.get("model_type", "gemma3")

        port = config.get("port", 5558)
        host = config.get("host", "127.0.0.1")
        self._url = f"http://{host}:{port}"

        self._timeout = config.get("timeout", 30.0)
        self._max_tokens = config.get("max_tokens", 300)
        self._prompt = config.get("prompt", "Describe what you see in this image.")

        # Validate model config
        model_config = get_model_config(self._model_type)
        is_valid, error = model_config.validate_config(
            model_path=config.get("model_path", ""),
            mmproj_path=config.get("mmproj_path"),
        )
        if not is_valid:
            raise ValueError(f"Invalid config for {self._model_type}: {error}")

        # Verify server is reachable
        if not self.health_check():
            logger.warning(
                f"LlamaVLMProvider: Server at {self._url} not responding. "
                f"Ensure llama-server is running with a multimodal model."
            )

        self._initialized = True
        logger.info(
            f"LlamaVLMProvider initialized: {self._model_type} at {self._url}"
        )

    def get_properties(self) -> VLMProperties:
        """Return provider capabilities."""
        model_name = "unknown"
        if self._config:
            # Extract model name from path
            model_path = self._config.get("model_path", "")
            if model_path:
                model_name = model_path.split("/")[-1].split("\\")[-1]
                # Remove .gguf extension
                if model_name.endswith(".gguf"):
                    model_name = model_name[:-5]

        return VLMProperties(
            name=model_name,
            is_local=True,
            supports_streaming=False,  # VLM description typically not streamed
            max_image_size=None,  # Depends on model
        )

    def describe(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        max_tokens: int = 300,
        **kwargs,
    ) -> VLMResponse:
        """
        Send image to llama-server and get description.

        Uses OpenAI-compatible multimodal format with base64 image.

        Args:
            image_base64: Base64-encoded JPEG image
            prompt: Description prompt (uses default if None)
            max_tokens: Max response tokens
            **kwargs: Additional options (temperature, etc.)

        Returns:
            VLMResponse with description and usage stats

        Raises:
            RuntimeError: Request failed
            ConnectionError: Server unreachable
            TimeoutError: Request timed out
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        effective_prompt = prompt or self._prompt
        effective_max_tokens = max_tokens or self._max_tokens

        # Build OpenAI-compatible multimodal message
        # Note: cache_prompt=false prevents llama-server from reusing KV cache
        # from previous requests, ensuring the new image is actually processed.
        payload = {
            "model": "local-vlm",  # Ignored by llama-server
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
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self._url}/v1/chat/completions",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"VLM request timed out after {self._timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to llama-server at {self._url}: {e}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"VLM request failed: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse response
        data = response.json()

        # Extract content from OpenAI-format response
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("VLM returned empty response")

        content = choices[0].get("message", {}).get("content", "")

        # Extract token counts
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        logger.debug(
            f"VLM describe: {output_tokens} tokens in {elapsed_ms:.0f}ms "
            f"({output_tokens / (elapsed_ms / 1000):.1f} tok/s)"
        )

        return VLMResponse(
            description=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        """Check if llama-server is responding."""
        if not self._url:
            return False

        try:
            response = requests.get(
                f"{self._url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    def shutdown(self) -> None:
        """Cleanup (server process managed by launcher)."""
        self._initialized = False
        logger.info("LlamaVLMProvider shut down")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        """
        Validate provider configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not config.get("executable_path"):
            errors.append("executable_path is required (full path to llama-server executable)")

        if not config.get("model_type"):
            errors.append("model_type is required")

        if not config.get("model_path"):
            errors.append("model_path is required")

        # Validate model-specific config
        model_type = config.get("model_type", "gemma3")
        try:
            model_config = get_model_config(model_type)
            is_valid, error = model_config.validate_config(
                model_path=config.get("model_path", ""),
                mmproj_path=config.get("mmproj_path"),
            )
            if not is_valid and error:
                errors.append(error)
        except ValueError as e:
            errors.append(str(e))

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        """
        Generate llama-server launch command from config.

        Uses model config system to build architecture-specific command.

        Returns:
            Shell command string to start llama-server
        """
        model_type = config.get("model_type", "gemma3")
        executable_path = config.get("executable_path")

        if not executable_path:
            logger.error("executable_path is required in vision.providers.llama config")
            return None

        # Coerce YAML booleans in extra_args (on/off/yes/no → True/False in YAML 1.1)
        extra_args = config.get("extra_args")
        if extra_args and isinstance(extra_args, list):
            _BOOL_MAP = {
                True: "on", False: "off",
                "True": "on", "False": "off",
                "Yes": "on", "No": "off",
            }
            extra_args = [
                _BOOL_MAP.get(x, x) if isinstance(x, bool)
                else _BOOL_MAP.get(x, x) if isinstance(x, str) and x in _BOOL_MAP
                else str(x) if not isinstance(x, str)
                else x
                for x in extra_args
            ]

        try:
            model_config = get_model_config(model_type)
            launch_config = model_config.generate_launch_config(
                executable_path=executable_path,
                model_path=config.get("model_path", ""),
                mmproj_path=config.get("mmproj_path"),
                port=config.get("port", 5558),
                context_size=config.get("context_size", 8192),
                gpu_layers=config.get("gpu_layers", 99),
                extra_args=extra_args,
                device=config.get("device"),
                tensor_split=config.get("tensor_split"),
            )
            return launch_config.command
        except (ValueError, KeyError) as e:
            logger.error(f"Cannot generate server command: {e}")
            return None

    @classmethod
    def get_health_url(cls, config: dict) -> Optional[str]:
        """
        Get health check URL for launcher.

        Returns:
            Health check URL string
        """
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 5558)
        return f"http://{host}:{port}/health"

    @classmethod
    def is_cloud_provider(cls) -> bool:
        """Local server provider."""
        return False
