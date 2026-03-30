"""
LLM Vision Provider - Route VLM calls through the existing LLM endpoint.

Bridges the VLM interface to the user's already-configured LLM service,
allowing multimodal models to handle both text generation and image description
without a separate VLM process.

Supports:
    - Local llama-server with multimodal models (e.g., Gemma 3, Qwen 2.5-VL)
    - Cloud APIs with vision support (GPT-4o, Claude, etc.)
"""

from .provider import LLMVisionProvider

__all__ = ["LLMVisionProvider"]
