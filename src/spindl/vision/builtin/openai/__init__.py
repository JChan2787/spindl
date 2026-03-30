"""
OpenAI VLM Provider - Cloud VLM via OpenAI-compatible API.

Supports any VLM endpoint that implements the OpenAI vision API:
    - OpenAI (GPT-4o, GPT-4-vision)
    - Together.ai
    - Fireworks.ai
    - Other OpenAI-compatible providers
"""

from .provider import OpenAIVLMProvider

__all__ = ["OpenAIVLMProvider"]
