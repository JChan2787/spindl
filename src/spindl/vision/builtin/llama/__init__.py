"""
Llama VLM Provider - Local VLM via llama-server.

Supports multimodal models served by llama-server:
    - Gemma 3 (X-Ray_Alpha, etc.) - requires mmproj
    - Qwen2-VL - unified GGUF (future)
    - LLaVA 1.5/1.6 - requires mmproj (future)
    - MiniCPM-V - requires mmproj (future)

The model config system handles architecture-specific launch commands.
"""

from .provider import LlamaVLMProvider

__all__ = ["LlamaVLMProvider"]
