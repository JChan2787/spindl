# TTS Module - Provider-based TTS architecture
#
# Core abstractions:
#   TTSProvider     - Abstract base class for all TTS implementations
#   TTSProperties   - Provider capability descriptor
#   AudioResult     - Synthesized audio chunk
#
# Registry:
#   TTSProviderRegistry    - Discovery and instantiation of providers
#   ProviderNotFoundError  - Raised when provider not found
#   create_default_registry - Factory for registry with default config
#
# Built-in providers (via builtin/):
#   KokoroTTS       - TCP client for Kokoro TTS server

from .base import AudioResult, TTSProperties, TTSProvider
from .builtin.kokoro import KokoroTTS
from .registry import (
    ProviderNotFoundError,
    TTSProviderRegistry,
    create_default_registry,
)

__all__ = [
    # Core abstractions
    "TTSProvider",
    "TTSProperties",
    "AudioResult",
    # Registry
    "TTSProviderRegistry",
    "ProviderNotFoundError",
    "create_default_registry",
    # Built-in providers
    "KokoroTTS",
]
