# STT Module - Provider-based STT architecture (NANO-061a)
#
# Core abstractions:
#   STTProvider         - Abstract base class for all STT implementations
#   STTProperties       - Provider capability descriptor
#
# Registry:
#   STTProviderRegistry        - Discovery and instantiation of providers
#   STTProviderNotFoundError   - Raised when provider not found
#   create_default_registry    - Factory for registry with default config
#
# Built-in providers (via builtin/):
#   ParakeetSTT         - TCP client for NeMo Parakeet server
#   WhisperSTT          - HTTP client for whisper.cpp server

from .base import STTProperties, STTProvider
from .builtin.parakeet import ParakeetSTT
from .builtin.whisper import WhisperSTT
from .registry import (
    STTProviderNotFoundError,
    STTProviderRegistry,
    create_default_registry,
)

__all__ = [
    # Core abstractions
    "STTProvider",
    "STTProperties",
    # Registry
    "STTProviderRegistry",
    "STTProviderNotFoundError",
    "create_default_registry",
    # Built-in providers
    "ParakeetSTT",
    "WhisperSTT",
]
