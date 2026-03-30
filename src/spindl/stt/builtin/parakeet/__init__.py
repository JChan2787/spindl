# Parakeet STT Provider - Built-in STT using NVIDIA NeMo Parakeet.
#
# Components:
#   ParakeetSTT         - TCP client for transcription requests
#   ParakeetSTTProvider - STTProvider implementation wrapping ParakeetSTT

from .client import ParakeetSTT
from .provider import ParakeetSTTProvider

__all__ = ["ParakeetSTT", "ParakeetSTTProvider"]
