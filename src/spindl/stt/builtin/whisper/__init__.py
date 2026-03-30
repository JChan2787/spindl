# Whisper.cpp STT Provider - Built-in STT using whisper.cpp HTTP server.
#
# Components:
#   WhisperSTT          - HTTP client for transcription requests
#   WhisperSTTProvider  - STTProvider implementation wrapping WhisperSTT

from .client import WhisperSTT
from .provider import WhisperSTTProvider

__all__ = ["WhisperSTT", "WhisperSTTProvider"]
