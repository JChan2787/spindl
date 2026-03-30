# Kokoro TTS Provider - Built-in TTS using Kokoro-82M model.
#
# Components:
#   KokoroTTS         - TCP client for synthesis requests
#   KokoroTTSProvider - TTSProvider implementation wrapping KokoroTTS
#   server.py         - TCP server bridge (run separately with CUDA)

from .client import KokoroTTS
from .provider import KokoroTTSProvider

__all__ = ["KokoroTTS", "KokoroTTSProvider"]
