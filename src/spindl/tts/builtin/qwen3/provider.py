"""
Qwen3-TTS Provider — TTSProvider implementation for Qwen3-TTS GGUF+ONNX server.

Wraps Qwen3TTSClient with the standardized TTSProvider interface.
The server is externally managed — SpindL does not spawn it.

Properties:
    - Sample rate: 24000 Hz
    - Format: pcm_f32le (32-bit float, little-endian)
    - Channels: 1 (mono)
    - Streaming: Yes (session-based per-sentence streaming)
"""

import logging
from typing import Iterator, Optional

import numpy as np

from ...base import AudioResult, TTSProperties, TTSProvider
from .client import Qwen3TTSClient

logger = logging.getLogger(__name__)


class Qwen3TTSProvider(TTSProvider):
    SAMPLE_RATE = 24000
    AUDIO_FORMAT = "pcm_f32le"
    CHANNELS = 1

    def __init__(self):
        self._client: Optional[Qwen3TTSClient] = None
        self._config: dict = {}
        self._speaker: str = "danny"
        self._temperature: float = 0.6
        self._emit_every_frames: int = 32
        self._instruct_template: str = ""
        self._initialized: bool = False

    def initialize(self, config: dict) -> None:
        self._config = config

        host = config.get("host", "127.0.0.1")
        port = config.get("port", 5557)
        timeout = config.get("timeout", 60.0)

        self._speaker = config.get("speaker", "danny")
        self._temperature = config.get("temperature", 0.6)
        self._emit_every_frames = config.get("emit_every_frames", 32)
        self._instruct_template = config.get("instruct_template", "")

        self._client = Qwen3TTSClient(host=host, port=port, timeout=timeout)

        if not self._client.is_server_available():
            raise ConnectionError(
                f"Qwen3-TTS server not available at {host}:{port}. "
                "The server must be started manually before launching SpindL."
            )

        self._initialized = True
        logger.info(f"Qwen3TTSProvider initialized: {host}:{port}")

    @property
    def instruct_template(self) -> str:
        return self._instruct_template

    def get_properties(self) -> TTSProperties:
        return TTSProperties(
            sample_rate=self.SAMPLE_RATE,
            audio_format=self.AUDIO_FORMAT,
            channels=self.CHANNELS,
            supports_streaming=True,
        )

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
        if not self._initialized or self._client is None:
            raise RuntimeError("Qwen3TTSProvider not initialized. Call initialize() first.")

        speaker = kwargs.get("speaker") or self._speaker
        temperature = kwargs.get("temperature") or self._temperature
        instruct = kwargs.get("instruct")

        audio = self._client.synthesize(
            text=text,
            speaker=speaker,
            temperature=temperature,
            instruct=instruct,
        )

        return AudioResult(
            data=audio.tobytes(),
            sample_rate=self.SAMPLE_RATE,
            format=self.AUDIO_FORMAT,
            is_final=True,
        )

    def begin_session(self) -> None:
        if not self._initialized or self._client is None:
            raise RuntimeError("Qwen3TTSProvider not initialized. Call initialize() first.")
        resp = self._client.begin_session()
        if resp.get("status") != "success":
            raise RuntimeError(f"begin_session failed: {resp}")

    def synthesize_next(
        self,
        text: str,
        speaker: Optional[str] = None,
        temperature: Optional[float] = None,
        instruct: Optional[str] = None,
        is_last: bool = False,
    ) -> AudioResult:
        if not self._initialized or self._client is None:
            raise RuntimeError("Qwen3TTSProvider not initialized. Call initialize() first.")

        spk = speaker or self._speaker
        temp = temperature or self._temperature

        resp = self._client.synthesize_next(
            text=text,
            speaker=spk,
            temperature=temp,
            instruct=instruct,
            is_last=is_last,
        )

        if resp.get("status") == "error":
            raise RuntimeError(f"synthesize_next error: {resp.get('message')}")

        audio_hex = resp.get("audio", "")
        if not audio_hex:
            return AudioResult(
                data=np.array([], dtype=np.float32).tobytes(),
                sample_rate=self.SAMPLE_RATE,
                format=self.AUDIO_FORMAT,
                is_final=is_last,
            )

        audio = np.frombuffer(bytes.fromhex(audio_hex), dtype=np.float32)
        return AudioResult(
            data=audio.tobytes(),
            sample_rate=self.SAMPLE_RATE,
            format=self.AUDIO_FORMAT,
            is_final=is_last,
        )

    def synthesize_stream(self, text: str, voice: Optional[str] = None, **kwargs) -> Iterator[AudioResult]:
        if not self._initialized or self._client is None:
            raise RuntimeError("Qwen3TTSProvider not initialized. Call initialize() first.")

        speaker = kwargs.get("speaker") or self._speaker
        temperature = kwargs.get("temperature") or self._temperature
        instruct = kwargs.get("instruct")
        instruct_per_sentence = kwargs.get("instruct_per_sentence")

        if instruct_per_sentence:
            print(f"[Qwen3-TTS] Session request: speaker={speaker}, temp={temperature}, instruct_per_sentence={instruct_per_sentence}", flush=True)
        elif instruct:
            print(f"[Qwen3-TTS] Session request: speaker={speaker}, temp={temperature}, instruct={instruct}", flush=True)
        else:
            print(f"[Qwen3-TTS] Session request: speaker={speaker}, temp={temperature}, no instruct", flush=True)

        for resp in self._client.synthesize_session(
            text=text,
            speaker=speaker,
            temperature=temperature,
            instruct=instruct,
            instruct_per_sentence=instruct_per_sentence,
        ):
            audio_hex = resp.get("audio", "")
            if not audio_hex:
                continue

            audio = np.frombuffer(bytes.fromhex(audio_hex), dtype=np.float32)
            yield AudioResult(
                data=audio.tobytes(),
                sample_rate=self.SAMPLE_RATE,
                format=self.AUDIO_FORMAT,
                is_final=resp.get("is_final", False),
            )

    def interrupt(self) -> None:
        if self._client is not None:
            self._client.send_interrupt()

    def list_voices(self) -> list[str]:
        if not self._initialized or self._client is None:
            return [self._speaker]
        try:
            return self._client.list_speakers()
        except (ConnectionError, RuntimeError) as e:
            logger.warning(f"Failed to list speakers: {e}")
            return [self._speaker]

    def health_check(self) -> bool:
        if self._client is None:
            return False
        return self._client.is_server_available()

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()
            self._client = None
        self._initialized = False
        logger.debug("Qwen3TTSProvider shut down (server sent shutdown command)")

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        errors = []

        host = config.get("host")
        if host is None:
            errors.append("Missing required field: host")
        elif not isinstance(host, str) or not host.strip():
            errors.append("host must be a non-empty string")

        port = config.get("port")
        if port is None:
            errors.append("Missing required field: port")
        elif not isinstance(port, int):
            errors.append(f"port must be an integer, got {type(port).__name__}")
        elif not (1 <= port <= 65535):
            errors.append(f"port must be between 1 and 65535, got {port}")

        speaker = config.get("speaker")
        if speaker is None:
            errors.append("Missing required field: speaker")
        elif not isinstance(speaker, str) or not speaker.strip():
            errors.append("speaker must be a non-empty string")

        temperature = config.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                errors.append(f"temperature must be a number, got {type(temperature).__name__}")
            elif not (0 <= temperature <= 2):
                errors.append(f"temperature must be between 0 and 2, got {temperature}")

        emit_every_frames = config.get("emit_every_frames")
        if emit_every_frames is not None:
            if not isinstance(emit_every_frames, int):
                errors.append(f"emit_every_frames must be an integer, got {type(emit_every_frames).__name__}")
            elif not (1 <= emit_every_frames <= 128):
                errors.append(f"emit_every_frames must be between 1 and 128, got {emit_every_frames}")

        return errors

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        return None
