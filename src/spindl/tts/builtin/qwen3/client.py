"""
Qwen3-TTS TCP client — session streaming + interrupt support.

Communicates with the Qwen3-TTS server (mad-labs/qwen3-tts-redux/server.py)
over localhost TCP using JSON-over-newline protocol.

Unlike the Kokoro client (one socket per request), this client maintains
a persistent connection for session-based streaming and mid-session interrupt.
"""

import json
import logging
import socket
import threading
import time
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Qwen3TTSClient:
    SAMPLE_RATE = 24000

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5557,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._sock: Optional[socket.socket] = None
        self._interrupted = threading.Event()

    def _ensure_connected(self) -> socket.socket:
        if self._sock is not None:
            return self._sock

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                self._sock = sock
                return sock
            except (ConnectionError, socket.error, OSError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        raise ConnectionError(
            f"Failed to connect to Qwen3-TTS server at {self.host}:{self.port} "
            f"after {self.max_retries} attempts: {last_error}"
        )

    def _send(self, request: dict) -> None:
        sock = self._ensure_connected()
        data = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
        sock.sendall(data)

    def _recv_line(self) -> dict:
        sock = self._ensure_connected()
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(1048576)
            if not chunk:
                self._sock = None
                raise ConnectionError("Server closed connection")
            buf += chunk
        line, _ = buf.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def _send_recv(self, request: dict) -> dict:
        self._send(request)
        return self._recv_line()

    def health(self) -> dict:
        return self._send_recv({"action": "health"})

    def is_server_available(self) -> bool:
        try:
            resp = self.health()
            return resp.get("status") == "success" and resp.get("model_loaded", False)
        except (ConnectionError, socket.error, OSError, json.JSONDecodeError):
            return False

    def synthesize(
        self,
        text: str,
        speaker: str = "ryan",
        temperature: float = 0.6,
        instruct: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        request: dict = {
            "action": "synthesize",
            "text": text,
            "speaker": speaker,
            "temperature": temperature,
        }
        if instruct:
            request["instruct"] = instruct
        if seed is not None:
            request["seed"] = seed

        resp = self._send_recv(request)

        if resp.get("status") == "error":
            raise RuntimeError(f"Qwen3-TTS server error: {resp.get('message')}")

        audio_hex = resp.get("audio", "")
        if not audio_hex:
            return np.array([], dtype=np.float32)

        return np.frombuffer(bytes.fromhex(audio_hex), dtype=np.float32)

    def synthesize_session(
        self,
        text: str,
        speaker: str = "ryan",
        temperature: float = 0.6,
        instruct: Optional[str] = None,
        instruct_per_sentence: Optional[list[str]] = None,
    ) -> Iterator[dict]:
        request: dict = {
            "action": "synthesize_session",
            "text": text,
            "speaker": speaker,
            "temperature": temperature,
        }
        if instruct:
            request["instruct"] = instruct
        if instruct_per_sentence:
            request["instruct_per_sentence"] = instruct_per_sentence

        self._interrupted.clear()
        self._send(request)

        sock = self._ensure_connected()
        original_timeout = sock.gettimeout()
        sock.settimeout(min(0.5, self.timeout))
        buf = b""
        try:
            while True:
                if self._interrupted.is_set():
                    return

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    resp = json.loads(line.decode("utf-8"))

                    if resp.get("status") == "interrupted":
                        return
                    if resp.get("status") == "error":
                        raise RuntimeError(f"Qwen3-TTS session error: {resp.get('message')}")

                    yield resp

                    if resp.get("is_final", False):
                        return

                try:
                    chunk = sock.recv(1048576)
                except socket.timeout:
                    continue
                if not chunk:
                    self._sock = None
                    raise ConnectionError("Server closed connection during session")
                buf += chunk
        finally:
            if self._sock is not None:
                self._sock.settimeout(original_timeout)

    def send_interrupt(self) -> None:
        self._interrupted.set()
        try:
            self._send({"action": "interrupt"})
        except (ConnectionError, socket.error, OSError) as e:
            logger.warning(f"Failed to send interrupt: {e}")

    def list_speakers(self) -> list[str]:
        resp = self._send_recv({"action": "list_speakers"})
        if resp.get("status") == "error":
            raise RuntimeError(f"Qwen3-TTS server error: {resp.get('message')}")
        return resp.get("speakers", [])

    def shutdown(self) -> None:
        try:
            self._send({"action": "shutdown"})
            # Read acknowledgment but don't require it
            try:
                self._recv_line()
            except (ConnectionError, socket.error, OSError):
                pass
        except (ConnectionError, socket.error, OSError) as e:
            logger.debug(f"Shutdown send failed (server may already be down): {e}")
        finally:
            self.close()

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
