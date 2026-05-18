#!/usr/bin/env python3
"""
Kokoro TTS Server - TCP bridge for Kokoro speech synthesis.

Runs on Windows with CUDA, accepts text from orchestrator via TCP.

Protocol:
    Request:  {"action": "synthesize", "text": "...", "voice": "af_bella", "lang": "a"}\n
    Response: {"status": "success", "audio": "<hex>", "sample_rate": 24000, "duration_ms": N}\n
    Error:    {"status": "error", "error": "..."}\n

Usage:
    python -m spindl.tts.builtin.kokoro.server --models-dir /path/to/tts/models

    Or via environment variable:
    KOKORO_MODELS_DIR=/path/to/tts/models python -m spindl.tts.builtin.kokoro.server
"""

import argparse
import json
import os
import signal
import socket
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Suppress non-blocking warnings from Kokoro/PyTorch internals
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")

# Kokoro imports
from kokoro import KPipeline
from kokoro.model import KModel


class KokoroTTSServer:
    """TCP server for Kokoro-based text-to-speech."""

    # Audio output sample rate (Kokoro's native rate)
    SAMPLE_RATE = 24000

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        models_dir: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.host = host
        self.port = port
        self.device = device
        self.model: Optional[KModel] = None
        self.pipeline: Optional[KPipeline] = None
        self._socket: Optional[socket.socket] = None
        self._shutdown: bool = False
        self._blended_voice: Optional[torch.FloatTensor] = None

        # Model paths - must be provided externally
        if models_dir is None:
            # Try environment variable
            env_path = os.environ.get("KOKORO_MODELS_DIR")
            if env_path:
                models_dir = Path(env_path)
            else:
                raise ValueError(
                    "Models directory not specified. "
                    "Use --models-dir argument or set KOKORO_MODELS_DIR environment variable."
                )

        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / "config.json"
        self.model_path = self.models_dir / "kokoro-v1_0.pth"
        self.voices_dir = self.models_dir / "voices"

    def load_model(self, default_lang: str = "a") -> None:
        """Load the Kokoro model. This is slow (~10-15s cold)."""
        print("Loading Kokoro model...", flush=True)
        print(f"Models directory: {self.models_dir}", flush=True)
        print(f"Device: {self.device}", flush=True)
        start = time.time()

        # Validate model files exist
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.voices_dir.exists():
            raise FileNotFoundError(f"Voices dir not found: {self.voices_dir}")

        # Load model from local files
        self.model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(self.config_path),
            model=str(self.model_path),
        ).to(self.device).eval()

        # Create pipeline with pre-loaded model
        self.pipeline = KPipeline(
            lang_code=default_lang,
            repo_id="hexgrad/Kokoro-82M",
            model=self.model,
        )

        elapsed = time.time() - start
        print(f"Model loaded in {elapsed:.1f}s", flush=True)

        # List available voices
        voices = [f.stem for f in self.voices_dir.glob("*.pt")]
        print(f"Available voices ({len(voices)}): {', '.join(sorted(voices)[:5])}...", flush=True)

    def get_voice_path(self, voice: str) -> Path:
        """Resolve voice name to file path."""
        voice_path = self.voices_dir / f"{voice}.pt"
        if not voice_path.exists():
            available = sorted([f.stem for f in self.voices_dir.glob("*.pt")])
            raise ValueError(f"Voice '{voice}' not found. Available: {available}")
        return voice_path

    def synthesize(self, text: str, voice: str = "af_bella", lang: str = "a", speed: float = 1.0) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (e.g., 'af_bella', 'am_adam')
            lang: Language code ('a' for American, 'b' for British)
            speed: Playback speed multiplier (default: 1.0)

        Returns:
            float32 numpy array of audio samples at 24000 Hz
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        # Resolve voice to path
        voice_path = self.get_voice_path(voice)

        # Update pipeline language if different
        if self.pipeline.lang_code != lang:
            self.pipeline = KPipeline(
                lang_code=lang,
                repo_id="hexgrad/Kokoro-82M",
                model=self.model,
            )

        # Generate audio (may yield multiple chunks)
        generator = self.pipeline(text, voice=str(voice_path), speed=speed)

        # Concatenate all chunks
        audio_chunks = []
        for gs, ps, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_chunks).astype(np.float32)

    def synthesize_with_blend(self, text: str, lang: str = "a", speed: float = 1.0) -> np.ndarray:
        """Synthesize using the cached blended voice tensor."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")
        if self._blended_voice is None:
            raise RuntimeError("No blended voice cached")

        if self.pipeline.lang_code != lang:
            self.pipeline = KPipeline(
                lang_code=lang,
                repo_id="hexgrad/Kokoro-82M",
                model=self.model,
            )

        generator = self.pipeline(text, voice=self._blended_voice, speed=speed)
        audio_chunks = []
        for gs, ps, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_chunks).astype(np.float32)

    def compute_blend(self, weights: dict[str, float]) -> tuple[Optional[torch.FloatTensor], list[str]]:
        """Weighted interpolation of voice embeddings. Returns (tensor, missing_voices)."""
        tensors = []
        w_vals = []
        missing = []
        for voice_id, weight in weights.items():
            if weight <= 0:
                continue
            try:
                voice_path = self.get_voice_path(voice_id)
                tensor = torch.load(str(voice_path), weights_only=True)
                tensors.append(tensor * weight)
                w_vals.append(weight)
            except ValueError:
                missing.append(voice_id)
                print(f"[WARN] Voice '{voice_id}' not found — skipped from blend", flush=True)
        if not tensors:
            return None, missing
        total_weight = sum(w_vals)
        blended = torch.stack(tensors).sum(dim=0) / total_weight
        return blended, missing

    def handle_request(self, data: str) -> dict:
        """
        Process a single JSON request.

        Args:
            data: JSON string (without trailing newline)

        Returns:
            Response dict with status, audio/error, sample_rate, duration_ms
        """
        start = time.time()

        try:
            request = json.loads(data)
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}

        action = request.get("action")

        if action == "synthesize":
            return self._handle_synthesize(request, start)
        elif action == "blend_voices":
            return self._handle_blend_voices(request, start)
        elif action == "list_voices":
            return self._handle_list_voices()
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    def _handle_synthesize(self, request: dict, start: float) -> dict:
        text = request.get("text")
        if not text:
            return {"status": "error", "error": "Missing 'text' field"}

        voice = request.get("voice", "af_bella")
        lang = request.get("lang", "a")
        use_blend = request.get("use_blend", False)
        speed = request.get("speed", 1.0)

        if lang not in ("a", "b"):
            return {"status": "error", "error": f"Invalid lang: {lang} (expected 'a' or 'b')"}

        try:
            if use_blend and self._blended_voice is not None:
                print(f"[TTS] Synthesize via blended voice ({len(text)} chars, speed={speed})", flush=True)
                audio = self.synthesize_with_blend(text, lang, speed=speed)
            else:
                print(f"[TTS] Synthesize via single voice '{voice}' ({len(text)} chars, speed={speed})", flush=True)
                audio = self.synthesize(text, voice, lang, speed=speed)
        except ValueError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": f"Synthesis failed: {e}"}

        audio_hex = audio.tobytes().hex()
        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "status": "success",
            "audio": audio_hex,
            "sample_rate": self.SAMPLE_RATE,
            "duration_ms": elapsed_ms,
        }

    def _handle_blend_voices(self, request: dict, start: float) -> dict:
        weights = request.get("weights", {})
        if not isinstance(weights, dict):
            return {"status": "error", "error": "weights must be a dict of voice_id -> float"}

        active_weights = {k: v for k, v in weights.items() if isinstance(v, (int, float)) and v > 0}

        if not active_weights:
            self._blended_voice = None
            print("[BLEND] Cleared — no active weights", flush=True)
            return {"status": "success", "voice_count": 0, "total_weight": 0, "missing": []}

        blended, missing = self.compute_blend(active_weights)
        self._blended_voice = blended

        voice_count = len(active_weights) - len(missing)
        total_weight = sum(v for k, v in active_weights.items() if k not in missing)
        elapsed_ms = int((time.time() - start) * 1000)

        if blended is not None:
            print(
                f"[BLEND] Computed: {voice_count} voices, "
                f"total_weight={total_weight:.2f}, {elapsed_ms}ms"
                + (f", missing: {missing}" if missing else ""),
                flush=True,
            )
        else:
            print(f"[BLEND] Failed — all voices missing: {missing}", flush=True)

        return {
            "status": "success",
            "voice_count": voice_count,
            "total_weight": total_weight,
            "missing": missing,
        }

    def _handle_list_voices(self) -> dict:
        voices = sorted([f.stem for f in self.voices_dir.glob("*.pt")])
        return {"status": "success", "voices": voices}

    def handle_connection(self, conn: socket.socket, addr: tuple) -> None:
        """Handle a single client connection."""
        print(f"Connection from {addr}", flush=True)

        try:
            # Read until newline
            buffer = b""
            while True:
                chunk = conn.recv(65536)  # 64KB chunks
                if not chunk:
                    break
                buffer += chunk
                if b"\n" in buffer:
                    break

            if not buffer:
                return

            # Parse request (strip newline)
            data = buffer.decode("utf-8").strip()
            if not data:
                return

            # Process and respond
            response = self.handle_request(data)
            response_json = json.dumps(response) + "\n"
            conn.sendall(response_json.encode("utf-8"))

            # Log result
            if response.get("status") == "success":
                audio_bytes = len(response.get("audio", "")) // 2  # hex is 2 chars per byte
                audio_samples = audio_bytes // 4  # float32 is 4 bytes
                audio_duration_ms = int(audio_samples / self.SAMPLE_RATE * 1000)
                print(
                    f"  -> success: {audio_duration_ms}ms audio, "
                    f"{response.get('duration_ms', 0)}ms inference",
                    flush=True,
                )
            else:
                print(f"  -> error: {response.get('error')}", flush=True)

        except Exception as e:
            print(f"Connection error: {e}", flush=True)
            try:
                error_response = json.dumps({"status": "error", "error": str(e)}) + "\n"
                conn.sendall(error_response.encode("utf-8"))
            except:
                pass

        finally:
            conn.close()

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals (SIGINT, SIGTERM)."""
        print("\nShutdown signal received...", flush=True)
        self._shutdown = True

    def serve_forever(self) -> None:
        """Start the server and listen for connections."""
        # Register signal handlers for graceful shutdown on Windows
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(1)

        # Set timeout so accept() doesn't block forever - allows signal checking
        self._socket.settimeout(1.0)

        print(f"Kokoro TTS server ready on {self.host}:{self.port}", flush=True)

        try:
            while not self._shutdown:
                try:
                    conn, addr = self._socket.accept()
                    self.handle_connection(conn, addr)
                except socket.timeout:
                    # Check shutdown flag and loop
                    continue
        except KeyboardInterrupt:
            print("\nShutting down...", flush=True)
        finally:
            print("Server stopped.", flush=True)
            self._socket.close()


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro TTS Server - TCP bridge for speech synthesis"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="Port to listen on (default: 5556)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Path to models directory containing config.json, kokoro-v1_0.pth, and voices/",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device for model inference (default: cuda). Options: cuda, cpu, cuda:0, cuda:1",
    )

    args = parser.parse_args()

    server = KokoroTTSServer(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        device=args.device,
    )
    server.load_model()
    server.serve_forever()


if __name__ == "__main__":
    main()
