#!/usr/bin/env python3
"""
NeMo STT Server - TCP bridge for NVIDIA Parakeet transcription.

Runs in WSL with CUDA, accepts audio from Windows via TCP.

Protocol:
    Request:  {"action": "transcribe", "audio": "<hex>", "sample_rate": 16000}\n
    Response: {"status": "success", "text": "...", "duration_ms": N}\n
    Error:    {"status": "error", "error": "..."}\n
"""

import json
import socket
import sys
import time
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

# NeMo imports (heavy - loaded once at startup)
import nemo.collections.asr as nemo_asr


class NeMoSTTServer:
    """TCP server for NeMo-based speech-to-text."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        self.model: Optional[nemo_asr.models.EncDecRNNTBPEModel] = None
        self._socket: Optional[socket.socket] = None

    def load_model(self) -> None:
        """Load the Parakeet model. This is slow (~15-20s cold, ~8s warm)."""
        print("Loading Parakeet model...", flush=True)
        start = time.time()

        # nvidia/parakeet-tdt-0.6b-v2 is a FastConformer Transducer model
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2"
        )

        # Move to GPU if available
        if hasattr(self.model, "cuda"):
            self.model = self.model.cuda()

        # CRITICAL: Change decoding strategy from 'greedy_batch' to 'greedy'
        # to avoid CUDA Graphs issues in WSL2 (error 35)
        decoding_cfg = OmegaConf.create({
            'strategy': 'greedy',  # NOT 'greedy_batch' - that uses CUDA Graphs
            'model_type': 'tdt',
            'durations': [0, 1, 2, 3, 4],
            'greedy': {'max_symbols': 10}
        })
        self.model.change_decoding_strategy(decoding_cfg)
        print("Decoding strategy set to 'greedy' (CUDA Graphs bypassed)", flush=True)

        elapsed = time.time() - start
        print(f"Model loaded in {elapsed:.1f}s", flush=True)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using the loaded model.

        Args:
            audio: float32 numpy array, mono
            sample_rate: Sample rate in Hz (model expects 16000)

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # NeMo expects audio as a list of numpy arrays or file paths
        # For in-memory audio, we pass a list with one array
        transcriptions = self.model.transcribe([audio])

        # RNNT models (like Parakeet) return Hypothesis objects, not strings
        # Extract the text from the first result
        if transcriptions and len(transcriptions) > 0:
            result = transcriptions[0]
            # Handle both string and Hypothesis object returns
            if hasattr(result, 'text'):
                return result.text
            return str(result)
        return ""

    def handle_request(self, data: str) -> dict:
        """
        Process a single JSON request.

        Args:
            data: JSON string (without trailing newline)

        Returns:
            Response dict with status, text/error, duration_ms
        """
        start = time.time()

        try:
            request = json.loads(data)
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}

        action = request.get("action")
        if action != "transcribe":
            return {"status": "error", "error": f"Unknown action: {action}"}

        # Decode audio from hex
        audio_hex = request.get("audio")
        if not audio_hex:
            return {"status": "error", "error": "Missing 'audio' field"}

        try:
            audio_bytes = bytes.fromhex(audio_hex)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
        except (ValueError, TypeError) as e:
            return {"status": "error", "error": f"Invalid audio encoding: {e}"}

        sample_rate = request.get("sample_rate", 16000)

        # Validate audio
        if len(audio) == 0:
            return {"status": "error", "error": "Empty audio buffer"}

        if sample_rate != 16000:
            return {
                "status": "error",
                "error": f"Unsupported sample rate: {sample_rate} (expected 16000)",
            }

        # Transcribe
        try:
            text = self.transcribe(audio, sample_rate)
        except Exception as e:
            return {"status": "error", "error": f"Transcription failed: {e}"}

        elapsed_ms = int((time.time() - start) * 1000)

        return {"status": "success", "text": text, "duration_ms": elapsed_ms}

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

            print(
                f"  -> {response.get('status')}: "
                f"{len(response.get('text', ''))} chars, "
                f"{response.get('duration_ms', 0)}ms",
                flush=True,
            )

        except Exception as e:
            print(f"Connection error: {e}", flush=True)
            try:
                error_response = json.dumps({"status": "error", "error": str(e)}) + "\n"
                conn.sendall(error_response.encode("utf-8"))
            except:
                pass

        finally:
            conn.close()

    def serve_forever(self) -> None:
        """Start the server and listen for connections."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(1)

        print(f"NeMo server ready on {self.host}:{self.port}", flush=True)

        try:
            while True:
                conn, addr = self._socket.accept()
                self.handle_connection(conn, addr)
        except KeyboardInterrupt:
            print("\nShutting down...", flush=True)
        finally:
            self._socket.close()


def main():
    server = NeMoSTTServer(host="127.0.0.1", port=5555)
    server.load_model()
    server.serve_forever()


if __name__ == "__main__":
    main()
