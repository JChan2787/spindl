#!/usr/bin/env python3
"""
SpindL TTS utility - send text to TTS server and play through speakers.

Usage:
    python speak_to_user.py "Hello User"
    python speak_to_user.py --voice af_bella "Hello User"
"""

import argparse
import json
import socket
import sys

import numpy as np
import sounddevice as sd


def speak(text: str, voice: str = "af_bella", lang: str = "a") -> None:
    """Send text to TTS server and play audio."""
    # Connect to TTS server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', 5556))
    except ConnectionRefusedError:
        print("Error: TTS server not running (port 5556)")
        sys.exit(1)

    # Send request
    request = {
        'action': 'synthesize',
        'text': text,
        'voice': voice,
        'lang': lang
    }
    sock.sendall((json.dumps(request) + '\n').encode('utf-8'))

    # Read response
    buffer = b''
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buffer += chunk
        if b'\n' in buffer:
            break

    sock.close()

    # Parse response
    response = json.loads(buffer.decode('utf-8').strip())
    if response['status'] == 'success':
        audio = np.frombuffer(bytes.fromhex(response['audio']), dtype=np.float32)
        print(f'Playing {len(audio)/24000:.2f}s of audio...')
        sd.play(audio, samplerate=24000)
        sd.wait()
    else:
        print(f"Error: {response.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpindL TTS utility")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--voice", default="af_bella", help="Voice ID (default: af_bella)")
    parser.add_argument("--lang", default="a", choices=["a", "b"], help="Language: a=American, b=British")

    args = parser.parse_args()
    speak(args.text, args.voice, args.lang)
