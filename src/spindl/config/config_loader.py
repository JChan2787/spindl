"""
Configuration loader for spindl.

Loads the main YAML configuration file with settings for all services.
"""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "./config/spindl.yaml"


def get_config_path() -> str:
    """Return the config file path from SPINDL_CONFIG env var or default."""
    return os.environ.get("SPINDL_CONFIG", DEFAULT_CONFIG_PATH)


def load_config(path: str | None = None) -> dict:
    """
    Load main configuration from YAML file.

    Args:
        path: Path to configuration file. If None, reads from
              SPINDL_CONFIG env var, falling back to ./config/spindl.yaml

    Returns:
        Configuration dict with sections:
            - stt: STT service settings (host, port, timeout)
            - tts: TTS service settings (host, port, timeout)
            - llm: LLM service settings (url, timeout, temperature, etc.)
            - audio: Audio settings (capture_rate, playback_rate, chunk_size)
            - vad: VAD settings (threshold, min_speech_ms, min_silence_ms)
            - persona: Persona settings (default, directory)

    Raises:
        FileNotFoundError: Config file not found
        ValueError: Invalid config format or parse error
    """
    config_path = Path(path if path is not None else get_config_path())

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse config: {e}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Config must be a YAML mapping, got {type(config).__name__}"
        )

    return config
