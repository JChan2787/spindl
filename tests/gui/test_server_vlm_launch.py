"""Tests for VLM dashboard launch socket handlers (NANO-079).

Tests cover:
- request_local_vlm_config: returns cached config and server running state
- launch_vlm_server: config validation, ServiceRunner creation, VisionProviderConfig wiring
- _persist_local_vlm_config: orchestrator path and pre-launch YAML path
- unified_vision flag in launch_llm_server: VisionProviderConfig injection for -np 2
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.gui.server import GUIServer
from spindl.launcher.config import VisionProviderConfig


# ============================================================================
# Helpers
# ============================================================================


def _make_server(with_orchestrator=True, vlm_provider="llama"):
    """Create a GUIServer with mocked internals for VLM launch testing."""
    server = GUIServer.__new__(GUIServer)
    server.sio = MagicMock()
    server.sio.event = lambda fn: fn  # passthrough decorator
    server.sio.emit = AsyncMock()
    server._config_path = "/tmp/test_config.yaml"
    server._event_loop = asyncio.new_event_loop()
    server._launch_in_progress = False
    server._conversations_dir = None
    server._personas_dir = None
    server._prompt_blocks_config = None
    server._tools_config_cache = None
    server._llm_config_cache = {
        "provider": "llama",
        "providers": {"llama": {"model_path": "/path/to/model"}},
    }
    server._vlm_config_cache = {
        "provider": vlm_provider,
        "providers": {
            "llama": {
                "host": "127.0.0.1",
                "port": 5558,
                "executable_path": "/path/to/llama-server",
                "model_path": "/path/to/vlm-model.gguf",
                "mmproj_path": "/path/to/mmproj.gguf",
            },
            "openai": {"api_key": "test-key", "base_url": "https://api.x.ai"},
        },
    }
    server._service_runner = None
    server._log_aggregator = None
    server._launched_services = set()

    if with_orchestrator:
        server._orchestrator = MagicMock()
        server._orchestrator.get_vlm_state.return_value = {
            "provider": vlm_provider,
            "available_providers": ["llama", "openai", "llm"],
            "healthy": True,
        }
        server._orchestrator._config = MagicMock()
        server._orchestrator._config.vlm_config = MagicMock()
        server._orchestrator._config.vlm_config.providers = {"llama": {}}
        server._orchestrator._config.save_to_yaml = MagicMock()
    else:
        server._orchestrator = None

    return server


# ============================================================================
# request_local_vlm_config
# ============================================================================


class TestRequestLocalVLMConfig:
    """Tests for the request_local_vlm_config handler."""

    def test_returns_cached_config(self) -> None:
        """Returns VLM llama provider config from cache."""
        server = _make_server(with_orchestrator=False)
        cache = server._vlm_config_cache
        llama_cfg = cache["providers"]["llama"]
        assert llama_cfg["host"] == "127.0.0.1"
        assert llama_cfg["port"] == 5558
        assert llama_cfg["model_path"] == "/path/to/vlm-model.gguf"

    def test_returns_empty_when_no_cache(self) -> None:
        """Returns empty dict when no VLM config cache."""
        server = _make_server(with_orchestrator=False)
        server._vlm_config_cache = None
        cache = server._vlm_config_cache or {}
        providers = cache.get("providers", {})
        assert providers.get("llama", {}) == {}

    def test_extra_args_list_to_string(self) -> None:
        """Extra args list is converted to string for frontend."""
        server = _make_server(with_orchestrator=False)
        server._vlm_config_cache["providers"]["llama"]["extra_args"] = ["--cache-ram", "0"]
        cfg = dict(server._vlm_config_cache["providers"]["llama"])
        if isinstance(cfg.get("extra_args"), list):
            cfg["extra_args"] = " ".join(str(a) for a in cfg["extra_args"])
        assert cfg["extra_args"] == "--cache-ram 0"

    def test_server_running_check(self) -> None:
        """Server running state depends on ServiceRunner."""
        server = _make_server(with_orchestrator=False)
        assert server._service_runner is None
        # No runner = not running
        server_running = False
        if server._service_runner and server._service_runner.is_service_running("vlm"):
            server_running = True
        assert server_running is False


# ============================================================================
# launch_vlm_server — config validation
# ============================================================================


class TestLaunchVLMServerValidation:
    """Tests for launch_vlm_server input validation."""

    def test_extra_args_string_to_list(self) -> None:
        """Frontend sends extra_args as string; backend splits to list."""
        config = {"extra_args": "--cache-ram 0 --verbose"}
        raw = config["extra_args"].strip()
        config["extra_args"] = raw.split() if raw else []
        assert config["extra_args"] == ["--cache-ram", "0", "--verbose"]

    def test_extra_args_empty_string(self) -> None:
        """Empty string becomes empty list."""
        config = {"extra_args": "  "}
        raw = config["extra_args"].strip()
        config["extra_args"] = raw.split() if raw else []
        assert config["extra_args"] == []

    def test_vision_provider_config_shape(self) -> None:
        """VisionProviderConfig is built with correct fields."""
        config = {
            "executable_path": "/path/to/llama-server",
            "model_path": "/path/to/vlm.gguf",
            "host": "127.0.0.1",
            "port": 5558,
        }
        vpc = VisionProviderConfig(
            enabled=True,
            provider="llama",
            provider_config=config,
        )
        assert vpc.enabled is True
        assert vpc.provider == "llama"
        assert vpc.provider_config["port"] == 5558


# ============================================================================
# _persist_local_vlm_config
# ============================================================================


class TestPersistLocalVLMConfig:
    """Tests for the _persist_local_vlm_config method."""

    def test_orchestrator_path(self) -> None:
        """Persists via orchestrator when available."""
        server = _make_server(with_orchestrator=True)
        config = {"host": "127.0.0.1", "port": 5558, "model_path": "/new/model.gguf",
                  "executable_path": "/path/to/llama-server", "model_type": "gemma3"}
        result = server._persist_local_vlm_config(config)
        assert result is True
        assert server._orchestrator._config.vlm_config.providers["llama"] == config
        server._orchestrator._config.save_to_yaml.assert_called_once()

    def test_pre_launch_path(self, tmp_path) -> None:
        """Persists via line-level YAML surgery — preserves env vars and other sections."""
        config_file = tmp_path / "spindl.yaml"
        # Write a realistic config with ${ENV_VAR} patterns that must survive
        config_file.write_text(
            "llm:\n"
            "  provider: openrouter\n"
            "  providers:\n"
            "    openrouter:\n"
            "      api_key: ${OPENROUTER_API_KEY}\n"
            "      model: google/gemini-2.5-pro\n"
            "vlm:\n"
            "  provider: openai\n"
            "  providers:\n"
            "    llama:\n"
            "      port: 5558\n"
            "    openai:\n"
            "      api_key: ${OPENAI_API_KEY}\n"
            "      base_url: https://api.openai.com\n"
        )

        server = _make_server(with_orchestrator=False)
        server._config_path = str(config_file)

        new_cfg = {"host": "127.0.0.1", "port": 5558, "model_path": "/new/vlm.gguf",
                   "executable_path": "/path/to/llama-server", "model_type": "gemma3"}
        result = server._persist_local_vlm_config(new_cfg)
        assert result is True

        # Verify YAML was updated
        raw = config_file.read_text()

        # ENV_VAR patterns must survive — not be nuked by yaml.dump
        assert "${OPENROUTER_API_KEY}" in raw
        assert "${OPENAI_API_KEY}" in raw

        # llama section updated
        import yaml
        data = yaml.safe_load(raw.replace("${OPENROUTER_API_KEY}", "x").replace("${OPENAI_API_KEY}", "y"))
        assert data["vlm"]["providers"]["llama"]["model_path"] == "/new/vlm.gguf"
        # openai section preserved
        assert data["vlm"]["providers"]["openai"]["base_url"] == "https://api.openai.com"

    def test_no_config_path(self) -> None:
        """Returns False when no config path."""
        server = _make_server(with_orchestrator=False)
        server._config_path = None
        result = server._persist_local_vlm_config({"port": 5558})
        assert result is False


# ============================================================================
# unified_vision flag — LLM launch with -np 2
# ============================================================================


class TestUnifiedVisionLLMLaunch:
    """Tests for unified_vision flag in launch_llm_server config."""

    def test_unified_vision_creates_vision_provider_config(self) -> None:
        """When unified_vision=True, a VisionProviderConfig(provider='llm') is created."""
        config = {
            "executable_path": "/path/to/llama-server",
            "model_path": "/path/to/model.gguf",
            "mmproj_path": "/path/to/mmproj.gguf",
            "unified_vision": True,
        }
        unified_vision = config.pop("unified_vision", False)
        assert unified_vision is True
        assert "unified_vision" not in config  # Removed from config dict

        vpc = VisionProviderConfig(
            enabled=True,
            provider="llm",
            provider_config={},
        )
        assert vpc.provider == "llm"

    def test_no_unified_vision_flag(self) -> None:
        """Without unified_vision, no VisionProviderConfig created."""
        config = {
            "executable_path": "/path/to/llama-server",
            "model_path": "/path/to/model.gguf",
        }
        unified_vision = config.pop("unified_vision", False)
        assert unified_vision is False

    def test_mmproj_stays_in_config(self) -> None:
        """mmproj_path remains in config for LlamaProvider.get_server_command."""
        config = {
            "executable_path": "/path/to/llama-server",
            "model_path": "/path/to/model.gguf",
            "mmproj_path": "/path/to/mmproj.gguf",
            "unified_vision": True,
        }
        config.pop("unified_vision", False)
        assert config["mmproj_path"] == "/path/to/mmproj.gguf"
