"""
Tests for LauncherConfig (NANO-015 Session 5).

Validates provider-based TTS configuration and service config parsing.
"""

import pytest
from pathlib import Path

from spindl.launcher.config import (
    LauncherConfig,
    ServiceConfig,
    HealthCheckConfig,
    TTSProviderConfig,
    load_launcher_config,
)


class TestHealthCheckConfig:
    """Tests for HealthCheckConfig dataclass."""

    def test_tcp_requires_host_and_port(self) -> None:
        """TCP health check requires both host and port."""
        with pytest.raises(ValueError, match="host.*port"):
            HealthCheckConfig(type="tcp", host=None, port=None)

    def test_tcp_with_host_only_fails(self) -> None:
        """TCP health check with only host fails."""
        with pytest.raises(ValueError, match="host.*port"):
            HealthCheckConfig(type="tcp", host="127.0.0.1", port=None)

    def test_http_requires_url(self) -> None:
        """HTTP health check requires URL."""
        with pytest.raises(ValueError, match="url"):
            HealthCheckConfig(type="http", url=None)

    def test_provider_type_requires_no_additional_fields(self) -> None:
        """Provider health check doesn't require additional fields."""
        config = HealthCheckConfig(type="provider", timeout=60)
        assert config.type == "provider"
        assert config.timeout == 60

    def test_none_type_requires_nothing(self) -> None:
        """None health check requires no configuration."""
        config = HealthCheckConfig(type="none")
        assert config.type == "none"


class TestServiceConfig:
    """Tests for ServiceConfig dataclass."""

    def test_command_can_be_none(self) -> None:
        """Service command can be None for provider-driven services."""
        config = ServiceConfig(
            name="tts",
            platform="native",
            command=None,
            health_check=HealthCheckConfig(type="provider"),
        )
        assert config.command is None
        assert config.name == "tts"

    def test_wsl_requires_distro(self) -> None:
        """WSL platform requires wsl_distro."""
        with pytest.raises(ValueError, match="wsl_distro"):
            ServiceConfig(
                name="stt",
                platform="wsl",
                command="python server.py",
                health_check=HealthCheckConfig(type="none"),
                wsl_distro=None,
            )


class TestTTSProviderConfig:
    """Tests for TTSProviderConfig dataclass."""

    def test_default_values(self) -> None:
        """TTSProviderConfig has sensible defaults."""
        config = TTSProviderConfig()
        assert config.provider == "kokoro"
        assert config.plugin_paths == []
        assert config.provider_config == {}


class TestLoadLauncherConfig:
    """Tests for load_launcher_config function."""

    def test_loads_tts_provider_config(self, tmp_path: Path) -> None:
        """load_launcher_config parses TTS provider configuration."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
tts:
  provider: "kokoro"
  plugin_paths:
    - "./plugins/tts"
  providers:
    kokoro:
      host: "127.0.0.1"
      port: 5556
      voice: "af_bella"

launcher:
  services:
    tts:
      enabled: true
      platform: "native"
      health_check:
        type: "provider"
        timeout: 60
""")

        config = load_launcher_config(str(config_file))

        assert config.tts_provider_config is not None
        assert config.tts_provider_config.provider == "kokoro"
        assert config.tts_provider_config.plugin_paths == ["./plugins/tts"]
        assert config.tts_provider_config.provider_config["host"] == "127.0.0.1"
        assert config.tts_provider_config.provider_config["port"] == 5556

    def test_service_without_command(self, tmp_path: Path) -> None:
        """Service can be defined without command field."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    tts:
      enabled: true
      platform: "native"
      health_check:
        type: "provider"
""")

        config = load_launcher_config(str(config_file))

        assert config.services["tts"].command is None

    def test_provider_health_check_type(self, tmp_path: Path) -> None:
        """Provider health check type is parsed correctly."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    tts:
      platform: "native"
      command: "python server.py"
      health_check:
        type: "provider"
        timeout: 60
""")

        config = load_launcher_config(str(config_file))

        assert config.services["tts"].health_check.type == "provider"
        assert config.services["tts"].health_check.timeout == 60

    def test_missing_tts_section_returns_none(self, tmp_path: Path) -> None:
        """Missing tts section results in None tts_provider_config."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    llm:
      platform: "native"
      command: "llama-server"
      health_check:
        type: "http"
        url: "http://127.0.0.1:5557/health"
""")

        config = load_launcher_config(str(config_file))

        assert config.tts_provider_config is None

    def test_backward_compatible_with_command(self, tmp_path: Path) -> None:
        """Services with explicit command still work."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    stt:
      platform: "wsl"
      wsl_distro: "Ubuntu"
      command: "python nemo_server.py"
      health_check:
        type: "tcp"
        host: "127.0.0.1"
        port: 5555
""")

        config = load_launcher_config(str(config_file))

        assert config.services["stt"].command == "python nemo_server.py"
        assert config.services["stt"].platform == "wsl"
        assert config.services["stt"].wsl_distro == "Ubuntu"


class TestEmbeddingServiceConfig:
    """Tests for embedding server as a generic command-driven service (NANO-043 Phase 5)."""

    def test_embedding_parsed_as_generic_service(self, tmp_path: Path) -> None:
        """Embedding service is parsed by generic _parse_service."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    embedding:
      enabled: true
      platform: native
      command: '"llama-server" --embedding -m "model.gguf" --port 5559'
      health_check:
        type: http
        url: http://127.0.0.1:5559/health
        timeout: 60
""")
        config = load_launcher_config(str(config_file))
        assert "embedding" in config.services
        emb = config.services["embedding"]
        assert emb.enabled is True
        assert emb.platform == "native"
        assert emb.command is not None
        assert "--embedding" in emb.command
        assert emb.health_check.type == "http"
        assert emb.health_check.url == "http://127.0.0.1:5559/health"
        assert emb.health_check.timeout == 60

    def test_embedding_in_startup_order(self, tmp_path: Path) -> None:
        """Embedding service appears in startup order."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    embedding:
      enabled: true
      platform: native
      command: 'llama-server --embedding'
      health_check:
        type: http
        url: http://127.0.0.1:5559/health
    llm:
      enabled: true
      platform: native
      command: 'llama-server'
      health_check:
        type: http
        url: http://127.0.0.1:5557/health
""")
        config = load_launcher_config(str(config_file))
        order = config.get_startup_order()
        assert "embedding" in order
        assert "llm" in order

    def test_embedding_disabled(self, tmp_path: Path) -> None:
        """Disabled embedding service is still parsed."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    embedding:
      enabled: false
      platform: native
      command: ''
      health_check:
        type: none
""")
        config = load_launcher_config(str(config_file))
        assert "embedding" in config.services
        assert config.services["embedding"].enabled is False

    def test_embedding_coexists_with_other_services(self, tmp_path: Path) -> None:
        """Embedding service coexists with LLM, STT, TTS services."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
launcher:
  services:
    llm:
      enabled: true
      platform: native
      health_check:
        type: provider
        timeout: 90
    stt:
      enabled: true
      platform: native
      command: 'python nemo_server.py'
      health_check:
        type: tcp
        host: 127.0.0.1
        port: 5555
    tts:
      enabled: true
      platform: native
      health_check:
        type: provider
        timeout: 60
    embedding:
      enabled: true
      platform: native
      command: '"llama-server" --embedding -m "model.gguf" --port 5559'
      health_check:
        type: http
        url: http://127.0.0.1:5559/health
""")
        config = load_launcher_config(str(config_file))
        assert len(config.services) == 4
        assert "embedding" in config.services
        assert "llm" in config.services
        assert "stt" in config.services
        assert "tts" in config.services
