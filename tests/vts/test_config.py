"""Tests for VTubeStudioConfig parsing and persistence (NANO-060a)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.config import VTubeStudioConfig, OrchestratorConfig


class TestVTubeStudioConfig:
    """Tests for VTubeStudioConfig dataclass."""

    def test_defaults(self):
        cfg = VTubeStudioConfig()
        assert cfg.enabled is False
        assert cfg.host == "localhost"
        assert cfg.port == 8001
        assert cfg.token_path == "./vtubeStudio_token.txt"
        assert cfg.plugin_name == "spindl"
        assert cfg.developer == "spindl"
        assert cfg.expressions == {}
        assert cfg.positions == {}
        assert cfg.thinking_hotkey == ""
        assert cfg.idle_hotkey == ""

    def test_from_dict_full(self):
        data = {
            "enabled": True,
            "host": "192.168.1.100",
            "port": 9001,
            "token_path": "/tmp/vts_token.txt",
            "plugin_name": "my-agent",
            "developer": "my-dev",
            "expressions": {
                "happy": "expression_happy.exp3.json",
                "sad": "expression_sad.exp3.json",
            },
            "positions": {
                "chat": {"x": 0.4, "y": -1.4, "size": -35, "rotation": 0},
            },
        }
        cfg = VTubeStudioConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.host == "192.168.1.100"
        assert cfg.port == 9001
        assert cfg.token_path == "/tmp/vts_token.txt"
        assert cfg.plugin_name == "my-agent"
        assert cfg.developer == "my-dev"
        assert cfg.expressions["happy"] == "expression_happy.exp3.json"
        assert cfg.positions["chat"]["x"] == 0.4

    def test_from_dict_empty(self):
        """Empty dict uses all defaults."""
        cfg = VTubeStudioConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.host == "localhost"
        assert cfg.port == 8001
        assert cfg.expressions == {}

    def test_from_dict_minimal(self):
        """Only 'enabled' set, everything else defaults."""
        data = {"enabled": True}
        cfg = VTubeStudioConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.host == "localhost"
        assert cfg.port == 8001

    def test_from_dict_partial(self):
        """Partial config merges with defaults."""
        data = {
            "enabled": True,
            "port": 9999,
            "expressions": {"angry": "angry.exp3.json"},
        }
        cfg = VTubeStudioConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.host == "localhost"  # default
        assert cfg.port == 9999
        assert cfg.expressions == {"angry": "angry.exp3.json"}
        assert cfg.positions == {}  # default

    def test_from_dict_hotkey_fields(self):
        """thinking_hotkey and idle_hotkey parse from dict."""
        data = {
            "thinking_hotkey": "ThinkingAnim",
            "idle_hotkey": "IdleBreathing",
        }
        cfg = VTubeStudioConfig.from_dict(data)
        assert cfg.thinking_hotkey == "ThinkingAnim"
        assert cfg.idle_hotkey == "IdleBreathing"

    def test_from_dict_hotkey_fields_default_empty(self):
        """Missing hotkey fields default to empty string."""
        cfg = VTubeStudioConfig.from_dict({})
        assert cfg.thinking_hotkey == ""
        assert cfg.idle_hotkey == ""


class TestOrchestratorConfigVTubeStudio:
    """Tests for VTubeStudio in OrchestratorConfig._from_dict()."""

    def test_from_dict_includes_vtubestudio(self):
        """OrchestratorConfig._from_dict() parses vtubestudio section."""
        data = {
            "vtubestudio": {
                "enabled": True,
                "host": "10.0.0.1",
                "port": 8002,
                "expressions": {"happy": "happy.exp3.json"},
            },
        }
        config = OrchestratorConfig._from_dict(data)
        assert config.vtubestudio_config.enabled is True
        assert config.vtubestudio_config.host == "10.0.0.1"
        assert config.vtubestudio_config.port == 8002
        assert config.vtubestudio_config.expressions["happy"] == "happy.exp3.json"

    def test_from_dict_no_vtubestudio_uses_defaults(self):
        """Missing vtubestudio section uses all defaults."""
        config = OrchestratorConfig._from_dict({})
        assert config.vtubestudio_config.enabled is False
        assert config.vtubestudio_config.host == "localhost"
        assert config.vtubestudio_config.port == 8001

    def test_from_yaml_with_vtubestudio(self, tmp_path: Path):
        """Full YAML round-trip with vtubestudio section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
vtubestudio:
  enabled: true
  host: "127.0.0.1"
  port: 8001
  token_path: "./my_token.txt"
  plugin_name: "test-plugin"
  developer: "test-dev"
  expressions:
    happy: "happy.exp3.json"
    sad: "sad.exp3.json"
  positions:
    chat:
      x: 0.4
      y: -1.4
      size: -35
      rotation: 0
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        assert config.vtubestudio_config.enabled is True
        assert config.vtubestudio_config.host == "127.0.0.1"
        assert config.vtubestudio_config.plugin_name == "test-plugin"
        assert config.vtubestudio_config.expressions["happy"] == "happy.exp3.json"
        assert config.vtubestudio_config.positions["chat"]["x"] == 0.4


class TestSaveToYamlVTubeStudio:
    """Tests for save_to_yaml VTubeStudio enabled persistence."""

    def test_save_to_yaml_updates_vts_enabled(self, tmp_path: Path):
        """save_to_yaml persists vtubestudio enabled flag."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
vtubestudio:
  enabled: false
  host: "localhost"
  port: 8001
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vtubestudio_config.enabled = True
        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.vtubestudio_config.enabled is True

    def test_save_to_yaml_vts_section_isolated(self, tmp_path: Path):
        """save_to_yaml only updates enabled in vtubestudio section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
stimuli:
  enabled: true
  patience:
    seconds: 60

vtubestudio:
  enabled: false
  host: "localhost"
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vtubestudio_config.enabled = True
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        # Stimuli enabled should remain true (not changed by VTS write)
        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.vtubestudio_config.enabled is True
        # Verify stimuli wasn't touched - check raw content
        lines = content.split("\n")
        in_stimuli = False
        for line in lines:
            if line.strip() == "stimuli:":
                in_stimuli = True
            elif line.strip() == "vtubestudio:":
                in_stimuli = False
            elif in_stimuli and "enabled:" in line:
                assert "true" in line, f"Stimuli enabled was modified: {line}"
