"""Tests for character.default YAML persistence (NANO-077).

Tests cover:
- save_to_yaml updates character.default value
- save_to_yaml does not corrupt other sections
- save_to_yaml preserves comments and formatting
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.config import OrchestratorConfig


SAMPLE_YAML = """\
# spindl configuration
character:
  default: spindle
  directory: ./characters

llm:
  provider: llama
  providers:
    llama:
      host: 127.0.0.1
      port: 5557

vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 500

pipeline:
  conversations_dir: ./conversations
  resume_session: true
  summarization_threshold: 0.6
"""


class TestCharacterDefaultPersistence:

    def test_character_default_updated(self, tmp_path) -> None:
        yaml_path = tmp_path / "spindl.yaml"
        yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")

        config = OrchestratorConfig.from_yaml(str(yaml_path))
        assert config.character_id == "spindle"

        config.character_id = "mryummers"
        config.save_to_yaml(str(yaml_path))

        content = yaml_path.read_text(encoding="utf-8")
        assert "default: mryummers" in content

    def test_other_sections_preserved(self, tmp_path) -> None:
        yaml_path = tmp_path / "spindl.yaml"
        yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")

        config = OrchestratorConfig.from_yaml(str(yaml_path))
        config.character_id = "mryummers"
        config.save_to_yaml(str(yaml_path))

        content = yaml_path.read_text(encoding="utf-8")
        # Other values should be untouched
        assert "provider: llama" in content
        assert "threshold: 0.5" in content
        assert "conversations_dir: ./conversations" in content

    def test_no_full_file_rewrite_artifacts(self, tmp_path) -> None:
        yaml_path = tmp_path / "spindl.yaml"
        yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")

        config = OrchestratorConfig.from_yaml(str(yaml_path))
        config.character_id = "mryummers"
        config.save_to_yaml(str(yaml_path))

        content = yaml_path.read_text(encoding="utf-8")
        # Comments should survive
        assert "# spindl configuration" in content
        # No yaml.dump continuation artifacts
        lines = content.split("\n")
        for line in lines:
            # No lines should have YAML continuation markers from dump()
            assert not line.strip().startswith(">-"), f"Continuation artifact found: {line}"

    def test_roundtrip_preserves_value(self, tmp_path) -> None:
        yaml_path = tmp_path / "spindl.yaml"
        yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")

        config = OrchestratorConfig.from_yaml(str(yaml_path))
        config.character_id = "mryummers"
        config.save_to_yaml(str(yaml_path))

        # Re-read and verify
        config2 = OrchestratorConfig.from_yaml(str(yaml_path))
        assert config2.character_id == "mryummers"
