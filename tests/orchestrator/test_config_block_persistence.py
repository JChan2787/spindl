"""
Tests for NANO-045c-1 — Block config YAML persistence.

Tests the save_block_config_to_yaml() method on OrchestratorConfig,
which handles section-level replacement for the nested prompt_blocks dict.
"""

import pytest
from pathlib import Path

from spindl.orchestrator.config import OrchestratorConfig


SAMPLE_YAML = """\
# spindl configuration
vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 300

pipeline:
  summarization_threshold: 0.75

memory:
  top_k: 5
  relevance_threshold: 0.64

prompt:
  rag_prefix: "Relevant memories:"
  rag_suffix: "End of memories."
  codex_prefix: "Facts:"
  codex_suffix: ""
"""

SAMPLE_YAML_WITH_BLOCKS = """\
# spindl configuration
vad:
  threshold: 0.5
  min_speech_ms: 250

prompt_blocks:
  disabled:
  - voice_state
  order:
  - persona_name
  - persona_appearance

pipeline:
  summarization_threshold: 0.75
"""


class TestSaveBlockConfigToYaml:
    """Tests for OrchestratorConfig.save_block_config_to_yaml()."""

    def test_insert_new_section(self, tmp_path: Path) -> None:
        """Inserts prompt_blocks section when none exists."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(SAMPLE_YAML)

        config = OrchestratorConfig()
        config.prompt_blocks = {
            "disabled": ["voice_state"],
            "order": ["persona_name", "persona_appearance"],
        }

        config.save_block_config_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "prompt_blocks:" in content
        assert "- voice_state" in content
        assert "- persona_name" in content
        # Original content preserved
        assert "threshold: 0.5" in content
        assert "# spindl configuration" in content

    def test_replace_existing_section(self, tmp_path: Path) -> None:
        """Replaces existing prompt_blocks section with updated config."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(SAMPLE_YAML_WITH_BLOCKS)

        config = OrchestratorConfig()
        config.prompt_blocks = {
            "disabled": ["rag_context", "codex_context"],
            "overrides": {"persona_name": "Custom agent name"},
        }

        config.save_block_config_to_yaml(str(config_file))

        content = config_file.read_text()
        # New values present
        assert "- rag_context" in content
        assert "- codex_context" in content
        assert "Custom agent name" in content
        # Old values gone
        assert "- voice_state" not in content
        # Surrounding sections preserved
        assert "threshold: 0.5" in content

    def test_remove_section_on_empty_config(self, tmp_path: Path) -> None:
        """Removes prompt_blocks section when config is empty."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(SAMPLE_YAML_WITH_BLOCKS)

        config = OrchestratorConfig()
        config.prompt_blocks = {}

        config.save_block_config_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "prompt_blocks:" not in content
        # Surrounding sections preserved
        assert "threshold: 0.5" in content

    def test_remove_section_on_none_config(self, tmp_path: Path) -> None:
        """Removes prompt_blocks section when config is None."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(SAMPLE_YAML_WITH_BLOCKS)

        config = OrchestratorConfig()
        config.prompt_blocks = None

        config.save_block_config_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "prompt_blocks:" not in content

    def test_noop_when_no_section_and_empty_config(self, tmp_path: Path) -> None:
        """Does nothing when no section exists and config is empty."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(SAMPLE_YAML)
        original = config_file.read_text()

        config = OrchestratorConfig()
        config.prompt_blocks = {}

        config.save_block_config_to_yaml(str(config_file))

        assert config_file.read_text() == original

    def test_preserves_comments(self, tmp_path: Path) -> None:
        """Comments outside prompt_blocks section are preserved."""
        yaml_with_comments = """\
# Main config
vad:
  threshold: 0.5  # voice activation threshold

# Pipeline settings
pipeline:
  summarization_threshold: 0.75
"""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(yaml_with_comments)

        config = OrchestratorConfig()
        config.prompt_blocks = {"disabled": ["voice_state"]}

        config.save_block_config_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "# Main config" in content
        assert "# voice activation threshold" in content
        assert "# Pipeline settings" in content
        assert "prompt_blocks:" in content

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing file."""
        config = OrchestratorConfig()
        config.prompt_blocks = {"disabled": ["voice_state"]}

        with pytest.raises(FileNotFoundError):
            config.save_block_config_to_yaml(str(tmp_path / "nonexistent.yaml"))

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save → load → compare: block config survives round-trip."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(SAMPLE_YAML)

        original_blocks = {
            "order": ["persona_name", "modality_context", "persona_rules"],
            "disabled": ["voice_state", "codex_context"],
            "overrides": {"persona_name": "You are TestBot."},
        }

        # Save
        config = OrchestratorConfig()
        config.prompt_blocks = original_blocks
        config.save_block_config_to_yaml(str(config_file))

        # Load back
        import yaml
        with open(config_file, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["prompt_blocks"]["order"] == original_blocks["order"]
        assert loaded["prompt_blocks"]["disabled"] == original_blocks["disabled"]
        assert loaded["prompt_blocks"]["overrides"] == original_blocks["overrides"]

    def test_overrides_with_none_values(self, tmp_path: Path) -> None:
        """Overrides with None values (cleared overrides) persist correctly."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(SAMPLE_YAML)

        config = OrchestratorConfig()
        config.prompt_blocks = {
            "overrides": {"persona_name": "Custom", "voice_state": None},
        }

        config.save_block_config_to_yaml(str(config_file))

        import yaml
        with open(config_file, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["prompt_blocks"]["overrides"]["persona_name"] == "Custom"
        assert loaded["prompt_blocks"]["overrides"]["voice_state"] is None
