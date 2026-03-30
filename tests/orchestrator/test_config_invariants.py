"""
Tests for NANO-089 Phase 3: Write-protection invariants.

Proves that config write paths preserve structural invariants:
1. VLM section never deleted, only provider: "none"
2. ${ENV_VAR} patterns survive all write paths
3. Provider field always set when providers block exists
4. Full config roundtrip: write → read → validate matches
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from spindl.orchestrator.config import (
    LLMConfig,
    OrchestratorConfig,
    VLMConfig,
)


# ---------------------------------------------------------------------------
# Shared YAML fixtures
# ---------------------------------------------------------------------------

FULL_CONFIG_YAML = """\
vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 500

pipeline:
  summarization_threshold: 0.6
  budget_strategy: truncate

llm:
  provider: openrouter
  providers:
    openrouter:
      api_key: ${OPENROUTER_API_KEY}
      model: google/gemini-2.5-pro
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95

vlm:
  provider: llama
  providers:
    llama:
      port: 5558
      model_path: /path/to/vlm.gguf
    openai:
      api_key: ${XAI_API_KEY}
      base_url: https://api.x.ai/v1

stt:
  provider: parakeet

tts:
  provider: kokoro

tools:
  enabled: true
  max_iterations: 3

memory:
  rag_top_k: 5
  relevance_threshold: 0.7
  embedding_timeout: 30

stimuli:
  patience:
    seconds: 30.0
    prompt: Are you still there?

character:
  default: yumi
  directory: ./characters
"""


# ===========================================================================
# Invariant 1: VLM section never deleted
# ===========================================================================


class TestVLMSectionPreservation:
    """VLM section must survive save_to_yaml regardless of provider value."""

    def test_vlm_section_survives_provider_none(self, tmp_path: Path) -> None:
        """Setting VLM provider to 'none' must not delete the section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vlm_config.provider = "none"
        config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        assert "vlm:" in raw, "vlm: section was deleted from YAML"

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.vlm_config.provider == "none"

    def test_vlm_providers_preserved_when_disabled(self, tmp_path: Path) -> None:
        """Provider sub-configs must survive when VLM is set to 'none'."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vlm_config.provider = "none"
        config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        # The llama and openai provider blocks must still be in the file
        assert "llama:" in raw
        assert "openai:" in raw

    def test_vlm_section_intact_through_multiple_saves(self, tmp_path: Path) -> None:
        """VLM section survives multiple save cycles with other field mutations."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))

        # Mutate non-VLM fields across 3 save cycles
        for threshold in (0.6, 0.7, 0.8):
            config.vad_threshold = threshold
            config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        assert "vlm:" in raw
        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.vlm_config.provider == "llama"
        assert reloaded.vad_threshold == 0.8


# ===========================================================================
# Invariant 2: ${ENV_VAR} patterns survive all write paths
# ===========================================================================


class TestEnvVarSurvival:
    """${ENV_VAR} patterns must survive save_to_yaml without being resolved or stripped."""

    def test_save_to_yaml_preserves_llm_env_var(self, tmp_path: Path) -> None:
        """${OPENROUTER_API_KEY} in LLM provider survives save_to_yaml."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vad_threshold = 0.9  # Mutate something to trigger a real write
        config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        assert "${OPENROUTER_API_KEY}" in raw, "LLM env var was nuked by save_to_yaml"

    def test_save_to_yaml_preserves_vlm_env_var(self, tmp_path: Path) -> None:
        """${XAI_API_KEY} in VLM provider survives save_to_yaml."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vad_threshold = 0.9  # Mutate something to trigger a real write
        config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        assert "${XAI_API_KEY}" in raw, "VLM env var was nuked by save_to_yaml"

    def test_save_to_yaml_preserves_multiple_env_vars(self, tmp_path: Path) -> None:
        """All ${ENV_VAR} patterns survive a single save cycle."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vad_threshold = 0.45
        config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        assert "${OPENROUTER_API_KEY}" in raw
        assert "${XAI_API_KEY}" in raw


# ===========================================================================
# Invariant 3: Provider field always set when providers block exists
# ===========================================================================


class TestProviderFieldPresence:
    """Provider routing key must exist when provider configs are present."""

    def test_vlm_from_dict_defaults_provider_when_missing(self) -> None:
        """VLMConfig.from_dict with no provider key defaults to 'llama'."""
        cfg = VLMConfig.from_dict({"providers": {"llama": {"port": 5558}}})
        assert cfg.provider == "llama"

    def test_llm_from_dict_defaults_provider_when_missing(self) -> None:
        """LLMConfig.from_dict with no provider key defaults to 'llama'."""
        cfg = LLMConfig.from_dict({"providers": {"llama": {"port": 5557}}})
        assert cfg.provider == "llama"

    def test_vlm_empty_provider_allowed_when_no_providers(self) -> None:
        """Empty provider is valid when providers dict is empty (unconfigured VLM)."""
        cfg = VLMConfig(provider="")
        assert cfg.provider == ""

    def test_vlm_empty_provider_rejected_with_providers(self) -> None:
        """Empty provider with non-empty providers dict is invalid — no routing key."""
        with pytest.raises(ValidationError, match="provider cannot be empty"):
            VLMConfig(provider="", providers={"llama": {"port": 5558}})

    def test_provider_fields_intact_after_roundtrip(self, tmp_path: Path) -> None:
        """Provider routing keys survive from_yaml → save_to_yaml → from_yaml."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.llm_config.provider == "openrouter"
        assert reloaded.vlm_config.provider == "llama"
        assert reloaded.stt_config.provider == "parakeet"
        assert reloaded.tts_config.provider == "kokoro"


# ===========================================================================
# Invariant 4: Full config roundtrip
# ===========================================================================


class TestConfigRoundtrip:
    """Write config → read back → all values match and structure intact."""

    def test_full_config_roundtrip(self, tmp_path: Path) -> None:
        """All dashboard-editable fields survive a full roundtrip."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))

        # Mutate dashboard-editable values
        config.vad_threshold = 0.85
        config.summarization_threshold = 0.75
        config.llm_config.provider = "llama"
        config.vlm_config.provider = "openai"
        config.tools_config.enabled = False

        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.vad_threshold == 0.85
        assert reloaded.summarization_threshold == 0.75
        assert reloaded.llm_config.provider == "llama"
        assert reloaded.vlm_config.provider == "openai"
        assert reloaded.tools_config.enabled is False

    def test_roundtrip_preserves_env_vars_in_raw_text(self, tmp_path: Path) -> None:
        """${ENV_VAR} strings survive the full from_yaml → save_to_yaml cycle in raw text."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        assert "${OPENROUTER_API_KEY}" in raw
        assert "${XAI_API_KEY}" in raw

    def test_roundtrip_preserves_non_dashboard_fields(self, tmp_path: Path) -> None:
        """Fields not touched by save_to_yaml survive the roundtrip."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(FULL_CONFIG_YAML)

        config = OrchestratorConfig.from_yaml(str(config_file))
        # Mutate a dashboard field to trigger a real write
        config.vad_threshold = 0.55
        config.save_to_yaml(str(config_file))

        raw = config_file.read_text()
        # These sections are not replaced by save_to_yaml's regex patterns —
        # they pass through as-is
        assert "provider: parakeet" in raw  # STT
        assert "provider: kokoro" in raw    # TTS
        assert "directory: ./characters" in raw
