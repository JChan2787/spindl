"""
Tests for NANO-089 config validation layer.

Validates Pydantic constraints on config models, validate-before-write gates,
and response schema enforcement.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from spindl.orchestrator.config import (
    LLMConfig,
    MemoryConfig,
    OrchestratorConfig,
    STTConfig,
    StimuliConfig,
    ToolsConfig,
    TTSConfig,
    VLMConfig,
    VTubeStudioConfig,
)
from spindl.gui.response_models import (
    LLMConfigResponse,
    VLMConfigResponse,
)


class TestProviderValidation:
    """Provider non-empty validators on STT, TTS, LLM configs."""

    def test_stt_rejects_empty_provider(self) -> None:
        with pytest.raises(ValidationError, match="STT provider cannot be empty"):
            STTConfig(provider="")

    def test_stt_rejects_whitespace_provider(self) -> None:
        with pytest.raises(ValidationError, match="STT provider cannot be empty"):
            STTConfig(provider="   ")

    def test_tts_rejects_empty_provider(self) -> None:
        with pytest.raises(ValidationError, match="TTS provider cannot be empty"):
            TTSConfig(provider="")

    def test_llm_rejects_empty_provider(self) -> None:
        with pytest.raises(ValidationError, match="LLM provider cannot be empty"):
            LLMConfig(provider="")

    def test_stt_accepts_valid_provider(self) -> None:
        cfg = STTConfig(provider="whisper")
        assert cfg.provider == "whisper"

    def test_tts_accepts_valid_provider(self) -> None:
        cfg = TTSConfig(provider="kokoro")
        assert cfg.provider == "kokoro"

    def test_llm_accepts_valid_provider(self) -> None:
        cfg = LLMConfig(provider="llama")
        assert cfg.provider == "llama"


class TestVLMConfig:
    """VLM config validation."""

    def test_accepts_standard_providers(self) -> None:
        for provider in ["llama", "openai", "llm", "none", ""]:
            cfg = VLMConfig(provider=provider)
            assert cfg.provider == provider

    def test_accepts_plugin_provider_names(self) -> None:
        """Plugin providers can use arbitrary names."""
        cfg = VLMConfig(provider="my_custom_vlm")
        assert cfg.provider == "my_custom_vlm"

    def test_default_provider_is_llama(self) -> None:
        cfg = VLMConfig()
        assert cfg.provider == "llama"


class TestOrchestratorFieldConstraints:
    """Field-level constraints on OrchestratorConfig."""

    def test_vad_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(vad_threshold=1.5)

    def test_vad_threshold_negative(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(vad_threshold=-0.1)

    def test_vad_threshold_valid(self) -> None:
        cfg = OrchestratorConfig(vad_threshold=0.45)
        assert cfg.vad_threshold == 0.45

    def test_min_speech_ms_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(min_speech_ms=0)

    def test_min_silence_ms_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(min_silence_ms=-1)

    def test_budget_strategy_enum(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(budget_strategy="invalid")

    def test_budget_strategy_valid_values(self) -> None:
        for strategy in ["truncate", "drop", "reject"]:
            cfg = OrchestratorConfig(budget_strategy=strategy)
            assert cfg.budget_strategy == strategy

    def test_summarization_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(summarization_threshold=0.0)  # gt=0.0, must be > 0

    def test_summarization_threshold_above_one(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(summarization_threshold=1.1)


class TestSubConfigConstraints:
    """Constraints on sub-config models."""

    def test_tools_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ToolsConfig(max_iterations=0)

    def test_memory_rag_top_k_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(rag_top_k=0)

    def test_memory_relevance_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(relevance_threshold=1.5)

    def test_stimuli_patience_seconds_minimum(self) -> None:
        with pytest.raises(ValidationError):
            StimuliConfig(patience_seconds=0.5)

    def test_vtubestudio_port_range(self) -> None:
        with pytest.raises(ValidationError):
            VTubeStudioConfig(port=0)

    def test_vtubestudio_port_max(self) -> None:
        with pytest.raises(ValidationError):
            VTubeStudioConfig(port=70000)


class TestValidateBeforeWrite:
    """Tests for validate-before-write gate in save_to_yaml."""

    def test_valid_config_can_save(self, tmp_path: Path) -> None:
        """A valid config saves without error."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text(
            "vad:\n  threshold: 0.5\n  min_speech_ms: 250\n  min_silence_ms: 500\n"
            "pipeline:\n  summarization_threshold: 0.6\n"
            "  budget_strategy: truncate\n"
        )
        config = OrchestratorConfig()
        # Should not raise
        config.save_to_yaml(str(config_file))

    def test_invalid_config_rejects_save(self, tmp_path: Path) -> None:
        """An invalid config raises ValueError before writing."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("vad:\n  threshold: 0.5\n")

        config = OrchestratorConfig()
        # Mutate to invalid state by bypassing Pydantic
        object.__setattr__(config, "vad_threshold", 2.0)

        with pytest.raises(ValueError, match="Config validation failed"):
            config.save_to_yaml(str(config_file))


class TestPersistValidation:
    """Tests for persist method validation gates."""

    def test_llm_config_validates_required_fields(self) -> None:
        """Missing executable_path and model_path produces errors."""
        from spindl.gui.server import GUIServer

        errors = GUIServer._validate_local_llm_config({})
        assert any("executable_path" in e or "model_path" in e or "url" in e for e in errors)

    def test_llm_config_accepts_local(self) -> None:
        from spindl.gui.server import GUIServer

        errors = GUIServer._validate_local_llm_config({
            "executable_path": "/path/to/llama-server",
            "model_path": "/path/to/model.gguf",
            "port": 5557,
        })
        assert errors == []

    def test_llm_config_accepts_url(self) -> None:
        from spindl.gui.server import GUIServer

        errors = GUIServer._validate_local_llm_config({
            "url": "http://127.0.0.1:5557",
        })
        assert errors == []

    def test_llm_config_rejects_bad_port(self) -> None:
        from spindl.gui.server import GUIServer

        errors = GUIServer._validate_local_llm_config({
            "executable_path": "/path/to/llama-server",
            "model_path": "/path/to/model.gguf",
            "port": 99999,
        })
        assert any("port" in e for e in errors)

    def test_vlm_config_validates_required_fields(self) -> None:
        from spindl.gui.server import GUIServer

        errors = GUIServer._validate_local_vlm_config({})
        assert len(errors) == 3  # executable_path, model_path, model_type

    def test_vlm_config_accepts_valid(self) -> None:
        from spindl.gui.server import GUIServer

        errors = GUIServer._validate_local_vlm_config({
            "executable_path": "/path/to/llama-server",
            "model_path": "/path/to/model.gguf",
            "model_type": "gemma3",
            "port": 5558,
        })
        assert errors == []


class TestResponseModels:
    """Tests for response schema models."""

    def test_llm_response_requires_provider(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfigResponse.model_validate({})

    def test_llm_response_minimal(self) -> None:
        resp = LLMConfigResponse(provider="llama")
        assert resp.provider == "llama"
        assert resp.available_providers == []

    def test_llm_response_full(self) -> None:
        resp = LLMConfigResponse(
            provider="openrouter",
            model="claude-opus-4.6",
            context_size=16384,
            available_providers=["llama", "openrouter", "deepseek"],
        )
        assert resp.provider == "openrouter"
        assert resp.model == "claude-opus-4.6"

    def test_vlm_response_requires_provider(self) -> None:
        with pytest.raises(ValidationError):
            VLMConfigResponse.model_validate({})

    def test_vlm_response_minimal(self) -> None:
        resp = VLMConfigResponse(provider="llama")
        assert resp.provider == "llama"
        assert resp.healthy is False
        assert resp.cloud_config is None

    def test_vlm_response_with_cloud_config(self) -> None:
        resp = VLMConfigResponse(
            provider="openai",
            available_providers=["llama", "openai"],
            healthy=True,
            cloud_config={
                "api_key": "sk-1...xyz",
                "model": "gpt-4o",
                "base_url": "https://api.openai.com",
            },
        )
        assert resp.cloud_config is not None
        assert resp.cloud_config.model == "gpt-4o"


class TestExtraFieldsIgnored:
    """Pydantic models should ignore unknown fields from YAML (ConfigDict extra='ignore')."""

    def test_stt_ignores_extra(self) -> None:
        cfg = STTConfig(provider="whisper", unknown_field="something")
        assert cfg.provider == "whisper"
        assert not hasattr(cfg, "unknown_field")

    def test_orchestrator_ignores_extra(self) -> None:
        cfg = OrchestratorConfig(debug=True, future_flag="test")
        assert cfg.debug is True
        assert not hasattr(cfg, "future_flag")
