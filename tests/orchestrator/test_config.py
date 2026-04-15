"""
Tests for OrchestratorConfig (NANO-015 Session 5).

Validates the new provider-based TTS configuration parsing.
"""

import pytest
import tempfile
from pathlib import Path

from pydantic import ValidationError

from spindl.orchestrator.config import LLMConfig, OrchestratorConfig, PromptConfig, TTSConfig, VLMConfig


class TestTTSConfig:
    """Tests for TTSConfig model."""

    def test_from_dict_with_full_config(self) -> None:
        """TTSConfig parses complete provider configuration."""
        data = {
            "provider": "kokoro",
            "plugin_paths": ["./plugins/tts"],
            "providers": {
                "kokoro": {
                    "host": "127.0.0.1",
                    "port": 5556,
                    "voice": "af_bella",
                    "language": "a",
                }
            }
        }

        config = TTSConfig.from_dict(data)

        assert config.provider == "kokoro"
        assert config.plugin_paths == ["./plugins/tts"]
        assert config.provider_config["host"] == "127.0.0.1"
        assert config.provider_config["port"] == 5556
        assert config.provider_config["voice"] == "af_bella"

    def test_from_dict_with_defaults(self) -> None:
        """TTSConfig uses defaults when fields are missing."""
        config = TTSConfig.from_dict({})

        assert config.provider == "kokoro"
        assert config.plugin_paths == []
        assert config.provider_config == {}

    def test_from_dict_with_provider_only(self) -> None:
        """TTSConfig handles provider without config section."""
        data = {"provider": "qwen3"}

        config = TTSConfig.from_dict(data)

        assert config.provider == "qwen3"
        assert config.provider_config == {}

    def test_from_dict_extracts_correct_provider_config(self) -> None:
        """TTSConfig extracts config for the selected provider only."""
        data = {
            "provider": "qwen3",
            "providers": {
                "kokoro": {"host": "127.0.0.1", "port": 5556},
                "qwen3": {"model": "Qwen/Qwen3-TTS", "device": "cuda:0"},
            }
        }

        config = TTSConfig.from_dict(data)

        assert config.provider == "qwen3"
        assert config.provider_config["model"] == "Qwen/Qwen3-TTS"
        assert config.provider_config["device"] == "cuda:0"
        assert "host" not in config.provider_config


class TestOrchestratorConfigTTS:
    """Tests for OrchestratorConfig TTS configuration parsing."""

    def test_from_yaml_parses_tts_config(self, tmp_path: Path) -> None:
        """OrchestratorConfig.from_yaml parses provider-based TTS config."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
stt:
  host: "127.0.0.1"
  port: 5555

tts:
  provider: "kokoro"
  plugin_paths:
    - "./plugins/tts"
  providers:
    kokoro:
      host: "127.0.0.1"
      port: 5556
      voice: "af_bella"
      language: "a"
      timeout: 30

llm:
  url: "http://127.0.0.1:5557"
""")

        config = OrchestratorConfig.from_yaml(str(config_file))

        assert config.tts_config.provider == "kokoro"
        assert config.tts_config.plugin_paths == ["./plugins/tts"]
        assert config.tts_config.provider_config["host"] == "127.0.0.1"
        assert config.tts_config.provider_config["port"] == 5556
        assert config.tts_config.provider_config["voice"] == "af_bella"

    def test_from_yaml_with_minimal_tts_config(self, tmp_path: Path) -> None:
        """OrchestratorConfig handles minimal TTS config."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
tts:
  provider: "kokoro"
""")

        config = OrchestratorConfig.from_yaml(str(config_file))

        assert config.tts_config.provider == "kokoro"
        assert config.tts_config.plugin_paths == []
        assert config.tts_config.provider_config == {}

    def test_from_yaml_without_tts_section_uses_defaults(self, tmp_path: Path) -> None:
        """OrchestratorConfig uses default TTSConfig when tts section is missing."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
stt:
  host: "127.0.0.1"
""")

        config = OrchestratorConfig.from_yaml(str(config_file))

        assert config.tts_config.provider == "kokoro"

    def test_to_dict_includes_tts_config(self) -> None:
        """OrchestratorConfig.to_dict serializes TTS config correctly."""
        config = OrchestratorConfig()
        config.tts_config = TTSConfig(
            provider="kokoro",
            plugin_paths=["./plugins/tts"],
            provider_config={"host": "127.0.0.1", "port": 5556}
        )

        result = config.to_dict()

        assert result["tts"]["provider"] == "kokoro"
        assert result["tts"]["plugin_paths"] == ["./plugins/tts"]
        assert result["tts"]["providers"]["kokoro"]["host"] == "127.0.0.1"
        assert result["tts"]["providers"]["kokoro"]["port"] == 5556

    def test_validate_catches_missing_provider(self) -> None:
        """Pydantic rejects empty TTS provider at construction time (NANO-089)."""
        with pytest.raises(ValidationError, match="TTS provider cannot be empty"):
            TTSConfig(provider="", plugin_paths=[], provider_config={})

    def test_validate_passes_with_complete_config(self, tmp_path: Path) -> None:
        """OrchestratorConfig loads valid TTS config without error (NANO-089)."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
stt:
  host: "127.0.0.1"
  port: 5555

tts:
  provider: "kokoro"
  providers:
    kokoro:
      host: "127.0.0.1"
      port: 5556

llm:
  provider: "llama"
  providers:
    llama:
      url: "http://127.0.0.1:5557"

vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 500

pipeline:
  conversations_dir: "./conversations"
  resume_session: true
  summarization_threshold: 0.6
  budget_strategy: "truncate"
""")

        # Should load without ValidationError
        config = OrchestratorConfig.from_yaml(str(config_file))
        assert config.tts_config.provider == "kokoro"


class TestOrchestratorConfigBackwardCompatibility:
    """Tests ensuring backward compatibility during migration."""

    def test_no_playback_sample_rate_attribute(self) -> None:
        """OrchestratorConfig no longer has playback_sample_rate attribute."""
        config = OrchestratorConfig()

        assert not hasattr(config, "playback_sample_rate")

    def test_audio_dict_no_playback_rate(self) -> None:
        """OrchestratorConfig.to_dict audio section has no playback_rate."""
        config = OrchestratorConfig()
        result = config.to_dict()

        assert "playback_rate" not in result["audio"]
        assert "capture_rate" in result["audio"]


class TestOrchestratorConfigPersistence:
    """Tests for YAML persistence (NANO-025 Phase 5)."""

    def test_save_to_yaml_updates_vad_threshold(self, tmp_path: Path) -> None:
        """save_to_yaml updates VAD threshold in YAML file."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
# VAD Settings
vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 500

pipeline:
  summarization_threshold: 0.6
""")

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vad_threshold = 0.85
        config.save_to_yaml(str(config_file))

        # Re-read and verify
        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.vad_threshold == 0.85

    def test_save_to_yaml_updates_min_speech_ms(self, tmp_path: Path) -> None:
        """save_to_yaml updates min_speech_ms in YAML file."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 500

pipeline:
  summarization_threshold: 0.6
""")

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.min_speech_ms = 400
        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.min_speech_ms == 400

    def test_save_to_yaml_updates_pipeline_settings(self, tmp_path: Path) -> None:
        """save_to_yaml updates pipeline settings in YAML file."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 500

pipeline:
  summarization_threshold: 0.6
""")

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.summarization_threshold = 0.75
        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.summarization_threshold == 0.75

    def test_save_to_yaml_preserves_comments(self, tmp_path: Path) -> None:
        """save_to_yaml preserves comments in YAML file."""
        config_file = tmp_path / "spindl.yaml"
        original_content = """# VAD Settings - tune for your microphone
vad:
  threshold: 0.5  # Speech detection sensitivity
  min_speech_ms: 250
  min_silence_ms: 500

# Pipeline configuration
pipeline:
  summarization_threshold: 0.6
"""
        config_file.write_text(original_content)

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vad_threshold = 0.9
        config.save_to_yaml(str(config_file))

        # Check that comments are preserved
        saved_content = config_file.read_text()
        assert "# VAD Settings - tune for your microphone" in saved_content
        assert "# Speech detection sensitivity" in saved_content

    def test_save_to_yaml_only_updates_vad_section_threshold(self, tmp_path: Path) -> None:
        """save_to_yaml doesn't update threshold values outside vad section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
some_other_section:
  threshold: 0.1

vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 500

pipeline:
  summarization_threshold: 0.6
""")

        config = OrchestratorConfig.from_yaml(str(config_file))
        config.vad_threshold = 0.9
        config.save_to_yaml(str(config_file))

        # Read raw content to check section isolation
        saved_content = config_file.read_text()
        lines = saved_content.split('\n')

        # Find threshold in some_other_section (should still be 0.1)
        in_other_section = False
        for line in lines:
            if 'some_other_section:' in line:
                in_other_section = True
            elif line.strip() and not line.startswith(' ') and ':' in line:
                in_other_section = False
            if in_other_section and 'threshold:' in line:
                assert '0.1' in line, f"Threshold in other section was modified: {line}"

    def test_save_to_yaml_raises_on_missing_file(self) -> None:
        """save_to_yaml raises FileNotFoundError for missing file."""
        config = OrchestratorConfig()

        with pytest.raises(FileNotFoundError):
            config.save_to_yaml("/nonexistent/path/spindl.yaml")


# =============================================================================
# Unified Vision Config Routing Tests (NANO-030 Phase 4)
# =============================================================================


def _build_vlm_config_for_screen_vision(
    vlm_provider: str,
    llm_config: LLMConfig,
    vlm_config: VLMConfig,
) -> dict:
    """
    Replicate the config routing logic from voice_agent._setup_tools().

    This is the exact conditional from voice_agent.py extracted for testability.
    """
    if vlm_provider == "llm":
        llm_cfg = llm_config.provider_config

        llm_url = llm_cfg.get("url")
        if not llm_url:
            llm_host = llm_cfg.get("host", "127.0.0.1")
            llm_port = llm_cfg.get("port", 5557)
            llm_url = f"http://{llm_host}:{llm_port}"

        vlm_provider_config = {
            "url": llm_url,
            "api_key": llm_cfg.get("api_key"),
            "model": llm_cfg.get("model", "local-llm"),
        }

        vlm_overrides = vlm_config.providers.get("llm", {})
        vlm_provider_config.update(vlm_overrides)

        return vlm_provider_config
    else:
        return vlm_config.providers.get(vlm_provider, {})


class TestUnifiedVisionConfigRouting:
    """Tests for NANO-030 Phase 4: unified vision config routing."""

    def test_llm_provider_uses_llm_url(self) -> None:
        """When vlm_provider is 'llm', config should use LLM provider's URL."""
        llm_config = LLMConfig(
            provider="llama",
            provider_config={"url": "http://127.0.0.1:5557"},
        )
        vlm_config = VLMConfig(provider="llm", providers={})

        result = _build_vlm_config_for_screen_vision("llm", llm_config, vlm_config)

        assert result["url"] == "http://127.0.0.1:5557"

    def test_llm_provider_constructs_url_from_host_port(self) -> None:
        """When LLM config has host/port but no url, URL should be constructed."""
        llm_config = LLMConfig(
            provider="llama",
            provider_config={"host": "192.168.1.100", "port": 8080},
        )
        vlm_config = VLMConfig(provider="llm", providers={})

        result = _build_vlm_config_for_screen_vision("llm", llm_config, vlm_config)

        assert result["url"] == "http://192.168.1.100:8080"

    def test_llm_provider_defaults_url_when_no_url_or_host(self) -> None:
        """When LLM config has neither url nor host/port, use defaults."""
        llm_config = LLMConfig(
            provider="llama",
            provider_config={},
        )
        vlm_config = VLMConfig(provider="llm", providers={})

        result = _build_vlm_config_for_screen_vision("llm", llm_config, vlm_config)

        assert result["url"] == "http://127.0.0.1:5557"

    def test_llm_provider_passes_api_key(self) -> None:
        """When LLM has api_key, it should be passed to VLM config."""
        llm_config = LLMConfig(
            provider="deepseek",
            provider_config={
                "url": "https://api.deepseek.com/v1",
                "api_key": "${DEEPSEEK_API_KEY}",
                "model": "deepseek-chat",
            },
        )
        vlm_config = VLMConfig(provider="llm", providers={})

        result = _build_vlm_config_for_screen_vision("llm", llm_config, vlm_config)

        assert result["api_key"] == "${DEEPSEEK_API_KEY}"
        assert result["model"] == "deepseek-chat"

    def test_llm_provider_no_api_key_when_local(self) -> None:
        """Local LLM config should have api_key=None."""
        llm_config = LLMConfig(
            provider="llama",
            provider_config={"url": "http://127.0.0.1:5557"},
        )
        vlm_config = VLMConfig(provider="llm", providers={})

        result = _build_vlm_config_for_screen_vision("llm", llm_config, vlm_config)

        assert result["api_key"] is None

    def test_vlm_overrides_take_precedence(self) -> None:
        """Vision-specific overrides from vlm.providers.llm should win."""
        llm_config = LLMConfig(
            provider="llama",
            provider_config={
                "url": "http://127.0.0.1:5557",
                "model": "gemma-3-12b",
                "timeout": 60.0,
            },
        )
        vlm_config = VLMConfig(
            provider="llm",
            providers={
                "llm": {
                    "prompt": "Describe what you see concisely.",
                    "max_tokens": 300,
                    "timeout": 15.0,
                }
            },
        )

        result = _build_vlm_config_for_screen_vision("llm", llm_config, vlm_config)

        assert result["prompt"] == "Describe what you see concisely."
        assert result["max_tokens"] == 300
        assert result["timeout"] == 15.0
        # Base LLM values still present
        assert result["url"] == "http://127.0.0.1:5557"
        assert result["model"] == "gemma-3-12b"

    def test_non_llm_provider_uses_vlm_config(self) -> None:
        """When vlm_provider is not 'llm', existing behavior is unchanged."""
        llm_config = LLMConfig(
            provider="llama",
            provider_config={"url": "http://127.0.0.1:5557"},
        )
        vlm_config = VLMConfig(
            provider="llama",
            providers={
                "llama": {
                    "host": "127.0.0.1",
                    "port": 5558,
                    "model_type": "gemma3",
                }
            },
        )

        result = _build_vlm_config_for_screen_vision("llama", llm_config, vlm_config)

        assert result["host"] == "127.0.0.1"
        assert result["port"] == 5558
        assert result["model_type"] == "gemma3"

    def test_non_llm_provider_missing_config_returns_empty(self) -> None:
        """When vlm_provider config section is missing, return empty dict."""
        llm_config = LLMConfig(provider="llama", provider_config={})
        vlm_config = VLMConfig(provider="openai", providers={})

        result = _build_vlm_config_for_screen_vision("openai", llm_config, vlm_config)

        assert result == {}


# ---------------------------------------------------------------------------
# Tests: PromptConfig (NANO-045d)
# ---------------------------------------------------------------------------

class TestPromptConfig:
    """Tests for PromptConfig model — injection wrapper configuration."""

    def test_defaults(self) -> None:
        """Default values are the directive strings."""
        pc = PromptConfig()
        assert "Use them to inform your response" in pc.rag_prefix
        assert pc.rag_suffix == "End of memories."
        assert "always true in this context" in pc.codex_prefix
        assert pc.codex_suffix == ""
        assert "style reference" in pc.example_dialogue_prefix
        assert pc.example_dialogue_suffix == "End of style examples."

    def test_from_dict_full(self) -> None:
        """from_dict with all fields populates correctly."""
        data = {
            "rag_prefix": "MEMORIES:",
            "rag_suffix": "END.",
            "codex_prefix": "FACTS:",
            "codex_suffix": "END FACTS.",
            "example_dialogue_prefix": "EXAMPLES:",
            "example_dialogue_suffix": "END EXAMPLES.",
        }
        pc = PromptConfig.from_dict(data)
        assert pc.rag_prefix == "MEMORIES:"
        assert pc.rag_suffix == "END."
        assert pc.codex_prefix == "FACTS:"
        assert pc.codex_suffix == "END FACTS."
        assert pc.example_dialogue_prefix == "EXAMPLES:"
        assert pc.example_dialogue_suffix == "END EXAMPLES."

    def test_from_dict_partial(self) -> None:
        """from_dict with partial fields uses defaults for missing."""
        pc = PromptConfig.from_dict({"rag_prefix": "CUSTOM"})
        assert pc.rag_prefix == "CUSTOM"
        assert pc.rag_suffix == "End of memories."  # default

    def test_from_dict_empty(self) -> None:
        """from_dict with empty dict uses all defaults."""
        pc = PromptConfig.from_dict({})
        defaults = PromptConfig()
        assert pc.rag_prefix == defaults.rag_prefix
        assert pc.rag_suffix == defaults.rag_suffix
        assert pc.codex_prefix == defaults.codex_prefix
        assert pc.codex_suffix == defaults.codex_suffix
        assert pc.example_dialogue_prefix == defaults.example_dialogue_prefix
        assert pc.example_dialogue_suffix == defaults.example_dialogue_suffix

    def test_orchestrator_config_has_prompt_config(self) -> None:
        """OrchestratorConfig includes prompt_config field."""
        config = OrchestratorConfig()
        assert isinstance(config.prompt_config, PromptConfig)

    def test_orchestrator_from_dict_parses_prompt(self) -> None:
        """OrchestratorConfig._from_dict reads prompt section."""
        data = {
            "prompt": {
                "rag_prefix": "TEST PREFIX",
                "codex_suffix": "TEST SUFFIX",
            }
        }
        config = OrchestratorConfig._from_dict(data)
        assert config.prompt_config.rag_prefix == "TEST PREFIX"
        assert config.prompt_config.codex_suffix == "TEST SUFFIX"

    def test_from_dict_example_dialogue_partial(self) -> None:
        """from_dict with only example_dialogue_prefix uses default for suffix."""
        pc = PromptConfig.from_dict({"example_dialogue_prefix": "CUSTOM:"})
        assert pc.example_dialogue_prefix == "CUSTOM:"
        assert pc.example_dialogue_suffix == "End of style examples."

    def test_orchestrator_from_dict_parses_example_dialogue_wrappers(self) -> None:
        """OrchestratorConfig._from_dict reads example dialogue wrapper fields."""
        data = {
            "prompt": {
                "example_dialogue_prefix": "STYLE:",
                "example_dialogue_suffix": "END STYLE.",
            }
        }
        config = OrchestratorConfig._from_dict(data)
        assert config.prompt_config.example_dialogue_prefix == "STYLE:"
        assert config.prompt_config.example_dialogue_suffix == "END STYLE."

    def test_orchestrator_from_dict_without_prompt(self) -> None:
        """OrchestratorConfig._from_dict without prompt uses defaults."""
        config = OrchestratorConfig._from_dict({})
        defaults = PromptConfig()
        assert config.prompt_config.rag_prefix == defaults.rag_prefix
        assert config.prompt_config.example_dialogue_prefix == defaults.example_dialogue_prefix


# ---------------------------------------------------------------------------
# Tests: Generation Params Persistence (NANO-053)
# ---------------------------------------------------------------------------

class TestGenerationParamsPersistence:
    """Tests for save_to_yaml generation parameter persistence."""

    def test_save_to_yaml_updates_temperature(self, tmp_path: Path) -> None:
        """save_to_yaml updates temperature under llm section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
vad:
  threshold: 0.5

llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95

pipeline:
  summarization_threshold: 0.6
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["temperature"] = 1.2
        config.save_to_yaml(str(config_file))

        # Re-read raw content
        content = config_file.read_text()
        assert "temperature: 1.2" in content

    def test_save_to_yaml_updates_max_tokens(self, tmp_path: Path) -> None:
        """save_to_yaml updates max_tokens under llm section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["max_tokens"] = 4096
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "max_tokens: 4096" in content

    def test_save_to_yaml_updates_top_p(self, tmp_path: Path) -> None:
        """save_to_yaml updates top_p under llm section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["top_p"] = 0.8
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "top_p: 0.8" in content

    def test_save_to_yaml_gen_params_section_isolated(self, tmp_path: Path) -> None:
        """save_to_yaml only updates gen params in llm section, not elsewhere."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
some_other_section:
  temperature: 0.1
  max_tokens: 100

llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95

pipeline:
  summarization_threshold: 0.6
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["temperature"] = 1.5
        config.llm_config.provider_config["max_tokens"] = 2048
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        lines = content.split("\n")

        # Check that other section's values are untouched
        in_other_section = False
        for line in lines:
            if "some_other_section:" in line:
                in_other_section = True
            elif line.strip() and not line.startswith(" ") and ":" in line:
                in_other_section = False
            if in_other_section and "temperature:" in line:
                assert "0.1" in line, f"Temperature in other section was modified: {line}"
            if in_other_section and "max_tokens:" in line:
                assert "100" in line, f"max_tokens in other section was modified: {line}"

    def test_emit_config_includes_generation_params(self) -> None:
        """OrchestratorConfig with LLM provider_config exposes gen params."""
        config = OrchestratorConfig()
        config.llm_config = LLMConfig(
            provider="llama",
            provider_config={
                "temperature": 0.9,
                "max_tokens": 1024,
                "top_p": 0.85,
            },
        )
        # Verify the params are accessible from provider_config
        assert config.llm_config.provider_config["temperature"] == 0.9
        assert config.llm_config.provider_config["max_tokens"] == 1024
        assert config.llm_config.provider_config["top_p"] == 0.85

    # --- NANO-108: Repetition penalty params ---

    def test_save_to_yaml_updates_repeat_penalty(self, tmp_path: Path) -> None:
        """save_to_yaml persists repeat_penalty under llm provider section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["repeat_penalty"] = 1.5
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "repeat_penalty: 1.5" in content

    def test_save_to_yaml_updates_repeat_last_n(self, tmp_path: Path) -> None:
        """save_to_yaml persists repeat_last_n under llm provider section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["repeat_last_n"] = 128
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "repeat_last_n: 128" in content

    def test_save_to_yaml_updates_frequency_penalty(self, tmp_path: Path) -> None:
        """save_to_yaml persists frequency_penalty under llm provider section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["frequency_penalty"] = 0.5
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "frequency_penalty: 0.5" in content

    def test_save_to_yaml_updates_presence_penalty(self, tmp_path: Path) -> None:
        """save_to_yaml persists presence_penalty under llm provider section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["presence_penalty"] = -0.3
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "presence_penalty: -0.3" in content

    def test_save_to_yaml_defaults_for_missing_repetition_params(self, tmp_path: Path) -> None:
        """save_to_yaml writes defaults when repetition params not in provider_config."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "repeat_penalty: 1.1" in content
        assert "repeat_last_n: 64" in content
        assert "frequency_penalty: 0.0" in content
        assert "presence_penalty: 0.0" in content

    # --- NANO-115: Tail sampling params (top_k / min_p) ---

    def test_save_to_yaml_updates_top_k(self, tmp_path: Path) -> None:
        """save_to_yaml persists top_k under llm provider section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["top_k"] = 50
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "top_k: 50" in content

    def test_save_to_yaml_updates_min_p(self, tmp_path: Path) -> None:
        """save_to_yaml persists min_p under llm provider section."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["min_p"] = 0.1
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "min_p: 0.1" in content

    def test_save_to_yaml_defaults_for_missing_tail_sampling_params(self, tmp_path: Path) -> None:
        """save_to_yaml writes defaults when tail sampling params not in provider_config."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        assert "top_k: 40" in content
        assert "min_p: 0.05" in content

    def test_save_to_yaml_roundtrips_all_sampling_params_together(self, tmp_path: Path) -> None:
        """All sampler params (tail + repetition + penalty) round-trip in a single save."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
llm:
  provider: llama
  providers:
    llama:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.95
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.llm_config.provider_config["top_k"] = 60
        config.llm_config.provider_config["min_p"] = 0.07
        config.llm_config.provider_config["repeat_penalty"] = 1.15
        config.llm_config.provider_config["repeat_last_n"] = 96
        config.llm_config.provider_config["frequency_penalty"] = 0.2
        config.llm_config.provider_config["presence_penalty"] = -0.1
        config.save_to_yaml(str(config_file))

        # Reload and confirm all values survive the round trip
        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.llm_config.provider_config["top_k"] == 60
        assert reloaded.llm_config.provider_config["min_p"] == 0.07
        assert reloaded.llm_config.provider_config["repeat_penalty"] == 1.15
        assert reloaded.llm_config.provider_config["repeat_last_n"] == 96
        assert reloaded.llm_config.provider_config["frequency_penalty"] == 0.2
        assert reloaded.llm_config.provider_config["presence_penalty"] == -0.1


class TestSaveToYamlStimuliPrompt:
    """Tests for save_to_yaml patience prompt persistence."""

    def test_save_to_yaml_updates_patience_prompt(self, tmp_path: Path) -> None:
        """save_to_yaml persists patience_prompt to YAML."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
stimuli:
  enabled: false
  patience:
    enabled: true
    seconds: 20
    prompt: Old default prompt text.
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.stimuli_config.patience_prompt = "Say something funny about cats."
        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.stimuli_config.patience_prompt == "Say something funny about cats."

    def test_save_to_yaml_prompt_section_isolated(self, tmp_path: Path) -> None:
        """save_to_yaml only updates prompt under stimuli, not other sections."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
prompt:
  prompt: This is a prompt section value
stimuli:
  enabled: false
  patience:
    enabled: true
    seconds: 30
    prompt: Original patience prompt.
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.stimuli_config.patience_prompt = "Updated patience prompt."
        config.save_to_yaml(str(config_file))

        content = config_file.read_text()
        # The prompt section's prompt should be untouched
        assert "This is a prompt section value" in content
        # The stimuli section's prompt should be updated
        assert "Updated patience prompt." in content


class TestSaveToYamlReflectionPrompt:
    """Tests for YAML round-trip of reflection prompt fields (NANO-104/105).

    The reflection_prompt field is multi-line and may contain quotes, braces,
    and newlines. YAML surgery must escape these correctly so the file remains
    parseable. Regression test for the 591 corruption bug.
    """

    def test_multiline_prompt_round_trip(self, tmp_path: Path) -> None:
        """Multi-line reflection_prompt with quotes and braces survives save_to_yaml."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
memory:
  enabled: true
  reflection_interval: 5
  reflection_prompt: null
  reflection_system_message: null
  reflection_delimiter: "{qa}"
""")
        config = OrchestratorConfig.from_yaml(str(config_file))

        # Set a prompt with inner quotes, braces, and newlines — the exact
        # pattern that caused the 591 corruption
        nasty_prompt = (
            'From the conversation below, extract the "most important" fact.\n\n'
            "Separate entries with \"{qa}\" delimiter.\n\n"
            "Conversation:\n{transcript}"
        )
        config.memory_config.reflection_prompt = nasty_prompt
        config.save_to_yaml(str(config_file))

        # The file must parse without error
        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.memory_config.reflection_prompt == nasty_prompt

    def test_prompt_with_inner_double_quotes(self, tmp_path: Path) -> None:
        """Prompt containing double quotes doesn't break YAML structure."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
memory:
  enabled: true
  reflection_interval: 20
  reflection_prompt: null
  reflection_delimiter: "{qa}"
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.memory_config.reflection_prompt = 'Extract "key facts" from {transcript}'
        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.memory_config.reflection_prompt == 'Extract "key facts" from {transcript}'

    def test_yaml_remains_parseable_after_save(self, tmp_path: Path) -> None:
        """After save_to_yaml, the entire YAML file parses without error."""
        import yaml

        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
memory:
  enabled: true
  reflection_interval: 5
  reflection_prompt: null
  reflection_system_message: null
  reflection_delimiter: "{qa}"
  embedding:
    base_url: http://127.0.0.1:5559
    timeout: 10
vad:
  threshold: 0.5
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.memory_config.reflection_prompt = (
            "Extract facts.\n\nUse \"{qa}\" between entries.\n\n{transcript}"
        )
        config.memory_config.reflection_system_message = (
            'You are a "fact extraction" assistant.'
        )
        config.save_to_yaml(str(config_file))

        # Must not raise
        content = config_file.read_text()
        parsed = yaml.safe_load(content)
        assert parsed is not None
        assert "memory" in parsed

    def test_null_prompt_round_trip(self, tmp_path: Path) -> None:
        """Setting prompt back to None writes 'null' and round-trips cleanly."""
        config_file = tmp_path / "spindl.yaml"
        config_file.write_text("""
memory:
  enabled: true
  reflection_prompt: "some old prompt with {transcript}"
  reflection_delimiter: "{qa}"
""")
        config = OrchestratorConfig.from_yaml(str(config_file))
        config.memory_config.reflection_prompt = None
        config.save_to_yaml(str(config_file))

        reloaded = OrchestratorConfig.from_yaml(str(config_file))
        assert reloaded.memory_config.reflection_prompt is None
