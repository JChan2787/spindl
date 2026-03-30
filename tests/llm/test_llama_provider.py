"""
Tests for LlamaProvider (llama.cpp LLM provider).

Tests cover:
- get_server_command() - Command generation for local mode
- get_health_url() - Health check URL generation
- validate_config() - Config validation for local and cloud modes
"""

import pytest

from spindl.llm.builtin.llama.provider import LlamaProvider


# =============================================================================
# get_server_command() Tests
# =============================================================================


class TestGetServerCommand:
    """Tests for LlamaProvider.get_server_command()."""

    def test_missing_executable_returns_none(self):
        """Should return None when executable_path is missing."""
        config = {"model_path": "model.gguf"}
        assert LlamaProvider.get_server_command(config) is None

    def test_missing_model_returns_none(self):
        """Should return None when model_path is missing."""
        config = {"executable_path": "/bin/llama-server"}
        assert LlamaProvider.get_server_command(config) is None

    def test_missing_both_returns_none(self):
        """Should return None when both paths are missing."""
        config = {}
        assert LlamaProvider.get_server_command(config) is None

    def test_minimal_config(self):
        """Should generate command with defaults for minimal config."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
        }
        result = LlamaProvider.get_server_command(config)

        assert result is not None
        assert "/bin/llama-server" in result
        assert "-m /models/model.gguf" in result
        assert "--host 127.0.0.1" in result
        assert "--port 5557" in result
        assert "-ngl 99" in result

    def test_custom_host_port(self):
        """Should use custom host and port."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "host": "0.0.0.0",
            "port": 8080,
        }
        result = LlamaProvider.get_server_command(config)

        assert "--host 0.0.0.0" in result
        assert "--port 8080" in result

    def test_custom_gpu_layers(self):
        """Should use custom gpu_layers value."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "gpu_layers": 50,
        }
        result = LlamaProvider.get_server_command(config)

        assert "-ngl 50" in result

    def test_single_gpu_device(self):
        """Should include --device flag for single GPU selection."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "device": "CUDA1",
        }
        result = LlamaProvider.get_server_command(config)

        assert "--device CUDA1" in result

    def test_multi_gpu_tensor_split(self):
        """Should include --tensor-split for multi-GPU distribution."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "tensor_split": [0.5, 0.5],
        }
        result = LlamaProvider.get_server_command(config)

        assert "--tensor-split 0.5,0.5" in result
        assert "--device" not in result

    def test_tensor_split_precedence_over_device(self):
        """tensor_split should take precedence over device when both present."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "device": "CUDA1",
            "tensor_split": [0.6, 0.4],
        }
        result = LlamaProvider.get_server_command(config)

        assert "--tensor-split 0.6,0.4" in result
        assert "--device" not in result

    def test_extra_args(self):
        """Should append extra_args to command."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "extra_args": ["-fa", "on"],
        }
        result = LlamaProvider.get_server_command(config)

        assert result.endswith("-fa on")

    def test_empty_extra_args(self):
        """Should not append anything for empty extra_args."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "extra_args": [],
        }
        result = LlamaProvider.get_server_command(config)

        # Should end with the last standard arg, not have trailing space
        assert not result.endswith(" ")

    def test_integer_tensor_split(self):
        """Should handle integer tensor_split values."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "tensor_split": [3, 1],
        }
        result = LlamaProvider.get_server_command(config)

        assert "--tensor-split 3,1" in result

    def test_context_size(self):
        """Should include -c flag when context_size is specified."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "context_size": 8192,
        }
        result = LlamaProvider.get_server_command(config)

        assert "-c 8192" in result

    def test_windows_paths(self):
        """Should handle Windows-style paths."""
        config = {
            "executable_path": "X:\\AI_LLM\\llama-server.exe",
            "model_path": "X:\\Models\\model.gguf",
        }
        result = LlamaProvider.get_server_command(config)

        assert "X:\\AI_LLM\\llama-server.exe" in result
        assert "X:\\Models\\model.gguf" in result

    def test_mmproj_path(self):
        """Should include --mmproj flag when mmproj_path is set."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/gemma-3-12b.gguf",
            "mmproj_path": "/models/mmproj-gemma-3-12b.gguf",
        }
        result = LlamaProvider.get_server_command(config)

        assert "--mmproj /models/mmproj-gemma-3-12b.gguf" in result

    def test_mmproj_path_absent(self):
        """Should NOT include --mmproj when mmproj_path is not set."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
        }
        result = LlamaProvider.get_server_command(config)

        assert "--mmproj" not in result

    def test_mmproj_path_with_windows_paths(self):
        """Should handle Windows-style mmproj paths."""
        config = {
            "executable_path": "X:\\AI_LLM\\llama-server.exe",
            "model_path": "X:\\Models\\gemma-3-12b.gguf",
            "mmproj_path": "X:\\Models\\mmproj-gemma-3-12b.gguf",
        }
        result = LlamaProvider.get_server_command(config)

        assert "--mmproj X:\\Models\\mmproj-gemma-3-12b.gguf" in result

    def test_mmproj_path_ordering(self):
        """mmproj should appear after model path and before GPU selection."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "mmproj_path": "/models/mmproj.gguf",
            "tensor_split": [0.5, 0.5],
        }
        result = LlamaProvider.get_server_command(config)

        mmproj_pos = result.index("--mmproj")
        model_pos = result.index("-m")
        tensor_pos = result.index("--tensor-split")
        assert model_pos < mmproj_pos < tensor_pos

    # ----- NANO-042: Reasoning Launch Flags -----

    # Reasoning is force-disabled (--reasoning-budget 0) until runtime toggle
    # supports cloud LLMs. Tests reflect the current behavior.

    def test_reasoning_format_deepseek_with_budget(self):
        """reasoning_format injects --reasoning-format, budget from config."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/qwen3.gguf",
            "reasoning_format": "deepseek",
            "reasoning_budget": 0,
        }
        result = LlamaProvider.get_server_command(config)

        assert "--jinja" in result
        assert "--reasoning-format deepseek" in result
        assert "--reasoning-budget 0" in result

    def test_reasoning_budget_from_config(self):
        """Config reasoning_budget is used directly."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/qwen3.gguf",
            "reasoning_format": "deepseek",
            "reasoning_budget": -1,
        }
        result = LlamaProvider.get_server_command(config)

        assert "--reasoning-budget -1" in result

    def test_reasoning_format_none(self):
        """Should include --reasoning-format none when configured."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/qwen3.gguf",
            "reasoning_format": "none",
        }
        result = LlamaProvider.get_server_command(config)

        assert "--jinja" in result
        assert "--reasoning-format none" in result

    def test_no_reasoning_format_still_injects_jinja(self):
        """--jinja is always injected, even without reasoning_format."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
        }
        result = LlamaProvider.get_server_command(config)

        assert "--jinja" in result
        assert "--reasoning-format" not in result
        assert "--reasoning-budget -1" in result

    def test_reasoning_flags_before_extra_args(self):
        """Reasoning flags should appear before extra_args."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/qwen3.gguf",
            "reasoning_format": "deepseek",
            "extra_args": ["-fa", "on"],
        }
        result = LlamaProvider.get_server_command(config)

        reasoning_pos = result.index("--reasoning-format")
        extra_pos = result.index("-fa")
        assert reasoning_pos < extra_pos


# =============================================================================
# get_health_url() Tests
# =============================================================================


class TestGetHealthUrl:
    """Tests for LlamaProvider.get_health_url()."""

    def test_default_host_port(self):
        """Should return URL with default host and port."""
        config = {}
        result = LlamaProvider.get_health_url(config)

        assert result == "http://127.0.0.1:5557/health"

    def test_custom_host(self):
        """Should use custom host."""
        config = {"host": "0.0.0.0"}
        result = LlamaProvider.get_health_url(config)

        assert result == "http://0.0.0.0:5557/health"

    def test_custom_port(self):
        """Should use custom port."""
        config = {"port": 8080}
        result = LlamaProvider.get_health_url(config)

        assert result == "http://127.0.0.1:8080/health"

    def test_custom_host_and_port(self):
        """Should use both custom host and port."""
        config = {"host": "192.168.1.10", "port": 9000}
        result = LlamaProvider.get_health_url(config)

        assert result == "http://192.168.1.10:9000/health"


# =============================================================================
# validate_config() Tests
# =============================================================================


class TestValidateConfig:
    """Tests for LlamaProvider.validate_config()."""

    def test_valid_local_config(self):
        """Should pass with valid local config."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
        }
        errors = LlamaProvider.validate_config(config)

        assert errors == []

    def test_valid_cloud_config(self):
        """Should pass with valid cloud/external config."""
        config = {"url": "http://api.example.com/v1"}
        errors = LlamaProvider.validate_config(config)

        assert errors == []

    def test_partial_local_exe_only(self):
        """Should fail when only executable_path is specified."""
        config = {"executable_path": "/bin/llama-server"}
        errors = LlamaProvider.validate_config(config)

        assert len(errors) == 1
        assert "model_path is required" in errors[0]

    def test_partial_local_model_only(self):
        """Should fail when only model_path is specified."""
        config = {"model_path": "/models/model.gguf"}
        errors = LlamaProvider.validate_config(config)

        assert len(errors) == 1
        assert "executable_path is required" in errors[0]

    def test_empty_config(self):
        """Should fail with empty config."""
        config = {}
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("required" in e.lower() for e in errors)

    def test_invalid_url_scheme(self):
        """Should fail when URL doesn't have http/https scheme."""
        config = {"url": "ftp://example.com"}
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("http" in e for e in errors)

    def test_invalid_port_type(self):
        """Should fail when port is not an integer."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "port": "5557",
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("port" in e.lower() for e in errors)

    def test_invalid_port_range(self):
        """Should fail when port is out of valid range."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "port": 99999,
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("port" in e.lower() for e in errors)

    def test_invalid_tensor_split_type(self):
        """Should fail when tensor_split is not a list."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "tensor_split": "0.5,0.5",
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("tensor_split" in e for e in errors)

    def test_invalid_tensor_split_length(self):
        """Should fail when tensor_split has fewer than 2 values."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "tensor_split": [1.0],
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("tensor_split" in e for e in errors)

    def test_invalid_tensor_split_values(self):
        """Should fail when tensor_split contains non-numbers."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "tensor_split": [0.5, "half"],
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("tensor_split" in e for e in errors)

    def test_invalid_gpu_layers_type(self):
        """Should fail when gpu_layers is not an integer."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "gpu_layers": 99.5,
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("gpu_layers" in e for e in errors)

    def test_invalid_extra_args_type(self):
        """Should fail when extra_args is not a list."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "extra_args": "-fa on",
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) >= 1
        assert any("extra_args" in e for e in errors)

    def test_extra_args_non_strings_coerced(self):
        """Non-string extra_args should be coerced to strings (YAML boolean gotcha)."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "extra_args": ["-fa", 1, True],
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) == 0
        assert config["extra_args"] == ["-fa", "1", "on"]

    def test_extra_args_string_booleans_coerced(self):
        """String 'True'/'False' in extra_args coerced to 'on'/'off' (Dashboard path)."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "extra_args": ["-fa", "True", "--mlock", "False"],
        }
        errors = LlamaProvider.validate_config(config)

        assert len(errors) == 0
        assert config["extra_args"] == ["-fa", "on", "--mlock", "off"]

    def test_valid_full_local_config(self):
        """Should pass with all valid local config options."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "host": "0.0.0.0",
            "port": 8080,
            "gpu_layers": 50,
            "device": "CUDA1",
            "extra_args": ["-fa", "on"],
            "timeout": 60.0,
            "temperature": 0.8,
            "max_tokens": 512,
        }
        errors = LlamaProvider.validate_config(config)

        assert errors == []

    # ----- NANO-042: Reasoning Config Validation -----

    def test_valid_reasoning_format_deepseek(self):
        """Should pass with reasoning_format='deepseek'."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_format": "deepseek",
        }
        errors = LlamaProvider.validate_config(config)
        assert errors == []

    def test_valid_reasoning_format_none(self):
        """Should pass with reasoning_format='none'."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_format": "none",
        }
        errors = LlamaProvider.validate_config(config)
        assert errors == []

    def test_invalid_reasoning_format(self):
        """Should fail with invalid reasoning_format value."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_format": "openai",
        }
        errors = LlamaProvider.validate_config(config)
        assert len(errors) == 1
        assert "reasoning_format" in errors[0]

    def test_valid_reasoning_budget_unlimited(self):
        """Should pass with reasoning_budget=-1 (unlimited)."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_budget": -1,
        }
        errors = LlamaProvider.validate_config(config)
        assert errors == []

    def test_valid_reasoning_budget_disabled(self):
        """Should pass with reasoning_budget=0 (disabled)."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_budget": 0,
        }
        errors = LlamaProvider.validate_config(config)
        assert errors == []

    def test_valid_reasoning_budget_positive(self):
        """Should pass with positive reasoning_budget."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_budget": 1024,
        }
        errors = LlamaProvider.validate_config(config)
        assert errors == []

    def test_invalid_reasoning_budget_type(self):
        """Should fail when reasoning_budget is not an integer."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_budget": 1.5,
        }
        errors = LlamaProvider.validate_config(config)
        assert len(errors) == 1
        assert "reasoning_budget" in errors[0]

    def test_invalid_reasoning_budget_too_low(self):
        """Should fail when reasoning_budget < -1."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "reasoning_budget": -2,
        }
        errors = LlamaProvider.validate_config(config)
        assert len(errors) == 1
        assert "reasoning_budget" in errors[0]

    def test_reasoning_config_absent_is_valid(self):
        """Should pass when reasoning config is entirely absent."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
        }
        errors = LlamaProvider.validate_config(config)
        assert errors == []

    def test_unified_vision_flag_passes_validation(self):
        """unified_vision flag should not cause validation errors (NANO-087)."""
        config = {
            "executable_path": "/bin/llama-server",
            "model_path": "/models/model.gguf",
            "unified_vision": True,
        }
        errors = LlamaProvider.validate_config(config)
        assert errors == []


# =============================================================================
# Unified Vision Slot Pinning Tests (NANO-087)
# =============================================================================


class TestUnifiedVisionSlotPinning:
    """Tests for id_slot injection when unified_vision is active."""

    def _make_provider(self, unified_vision: bool = False):
        """Create a provider configured for generate() testing."""
        from unittest.mock import MagicMock, patch

        provider = LlamaProvider()
        provider._base_url = "http://127.0.0.1:5557"
        provider._timeout = 30.0
        provider._max_retries = 1
        provider._retry_delay = 0.1
        provider._default_temperature = 0.7
        provider._default_max_tokens = 256
        provider._default_top_p = 0.95
        provider._model_name = "test-model"
        provider._unified_vision = unified_vision
        provider._initialized = True
        return provider

    def _mock_response(self):
        """Create a mock successful response."""
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "Hello", "role": "assistant"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp.raise_for_status = MagicMock()
        return resp

    def test_unified_vision_adds_id_slot_zero(self):
        """When unified_vision=True, payload includes id_slot=0 (NANO-087)."""
        from unittest.mock import patch

        provider = self._make_provider(unified_vision=True)

        with patch("spindl.llm.builtin.llama.provider.requests.post") as mock_post:
            mock_post.return_value = self._mock_response()
            provider.generate(messages=[{"role": "user", "content": "hi"}])

            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["id_slot"] == 0

    def test_no_unified_vision_no_id_slot(self):
        """When unified_vision=False (default), payload has no id_slot."""
        from unittest.mock import patch

        provider = self._make_provider(unified_vision=False)

        with patch("spindl.llm.builtin.llama.provider.requests.post") as mock_post:
            mock_post.return_value = self._mock_response()
            provider.generate(messages=[{"role": "user", "content": "hi"}])

            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert "id_slot" not in payload

    def test_unified_vision_from_config(self):
        """initialize() reads unified_vision from config dict."""
        from unittest.mock import patch

        provider = LlamaProvider()

        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "url": "http://127.0.0.1:5557",
                "unified_vision": True,
            })

        assert provider._unified_vision is True

    def test_unified_vision_default_false(self):
        """Without unified_vision in config, defaults to False."""
        from unittest.mock import patch

        provider = LlamaProvider()

        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "url": "http://127.0.0.1:5557",
            })

        assert provider._unified_vision is False


# =============================================================================
# Architecture Detection + supports_tool_role Tests (NANO-087e)
# =============================================================================


class TestArchitectureDetection:
    """Tests for _detect_architecture() and supports_tool_role."""

    def test_detect_gemma3_from_chat_template(self):
        """Detects gemma3 from <start_of_turn> in chat template."""
        arch = LlamaProvider._detect_architecture("model.gguf", {
            "chat_template": "{{ bos_token }}\n<start_of_turn>user\n..."
        })
        assert arch == "gemma3"

    def test_detect_gemma3_from_filename_hyphen(self):
        """Detects gemma3 from 'gemma-3' in filename."""
        arch = LlamaProvider._detect_architecture(
            "google_gemma-3-27b-it-Q6_K_L.gguf", {}
        )
        assert arch == "gemma3"

    def test_detect_gemma3_from_filename_no_hyphen(self):
        """Detects gemma3 from 'gemma3' in filename."""
        arch = LlamaProvider._detect_architecture(
            "SicariusSicariiStuff_X-Ray_Alpha-Q8_0.gguf", {}
        )
        # X-Ray doesn't contain gemma3 in name — should return None
        assert arch is None

    def test_detect_gemma3_finetune_with_gemma3_in_name(self):
        """Detects gemma3 architecture when name contains gemma3."""
        arch = LlamaProvider._detect_architecture(
            "some-gemma3-finetune-Q4_K_M.gguf", {}
        )
        assert arch == "gemma3"

    def test_unknown_model_returns_none(self):
        """Unknown model returns None."""
        arch = LlamaProvider._detect_architecture("llama-3.2-3b.gguf", {})
        assert arch is None

    def test_empty_props_returns_none(self):
        """Empty props with non-matching name returns None."""
        arch = LlamaProvider._detect_architecture("mistral-7b.gguf", {})
        assert arch is None

    def test_chat_template_takes_priority(self):
        """Chat template detection runs before filename matching."""
        arch = LlamaProvider._detect_architecture("random-model.gguf", {
            "chat_template": "blah <start_of_turn> blah"
        })
        assert arch == "gemma3"

    def test_gemma3_supports_tool_role_false(self):
        """Gemma3 architecture returns supports_tool_role=False."""
        provider = LlamaProvider()
        provider._model_name = "gemma-3-27b"
        provider._model_architecture = "gemma3"
        provider._context_length = 8192

        props = provider.get_properties()
        assert props.supports_tool_role is False

    def test_unknown_arch_supports_tool_role_false(self):
        """Unknown architecture defaults to supports_tool_role=False (safe)."""
        provider = LlamaProvider()
        provider._model_name = "some-model"
        provider._model_architecture = None
        provider._context_length = 8192

        props = provider.get_properties()
        assert props.supports_tool_role is False

    def test_non_strict_arch_supports_tool_role_true(self):
        """Non-strict architecture returns supports_tool_role=True."""
        provider = LlamaProvider()
        provider._model_name = "llama-3.2"
        provider._model_architecture = "llama"
        provider._context_length = 8192

        props = provider.get_properties()
        assert props.supports_tool_role is True
