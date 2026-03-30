"""Tests for VLM model configurations."""

import pytest

from spindl.vision.builtin.llama.models import (
    get_model_config,
    list_model_types,
    Gemma3ModelConfig,
    MODEL_CONFIGS,
)
from spindl.vision.builtin.llama.models.base import (
    ModelLaunchConfig,
    VLMModelConfig,
)


class TestModelConfigRegistry:
    """Tests for model config registry functions."""

    def test_list_model_types_includes_gemma3(self) -> None:
        """list_model_types includes gemma3."""
        types = list_model_types()
        assert "gemma3" in types

    def test_get_model_config_returns_instance(self) -> None:
        """get_model_config returns an instantiated config."""
        config = get_model_config("gemma3")
        assert isinstance(config, VLMModelConfig)
        assert isinstance(config, Gemma3ModelConfig)

    def test_get_model_config_unknown_type_raises(self) -> None:
        """get_model_config raises ValueError for unknown type."""
        with pytest.raises(ValueError) as exc_info:
            get_model_config("nonexistent_model")

        assert "Unknown model type" in str(exc_info.value)
        assert "nonexistent_model" in str(exc_info.value)


class TestGemma3ModelConfig:
    """Tests for Gemma3ModelConfig."""

    def test_model_type_property(self) -> None:
        """model_type returns 'gemma3'."""
        config = Gemma3ModelConfig()
        assert config.model_type == "gemma3"

    def test_requires_mmproj_true(self) -> None:
        """requires_mmproj returns True for Gemma3."""
        config = Gemma3ModelConfig()
        assert config.requires_mmproj is True

    def test_generate_launch_config_success(self) -> None:
        """generate_launch_config produces valid command."""
        config = Gemma3ModelConfig()

        launch = config.generate_launch_config(
            executable_path="/opt/llama/llama-server",
            model_path="/path/to/model.gguf",
            mmproj_path="/path/to/mmproj.gguf",
            port=5558,
            context_size=8192,
            gpu_layers=99,
        )

        assert isinstance(launch, ModelLaunchConfig)
        assert "/opt/llama/llama-server" in launch.command
        assert "/path/to/model.gguf" in launch.command
        assert "--mmproj" in launch.command
        assert "/path/to/mmproj.gguf" in launch.command
        assert "-c 8192" in launch.command
        assert "--port 5558" in launch.command
        assert "-ngl 99" in launch.command
        assert launch.port == 5558
        assert launch.health_url == "http://127.0.0.1:5558/health"

    def test_generate_launch_config_without_mmproj_raises(self) -> None:
        """generate_launch_config raises ValueError without mmproj."""
        config = Gemma3ModelConfig()

        with pytest.raises(ValueError) as exc_info:
            config.generate_launch_config(
                executable_path="/opt/llama/llama-server",
                model_path="/path/to/model.gguf",
                mmproj_path=None,
            )

        assert "mmproj" in str(exc_info.value).lower()

    def test_generate_launch_config_with_extra_args(self) -> None:
        """generate_launch_config includes extra_args."""
        config = Gemma3ModelConfig()

        launch = config.generate_launch_config(
            executable_path="/opt/llama/llama-server",
            model_path="/path/to/model.gguf",
            mmproj_path="/path/to/mmproj.gguf",
            extra_args=["--flash-attn", "--threads", "8"],
        )

        assert "--flash-attn" in launch.command
        assert "--threads 8" in launch.command

    def test_validate_config_valid(self) -> None:
        """validate_config returns (True, None) for valid config."""
        config = Gemma3ModelConfig()

        is_valid, error = config.validate_config(
            model_path="/path/to/model.gguf",
            mmproj_path="/path/to/mmproj.gguf",
        )

        assert is_valid is True
        assert error is None

    def test_validate_config_missing_model_path(self) -> None:
        """validate_config returns error for missing model_path."""
        config = Gemma3ModelConfig()

        is_valid, error = config.validate_config(
            model_path="",
            mmproj_path="/path/to/mmproj.gguf",
        )

        assert is_valid is False
        assert "model_path" in error

    def test_validate_config_missing_mmproj(self) -> None:
        """validate_config returns error for missing mmproj."""
        config = Gemma3ModelConfig()

        is_valid, error = config.validate_config(
            model_path="/path/to/model.gguf",
            mmproj_path=None,
        )

        assert is_valid is False
        assert "mmproj" in error.lower()
