"""
Tests for GUI server utilities.

Tests cover:
- _extract_model_name() - Model name extraction and quantization suffix stripping
"""

import pytest

from spindl.gui.server import GUIServer


# =============================================================================
# _extract_model_name() Tests
# =============================================================================


class TestExtractModelName:
    """Tests for GUIServer._extract_model_name()."""

    def test_full_path_with_q8_0(self):
        """Should extract model name and strip Q8_0 suffix."""
        model_path = "X:/Models/Rocinante-X-12B-v1b-Q8_0.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Rocinante-X-12B-v1b"

    def test_full_path_with_q4_k_m(self):
        """Should extract model name and strip Q4_K_M suffix."""
        model_path = "/models/Llama-3-70B-Q4_K_M.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Llama-3-70B"

    def test_full_path_with_iq4_xs(self):
        """Should extract model name and strip IQ4_XS suffix."""
        model_path = "./models/Mistral-7B-IQ4_XS.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Mistral-7B"

    def test_no_quantization_suffix(self):
        """Should keep model name intact when no quantization suffix."""
        model_path = "/models/CustomModel.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "CustomModel"

    def test_windows_backslash_path(self):
        """Should handle Windows-style backslash paths."""
        model_path = "X:\\Models\\Model-Q8_0.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Model"

    def test_q6_k_suffix(self):
        """Should strip Q6_K suffix."""
        model_path = "/models/Phi-3-mini-Q6_K.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Phi-3-mini"

    def test_q5_k_m_suffix(self):
        """Should strip Q5_K_M suffix."""
        model_path = "/models/Qwen-14B-Q5_K_M.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Qwen-14B"

    def test_f16_suffix(self):
        """Should strip f16 suffix."""
        model_path = "/models/TinyLlama-f16.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "TinyLlama"

    def test_bf16_suffix(self):
        """Should strip bf16 suffix."""
        model_path = "/models/Gemma-2B-bf16.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Gemma-2B"

    def test_iq3_xxs_suffix(self):
        """Should strip IQ3_XXS suffix (importance matrix quant)."""
        model_path = "/models/DeepSeek-7B-IQ3_XXS.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "DeepSeek-7B"

    def test_q4_k_l_suffix(self):
        """Should strip Q4_K_L suffix (large K-quant)."""
        model_path = "/models/Command-R-35B-Q4_K_L.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Command-R-35B"

    def test_relative_path(self):
        """Should handle relative paths."""
        model_path = "../models/Model-Q4_0.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Model"

    def test_filename_only(self):
        """Should handle filename without directory path."""
        model_path = "Model-Q8_0.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Model"

    def test_model_with_version_numbers(self):
        """Should preserve version numbers in model name."""
        model_path = "/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "Llama-3.1-8B-Instruct"

    def test_model_with_dashes_in_name(self):
        """Should handle model names with many dashes."""
        model_path = "/models/my-custom-fine-tuned-model-v2-Q8_0.gguf"
        result = GUIServer._extract_model_name(model_path)

        assert result == "my-custom-fine-tuned-model-v2"
