"""Tests for generation parameter resolution (NANO-108)."""

import pytest

from spindl.llm.pipeline import LLMPipeline


class TestResolveGenerationParams:
    """Tests for _resolve_generation_params() repetition penalty support."""

    def _make_pipeline(self) -> LLMPipeline:
        """Create a minimal pipeline for testing param resolution."""
        # Pipeline can be instantiated without a provider for param resolution
        pipeline = LLMPipeline.__new__(LLMPipeline)
        return pipeline

    def test_defaults_include_repetition_params(self) -> None:
        """Hardcoded defaults include all 4 repetition params."""
        pipeline = self._make_pipeline()
        params = pipeline._resolve_generation_params({}, None)

        assert params["repeat_penalty"] == 1.1
        assert params["repeat_last_n"] == 64
        assert params["frequency_penalty"] == 0.0
        assert params["presence_penalty"] == 0.0

    def test_defaults_include_original_params(self) -> None:
        """Hardcoded defaults still include temperature, max_tokens, top_p."""
        pipeline = self._make_pipeline()
        params = pipeline._resolve_generation_params({}, None)

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 256
        assert params["top_p"] == 0.95

    def test_persona_overrides_repetition_params(self) -> None:
        """Persona generation block overrides repetition defaults."""
        pipeline = self._make_pipeline()
        persona = {
            "generation": {
                "repeat_penalty": 1.5,
                "repeat_last_n": 128,
                "frequency_penalty": 0.3,
                "presence_penalty": -0.2,
            }
        }
        params = pipeline._resolve_generation_params(persona, None)

        assert params["repeat_penalty"] == 1.5
        assert params["repeat_last_n"] == 128
        assert params["frequency_penalty"] == 0.3
        assert params["presence_penalty"] == -0.2

    def test_persona_partial_override(self) -> None:
        """Persona with only some params leaves others at defaults."""
        pipeline = self._make_pipeline()
        persona = {
            "generation": {
                "repeat_penalty": 1.3,
            }
        }
        params = pipeline._resolve_generation_params(persona, None)

        assert params["repeat_penalty"] == 1.3
        assert params["repeat_last_n"] == 64  # default
        assert params["frequency_penalty"] == 0.0  # default
        assert params["presence_penalty"] == 0.0  # default

    def test_override_takes_priority_over_persona(self) -> None:
        """Explicit override dict beats persona generation values."""
        pipeline = self._make_pipeline()
        persona = {
            "generation": {
                "repeat_penalty": 1.5,
                "frequency_penalty": 0.3,
            }
        }
        override = {
            "repeat_penalty": 1.8,
            "frequency_penalty": 0.7,
        }
        params = pipeline._resolve_generation_params(persona, override)

        assert params["repeat_penalty"] == 1.8
        assert params["frequency_penalty"] == 0.7

    def test_override_takes_priority_over_defaults(self) -> None:
        """Explicit override dict beats hardcoded defaults."""
        pipeline = self._make_pipeline()
        override = {
            "presence_penalty": 0.5,
            "repeat_last_n": 256,
        }
        params = pipeline._resolve_generation_params({}, override)

        assert params["presence_penalty"] == 0.5
        assert params["repeat_last_n"] == 256

    def test_all_seven_params_present(self) -> None:
        """Result always contains all 7 generation params."""
        pipeline = self._make_pipeline()
        params = pipeline._resolve_generation_params({}, None)

        expected_keys = {
            "temperature", "max_tokens", "top_p",
            "repeat_penalty", "repeat_last_n",
            "frequency_penalty", "presence_penalty",
        }
        assert expected_keys.issubset(set(params.keys()))
