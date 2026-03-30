"""
Tests for OpenRouterProvider (OpenRouter cloud LLM provider).

Tests cover:
- validate_config() — Config validation for required/optional fields
- is_cloud_provider() / get_server_command() — Cloud provider classification
- initialize() — API key resolution, model requirement, health check
- generate() — Response parsing, reasoning field, tool calls
- generate_stream() — SSE parsing, keep-alive handling, reasoning in delta
- count_tokens() — Char-ratio estimation
- health_check() — API reachability
- _build_headers() — OpenRouter-specific headers
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from spindl.llm.builtin.openrouter.provider import OpenRouterProvider, resolve_env_vars


# =============================================================================
# resolve_env_vars() Tests
# =============================================================================


class TestResolveEnvVars:
    """Tests for environment variable substitution."""

    def test_resolves_env_var(self):
        """Should replace ${VAR} with env value."""
        with patch.dict("os.environ", {"MY_KEY": "secret123"}):
            assert resolve_env_vars("${MY_KEY}") == "secret123"

    def test_missing_env_var_raises(self):
        """Should raise ValueError for unset env var."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MISSING_VAR"):
                resolve_env_vars("${MISSING_VAR}")

    def test_passthrough_plain_string(self):
        """Should return plain strings unchanged."""
        assert resolve_env_vars("sk-plain-key") == "sk-plain-key"

    def test_multiple_vars(self):
        """Should resolve multiple env vars in one string."""
        with patch.dict("os.environ", {"A": "hello", "B": "world"}):
            assert resolve_env_vars("${A}-${B}") == "hello-world"


# =============================================================================
# validate_config() Tests
# =============================================================================


class TestValidateConfig:
    """Tests for OpenRouterProvider.validate_config()."""

    def test_valid_minimal_config(self):
        """Should pass with required fields only."""
        config = {
            "api_key": "${OPENROUTER_API_KEY}",
            "model": "google/gemini-2.5-pro",
        }
        errors = OpenRouterProvider.validate_config(config)
        assert errors == []

    def test_valid_full_config(self):
        """Should pass with all valid fields."""
        config = {
            "api_key": "sk-or-v1-test",
            "model": "openai/gpt-4o",
            "url": "https://openrouter.ai/api/v1",
            "timeout": 120,
            "temperature": 0.8,
            "max_tokens": 2048,
        }
        errors = OpenRouterProvider.validate_config(config)
        assert errors == []

    def test_missing_api_key(self):
        """Should fail when api_key is missing."""
        config = {"model": "openai/gpt-4o"}
        errors = OpenRouterProvider.validate_config(config)
        assert len(errors) == 1
        assert "api_key" in errors[0]

    def test_missing_model(self):
        """Should fail when model is missing."""
        config = {"api_key": "sk-test"}
        errors = OpenRouterProvider.validate_config(config)
        assert len(errors) == 1
        assert "model" in errors[0]

    def test_empty_config(self):
        """Should fail with both api_key and model errors."""
        errors = OpenRouterProvider.validate_config({})
        assert len(errors) == 2
        assert any("api_key" in e for e in errors)
        assert any("model" in e for e in errors)

    def test_invalid_api_key_type(self):
        """Should fail when api_key is not a string."""
        config = {"api_key": 12345, "model": "openai/gpt-4o"}
        errors = OpenRouterProvider.validate_config(config)
        assert any("api_key" in e and "string" in e for e in errors)

    def test_invalid_model_type(self):
        """Should fail when model is not a string."""
        config = {"api_key": "sk-test", "model": 42}
        errors = OpenRouterProvider.validate_config(config)
        assert any("model" in e and "string" in e for e in errors)

    def test_invalid_url_type(self):
        """Should fail when url is not a string."""
        config = {"api_key": "sk-test", "model": "openai/gpt-4o", "url": 123}
        errors = OpenRouterProvider.validate_config(config)
        assert any("url" in e for e in errors)

    def test_invalid_timeout_type(self):
        """Should fail when timeout is not a number."""
        config = {"api_key": "sk-test", "model": "openai/gpt-4o", "timeout": "fast"}
        errors = OpenRouterProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_negative_timeout(self):
        """Should fail when timeout is non-positive."""
        config = {"api_key": "sk-test", "model": "openai/gpt-4o", "timeout": -1}
        errors = OpenRouterProvider.validate_config(config)
        assert any("timeout" in e for e in errors)

    def test_invalid_temperature_range(self):
        """Should fail when temperature is out of range."""
        config = {"api_key": "sk-test", "model": "openai/gpt-4o", "temperature": 3.0}
        errors = OpenRouterProvider.validate_config(config)
        assert any("temperature" in e for e in errors)

    def test_invalid_max_tokens_type(self):
        """Should fail when max_tokens is not an integer."""
        config = {"api_key": "sk-test", "model": "openai/gpt-4o", "max_tokens": 1.5}
        errors = OpenRouterProvider.validate_config(config)
        assert any("max_tokens" in e for e in errors)

    def test_zero_max_tokens(self):
        """Should fail when max_tokens is less than 1."""
        config = {"api_key": "sk-test", "model": "openai/gpt-4o", "max_tokens": 0}
        errors = OpenRouterProvider.validate_config(config)
        assert any("max_tokens" in e for e in errors)


# =============================================================================
# Cloud Provider Classification Tests
# =============================================================================


class TestCloudClassification:
    """Tests for cloud provider class methods."""

    def test_is_cloud_provider(self):
        """OpenRouter should be classified as cloud provider."""
        assert OpenRouterProvider.is_cloud_provider() is True

    def test_get_server_command_returns_none(self):
        """Cloud provider should not generate server commands."""
        config = {"api_key": "sk-test", "model": "openai/gpt-4o"}
        assert OpenRouterProvider.get_server_command(config) is None


# =============================================================================
# initialize() Tests
# =============================================================================


class TestInitialize:
    """Tests for OpenRouterProvider.initialize()."""

    def test_missing_api_key_raises(self):
        """Should raise ValueError when api_key is missing."""
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="API key not configured"):
            provider.initialize({"model": "openai/gpt-4o"})

    def test_empty_api_key_raises(self):
        """Should raise ValueError when api_key is empty string."""
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="API key not configured"):
            provider.initialize({"api_key": "", "model": "openai/gpt-4o"})

    def test_missing_model_raises(self):
        """Should raise ValueError when model is missing."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            with pytest.raises(ValueError, match="model not configured"):
                provider.initialize({"api_key": "sk-test"})

    def test_env_var_resolution(self):
        """Should resolve ${ENV_VAR} in api_key."""
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-resolved"}):
            with patch.object(provider, "_health_check_internal", return_value=True):
                provider.initialize({
                    "api_key": "${OPENROUTER_API_KEY}",
                    "model": "openai/gpt-4o",
                })
                assert provider._api_key == "sk-resolved"

    def test_unset_env_var_raises(self):
        """Should raise ValueError for unset env var in api_key."""
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="NONEXISTENT_KEY"):
                provider.initialize({
                    "api_key": "${NONEXISTENT_KEY}",
                    "model": "openai/gpt-4o",
                })

    def test_health_check_failure_raises(self):
        """Should raise ConnectionError when API is unreachable."""
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test"}):
            with patch.object(provider, "_health_check_internal", return_value=False):
                with pytest.raises(ConnectionError, match="not reachable"):
                    provider.initialize({
                        "api_key": "${OPENROUTER_API_KEY}",
                        "model": "openai/gpt-4o",
                    })

    def test_successful_initialization(self):
        """Should set all fields and mark initialized."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-direct-key",
                "model": "anthropic/claude-4-sonnet",
                "url": "https://custom.openrouter.ai/api/v1",
                "timeout": 180,
                "temperature": 0.5,
                "max_tokens": 4096,
            })

        assert provider._initialized is True
        assert provider._api_key == "sk-direct-key"
        assert provider._model == "anthropic/claude-4-sonnet"
        assert provider._base_url == "https://custom.openrouter.ai/api/v1"
        assert provider._timeout == 180
        assert provider._default_temperature == 0.5
        assert provider._default_max_tokens == 4096

    def test_default_values(self):
        """Should use defaults for optional fields."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-test",
                "model": "openai/gpt-4o",
            })

        assert provider._base_url == "https://openrouter.ai/api/v1"
        assert provider._timeout == 120.0
        assert provider._default_temperature == 0.7
        assert provider._default_max_tokens == 256

    def test_url_trailing_slash_stripped(self):
        """Should strip trailing slash from base URL."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-test",
                "model": "openai/gpt-4o",
                "url": "https://openrouter.ai/api/v1/",
            })

        assert provider._base_url == "https://openrouter.ai/api/v1"


# =============================================================================
# get_properties() Tests
# =============================================================================


class TestGetProperties:
    """Tests for OpenRouterProvider.get_properties()."""

    def test_properties_reflect_model(self):
        """Should return properties matching configured model."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-test",
                "model": "meta-llama/llama-3.3-70b",
            })

        props = provider.get_properties()
        assert props.model_name == "meta-llama/llama-3.3-70b"
        assert props.supports_streaming is True
        assert props.supports_tools is True
        assert props.context_length == 8192  # Fallback when no context_size configured


# =============================================================================
# _build_headers() Tests
# =============================================================================


class TestBuildHeaders:
    """Tests for OpenRouter-specific HTTP headers."""

    def test_includes_required_headers(self):
        """Should include Authorization, HTTP-Referer, and X-Title."""
        provider = OpenRouterProvider()
        provider._api_key = "sk-test-key"

        headers = provider._build_headers()

        assert headers["Authorization"] == "Bearer sk-test-key"
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers
        assert headers["Content-Type"] == "application/json"

    def test_referer_is_github_url(self):
        """HTTP-Referer should point to project GitHub."""
        provider = OpenRouterProvider()
        provider._api_key = "sk-test"

        headers = provider._build_headers()

        assert "github.com" in headers["HTTP-Referer"]

    def test_title_is_spindl(self):
        """X-Title should be SpindL."""
        provider = OpenRouterProvider()
        provider._api_key = "sk-test"

        headers = provider._build_headers()

        assert headers["X-Title"] == "SpindL"


# =============================================================================
# generate() Tests
# =============================================================================


class TestGenerate:
    """Tests for OpenRouterProvider.generate()."""

    def _make_provider(self):
        """Create an initialized provider with mocked health check."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-test",
                "model": "openai/gpt-4o",
            })
        return provider

    def test_not_initialized_raises(self):
        """Should raise RuntimeError if not initialized."""
        provider = OpenRouterProvider()
        with pytest.raises(RuntimeError, match="not initialized"):
            provider.generate([{"role": "user", "content": "hi"}])

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_basic_generation(self, mock_post):
        """Should parse a standard response correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello!", "role": "assistant"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        }
        mock_post.return_value = mock_response

        provider = self._make_provider()
        result = provider.generate([{"role": "user", "content": "hi"}])

        assert result.content == "Hello!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.finish_reason == "stop"
        assert result.reasoning is None
        assert result.tool_calls == []

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_reasoning_field(self, mock_post):
        """Should extract reasoning from OpenRouter's `reasoning` field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "The answer is 42.",
                    "reasoning": "Let me think step by step...",
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        }
        mock_post.return_value = mock_response

        provider = self._make_provider()
        result = provider.generate([{"role": "user", "content": "think about this"}])

        assert result.content == "The answer is 42."
        assert result.reasoning == "Let me think step by step..."
        assert result.reasoning_tokens == 12

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_tool_calls(self, mock_post):
        """Should parse tool calls from response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "screen_vision",
                            "arguments": '{"prompt": "describe"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        mock_post.return_value = mock_response

        provider = self._make_provider()
        result = provider.generate(
            [{"role": "user", "content": "look at screen"}],
            tools=[{"type": "function", "function": {"name": "screen_vision"}}],
        )

        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "screen_vision"
        assert result.tool_calls[0].arguments == {"prompt": "describe"}

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_401_raises_value_error(self, mock_post):
        """Should raise ValueError on auth failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid key"}}
        mock_response.text = "Invalid key"
        mock_post.return_value = mock_response

        provider = self._make_provider()
        with pytest.raises(ValueError, match="Invalid API key"):
            provider.generate([{"role": "user", "content": "hi"}])

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_402_raises_value_error(self, mock_post):
        """Should raise ValueError on insufficient credits."""
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.json.return_value = {"error": {"message": "No credits"}}
        mock_response.text = "No credits"
        mock_post.return_value = mock_response

        provider = self._make_provider()
        with pytest.raises(ValueError, match="Insufficient credits"):
            provider.generate([{"role": "user", "content": "hi"}])

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_429_raises_runtime_error(self, mock_post):
        """Should raise RuntimeError on rate limit."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limited"}}
        mock_response.text = "Rate limited"
        mock_post.return_value = mock_response

        provider = self._make_provider()
        with pytest.raises(RuntimeError, match="Rate limited"):
            provider.generate([{"role": "user", "content": "hi"}])

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_502_raises_runtime_error(self, mock_post):
        """Should raise RuntimeError on upstream provider error."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.json.return_value = {"error": {"message": "Provider down"}}
        mock_response.text = "Provider down"
        mock_post.return_value = mock_response

        provider = self._make_provider()
        with pytest.raises(RuntimeError, match="Upstream provider error"):
            provider.generate([{"role": "user", "content": "hi"}])


# =============================================================================
# generate_stream() Tests
# =============================================================================


class TestGenerateStream:
    """Tests for OpenRouterProvider.generate_stream()."""

    def _make_provider(self):
        """Create an initialized provider with mocked health check."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-test",
                "model": "openai/gpt-4o",
            })
        return provider

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_basic_streaming(self, mock_post):
        """Should parse SSE stream into chunks."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_lines
        mock_post.return_value = mock_response

        provider = self._make_provider()
        chunks = list(provider.generate_stream([{"role": "user", "content": "hi"}]))

        content = "".join(c.content for c in chunks)
        assert content == "Hello world"
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "stop"

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_keep_alive_comments_ignored(self, mock_post):
        """Should skip SSE comment lines (: OPENROUTER PROCESSING)."""
        sse_lines = [
            b': OPENROUTER PROCESSING',
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            b': OPENROUTER PROCESSING',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_lines
        mock_post.return_value = mock_response

        provider = self._make_provider()
        chunks = list(provider.generate_stream([{"role": "user", "content": "hi"}]))

        content = "".join(c.content for c in chunks)
        assert content == "Hi"

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_reasoning_in_stream(self, mock_post):
        """Should extract reasoning from delta's `reasoning` field."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"reasoning":"Step 1..."},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":"Answer."},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_lines
        mock_post.return_value = mock_response

        provider = self._make_provider()
        chunks = list(provider.generate_stream([{"role": "user", "content": "think"}]))

        reasoning_chunks = [c for c in chunks if c.reasoning]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0].reasoning == "Step 1..."

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_usage_in_final_chunk(self, mock_post):
        """Should include token usage in the usage-only final chunk."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: {"usage":{"prompt_tokens":10,"completion_tokens":5,"completion_tokens_details":{"reasoning_tokens":2}}}',
            b'data: [DONE]',
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_lines
        mock_post.return_value = mock_response

        provider = self._make_provider()
        chunks = list(provider.generate_stream([{"role": "user", "content": "hi"}]))

        final = chunks[-1]
        assert final.is_final is True
        assert final.input_tokens == 10
        assert final.output_tokens == 5
        assert final.reasoning_tokens == 2

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_tool_calls_accumulated_across_chunks(self, mock_post):
        """Should accumulate tool call data across multiple SSE events."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"screen_vision","arguments":"{\\"pr"}}]},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ompt\\": \\"look\\"}"}}]},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            b'data: [DONE]',
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_lines
        mock_post.return_value = mock_response

        provider = self._make_provider()
        chunks = list(provider.generate_stream([{"role": "user", "content": "look"}]))

        final = [c for c in chunks if c.is_final]
        assert len(final) == 1
        assert len(final[0].tool_calls) == 1
        assert final[0].tool_calls[0].name == "screen_vision"
        assert final[0].tool_calls[0].arguments == {"prompt": "look"}

    @patch("spindl.llm.builtin.openrouter.provider.requests.post")
    def test_empty_lines_skipped(self, mock_post):
        """Should skip empty lines in SSE stream."""
        sse_lines = [
            b'',
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            b'',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_lines
        mock_post.return_value = mock_response

        provider = self._make_provider()
        chunks = list(provider.generate_stream([{"role": "user", "content": "hi"}]))

        content = "".join(c.content for c in chunks)
        assert content == "Hi"


# =============================================================================
# count_tokens() Tests
# =============================================================================


class TestCountTokens:
    """Tests for char-ratio token estimation."""

    def _make_provider(self):
        """Create an initialized provider."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-test",
                "model": "openai/gpt-4o",
            })
        return provider

    def test_not_initialized_raises(self):
        """Should raise RuntimeError if not initialized."""
        provider = OpenRouterProvider()
        with pytest.raises(RuntimeError, match="not initialized"):
            provider.count_tokens("hello")

    def test_short_text(self):
        """Should return at least 1 for short text."""
        provider = self._make_provider()
        assert provider.count_tokens("hi") >= 1

    def test_reasonable_estimate(self):
        """Should estimate ~1 token per 4 characters."""
        provider = self._make_provider()
        text = "a" * 400
        tokens = provider.count_tokens(text)
        assert tokens == 100

    def test_empty_string(self):
        """Should return 1 for empty string (minimum)."""
        provider = self._make_provider()
        assert provider.count_tokens("") == 1


# =============================================================================
# health_check() Tests
# =============================================================================


class TestHealthCheck:
    """Tests for API reachability check."""

    @patch("spindl.llm.builtin.openrouter.provider.requests.get")
    def test_healthy(self, mock_get):
        """Should return True when /models returns 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = OpenRouterProvider()
        provider._api_key = "sk-test"
        assert provider._health_check_internal() is True

    @patch("spindl.llm.builtin.openrouter.provider.requests.get")
    def test_unhealthy_status(self, mock_get):
        """Should return False when /models returns non-200."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        provider = OpenRouterProvider()
        provider._api_key = "sk-test"
        assert provider._health_check_internal() is False

    @patch("spindl.llm.builtin.openrouter.provider.requests.get")
    def test_connection_error(self, mock_get):
        """Should return False on connection error."""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError()

        provider = OpenRouterProvider()
        provider._api_key = "sk-test"
        assert provider._health_check_internal() is False


# =============================================================================
# shutdown() Tests
# =============================================================================


class TestShutdown:
    """Tests for provider cleanup."""

    def test_clears_state(self):
        """Should clear sensitive data and mark uninitialized."""
        provider = OpenRouterProvider()
        with patch.object(provider, "_health_check_internal", return_value=True):
            provider.initialize({
                "api_key": "sk-sensitive",
                "model": "openai/gpt-4o",
            })

        provider.shutdown()

        assert provider._initialized is False
        assert provider._api_key == ""
