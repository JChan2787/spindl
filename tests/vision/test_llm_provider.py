"""Tests for LLMVisionProvider."""

from unittest import mock

import pytest
import requests

from spindl.vision.builtin.llm.provider import LLMVisionProvider
from spindl.vision.base import VLMProvider, VLMResponse


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestLLMVisionProviderConfig:
    """Tests for validate_config()."""

    def test_validate_config_valid_local(self) -> None:
        """Valid local config (url only) returns no errors."""
        errors = LLMVisionProvider.validate_config({
            "url": "http://127.0.0.1:5557",
        })
        assert errors == []

    def test_validate_config_valid_cloud(self) -> None:
        """Valid cloud config (url + api_key) returns no errors."""
        errors = LLMVisionProvider.validate_config({
            "url": "https://api.openai.com",
            "api_key": "${OPENAI_API_KEY}",
        })
        assert errors == []

    def test_validate_config_valid_with_all_optional(self) -> None:
        """Full config with all optional fields returns no errors."""
        errors = LLMVisionProvider.validate_config({
            "url": "http://127.0.0.1:5557",
            "api_key": "sk-test",
            "timeout": 60.0,
            "max_tokens": 500,
        })
        assert errors == []

    def test_validate_config_missing_url(self) -> None:
        """Missing url returns an error."""
        errors = LLMVisionProvider.validate_config({})
        assert len(errors) == 1
        assert "url is required" in errors[0]

    def test_validate_config_invalid_url_scheme(self) -> None:
        """URL without http(s):// returns an error."""
        errors = LLMVisionProvider.validate_config({
            "url": "ftp://some-server",
        })
        assert len(errors) == 1
        assert "http://" in errors[0]

    def test_validate_config_invalid_url_type(self) -> None:
        """Non-string url returns an error."""
        errors = LLMVisionProvider.validate_config({
            "url": 12345,
        })
        assert len(errors) == 1
        assert "must be a string" in errors[0]

    def test_validate_config_invalid_timeout(self) -> None:
        """Negative timeout returns an error."""
        errors = LLMVisionProvider.validate_config({
            "url": "http://127.0.0.1:5557",
            "timeout": -5,
        })
        assert len(errors) == 1
        assert "positive" in errors[0]

    def test_validate_config_invalid_max_tokens(self) -> None:
        """Non-integer max_tokens returns an error."""
        errors = LLMVisionProvider.validate_config({
            "url": "http://127.0.0.1:5557",
            "max_tokens": 3.5,
        })
        assert len(errors) == 1
        assert "integer" in errors[0]

    def test_validate_config_zero_max_tokens(self) -> None:
        """Zero max_tokens returns an error."""
        errors = LLMVisionProvider.validate_config({
            "url": "http://127.0.0.1:5557",
            "max_tokens": 0,
        })
        assert len(errors) == 1
        assert "at least 1" in errors[0]


# ---------------------------------------------------------------------------
# Class methods (launcher interface)
# ---------------------------------------------------------------------------


class TestLLMVisionProviderClassMethods:
    """Tests for class-level methods used by launcher."""

    def test_is_cloud_provider_returns_true(self) -> None:
        """is_cloud_provider() returns True (no VLM server to launch)."""
        assert LLMVisionProvider.is_cloud_provider() is True

    def test_get_server_command_returns_none(self) -> None:
        """get_server_command() returns None."""
        assert LLMVisionProvider.get_server_command({}) is None

    def test_get_health_url_returns_none(self) -> None:
        """get_health_url() returns None."""
        assert LLMVisionProvider.get_health_url({}) is None

    def test_is_vlm_provider_subclass(self) -> None:
        """LLMVisionProvider is a VLMProvider subclass."""
        assert issubclass(LLMVisionProvider, VLMProvider)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestLLMVisionProviderURLConstruction:
    """Tests for _build_completions_url()."""

    def _make_provider(self, url: str) -> LLMVisionProvider:
        """Create a provider with a given URL (skip health check)."""
        provider = LLMVisionProvider()
        provider._url = url.rstrip("/")
        return provider

    def test_url_without_v1_appends_full_path(self) -> None:
        """Base URL gets /v1/chat/completions appended."""
        p = self._make_provider("http://127.0.0.1:5557")
        assert p._build_completions_url() == "http://127.0.0.1:5557/v1/chat/completions"

    def test_url_with_v1_appends_only_completions(self) -> None:
        """URL ending in /v1 gets /chat/completions appended."""
        p = self._make_provider("https://api.deepseek.com/v1")
        assert p._build_completions_url() == "https://api.deepseek.com/v1/chat/completions"

    def test_url_trailing_slash_stripped(self) -> None:
        """Trailing slash is stripped before path construction."""
        p = self._make_provider("http://127.0.0.1:5557/")
        assert p._build_completions_url() == "http://127.0.0.1:5557/v1/chat/completions"

    def test_url_with_v1_trailing_slash(self) -> None:
        """URL ending in /v1/ is handled correctly."""
        p = self._make_provider("https://api.deepseek.com/v1/")
        assert p._build_completions_url() == "https://api.deepseek.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


class TestLLMVisionProviderInitialize:
    """Tests for initialize()."""

    @mock.patch.object(LLMVisionProvider, "health_check", return_value=True)
    def test_initialize_local_defaults(self, _mock_hc) -> None:
        """Initialize with minimal config stores correct defaults."""
        provider = LLMVisionProvider()
        provider.initialize({"url": "http://127.0.0.1:5557"})

        assert provider._url == "http://127.0.0.1:5557"
        assert provider._api_key is None
        assert provider._model == "local-llm"
        assert provider._timeout == 30.0
        assert provider._max_tokens == 300
        assert provider._initialized is True

    @mock.patch.object(LLMVisionProvider, "health_check", return_value=True)
    def test_initialize_cloud_with_api_key(self, _mock_hc) -> None:
        """Initialize with api_key resolves env vars."""
        with mock.patch.dict("os.environ", {"TEST_KEY": "sk-secret"}):
            provider = LLMVisionProvider()
            provider.initialize({
                "url": "https://api.openai.com",
                "api_key": "${TEST_KEY}",
                "model": "gpt-4o",
            })

        assert provider._api_key == "sk-secret"
        assert provider._model == "gpt-4o"

    @mock.patch.object(LLMVisionProvider, "health_check", return_value=True)
    def test_initialize_unresolved_env_var_sets_none(self, _mock_hc) -> None:
        """Unresolved env var results in api_key = None."""
        with mock.patch.dict("os.environ", {}, clear=False):
            # Ensure the var does NOT exist
            import os
            os.environ.pop("NONEXISTENT_KEY_12345", None)

            provider = LLMVisionProvider()
            provider.initialize({
                "url": "https://api.openai.com",
                "api_key": "${NONEXISTENT_KEY_12345}",
            })

        assert provider._api_key is None

    @mock.patch.object(LLMVisionProvider, "health_check", return_value=True)
    def test_initialize_custom_values(self, _mock_hc) -> None:
        """Custom prompt, max_tokens, timeout are stored."""
        provider = LLMVisionProvider()
        provider.initialize({
            "url": "http://localhost:8080",
            "timeout": 60.0,
            "max_tokens": 500,
            "prompt": "What is on screen?",
        })

        assert provider._timeout == 60.0
        assert provider._max_tokens == 500
        assert provider._prompt == "What is on screen?"

    @mock.patch.object(LLMVisionProvider, "health_check", return_value=False)
    def test_initialize_warns_on_failed_health_check(self, _mock_hc) -> None:
        """Initialize warns but does not raise when health check fails."""
        provider = LLMVisionProvider()
        # Should not raise
        provider.initialize({"url": "http://127.0.0.1:5557"})
        assert provider._initialized is True


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestLLMVisionProviderProperties:
    """Tests for get_properties()."""

    def test_properties_local_provider(self) -> None:
        """No api_key → is_local=True."""
        provider = LLMVisionProvider()
        provider._api_key = None
        provider._model = "local-llm"

        props = provider.get_properties()
        assert props.is_local is True
        assert props.name == "local-llm"
        assert props.supports_streaming is False

    def test_properties_cloud_provider(self) -> None:
        """With api_key → is_local=False."""
        provider = LLMVisionProvider()
        provider._api_key = "sk-test"
        provider._model = "gpt-4o"

        props = provider.get_properties()
        assert props.is_local is False
        assert props.name == "gpt-4o"


# ---------------------------------------------------------------------------
# Describe
# ---------------------------------------------------------------------------


class TestLLMVisionProviderDescribe:
    """Tests for describe()."""

    def _make_initialized_provider(
        self, url: str = "http://127.0.0.1:5557", api_key: str = None
    ) -> LLMVisionProvider:
        """Create an initialized provider without hitting the network."""
        provider = LLMVisionProvider()
        provider._url = url
        provider._api_key = api_key
        provider._model = "test-model"
        provider._timeout = 30.0
        provider._max_tokens = 300
        provider._prompt = "Describe what you see."
        provider._initialized = True
        return provider

    def _mock_response(
        self, content: str = "A desktop screenshot.", input_tokens: int = 50, output_tokens: int = 10
    ) -> mock.MagicMock:
        """Create a mock requests.Response with OpenAI-format JSON."""
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
        }
        resp.raise_for_status = mock.MagicMock()
        return resp

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_sends_correct_payload(self, mock_post) -> None:
        """Payload has multimodal content, cache_prompt=False, correct model."""
        mock_post.return_value = self._mock_response()
        provider = self._make_initialized_provider()

        provider.describe(image_base64="abc123")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert payload["model"] == "test-model"
        assert payload["cache_prompt"] is False

        # Check multimodal content
        content = payload["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "abc123" in content[1]["image_url"]["url"]

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_pins_to_slot_1(self, mock_post) -> None:
        """Payload always includes id_slot=1 for vision describe (NANO-087)."""
        mock_post.return_value = self._mock_response()
        provider = self._make_initialized_provider()

        provider.describe(image_base64="abc123")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["id_slot"] == 1

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_returns_vlm_response(self, mock_post) -> None:
        """Mock 200 response returns populated VLMResponse."""
        mock_post.return_value = self._mock_response(
            content="A code editor.", input_tokens=100, output_tokens=20
        )
        provider = self._make_initialized_provider()

        result = provider.describe(image_base64="img_data")

        assert isinstance(result, VLMResponse)
        assert result.description == "A code editor."
        assert result.input_tokens == 100
        assert result.output_tokens == 20
        assert result.latency_ms > 0

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_uses_custom_prompt(self, mock_post) -> None:
        """Custom prompt appears in payload."""
        mock_post.return_value = self._mock_response()
        provider = self._make_initialized_provider()

        provider.describe(image_base64="img", prompt="What app is open?")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        text_content = payload["messages"][0]["content"][0]["text"]
        assert text_content == "What app is open?"

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_includes_auth_header_for_cloud(self, mock_post) -> None:
        """With api_key, Authorization header is sent."""
        mock_post.return_value = self._mock_response()
        provider = self._make_initialized_provider(api_key="sk-test-key")

        provider.describe(image_base64="img")

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers")
        assert headers["Authorization"] == "Bearer sk-test-key"

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_no_auth_header_for_local(self, mock_post) -> None:
        """Without api_key, no Authorization header."""
        mock_post.return_value = self._mock_response()
        provider = self._make_initialized_provider(api_key=None)

        provider.describe(image_base64="img")

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers")
        assert "Authorization" not in headers

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_raises_on_timeout(self, mock_post) -> None:
        """Timeout exception surfaces as TimeoutError."""
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        provider = self._make_initialized_provider()

        with pytest.raises(TimeoutError):
            provider.describe(image_base64="img")

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_raises_on_connection_error(self, mock_post) -> None:
        """Connection failure surfaces as ConnectionError."""
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        provider = self._make_initialized_provider()

        with pytest.raises(ConnectionError):
            provider.describe(image_base64="img")

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_raises_on_http_error(self, mock_post) -> None:
        """HTTP error surfaces as RuntimeError."""
        resp = mock.MagicMock()
        resp.status_code = 500
        resp.json.return_value = {"error": {"message": "Internal error"}}
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp
        )
        mock_post.return_value = resp
        provider = self._make_initialized_provider()

        with pytest.raises(RuntimeError, match="VLM request failed"):
            provider.describe(image_base64="img")

    @mock.patch("spindl.vision.builtin.llm.provider.requests.post")
    def test_describe_raises_on_empty_response(self, mock_post) -> None:
        """Empty choices raises RuntimeError."""
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": []}
        resp.raise_for_status = mock.MagicMock()
        mock_post.return_value = resp
        provider = self._make_initialized_provider()

        with pytest.raises(RuntimeError, match="empty response"):
            provider.describe(image_base64="img")

    def test_describe_not_initialized_raises(self) -> None:
        """Calling describe() without initialize() raises RuntimeError."""
        provider = LLMVisionProvider()

        with pytest.raises(RuntimeError, match="not initialized"):
            provider.describe(image_base64="img")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestLLMVisionProviderHealthCheck:
    """Tests for health_check()."""

    @mock.patch("spindl.vision.builtin.llm.provider.requests.get")
    def test_health_check_local_hits_health_endpoint(self, mock_get) -> None:
        """Local (no api_key): GET /health, returns True on 200."""
        resp = mock.MagicMock()
        resp.status_code = 200
        mock_get.return_value = resp

        provider = LLMVisionProvider()
        provider._url = "http://127.0.0.1:5557"
        provider._api_key = None

        assert provider.health_check() is True
        mock_get.assert_called_once_with(
            "http://127.0.0.1:5557/health", timeout=5.0
        )

    @mock.patch("spindl.vision.builtin.llm.provider.requests.get")
    def test_health_check_local_returns_false_on_error(self, mock_get) -> None:
        """Local: returns False on connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        provider = LLMVisionProvider()
        provider._url = "http://127.0.0.1:5557"
        provider._api_key = None

        assert provider.health_check() is False

    @mock.patch("spindl.vision.builtin.llm.provider.requests.get")
    def test_health_check_cloud_hits_models_endpoint(self, mock_get) -> None:
        """Cloud (has api_key): GET /v1/models with auth, True on 200."""
        resp = mock.MagicMock()
        resp.status_code = 200
        mock_get.return_value = resp

        provider = LLMVisionProvider()
        provider._url = "https://api.openai.com"
        provider._api_key = "sk-test"

        assert provider.health_check() is True
        mock_get.assert_called_once_with(
            "https://api.openai.com/v1/models",
            headers={"Authorization": "Bearer sk-test"},
            timeout=5.0,
        )

    @mock.patch("spindl.vision.builtin.llm.provider.requests.get")
    def test_health_check_cloud_returns_true_on_401(self, mock_get) -> None:
        """Cloud: returns True on 401 (server up, bad key)."""
        resp = mock.MagicMock()
        resp.status_code = 401
        mock_get.return_value = resp

        provider = LLMVisionProvider()
        provider._url = "https://api.openai.com"
        provider._api_key = "sk-bad"

        assert provider.health_check() is True

    @mock.patch("spindl.vision.builtin.llm.provider.requests.get")
    def test_health_check_local_non_200_returns_false(self, mock_get) -> None:
        """Local: non-200 status returns False."""
        resp = mock.MagicMock()
        resp.status_code = 503
        mock_get.return_value = resp

        provider = LLMVisionProvider()
        provider._url = "http://127.0.0.1:5557"
        provider._api_key = None

        assert provider.health_check() is False


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestLLMVisionProviderShutdown:
    """Tests for shutdown()."""

    def test_shutdown_clears_initialized(self) -> None:
        """Shutdown sets _initialized to False."""
        provider = LLMVisionProvider()
        provider._initialized = True

        provider.shutdown()

        assert provider._initialized is False
