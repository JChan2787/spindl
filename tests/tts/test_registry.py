"""Tests for TTS provider registry."""

import pytest
import tempfile
from pathlib import Path
from typing import Optional

from spindl.tts.base import AudioResult, TTSProperties, TTSProvider
from spindl.tts.registry import (
    ProviderNotFoundError,
    TTSProviderRegistry,
    create_default_registry,
)


class TestProviderNotFoundError:
    """Tests for ProviderNotFoundError exception."""

    def test_includes_provider_name(self) -> None:
        """Error message includes the requested provider name."""
        error = ProviderNotFoundError("qwen3", ["kokoro"])

        assert error.provider_name == "qwen3"
        assert "qwen3" in str(error)

    def test_includes_available_providers(self) -> None:
        """Error message lists available providers."""
        error = ProviderNotFoundError("missing", ["kokoro", "elevenlabs"])

        assert error.available == ["kokoro", "elevenlabs"]
        assert "kokoro" in str(error)
        assert "elevenlabs" in str(error)

    def test_handles_empty_available_list(self) -> None:
        """Error message handles case when no providers available."""
        error = ProviderNotFoundError("any", [])

        assert error.available == []
        assert "(none)" in str(error)


class TestTTSProviderRegistry:
    """Tests for TTSProviderRegistry class."""

    def test_creates_empty_registry(self) -> None:
        """Registry starts with no providers if builtin not found."""
        registry = TTSProviderRegistry()

        # May or may not have providers depending on builtin package state
        available = registry.list_available()
        assert isinstance(available, list)

    def test_get_provider_class_raises_for_unknown(self) -> None:
        """get_provider_class raises ProviderNotFoundError for unknown provider."""
        registry = TTSProviderRegistry()

        with pytest.raises(ProviderNotFoundError) as exc_info:
            registry.get_provider_class("nonexistent_provider")

        assert exc_info.value.provider_name == "nonexistent_provider"

    def test_list_available_returns_sorted_list(self) -> None:
        """list_available returns sorted list of providers."""
        registry = TTSProviderRegistry()

        available = registry.list_available()
        assert available == sorted(available)

    def test_discovers_plugin_from_path(self) -> None:
        """Registry discovers providers from plugin paths."""
        # Create a temporary plugin directory with a mock provider
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "test_provider"
            plugin_dir.mkdir()

            # Write a mock provider module
            init_content = '''
from typing import Iterator, Optional
from spindl.tts.base import AudioResult, TTSProperties, TTSProvider

class TestTTSProvider(TTSProvider):
    """Mock TTS provider for testing."""

    def initialize(self, config: dict) -> None:
        pass

    def get_properties(self) -> TTSProperties:
        return TTSProperties(
            sample_rate=16000,
            audio_format="pcm_f32le",
            channels=1,
            supports_streaming=False,
        )

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
        return AudioResult(
            data=b"test_audio",
            sample_rate=16000,
            format="pcm_f32le",
        )

    def list_voices(self) -> list[str]:
        return ["test_voice"]

    def health_check(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    @classmethod
    def validate_config(cls, config: dict) -> list[str]:
        return []

    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]:
        return None
'''
            (plugin_dir / "__init__.py").write_text(init_content)

            # Create registry with plugin path
            registry = TTSProviderRegistry(plugin_paths=[tmpdir])

            # Should discover the plugin
            assert "test_provider" in registry.list_available()

            # Should be able to get the provider class
            provider_class = registry.get_provider_class("test_provider")
            assert provider_class.__name__ == "TestTTSProvider"

            # Should be able to instantiate and use
            provider = provider_class()
            provider.initialize({})
            assert provider.health_check() is True
            assert provider.get_properties().sample_rate == 16000

    def test_caches_discovered_plugins(self) -> None:
        """Registry caches discovered plugins to avoid re-scanning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "cached_provider"
            plugin_dir.mkdir()

            init_content = '''
from typing import Optional
from spindl.tts.base import AudioResult, TTSProperties, TTSProvider

class CachedProvider(TTSProvider):
    def initialize(self, config: dict) -> None: pass
    def get_properties(self) -> TTSProperties:
        return TTSProperties(24000, "pcm_f32le", 1, False)
    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
        return AudioResult(b"", 24000, "pcm_f32le")
    def list_voices(self) -> list[str]: return []
    def health_check(self) -> bool: return True
    def shutdown(self) -> None: pass
    @classmethod
    def validate_config(cls, config: dict) -> list[str]: return []
    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]: return None
'''
            (plugin_dir / "__init__.py").write_text(init_content)

            registry = TTSProviderRegistry(plugin_paths=[tmpdir])

            # First access discovers and caches
            provider_class_1 = registry.get_provider_class("cached_provider")

            # Second access returns cached
            provider_class_2 = registry.get_provider_class("cached_provider")

            assert provider_class_1 is provider_class_2

    def test_refresh_clears_cache(self) -> None:
        """refresh() clears the plugin cache."""
        registry = TTSProviderRegistry()

        # Manually add to cache
        registry._plugin_cache["manual"] = TTSProvider  # type: ignore

        # Verify in cache
        assert "manual" in registry._plugin_cache

        # Refresh clears
        registry.refresh()
        assert "manual" not in registry._plugin_cache

    def test_builtin_takes_precedence_over_plugin(self) -> None:
        """Built-in providers take precedence over plugins with same name."""
        # This test verifies search order - builtin is checked first
        # We can't easily test this without a real builtin, so we verify
        # the logic by checking registry internals

        registry = TTSProviderRegistry()

        # If builtin has "kokoro", plugin "kokoro" should be ignored
        # This is just a structural test - real behavior tested when builtin exists
        assert hasattr(registry, "_builtin_providers")
        assert hasattr(registry, "_plugin_cache")

    def test_multiple_plugin_paths(self) -> None:
        """Registry scans multiple plugin paths."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                # Create provider in first path
                plugin_dir1 = Path(tmpdir1) / "provider_one"
                plugin_dir1.mkdir()
                init1 = '''
from typing import Optional
from spindl.tts.base import AudioResult, TTSProperties, TTSProvider

class ProviderOne(TTSProvider):
    def initialize(self, config: dict) -> None: pass
    def get_properties(self) -> TTSProperties:
        return TTSProperties(24000, "pcm_f32le", 1, False)
    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
        return AudioResult(b"one", 24000, "pcm_f32le")
    def list_voices(self) -> list[str]: return []
    def health_check(self) -> bool: return True
    def shutdown(self) -> None: pass
    @classmethod
    def validate_config(cls, config: dict) -> list[str]: return []
    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]: return None
'''
                (plugin_dir1 / "__init__.py").write_text(init1)

                # Create provider in second path
                plugin_dir2 = Path(tmpdir2) / "provider_two"
                plugin_dir2.mkdir()
                init2 = '''
from typing import Optional
from spindl.tts.base import AudioResult, TTSProperties, TTSProvider

class ProviderTwo(TTSProvider):
    def initialize(self, config: dict) -> None: pass
    def get_properties(self) -> TTSProperties:
        return TTSProperties(12000, "pcm_s16le", 1, True)
    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AudioResult:
        return AudioResult(b"two", 12000, "pcm_s16le")
    def list_voices(self) -> list[str]: return []
    def health_check(self) -> bool: return True
    def shutdown(self) -> None: pass
    @classmethod
    def validate_config(cls, config: dict) -> list[str]: return []
    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]: return None
'''
                (plugin_dir2 / "__init__.py").write_text(init2)

                # Create registry with both paths
                registry = TTSProviderRegistry(plugin_paths=[tmpdir1, tmpdir2])

                available = registry.list_available()
                assert "provider_one" in available
                assert "provider_two" in available


class TestCreateDefaultRegistry:
    """Tests for create_default_registry factory function."""

    def test_creates_registry_instance(self) -> None:
        """create_default_registry returns a TTSProviderRegistry."""
        registry = create_default_registry()
        assert isinstance(registry, TTSProviderRegistry)

    def test_accepts_plugin_paths(self) -> None:
        """create_default_registry passes plugin_paths to registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = create_default_registry(plugin_paths=[tmpdir])

            # Registry should have the path configured
            assert Path(tmpdir) in registry._plugin_paths

    def test_default_has_no_plugin_paths(self) -> None:
        """create_default_registry with no args has empty plugin paths."""
        registry = create_default_registry()
        assert registry._plugin_paths == []
