"""Tests for STT provider registry."""

import pytest
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from spindl.stt.base import STTProperties, STTProvider
from spindl.stt.registry import (
    STTProviderNotFoundError,
    STTProviderRegistry,
    create_default_registry,
)


class TestSTTProviderNotFoundError:
    """Tests for STTProviderNotFoundError exception."""

    def test_includes_provider_name(self) -> None:
        """Error message includes the requested provider name."""
        error = STTProviderNotFoundError("whisper", ["parakeet"])

        assert error.provider_name == "whisper"
        assert "whisper" in str(error)

    def test_includes_available_providers(self) -> None:
        """Error message lists available providers."""
        error = STTProviderNotFoundError("missing", ["parakeet", "whisper"])

        assert error.available == ["parakeet", "whisper"]
        assert "parakeet" in str(error)
        assert "whisper" in str(error)

    def test_handles_empty_available_list(self) -> None:
        """Error message handles case when no providers available."""
        error = STTProviderNotFoundError("any", [])

        assert error.available == []
        assert "(none)" in str(error)

    def test_is_exception_subclass(self) -> None:
        """STTProviderNotFoundError is a proper Exception."""
        error = STTProviderNotFoundError("test", [])
        assert isinstance(error, Exception)


class TestSTTProviderRegistry:
    """Tests for STTProviderRegistry class."""

    def test_creates_registry(self) -> None:
        """Registry can be created."""
        registry = STTProviderRegistry()
        available = registry.list_available()
        assert isinstance(available, list)

    def test_discovers_builtin_parakeet(self) -> None:
        """Registry discovers the built-in parakeet provider."""
        registry = STTProviderRegistry()
        available = registry.list_available()
        assert "parakeet" in available

    def test_get_provider_class_returns_parakeet(self) -> None:
        """get_provider_class returns ParakeetSTTProvider for 'parakeet'."""
        registry = STTProviderRegistry()
        provider_class = registry.get_provider_class("parakeet")

        assert issubclass(provider_class, STTProvider)
        assert provider_class.__name__ == "ParakeetSTTProvider"

    def test_get_provider_class_raises_for_unknown(self) -> None:
        """get_provider_class raises STTProviderNotFoundError for unknown provider."""
        registry = STTProviderRegistry()

        with pytest.raises(STTProviderNotFoundError) as exc_info:
            registry.get_provider_class("nonexistent_provider")

        assert exc_info.value.provider_name == "nonexistent_provider"

    def test_list_available_returns_sorted_list(self) -> None:
        """list_available returns sorted list of providers."""
        registry = STTProviderRegistry()
        available = registry.list_available()
        assert available == sorted(available)

    def test_discovers_plugin_from_path(self) -> None:
        """Registry discovers providers from plugin paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "test_stt_provider"
            plugin_dir.mkdir()

            init_content = '''
from typing import Optional
import numpy as np
from spindl.stt.base import STTProperties, STTProvider

class TestSTTPlugin(STTProvider):
    def initialize(self, config: dict) -> None:
        pass

    def get_properties(self) -> STTProperties:
        return STTProperties(sample_rate=16000, audio_format="pcm_f32le", supports_streaming=False)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return "plugin transcription"

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

            registry = STTProviderRegistry(plugin_paths=[tmpdir])
            provider_class = registry.get_provider_class("test_stt_provider")

            assert issubclass(provider_class, STTProvider)

            # Verify it works
            provider = provider_class()
            audio = np.zeros(16000, dtype=np.float32)
            assert provider.transcribe(audio) == "plugin transcription"

    def test_plugin_cached_after_first_discovery(self) -> None:
        """Once a plugin is discovered, it's cached for subsequent lookups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "cached_provider"
            plugin_dir.mkdir()

            init_content = '''
from typing import Optional
import numpy as np
from spindl.stt.base import STTProperties, STTProvider

class CachedProvider(STTProvider):
    def initialize(self, config: dict) -> None: pass
    def get_properties(self) -> STTProperties:
        return STTProperties(16000, "pcm_f32le", False)
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str: return ""
    def health_check(self) -> bool: return True
    def shutdown(self) -> None: pass
    @classmethod
    def validate_config(cls, config: dict) -> list[str]: return []
    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]: return None
'''
            (plugin_dir / "__init__.py").write_text(init_content)

            registry = STTProviderRegistry(plugin_paths=[tmpdir])

            # First lookup - discovers from path
            class1 = registry.get_provider_class("cached_provider")
            # Second lookup - should come from cache
            class2 = registry.get_provider_class("cached_provider")

            assert class1 is class2

    def test_builtin_takes_precedence_over_plugin(self) -> None:
        """Built-in providers take precedence over plugins with same name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin named "parakeet" (same as builtin)
            plugin_dir = Path(tmpdir) / "parakeet"
            plugin_dir.mkdir()

            init_content = '''
from typing import Optional
import numpy as np
from spindl.stt.base import STTProperties, STTProvider

class FakeParakeet(STTProvider):
    def initialize(self, config: dict) -> None: pass
    def get_properties(self) -> STTProperties:
        return STTProperties(16000, "pcm_f32le", False)
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str: return "fake"
    def health_check(self) -> bool: return True
    def shutdown(self) -> None: pass
    @classmethod
    def validate_config(cls, config: dict) -> list[str]: return []
    @classmethod
    def get_server_command(cls, config: dict) -> Optional[str]: return None
'''
            (plugin_dir / "__init__.py").write_text(init_content)

            registry = STTProviderRegistry(plugin_paths=[tmpdir])
            provider_class = registry.get_provider_class("parakeet")

            # Should be the builtin, not the fake
            assert provider_class.__name__ == "ParakeetSTTProvider"

    def test_refresh_clears_and_rediscovers(self) -> None:
        """refresh() clears caches and rediscovers builtin providers."""
        registry = STTProviderRegistry()
        initial = registry.list_available()

        registry.refresh()
        refreshed = registry.list_available()

        assert initial == refreshed

    def test_list_available_includes_plugin_paths(self) -> None:
        """list_available scans plugin paths for undiscovered providers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "undiscovered_provider"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("# empty")

            registry = STTProviderRegistry(plugin_paths=[tmpdir])
            available = registry.list_available()

            assert "undiscovered_provider" in available


class TestCreateDefaultRegistry:
    """Tests for the create_default_registry factory function."""

    def test_creates_registry_without_args(self) -> None:
        """Factory creates a working registry with no arguments."""
        registry = create_default_registry()
        assert isinstance(registry, STTProviderRegistry)
        assert "parakeet" in registry.list_available()

    def test_creates_registry_with_plugin_paths(self) -> None:
        """Factory passes plugin_paths through."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = create_default_registry(plugin_paths=[tmpdir])
            assert isinstance(registry, STTProviderRegistry)
