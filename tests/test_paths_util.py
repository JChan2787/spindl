"""Tests for spindl.utils.paths utilities."""

from pathlib import Path

import pytest

from spindl.utils.paths import get_project_root, resolve_relative_path


class TestGetProjectRoot:
    """Tests for get_project_root()."""

    def test_returns_path_with_pyproject_toml(self) -> None:
        """Project root contains pyproject.toml."""
        root = get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_returns_absolute_path(self) -> None:
        """Project root is an absolute path."""
        root = get_project_root()
        assert root.is_absolute()

    def test_is_consistent(self) -> None:
        """Repeated calls return the same path (cached)."""
        assert get_project_root() == get_project_root()


class TestResolveRelativePath:
    """Tests for resolve_relative_path()."""

    def test_absolute_path_unchanged(self) -> None:
        """Absolute paths pass through unchanged."""
        abs_path = str(Path(__file__).resolve())
        assert resolve_relative_path(abs_path) == abs_path

    def test_relative_path_resolved_to_project_root(self) -> None:
        """Relative paths resolve against project root."""
        result = resolve_relative_path("tts/models")
        root = get_project_root()
        assert result == str(root / "tts/models")

    def test_dotslash_relative_resolved(self) -> None:
        """Dot-slash relative paths also resolve against project root."""
        result = resolve_relative_path("./tts/models")
        root = get_project_root()
        assert result == str(root / "tts/models")

    @pytest.mark.skipif(
        __import__("sys").platform != "win32",
        reason="Windows-only path resolution",
    )
    def test_windows_absolute_path_unchanged(self) -> None:
        """Windows-style absolute paths pass through unchanged."""
        win_path = "C:/Users/someone/models"
        result = resolve_relative_path(win_path)
        assert result == win_path
