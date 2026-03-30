"""
Tests for CharacterLoader — VRM path resolution (NANO-097).
"""

import json
from pathlib import Path

import pytest

from spindl.characters.loader import CharacterLoader


@pytest.fixture
def char_dir(tmp_path: Path) -> Path:
    """Create a minimal character directory with card.json."""
    char = tmp_path / "test_char"
    char.mkdir()
    card = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Test",
            "extensions": {
                "spindl": {
                    "id": "test_char",
                }
            },
        },
    }
    (char / "card.json").write_text(json.dumps(card), encoding="utf-8")
    return tmp_path


class TestGetVrmPath:
    """Tests for CharacterLoader.get_vrm_path() (NANO-097)."""

    def test_returns_none_when_no_avatar_vrm_field(self, char_dir: Path):
        """Character without avatar_vrm field returns None."""
        loader = CharacterLoader(str(char_dir))
        assert loader.get_vrm_path("test_char") is None

    def test_returns_none_when_file_missing(self, char_dir: Path):
        """avatar_vrm field set but file doesn't exist returns None."""
        card_path = char_dir / "test_char" / "card.json"
        card = json.loads(card_path.read_text())
        card["data"]["extensions"]["spindl"]["avatar_vrm"] = "model.vrm"
        card_path.write_text(json.dumps(card), encoding="utf-8")

        loader = CharacterLoader(str(char_dir))
        assert loader.get_vrm_path("test_char") is None

    def test_returns_path_when_file_exists(self, char_dir: Path):
        """avatar_vrm field set and file exists returns absolute Path."""
        card_path = char_dir / "test_char" / "card.json"
        card = json.loads(card_path.read_text())
        card["data"]["extensions"]["spindl"]["avatar_vrm"] = "avatar.vrm"
        card_path.write_text(json.dumps(card), encoding="utf-8")

        # Create the VRM file
        vrm_file = char_dir / "test_char" / "avatar.vrm"
        vrm_file.write_bytes(b"fake vrm data")

        loader = CharacterLoader(str(char_dir))
        result = loader.get_vrm_path("test_char")
        assert result is not None
        assert result == vrm_file
        assert result.exists()

    def test_returns_none_for_nonexistent_character(self, char_dir: Path):
        """Nonexistent character returns None without raising."""
        loader = CharacterLoader(str(char_dir))
        assert loader.get_vrm_path("does_not_exist") is None


class TestGetAvatarAnimations:
    """Tests for CharacterLoader.get_avatar_animations() (NANO-098 Session 3)."""

    def test_returns_none_when_no_field(self, char_dir: Path):
        """Character without avatar_animations field returns None."""
        loader = CharacterLoader(str(char_dir))
        assert loader.get_avatar_animations("test_char") is None

    def test_returns_dict_when_set(self, char_dir: Path):
        """avatar_animations field set returns the dict."""
        config = {
            "default": "Breathing Idle",
            "emotions": {"amused": {"threshold": 0.75, "clip": "Happy"}},
        }
        card_path = char_dir / "test_char" / "card.json"
        card = json.loads(card_path.read_text())
        card["data"]["extensions"]["spindl"]["avatar_animations"] = config
        card_path.write_text(json.dumps(card), encoding="utf-8")

        loader = CharacterLoader(str(char_dir))
        result = loader.get_avatar_animations("test_char")
        assert result == config

    def test_returns_none_for_nonexistent_character(self, char_dir: Path):
        """Nonexistent character returns None without raising."""
        loader = CharacterLoader(str(char_dir))
        assert loader.get_avatar_animations("does_not_exist") is None


class TestGetCharacterAnimationsDir:
    """Tests for CharacterLoader.get_character_animations_dir() (NANO-098 Session 3)."""

    def test_returns_animations_subdir(self, char_dir: Path):
        """Returns characters_dir / character_id / animations."""
        loader = CharacterLoader(str(char_dir))
        result = loader.get_character_animations_dir("test_char")
        assert result == char_dir / "test_char" / "animations"
