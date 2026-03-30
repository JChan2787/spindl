"""Tests for StimuliConfig parsing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.orchestrator.config import StimuliConfig


class TestStimuliConfig:
    """Tests for StimuliConfig dataclass."""

    def test_defaults(self):
        cfg = StimuliConfig()
        assert cfg.enabled is False
        assert cfg.patience_enabled is False
        assert cfg.patience_seconds == 60.0
        assert "idle" in cfg.patience_prompt.lower()

    def test_from_dict_full(self):
        data = {
            "enabled": True,
            "patience": {
                "enabled": False,
                "seconds": 30.0,
                "prompt": "Say something!",
            },
        }
        cfg = StimuliConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.patience_enabled is False
        assert cfg.patience_seconds == 30.0
        assert cfg.patience_prompt == "Say something!"

    def test_from_dict_minimal(self):
        """Only 'enabled' set, patience uses defaults."""
        data = {"enabled": True}
        cfg = StimuliConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.patience_enabled is False
        assert cfg.patience_seconds == 60.0

    def test_from_dict_empty(self):
        """Empty dict uses all defaults."""
        cfg = StimuliConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.patience_enabled is False

    def test_from_dict_partial_patience(self):
        """Patience section with only seconds."""
        data = {
            "patience": {
                "seconds": 120.0,
            },
        }
        cfg = StimuliConfig.from_dict(data)
        assert cfg.patience_seconds == 120.0
        assert cfg.patience_enabled is False  # Default


class TestStimuliConfigTwitch:
    """Tests for Twitch fields in StimuliConfig (NANO-056b)."""

    def test_twitch_defaults(self):
        cfg = StimuliConfig()
        assert cfg.twitch_enabled is False
        assert cfg.twitch_channel == ""
        assert cfg.twitch_app_id == ""
        assert cfg.twitch_app_secret == ""
        assert cfg.twitch_buffer_size == 10
        assert cfg.twitch_max_message_length == 300
        assert "{messages}" in cfg.twitch_prompt_template

    def test_from_dict_with_twitch(self):
        data = {
            "enabled": True,
            "patience": {"enabled": True, "seconds": 60.0, "prompt": "idle"},
            "twitch": {
                "enabled": True,
                "channel": "testchannel",
                "app_id": "my_app_id",
                "app_secret": "my_secret",
                "buffer_size": 20,
                "max_message_length": 500,
                "prompt_template": "Custom: {messages}",
            },
        }
        cfg = StimuliConfig.from_dict(data)
        assert cfg.twitch_enabled is True
        assert cfg.twitch_channel == "testchannel"
        assert cfg.twitch_app_id == "my_app_id"
        assert cfg.twitch_app_secret == "my_secret"
        assert cfg.twitch_buffer_size == 20
        assert cfg.twitch_max_message_length == 500
        assert cfg.twitch_prompt_template == "Custom: {messages}"

    def test_from_dict_no_twitch_uses_defaults(self):
        data = {"enabled": True}
        cfg = StimuliConfig.from_dict(data)
        assert cfg.twitch_enabled is False
        assert cfg.twitch_channel == ""
        assert cfg.twitch_buffer_size == 10

    def test_from_dict_partial_twitch(self):
        data = {
            "twitch": {
                "enabled": True,
                "channel": "mychan",
            },
        }
        cfg = StimuliConfig.from_dict(data)
        assert cfg.twitch_enabled is True
        assert cfg.twitch_channel == "mychan"
        assert cfg.twitch_app_id == ""  # Default
        assert cfg.twitch_buffer_size == 10  # Default

    def test_buffer_size_validation(self):
        """Buffer size must be between 1 and 50."""
        with pytest.raises(Exception):
            StimuliConfig(twitch_buffer_size=0)
        with pytest.raises(Exception):
            StimuliConfig(twitch_buffer_size=51)

    def test_max_message_length_validation(self):
        """Max message length must be between 50 and 1000."""
        with pytest.raises(Exception):
            StimuliConfig(twitch_max_message_length=10)
        with pytest.raises(Exception):
            StimuliConfig(twitch_max_message_length=1001)
