"""Tests for TwitchModule."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.twitch import TwitchModule, TwitchMessage
from spindl.stimuli.models import StimulusSource


class TestTwitchModuleProperties:
    """Tests for TwitchModule basic properties."""

    def test_defaults(self):
        module = TwitchModule()
        assert module.name == "twitch"
        assert module.priority == 50
        assert module.enabled is False
        assert module.channel == ""
        assert module.app_id == ""
        assert module.app_secret == ""
        assert module.buffer_size == 10
        assert module.max_message_length == 300
        assert module.connected is False
        assert module.buffer_count == 0
        assert module.recent_messages == []

    def test_custom_config(self):
        module = TwitchModule(
            channel="testchannel",
            app_id="my_app_id",
            app_secret="my_secret",
            buffer_size=20,
            max_message_length=500,
            prompt_template="Custom: {messages}",
            enabled=True,
        )
        assert module.channel == "testchannel"
        assert module.app_id == "my_app_id"
        assert module.app_secret == "my_secret"
        assert module.buffer_size == 20
        assert module.max_message_length == 500
        assert module.prompt_template == "Custom: {messages}"
        assert module.enabled is True

    def test_enable_disable(self):
        module = TwitchModule()
        module.enabled = True
        assert module.enabled is True
        module.enabled = False
        assert module.enabled is False

    def test_disable_clears_buffer(self):
        module = TwitchModule(enabled=True)
        module._running = True
        module._buffer.append(TwitchMessage(username="user1", text="hello"))
        assert module.buffer_count == 1
        module.enabled = False
        assert module.buffer_count == 0

    def test_channel_setter(self):
        module = TwitchModule()
        module.channel = "newchannel"
        assert module.channel == "newchannel"

    def test_app_id_setter(self):
        module = TwitchModule()
        module.app_id = "new_id"
        assert module.app_id == "new_id"

    def test_app_secret_setter(self):
        module = TwitchModule()
        module.app_secret = "new_secret"
        assert module.app_secret == "new_secret"

    def test_buffer_size_setter(self):
        module = TwitchModule()
        module.buffer_size = 25
        assert module.buffer_size == 25

    def test_buffer_size_minimum_clamp(self):
        module = TwitchModule(buffer_size=0)
        assert module.buffer_size == 1

    def test_buffer_size_setter_preserves_messages(self):
        module = TwitchModule(buffer_size=5)
        for i in range(3):
            module._buffer.append(TwitchMessage(username=f"user{i}", text=f"msg{i}"))
        module.buffer_size = 10
        assert module.buffer_count == 3
        assert module.buffer_size == 10

    def test_max_message_length_setter(self):
        module = TwitchModule()
        module.max_message_length = 500
        assert module.max_message_length == 500

    def test_prompt_template_setter(self):
        module = TwitchModule()
        module.prompt_template = "New template: {messages}"
        assert module.prompt_template == "New template: {messages}"

    def test_health_check_disconnected(self):
        module = TwitchModule()
        assert module.health_check() is False

    def test_health_check_connected(self):
        module = TwitchModule()
        module._connected = True
        assert module.health_check() is True

    def test_recent_messages_formatting(self):
        module = TwitchModule()
        module._buffer.append(TwitchMessage(username="alice", text="hello world"))
        module._buffer.append(TwitchMessage(username="bob", text="hi there"))
        assert module.recent_messages == ["alice: hello world", "bob: hi there"]


class TestTwitchHasStimulus:
    """Tests for has_stimulus() logic."""

    def test_no_stimulus_when_empty(self):
        module = TwitchModule(enabled=True)
        module._running = True
        assert module.has_stimulus() is False

    def test_no_stimulus_when_disabled(self):
        module = TwitchModule(enabled=False)
        module._running = True
        module._buffer.append(TwitchMessage(username="user", text="msg"))
        assert module.has_stimulus() is False

    def test_no_stimulus_when_not_running(self):
        module = TwitchModule(enabled=True)
        module._buffer.append(TwitchMessage(username="user", text="msg"))
        assert module.has_stimulus() is False

    def test_has_stimulus_with_messages(self):
        module = TwitchModule(enabled=True)
        module._running = True
        module._buffer.append(TwitchMessage(username="user", text="msg"))
        assert module.has_stimulus() is True


class TestTwitchGetStimulus:
    """Tests for get_stimulus() behavior."""

    def test_returns_none_when_empty(self):
        module = TwitchModule(enabled=True)
        module._running = True
        assert module.get_stimulus() is None

    def test_returns_stimulus_data(self):
        """NANO-115: user_input now carries the directive + fenced batch directly."""
        module = TwitchModule(
            channel="testchannel",
            enabled=True,
            prompt_template="Chat:\n{messages}\nRespond.",
        )
        module._running = True
        module._buffer.append(TwitchMessage(username="alice", text="hello"))
        module._buffer.append(TwitchMessage(username="bob", text="world"))

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.source == StimulusSource.TWITCH
        assert "alice: hello" in stimulus.user_input
        assert "bob: world" in stimulus.user_input
        assert "Chat:" in stimulus.user_input
        assert "Respond." in stimulus.user_input
        assert "twitch_content" not in stimulus.metadata
        assert stimulus.metadata["message_count"] == 2
        assert stimulus.metadata["channel"] == "testchannel"
        assert len(stimulus.metadata["messages"]) == 2

    def test_drains_buffer(self):
        module = TwitchModule(enabled=True)
        module._running = True
        module._buffer.append(TwitchMessage(username="user", text="msg"))

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert module.buffer_count == 0
        assert module.has_stimulus() is False

    def test_one_shot_behavior(self):
        """get_stimulus() should return None on second call."""
        module = TwitchModule(enabled=True)
        module._running = True
        module._buffer.append(TwitchMessage(username="user", text="msg"))

        first = module.get_stimulus()
        second = module.get_stimulus()
        assert first is not None
        assert second is None

    def test_metadata_structure(self):
        module = TwitchModule(channel="ch", enabled=True)
        module._running = True
        module._buffer.append(
            TwitchMessage(username="viewer", text="hey", sent_timestamp_ms=1234567890000)
        )

        stimulus = module.get_stimulus()
        assert stimulus.metadata["messages"][0]["username"] == "viewer"
        assert stimulus.metadata["messages"][0]["text"] == "hey"
        assert stimulus.metadata["messages"][0]["sent_timestamp_ms"] == 1234567890000
        assert "twitch_content" not in stimulus.metadata

    def test_stimulus_user_input_carries_batch(self):
        """NANO-115: user_input IS the directive + fenced batch (not a short trigger)."""
        module = TwitchModule(
            channel="ch",
            enabled=True,
            prompt_template="Chat:\n{messages}\nRespond.",
        )
        module._running = True
        module._buffer.append(TwitchMessage(username="alice", text="hello"))

        stimulus = module.get_stimulus()
        assert "alice: hello" in stimulus.user_input
        assert "Chat:" in stimulus.user_input
        assert "Respond." in stimulus.user_input

    def test_char_cap_truncates_with_ellipsis_marker(self):
        """NANO-115: messages longer than char_cap are truncated and marked with '...'."""
        module = TwitchModule(
            channel="ch",
            enabled=True,
            prompt_template="{messages}",
            char_cap=50,
        )
        module._running = True
        long_text = "a" * 80
        module._buffer.append(TwitchMessage(username="spammer", text=long_text))

        stimulus = module.get_stimulus()
        assert "..." in stimulus.user_input
        assert ("a" * 80) not in stimulus.user_input
        assert ("a" * 50 + "...") in stimulus.user_input

    def test_missing_messages_placeholder_falls_back_to_default(self):
        """NANO-115: template missing {messages} falls back to default to avoid losing batch."""
        module = TwitchModule(
            channel="ch",
            enabled=True,
            prompt_template="No placeholder here — just instructions.",
        )
        module._running = True
        module._buffer.append(TwitchMessage(username="alice", text="hello"))

        stimulus = module.get_stimulus()
        assert "alice: hello" in stimulus.user_input


class TestTwitchShouldAccept:
    """Tests for message filtering."""

    def test_accepts_normal_message(self):
        module = TwitchModule(max_message_length=300)
        assert module._should_accept("user", "hello world") is True

    def test_rejects_too_long(self):
        module = TwitchModule(max_message_length=10)
        assert module._should_accept("user", "this is a very long message") is False

    def test_rejects_empty(self):
        module = TwitchModule()
        assert module._should_accept("user", "") is False

    def test_rejects_whitespace_only(self):
        module = TwitchModule()
        assert module._should_accept("user", "   ") is False

    def test_accepts_at_exact_limit(self):
        module = TwitchModule(max_message_length=5)
        assert module._should_accept("user", "hello") is True

    def test_rejects_one_over_limit(self):
        module = TwitchModule(max_message_length=5)
        assert module._should_accept("user", "helloo") is False


class TestTwitchBuffer:
    """Tests for buffer behavior."""

    def test_buffer_maxlen(self):
        module = TwitchModule(buffer_size=3)
        for i in range(5):
            module._buffer.append(TwitchMessage(username=f"user{i}", text=f"msg{i}"))
        assert module.buffer_count == 3
        # Oldest messages should be dropped
        usernames = [m.username for m in module._buffer]
        assert usernames == ["user2", "user3", "user4"]

    def test_buffer_clear_on_stop(self):
        module = TwitchModule(channel="ch", app_id="id", app_secret="secret", enabled=True)
        module._running = True
        module._buffer.append(TwitchMessage(username="user", text="msg"))
        module.stop()
        assert module.buffer_count == 0


class TestTwitchLifecycle:
    """Tests for start/stop lifecycle."""

    def test_start_without_config_does_nothing(self):
        """Start without channel/app_id should not start thread."""
        module = TwitchModule()
        module.start()
        assert module._running is False
        assert module._thread is None

    def test_start_without_app_id_does_nothing(self):
        module = TwitchModule(channel="ch")
        module.start()
        assert module._running is False

    def test_start_without_channel_does_nothing(self):
        module = TwitchModule(app_id="id", app_secret="secret")
        module.start()
        assert module._running is False

    @patch("spindl.stimuli.twitch.threading.Thread")
    def test_start_with_config_creates_thread(self, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        module = TwitchModule(
            channel="ch",
            app_id="id",
            app_secret="secret",
            enabled=True,
        )
        module.start()
        assert module._running is True
        mock_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once()

    @patch("spindl.stimuli.twitch.threading.Thread")
    def test_start_idempotent(self, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        module = TwitchModule(
            channel="ch", app_id="id", app_secret="secret"
        )
        module.start()
        module.start()  # Second call should be no-op
        assert mock_thread_cls.call_count == 1

    @patch("spindl.stimuli.twitch.threading.Thread")
    @patch.dict("os.environ", {"TWITCH_APP_ID": "env_id", "TWITCH_APP_SECRET": "env_secret"})
    def test_start_with_env_vars_only(self, mock_thread_cls):
        """Module starts when app_id/app_secret come from env vars."""
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        module = TwitchModule(channel="ch", enabled=True)
        # Config fields are empty, but env vars are set
        assert module.app_id == ""
        assert module.resolved_app_id == "env_id"
        assert module.resolved_app_secret == "env_secret"
        module.start()
        assert module._running is True
        mock_thread.start.assert_called_once()

    def test_stop_when_not_running(self):
        """Stop when not running should be no-op."""
        module = TwitchModule()
        module.stop()  # Should not raise
        assert module._running is False


class TestTwitchEnvVarFallback:
    """Tests for env var fallback on auth credentials."""

    def test_resolved_app_id_prefers_config(self):
        module = TwitchModule(app_id="config_id")
        assert module.resolved_app_id == "config_id"

    def test_resolved_app_secret_prefers_config(self):
        module = TwitchModule(app_secret="config_secret")
        assert module.resolved_app_secret == "config_secret"

    @patch.dict("os.environ", {"TWITCH_APP_ID": "env_id"})
    def test_resolved_app_id_falls_back_to_env(self):
        module = TwitchModule()
        assert module.resolved_app_id == "env_id"

    @patch.dict("os.environ", {"TWITCH_APP_SECRET": "env_secret"})
    def test_resolved_app_secret_falls_back_to_env(self):
        module = TwitchModule()
        assert module.resolved_app_secret == "env_secret"

    @patch.dict("os.environ", {}, clear=True)
    def test_resolved_empty_when_no_config_no_env(self):
        module = TwitchModule()
        assert module.resolved_app_id == ""
        assert module.resolved_app_secret == ""

    @patch.dict("os.environ", {"TWITCH_APP_ID": "env_id"})
    def test_config_takes_precedence_over_env(self):
        module = TwitchModule(app_id="config_id")
        assert module.resolved_app_id == "config_id"
