"""Tests for TwitchModule."""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.twitch import TwitchModule, TwitchMessage
from spindl.stimuli.twitch_selector import TwitchSelector, SelectionResult
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
        """NANO-130: get_stimulus returns single selected message."""
        module = TwitchModule(
            channel="testchannel",
            enabled=True,
            prompt_template="Chat:\n{messages}\nRespond.",
            selection_mode="heuristic",
        )
        module._running = True
        now_ms = int(time.time() * 1000)
        module._buffer.append(TwitchMessage(username="alice", text="what is happening in this game right now?", sent_timestamp_ms=now_ms))
        module._buffer.append(TwitchMessage(username="bob", text="I love this part of the stream so much", sent_timestamp_ms=now_ms))

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.source == StimulusSource.TWITCH
        assert "Chat:" in stimulus.user_input
        assert "Respond." in stimulus.user_input
        assert "twitch_content" not in stimulus.metadata
        assert stimulus.metadata["message_count"] == 1
        assert stimulus.metadata["channel"] == "testchannel"
        assert len(stimulus.metadata["messages"]) == 1

    def test_drains_buffer(self):
        module = TwitchModule(enabled=True, selection_mode="heuristic")
        module._running = True
        now_ms = int(time.time() * 1000)
        module._buffer.append(TwitchMessage(username="user", text="what is your favorite part of the game so far?", sent_timestamp_ms=now_ms))

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert module.buffer_count == 0
        assert module.has_stimulus() is False

    def test_one_shot_behavior(self):
        """get_stimulus() should return None on second call."""
        module = TwitchModule(enabled=True, selection_mode="heuristic")
        module._running = True
        now_ms = int(time.time() * 1000)
        module._buffer.append(TwitchMessage(username="user", text="what do you think about this boss fight?", sent_timestamp_ms=now_ms))

        first = module.get_stimulus()
        second = module.get_stimulus()
        assert first is not None
        assert second is None

    def test_metadata_structure(self):
        module = TwitchModule(channel="ch", enabled=True, selection_mode="heuristic")
        module._running = True
        now_ms = int(time.time() * 1000)
        module._buffer.append(
            TwitchMessage(username="viewer", text="what is the best strategy for this boss?", sent_timestamp_ms=now_ms)
        )

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.metadata["messages"][0]["username"] == "viewer"
        assert stimulus.metadata["messages"][0]["text"] == "what is the best strategy for this boss?"
        assert "twitch_content" not in stimulus.metadata
        assert "selection" in stimulus.metadata

    def test_stimulus_user_input_carries_selected_message(self):
        """NANO-130: user_input carries the single selected message."""
        module = TwitchModule(
            channel="ch",
            enabled=True,
            prompt_template="Chat:\n{messages}\nRespond.",
            selection_mode="heuristic",
        )
        module._running = True
        now_ms = int(time.time() * 1000)
        module._buffer.append(TwitchMessage(username="alice", text="how is the game going today?", sent_timestamp_ms=now_ms))

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "alice: how is the game going today?" in stimulus.user_input
        assert "Chat:" in stimulus.user_input
        assert "Respond." in stimulus.user_input

    def test_char_cap_truncates_with_ellipsis_marker(self):
        """NANO-115: messages longer than char_cap are truncated and marked with '...'."""
        module = TwitchModule(
            channel="ch",
            enabled=True,
            prompt_template="{messages}",
            char_cap=50,
            selection_mode="heuristic",
        )
        module._running = True
        now_ms = int(time.time() * 1000)
        long_text = "what do you think about " + "a" * 80
        module._buffer.append(TwitchMessage(username="spammer", text=long_text, sent_timestamp_ms=now_ms))

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "..." in stimulus.user_input

    def test_missing_messages_placeholder_falls_back_to_default(self):
        """NANO-115: template missing {messages} falls back to default to avoid losing batch."""
        module = TwitchModule(
            channel="ch",
            enabled=True,
            prompt_template="No placeholder here — just instructions.",
            selection_mode="heuristic",
        )
        module._running = True
        now_ms = int(time.time() * 1000)
        module._buffer.append(TwitchMessage(username="alice", text="what do you think about this fight?", sent_timestamp_ms=now_ms))

        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "alice:" in stimulus.user_input


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


class TestTwitchStalenessFilter:
    """Tests for NANO-130 staleness filter."""

    def _make_module(self, max_age: float = 15.0) -> TwitchModule:
        module = TwitchModule(
            enabled=True, selection_mode="heuristic",
            max_message_age_seconds=max_age,
        )
        module._running = True
        return module

    def _fresh_msg(self, username: str = "user", text: str = "what do you think about this part of the game?") -> TwitchMessage:
        return TwitchMessage(
            username=username, text=text,
            sent_timestamp_ms=int(time.time() * 1000),
        )

    def _stale_msg(self, username: str = "old", text: str = "what was happening back when this message was sent?") -> TwitchMessage:
        return TwitchMessage(
            username=username, text=text,
            sent_timestamp_ms=int(time.time() * 1000) - 60_000,
        )

    def test_fresh_messages_pass(self):
        module = self._make_module()
        module._buffer.append(self._fresh_msg())
        assert module.has_stimulus() is True
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.metadata["message_count"] == 1

    def test_all_stale_returns_false(self):
        module = self._make_module(max_age=5.0)
        module._buffer.append(self._stale_msg())
        assert module.has_stimulus() is False

    def test_stale_get_stimulus_returns_none(self):
        module = self._make_module(max_age=5.0)
        module._buffer.append(self._stale_msg())
        assert module.get_stimulus() is None

    def test_mixed_fresh_and_stale(self):
        module = self._make_module(max_age=5.0)
        module._buffer.append(self._stale_msg())
        module._buffer.append(self._fresh_msg(username="newuser", text="what is going on here"))
        assert module.has_stimulus() is True
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.metadata["selection"]["stale_dropped"] == 1
        assert "newuser" in stimulus.user_input

    def test_zero_timestamp_treated_as_fresh(self):
        module = self._make_module(max_age=5.0)
        module._buffer.append(TwitchMessage(username="notimed", text="no timestamp message here", sent_timestamp_ms=0))
        assert module.has_stimulus() is True

    def test_staleness_filter_drains_buffer(self):
        module = self._make_module(max_age=5.0)
        module._buffer.append(self._stale_msg())
        module.get_stimulus()
        assert module.buffer_count == 0

    def test_max_message_age_property(self):
        module = TwitchModule(max_message_age_seconds=30.0)
        assert module.max_message_age_seconds == 30.0
        module.max_message_age_seconds = 0.5
        assert module.max_message_age_seconds == 1.0
        module.max_message_age_seconds = 200.0
        assert module.max_message_age_seconds == 120.0


class TestTwitchSelectionPass:
    """Tests for NANO-130 selection pass integration."""

    def _make_module(self) -> TwitchModule:
        module = TwitchModule(
            enabled=True, selection_mode="heuristic",
        )
        module._running = True
        return module

    def test_single_message_selected(self):
        module = self._make_module()
        module._buffer.append(TwitchMessage(
            username="alice", text="how do you feel about the boss fight?",
            sent_timestamp_ms=int(time.time() * 1000),
        ))
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.metadata["message_count"] == 1
        assert stimulus.metadata["messages"][0]["username"] == "alice"
        assert "selection" in stimulus.metadata
        assert stimulus.metadata["selection"]["mode"] == "heuristic"

    def test_heuristic_rejects_low_quality(self):
        module = self._make_module()
        module._buffer.append(TwitchMessage(
            username="spammer", text="lol",
            sent_timestamp_ms=int(time.time() * 1000),
        ))
        stimulus = module.get_stimulus()
        assert stimulus is None

    def test_heuristic_selects_best_from_multiple(self):
        module = self._make_module()
        now_ms = int(time.time() * 1000)
        module._buffer.append(TwitchMessage(
            username="emote_user", text="PogChamp",
            sent_timestamp_ms=now_ms,
        ))
        module._buffer.append(TwitchMessage(
            username="asker", text="what boss ability did you just dodge there?",
            sent_timestamp_ms=now_ms,
        ))
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert stimulus.metadata["messages"][0]["username"] == "asker"

    def test_selection_metadata_in_stimulus(self):
        module = self._make_module()
        module._buffer.append(TwitchMessage(
            username="viewer", text="this stream is amazing, how long have you been playing?",
            sent_timestamp_ms=int(time.time() * 1000),
        ))
        stimulus = module.get_stimulus()
        assert stimulus is not None
        sel = stimulus.metadata["selection"]
        assert "mode" in sel
        assert "reason" in sel
        assert "candidates" in sel
        assert "stale_dropped" in sel

    def test_single_message_prompt_template(self):
        module = self._make_module()
        module._buffer.append(TwitchMessage(
            username="viewer", text="great stream today really enjoying it",
            sent_timestamp_ms=int(time.time() * 1000),
        ))
        stimulus = module.get_stimulus()
        assert stimulus is not None
        assert "viewer:" in stimulus.user_input
        assert "A viewer just said something" in stimulus.user_input

    def test_selection_mode_property(self):
        module = TwitchModule(selection_mode="heuristic")
        assert module.selection_mode == "heuristic"
        module.selection_mode = "llm"
        assert module.selection_mode == "llm"
        module.selection_mode = "invalid"
        assert module.selection_mode == "llm"


class TestTwitchSelector:
    """Tests for TwitchSelector heuristic scoring."""

    def test_empty_messages(self):
        selector = TwitchSelector()
        result = selector.select([], mode="heuristic")
        assert result.selected_index is None
        assert result.reason == "empty"

    def test_heuristic_selects_question(self):
        messages = [
            {"username": "a", "text": "lol"},
            {"username": "b", "text": "what build are you running for this fight?"},
        ]
        selector = TwitchSelector()
        result = selector.select(messages, mode="heuristic")
        assert result.selected_index == 1
        assert result.mode == "heuristic"

    def test_heuristic_rejects_all_short(self):
        messages = [
            {"username": "a", "text": "hi"},
            {"username": "b", "text": "lol"},
        ]
        selector = TwitchSelector()
        result = selector.select(messages, mode="heuristic")
        assert result.selected_index is None

    def test_heuristic_scores_longer_messages_higher(self):
        messages = [
            {"username": "short", "text": "nice"},
            {"username": "long", "text": "this is a really detailed and thoughtful message about the game"},
        ]
        selector = TwitchSelector()
        result = selector.select(messages, mode="heuristic")
        assert result.selected_index == 1

    def test_heuristic_penalizes_non_alpha(self):
        messages = [
            {"username": "emoji", "text": "🔥🔥🔥🔥🔥"},
            {"username": "text", "text": "I love this part of the game so much"},
        ]
        selector = TwitchSelector()
        result = selector.select(messages, mode="heuristic")
        assert result.selected_index == 1

    def test_llm_fallback_to_heuristic_when_not_configured(self):
        messages = [
            {"username": "viewer", "text": "what is the best strategy for this boss fight?"},
        ]
        selector = TwitchSelector()
        result = selector.select(messages, mode="llm")
        assert result.mode == "heuristic"

    def test_score_message_question_bonus(self):
        assert TwitchSelector._score_message("what do you think?") > TwitchSelector._score_message("I agree")


# ============================================================
# NANO-130 Phase 2: Chat-TTS config round-trip tests
# ============================================================


class TestChatTTSConfig:
    """Tests for chat-TTS config fields on StimuliConfig."""

    def test_defaults(self):
        from spindl.orchestrator.config import StimuliConfig
        cfg = StimuliConfig()
        assert cfg.twitch_chat_tts_enabled is False
        assert cfg.twitch_chat_tts_host == "127.0.0.1"
        assert cfg.twitch_chat_tts_port == 5560
        assert cfg.twitch_chat_tts_device == "cpu"
        assert cfg.twitch_chat_tts_voice == "af_sarah"
        assert cfg.twitch_chat_tts_speed == 1.1
        assert cfg.twitch_chat_tts_format == "{username} says: {message}"
        assert cfg.twitch_chat_tts_max_length == 100

    def test_from_dict_parses_chat_tts(self):
        from spindl.orchestrator.config import StimuliConfig
        cfg = StimuliConfig.from_dict({
            "twitch": {
                "chat_tts": {
                    "enabled": True,
                    "host": "192.168.1.10",
                    "port": 5570,
                    "device": "cuda:0",
                    "voice": "af_bella",
                    "speed": 1.3,
                    "format": "{username}: {message}",
                    "max_length": 200,
                }
            }
        })
        assert cfg.twitch_chat_tts_enabled is True
        assert cfg.twitch_chat_tts_host == "192.168.1.10"
        assert cfg.twitch_chat_tts_port == 5570
        assert cfg.twitch_chat_tts_device == "cuda:0"
        assert cfg.twitch_chat_tts_voice == "af_bella"
        assert cfg.twitch_chat_tts_speed == 1.3
        assert cfg.twitch_chat_tts_format == "{username}: {message}"
        assert cfg.twitch_chat_tts_max_length == 200

    def test_from_dict_missing_chat_tts_uses_defaults(self):
        from spindl.orchestrator.config import StimuliConfig
        cfg = StimuliConfig.from_dict({"twitch": {}})
        assert cfg.twitch_chat_tts_enabled is False
        assert cfg.twitch_chat_tts_port == 5560
        assert cfg.twitch_chat_tts_voice == "af_sarah"

    def test_chat_tts_port_validation(self):
        from spindl.orchestrator.config import StimuliConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StimuliConfig(twitch_chat_tts_port=0)
        with pytest.raises(ValidationError):
            StimuliConfig(twitch_chat_tts_port=70000)

    def test_chat_tts_speed_clamped(self):
        from spindl.orchestrator.config import StimuliConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StimuliConfig(twitch_chat_tts_speed=0.1)
        with pytest.raises(ValidationError):
            StimuliConfig(twitch_chat_tts_speed=3.0)

    def test_chat_tts_max_length_clamped(self):
        from spindl.orchestrator.config import StimuliConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StimuliConfig(twitch_chat_tts_max_length=5)
        with pytest.raises(ValidationError):
            StimuliConfig(twitch_chat_tts_max_length=1000)


class TestChatTTSSynthesisHelper:
    """Tests for OrchestratorCallbacks._synthesize_chat_tts."""

    def _make_callbacks(self):
        from spindl.orchestrator.callbacks import OrchestratorCallbacks
        cb = OrchestratorCallbacks(
            stt_client=None,
            tts_provider=None,
            llm_pipeline=MagicMock(),
            persona={},
        )
        return cb

    def test_returns_none_when_no_client(self):
        cb = self._make_callbacks()
        assert cb._synthesize_chat_tts("user", "hello") is None

    def test_returns_none_when_server_unavailable(self):
        cb = self._make_callbacks()
        mock_client = MagicMock()
        mock_client.is_server_available.return_value = False
        cb._chat_tts_client = mock_client
        assert cb._synthesize_chat_tts("user", "hello") is None

    def test_format_string_applied(self):
        import numpy as np
        cb = self._make_callbacks()
        mock_client = MagicMock()
        mock_client.is_server_available.return_value = True
        mock_client.synthesize.return_value = np.zeros(2400, dtype=np.float32)
        cb._chat_tts_client = mock_client
        cb._chat_tts_format = "{username} said: {message}"
        cb._chat_tts_voice = "af_test"
        cb._chat_tts_speed = 1.0
        cb._chat_tts_max_length = 100

        result = cb._synthesize_chat_tts("viewer1", "nice stream")
        assert result is not None
        mock_client.synthesize.assert_called_once()
        call_args = mock_client.synthesize.call_args
        assert call_args.kwargs["text"] == "viewer1 said: nice stream"

    def test_message_truncated_at_max_length(self):
        import numpy as np
        cb = self._make_callbacks()
        mock_client = MagicMock()
        mock_client.is_server_available.return_value = True
        mock_client.synthesize.return_value = np.zeros(2400, dtype=np.float32)
        cb._chat_tts_client = mock_client
        cb._chat_tts_format = "{username}: {message}"
        cb._chat_tts_voice = "af_test"
        cb._chat_tts_speed = 1.0
        cb._chat_tts_max_length = 10

        long_message = "a" * 200
        cb._synthesize_chat_tts("v", long_message)
        call_args = mock_client.synthesize.call_args
        assert call_args.kwargs["text"] == "v: " + "a" * 10

    def test_synthesis_exception_returns_none(self):
        cb = self._make_callbacks()
        mock_client = MagicMock()
        mock_client.is_server_available.return_value = True
        mock_client.synthesize.side_effect = ConnectionError("timeout")
        cb._chat_tts_client = mock_client
        cb._chat_tts_format = "{username}: {message}"
        cb._chat_tts_voice = "af_test"
        cb._chat_tts_speed = 1.0
        cb._chat_tts_max_length = 100

        assert cb._synthesize_chat_tts("user", "hello") is None

    def test_reconnect_creates_client(self):
        cb = self._make_callbacks()
        assert cb._chat_tts_client is None

        cfg = MagicMock()
        cfg.twitch_chat_tts_host = "127.0.0.1"
        cfg.twitch_chat_tts_port = 5560
        cfg.twitch_chat_tts_voice = "af_sarah"
        cfg.twitch_chat_tts_speed = 1.1
        cfg.twitch_chat_tts_format = "{username} says: {message}"
        cfg.twitch_chat_tts_max_length = 100

        cb._reconnect_chat_tts_client(cfg)
        assert cb._chat_tts_client is not None
        assert cb._chat_tts_client.host == "127.0.0.1"
        assert cb._chat_tts_client.port == 5560
        assert cb._chat_tts_voice == "af_sarah"
        assert cb._chat_tts_speed == 1.1
