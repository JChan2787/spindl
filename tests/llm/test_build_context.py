"""Tests for BuildContext and InputModality."""

import pytest

from spindl.llm.build_context import BuildContext, InputModality, Message


class TestInputModality:
    """Tests for InputModality enum."""

    def test_voice_modality(self) -> None:
        """VOICE modality has correct value."""
        assert InputModality.VOICE.value == "voice"

    def test_text_modality(self) -> None:
        """TEXT modality has correct value."""
        assert InputModality.TEXT.value == "text"

    def test_modalities_are_distinct(self) -> None:
        """All modalities have distinct values."""
        values = [m.value for m in InputModality]
        assert len(values) == len(set(values))


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self) -> None:
        """Message can be created with required fields."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is None

    def test_message_with_timestamp(self) -> None:
        """Message can include optional timestamp."""
        msg = Message(role="assistant", content="Hi there", timestamp=1234567890.0)
        assert msg.timestamp == 1234567890.0


class TestBuildContext:
    """Tests for BuildContext dataclass."""

    def test_minimal_creation(self) -> None:
        """BuildContext can be created with just input_content."""
        ctx = BuildContext(input_content="Hello")
        assert ctx.input_content == "Hello"
        assert ctx.input_modality == InputModality.TEXT  # Default
        assert ctx.input_metadata == {}
        assert ctx.conversation_state is None
        assert ctx.state_trigger is None
        assert ctx.persona == {}
        assert ctx.config == {}
        assert ctx.recent_messages == []
        assert ctx.summary is None

    def test_full_creation(self) -> None:
        """BuildContext can be created with all fields populated."""
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
        ctx = BuildContext(
            input_content="What's up?",
            input_modality=InputModality.VOICE,
            input_metadata={"trigger": "vad_speech_start"},
            conversation_state=None,  # Would be AgentState in real use
            state_trigger="barge_in",
            persona={"name": "Spindle", "personality": "Helpful"},
            config={"max_tokens": 256},
            recent_messages=messages,
            summary="Previous context about testing.",
        )

        assert ctx.input_content == "What's up?"
        assert ctx.input_modality == InputModality.VOICE
        assert ctx.input_metadata["trigger"] == "vad_speech_start"
        assert ctx.state_trigger == "barge_in"
        assert ctx.persona["name"] == "Spindle"
        assert ctx.config["max_tokens"] == 256
        assert len(ctx.recent_messages) == 2
        assert ctx.summary == "Previous context about testing."

    def test_with_updates_creates_new_instance(self) -> None:
        """with_updates creates a new BuildContext, not mutating original."""
        original = BuildContext(
            input_content="Original",
            input_modality=InputModality.TEXT,
        )

        updated = original.with_updates(
            input_content="Updated",
            summary="New summary",
        )

        # Original unchanged
        assert original.input_content == "Original"
        assert original.summary is None

        # New instance has updates
        assert updated.input_content == "Updated"
        assert updated.summary == "New summary"
        # Unchanged fields preserved
        assert updated.input_modality == InputModality.TEXT

    def test_with_updates_preserves_complex_fields(self) -> None:
        """with_updates preserves list/dict fields correctly."""
        messages = [Message(role="user", content="Test")]
        original = BuildContext(
            input_content="Test",
            recent_messages=messages,
            persona={"name": "Test"},
        )

        updated = original.with_updates(summary="Added summary")

        # Lists/dicts should be preserved (note: shallow copy from asdict)
        assert len(updated.recent_messages) == 1
        assert updated.persona["name"] == "Test"


class TestBuildContextWithAgentState:
    """Tests for BuildContext interaction with AgentState."""

    def test_can_accept_agent_state(self) -> None:
        """BuildContext can store AgentState from state machine."""
        # Import here to test the integration
        from spindl.core.state_machine import AgentState

        ctx = BuildContext(
            input_content="Hello",
            conversation_state=AgentState.USER_SPEAKING,
            state_trigger="vad_speech_start",
        )

        assert ctx.conversation_state == AgentState.USER_SPEAKING
        assert ctx.state_trigger == "vad_speech_start"
