"""Tests for ReflectionSystem — async memory entry generation.

NANO-043 Phase 3. NANO-104 editable prompt + format-agnostic parser.
"""

import os
import shutil
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from spindl.llm.base import LLMResponse
from spindl.llm.plugins.base import PipelineContext
from spindl.llm.plugins.conversation_history import ConversationHistoryManager
from spindl.memory.embedding_client import EmbeddingClient
from spindl.memory.memory_store import MemoryStore
from spindl.memory.reflection import (
    ReflectionSystem,
    format_transcript,
    _parse_memory_entries,
    _looks_complete,
    DEFAULT_REFLECTION_PROMPT,
    DEFAULT_REFLECTION_SYSTEM_MESSAGE,
    DEFAULT_DELIMITER,
    MIN_ENTRY_LENGTH,
)
from spindl.memory.reflection_monitor import ReflectionMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """Mock EmbeddingClient that returns deterministic fake embeddings."""
    client = MagicMock(spec=EmbeddingClient)

    def fake_embed(text: str) -> list[float]:
        h = hash(text) % 10000
        return [(h + i) / 10000.0 for i in range(64)]

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        return [fake_embed(t) for t in texts]

    client.embed.side_effect = fake_embed
    client.embed_batch.side_effect = fake_embed_batch
    return client


@pytest.fixture
def memory_dir() -> str:
    """Temporary directory for ChromaDB persistence."""
    d = tempfile.mkdtemp(prefix="nano_reflection_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(mock_embedding_client: MagicMock, memory_dir: str) -> MemoryStore:
    """MemoryStore with mock embeddings. Dedup disabled — reflection tests
    verify the reflection pipeline, not dedup behavior (see test_dedup.py)."""
    return MemoryStore(
        character_id="testchar",
        memory_dir=memory_dir,
        embedding_client=mock_embedding_client,
        dedup_threshold=None,
        global_memory_dir=os.path.join(memory_dir, "_global"),
    )


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Mock LLMProvider that returns predictable reflections."""
    provider = MagicMock()

    # Default: return 3 clean Q&A pairs
    provider.generate.return_value = LLMResponse(
        content=(
            "Q: What is Alex's profession?\n"
            "A: Alex is a software engineer."
            "{qa}"
            "Q: What does Alex want to build?\n"
            "A: Alex wants to build a side project using Python."
            "{qa}"
            "Q: What is Alex's favorite drink?\n"
            "A: Alex likes mango smoothies."
        ),
        input_tokens=200,
        output_tokens=80,
        finish_reason="stop",
    )
    return provider


@pytest.fixture
def history_manager() -> ConversationHistoryManager:
    """ConversationHistoryManager with no persistence."""
    d = tempfile.mkdtemp(prefix="nano_reflection_hist_")
    manager = ConversationHistoryManager(
        conversations_dir=d,
        resume_session=False,
    )
    manager.ensure_session("testchar")
    yield manager
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def populated_history(history_manager: ConversationHistoryManager) -> ConversationHistoryManager:
    """History manager with 20+ turns for reflection trigger."""
    for i in range(12):
        history_manager.stash_user_input(f"User message {i}")
        history_manager.store_turn(f"Assistant response {i}")
    # 12 pairs = 24 turns
    return history_manager


@pytest.fixture
def reflection_system(
    mock_llm_provider: MagicMock,
    store: MemoryStore,
    history_manager: ConversationHistoryManager,
) -> ReflectionSystem:
    """ReflectionSystem with mock provider and real ChromaDB."""
    return ReflectionSystem(
        llm_provider=mock_llm_provider,
        memory_store=store,
        history_manager=history_manager,
        reflection_interval=20,
        max_tokens=300,
    )


# ---------------------------------------------------------------------------
# format_transcript tests
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    """Tests for format_transcript helper."""

    def test_formats_user_and_assistant(self):
        turns = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = format_transcript(turns)
        assert result == "User: Hello!\nAssistant: Hi there!"

    def test_formats_summary_turns(self):
        turns = [
            {"role": "summary", "content": "They discussed food."},
            {"role": "user", "content": "What about drinks?"},
        ]
        result = format_transcript(turns)
        assert "[Previous summary]:" in result
        assert "User: What about drinks?" in result

    def test_empty_history(self):
        assert format_transcript([]) == ""

    def test_unknown_role(self):
        turns = [{"role": "unknown", "content": "Mystery"}]
        result = format_transcript(turns)
        assert result == ""  # Unknown roles produce no output


# ---------------------------------------------------------------------------
# _looks_complete tests
# ---------------------------------------------------------------------------


class TestLooksComplete:
    """Tests for _looks_complete validator."""

    def test_ends_with_period(self):
        assert _looks_complete("Alex is an engineer.") is True

    def test_ends_with_question_mark(self):
        assert _looks_complete("What is Alex's job?") is True

    def test_ends_with_exclamation(self):
        assert _looks_complete("That's great!") is True

    def test_ends_with_closing_quote(self):
        assert _looks_complete('He said "hello."') is True

    def test_ends_with_closing_paren(self):
        assert _looks_complete("(including Python)") is True

    def test_truncated_mid_word(self):
        assert _looks_complete("Alex likes man") is False

    def test_ends_with_comma(self):
        assert _looks_complete("Alex likes coding,") is False

    def test_empty_string(self):
        assert _looks_complete("") is False

    def test_whitespace_only(self):
        assert _looks_complete("   ") is False


# ---------------------------------------------------------------------------
# _parse_memory_entries tests (NANO-104: format-agnostic)
# ---------------------------------------------------------------------------


class TestParseMemoryEntries:
    """Tests for _parse_memory_entries — format-agnostic parser."""

    def test_three_clean_qa_pairs_backward_compat(self):
        """Default {qa} delimiter + Q&A format still works."""
        response = (
            "Q: What?\nA: Something interesting here."
            "{qa}"
            "Q: Who?\nA: Someone important here."
            "{qa}"
            "Q: Where?\nA: Somewhere far away."
        )
        entries = _parse_memory_entries(response)
        assert len(entries) == 3
        assert "What?" in entries[0]
        assert "Someone important here." in entries[1]
        assert "Somewhere far away." in entries[2]

    def test_plain_text_entries(self):
        """Non-Q&A plain text entries are accepted."""
        response = (
            "User's name is Alex and they are a software engineer."
            "{qa}"
            "Alex prefers Python for side projects and backend work."
            "{qa}"
            "Alex enjoys mango smoothies as their favorite drink."
        )
        entries = _parse_memory_entries(response)
        assert len(entries) == 3
        assert "Alex" in entries[0]
        assert "Python" in entries[1]
        assert "mango" in entries[2]

    def test_custom_delimiter(self):
        """Custom delimiter splits correctly."""
        response = (
            "User lives in Southern California."
            "\n---\n"
            "User is a software architect."
            "\n---\n"
            "User prefers dark mode interfaces."
        )
        entries = _parse_memory_entries(response, delimiter="\n---\n")
        assert len(entries) == 3
        assert "Southern California." in entries[0]

    def test_single_entry_no_delimiter(self):
        """Single block without delimiter is treated as one entry."""
        entries = _parse_memory_entries("User's name is Alex and they like coding.")
        assert len(entries) == 1

    def test_truncated_last_entry_discarded(self):
        """Truncated entry (no sentence-ending punctuation) is discarded."""
        response = (
            "User is a software engineer."
            "{qa}"
            "User likes man"  # truncated
        )
        entries = _parse_memory_entries(response)
        assert len(entries) == 1

    def test_short_entries_discarded(self):
        """Entries shorter than MIN_ENTRY_LENGTH are discarded."""
        response = (
            "User is a software engineer who works on side projects."
            "{qa}"
            "Short."  # 6 chars < MIN_ENTRY_LENGTH
        )
        entries = _parse_memory_entries(response)
        assert len(entries) == 1

    def test_empty_chunks_ignored(self):
        """Empty chunks between delimiters are ignored."""
        response = (
            "{qa}"
            "User is a software engineer."
            "{qa}"
            "{qa}"
        )
        entries = _parse_memory_entries(response)
        assert len(entries) == 1

    def test_empty_response(self):
        assert _parse_memory_entries("") == []
        assert _parse_memory_entries("  ") == []

    def test_none_response(self):
        assert _parse_memory_entries(None) == []

    def test_bullet_list_format(self):
        """Bullet list entries work with appropriate delimiter."""
        response = (
            "- User is a software engineer with 10 years experience."
            "\n\n"
            "- User prefers Python and TypeScript for development."
            "\n\n"
            "- User lives in Southern California area."
        )
        entries = _parse_memory_entries(response, delimiter="\n\n")
        assert len(entries) == 3

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped from entries."""
        response = "  User is an engineer.  {qa}  User likes coding.  "
        entries = _parse_memory_entries(response)
        assert len(entries) == 2
        assert entries[0] == "User is an engineer."
        assert entries[1] == "User likes coding."


# ---------------------------------------------------------------------------
# ReflectionSystem tests
# ---------------------------------------------------------------------------


class TestReflectionSystem:
    """Tests for ReflectionSystem core logic."""

    def test_no_reflection_below_threshold(self, reflection_system):
        """No reflection when turn count is below interval."""
        # 0 turns in history, threshold is 20
        cards = reflection_system.check_and_reflect()
        assert cards == []

    def test_reflection_triggers_at_threshold(
        self, mock_llm_provider, store, populated_history
    ):
        """Reflection triggers when turn count reaches interval."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        # populated_history has 24 turns, threshold is 20
        cards = system.check_and_reflect()
        assert len(cards) == 3
        assert mock_llm_provider.generate.call_count == 1

    def test_flash_cards_stored_in_chromadb(
        self, mock_llm_provider, store, populated_history
    ):
        """Flash cards are stored in the flashcards collection."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        system.check_and_reflect()

        all_cards = store.get_all("flashcards")
        assert len(all_cards) == 3
        # Verify metadata
        for card in all_cards:
            assert card["metadata"]["type"] == "flash_card"
            assert card["metadata"]["source"] == "reflection"
            assert "timestamp" in card["metadata"]

    def test_processed_count_advances(
        self, mock_llm_provider, store, populated_history
    ):
        """Processed turn count advances after reflection."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        assert system.processed_turn_count == 0
        system.check_and_reflect()
        assert system.processed_turn_count == 24  # 12 pairs = 24 turns

    def test_no_double_processing(
        self, mock_llm_provider, store, populated_history
    ):
        """Second check_and_reflect does not re-process same turns."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        system.check_and_reflect()
        cards = system.check_and_reflect()  # Second call
        assert cards == []  # No new turns to process
        assert mock_llm_provider.generate.call_count == 1

    def test_llm_failure_graceful(
        self, store, populated_history
    ):
        """LLM failure doesn't crash; processed count still advances."""
        provider = MagicMock()
        provider.generate.side_effect = ConnectionError("LLM down")

        system = ReflectionSystem(
            llm_provider=provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        cards = system.check_and_reflect()
        assert cards == []
        # Count should still advance so we don't retry the same turns
        assert system.processed_turn_count == 24

    def test_truncated_response_handled(
        self, store, populated_history
    ):
        """Truncated LLM response: last entry discarded, others kept."""
        provider = MagicMock()
        provider.generate.return_value = LLMResponse(
            content=(
                "Q: What?\nA: Something interesting."
                "{qa}"
                "Q: Who?\nA: Some"  # truncated
            ),
            input_tokens=200,
            output_tokens=50,
            finish_reason="length",  # Hit max_tokens
        )

        system = ReflectionSystem(
            llm_provider=provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        cards = system.check_and_reflect()
        assert len(cards) == 1  # Only the first complete entry

    def test_session_id_in_metadata(
        self, mock_llm_provider, store, populated_history
    ):
        """Session ID is included in flash card metadata when provided."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        system._session_id = "testchar_20260207_0930"
        system.check_and_reflect()

        all_cards = store.get_all("flashcards")
        for card in all_cards:
            assert card["metadata"]["session_id"] == "testchar_20260207_0930"

    def test_reflection_prompt_contains_transcript(
        self, mock_llm_provider, store, populated_history
    ):
        """The LLM call includes the conversation transcript."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        system.check_and_reflect()

        call_args = mock_llm_provider.generate.call_args
        messages = call_args.kwargs["messages"]

        # System message is fact extraction, not character card
        assert "fact extraction" in messages[0]["content"].lower()

        # User message contains the transcript
        user_content = messages[1]["content"]
        assert "User message 0" in user_content
        assert "Assistant response 0" in user_content

    def test_reflection_uses_low_temperature(
        self, mock_llm_provider, store, populated_history
    ):
        """Reflection LLM call uses low temperature (0.3)."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        system.check_and_reflect()

        call_args = mock_llm_provider.generate.call_args
        assert call_args.kwargs["temperature"] == 0.3

    def test_empty_llm_response(
        self, store, populated_history
    ):
        """Empty LLM response produces no flash cards."""
        provider = MagicMock()
        provider.generate.return_value = LLMResponse(
            content="",
            input_tokens=200,
            output_tokens=0,
            finish_reason="stop",
        )

        system = ReflectionSystem(
            llm_provider=provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
        )
        cards = system.check_and_reflect()
        assert cards == []

    # --- NANO-104: Custom prompt/system message/delimiter tests ---

    def test_custom_prompt_used(
        self, mock_llm_provider, store, populated_history
    ):
        """Custom reflection prompt is used instead of default."""
        custom_prompt = "Extract the single most important fact.\n\nConversation:\n{transcript}"
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
            reflection_prompt=custom_prompt,
        )
        system.check_and_reflect()

        call_args = mock_llm_provider.generate.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "Extract the single most important fact." in user_content

    def test_custom_system_message_used(
        self, mock_llm_provider, store, populated_history
    ):
        """Custom system message is used instead of default."""
        custom_sys = "You are a memory curator. Be selective."
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
            reflection_system_message=custom_sys,
        )
        system.check_and_reflect()

        call_args = mock_llm_provider.generate.call_args
        sys_content = call_args.kwargs["messages"][0]["content"]
        assert sys_content == custom_sys

    def test_custom_delimiter_used(
        self, store, populated_history
    ):
        """Custom delimiter splits entries correctly."""
        provider = MagicMock()
        provider.generate.return_value = LLMResponse(
            content=(
                "User is a software engineer."
                "\n---\n"
                "User prefers Python for projects."
            ),
            input_tokens=200,
            output_tokens=40,
            finish_reason="stop",
        )

        system = ReflectionSystem(
            llm_provider=provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
            reflection_delimiter="\n---\n",
        )
        cards = system.check_and_reflect()
        assert len(cards) == 2

    def test_none_prompt_uses_default(
        self, mock_llm_provider, store, populated_history
    ):
        """None prompt falls back to DEFAULT_REFLECTION_PROMPT."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
            reflection_prompt=None,
        )
        system.check_and_reflect()

        call_args = mock_llm_provider.generate.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "3 most salient" in user_content

    def test_none_system_message_uses_default(
        self, mock_llm_provider, store, populated_history
    ):
        """None system message falls back to DEFAULT_REFLECTION_SYSTEM_MESSAGE."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=populated_history,
            reflection_interval=20,
            max_tokens=300,
            reflection_system_message=None,
        )
        system.check_and_reflect()

        call_args = mock_llm_provider.generate.call_args
        sys_content = call_args.kwargs["messages"][0]["content"]
        assert "fact extraction" in sys_content.lower()


# ---------------------------------------------------------------------------
# ReflectionSystem threading tests
# ---------------------------------------------------------------------------


class TestReflectionSystemThreading:
    """Tests for ReflectionSystem background thread behavior."""

    def test_start_stop_lifecycle(self, reflection_system):
        """Thread starts and stops cleanly."""
        reflection_system.start(session_id="test_session")
        assert reflection_system._thread is not None
        assert reflection_system._thread.is_alive()

        reflection_system.stop()
        assert reflection_system._thread is None

    def test_notify_wakes_thread(
        self, mock_llm_provider, store, history_manager
    ):
        """notify() triggers reflection check in background thread."""
        system = ReflectionSystem(
            llm_provider=mock_llm_provider,
            memory_store=store,
            history_manager=history_manager,
            reflection_interval=20,
            max_tokens=300,
        )
        # Start with empty history (seeds _processed_turn_count to 0),
        # then add turns so they count as NEW since start().
        system.start()
        for i in range(12):
            history_manager.stash_user_input(f"User message {i}")
            history_manager.store_turn(f"Assistant response {i}")
        system.notify()

        # Give the thread time to process
        time.sleep(1.0)
        system.stop()

        # Should have generated flash cards
        all_cards = store.get_all("flashcards")
        assert len(all_cards) == 3

    def test_double_start_ignored(self, reflection_system):
        """Starting an already-running system is a no-op."""
        reflection_system.start()
        thread1 = reflection_system._thread

        reflection_system.start()  # Should be ignored
        thread2 = reflection_system._thread

        assert thread1 is thread2
        reflection_system.stop()

    def test_daemon_thread(self, reflection_system):
        """Reflection thread is a daemon (won't block process exit)."""
        reflection_system.start()
        assert reflection_system._thread.daemon is True
        reflection_system.stop()


# ---------------------------------------------------------------------------
# ReflectionMonitor tests
# ---------------------------------------------------------------------------


class TestReflectionMonitor:
    """Tests for ReflectionMonitor PostProcessor."""

    def test_name(self, reflection_system):
        monitor = ReflectionMonitor(reflection_system)
        assert monitor.name == "reflection_monitor"

    def test_returns_response_unchanged(self, reflection_system):
        """Monitor doesn't modify the response."""
        monitor = ReflectionMonitor(reflection_system)
        context = PipelineContext(
            user_input="Hello",
            persona={"id": "test", "name": "Test"},
        )
        result = monitor.process(context, "Original response")
        assert result == "Original response"

    def test_notifies_reflection_system(self, reflection_system):
        """Monitor calls notify() on the reflection system."""
        monitor = ReflectionMonitor(reflection_system)
        reflection_system.notify = MagicMock()

        context = PipelineContext(
            user_input="Hello",
            persona={"id": "test", "name": "Test"},
        )
        monitor.process(context, "Response")
        reflection_system.notify.assert_called_once()


# ---------------------------------------------------------------------------
# SummarizationTrigger live_mode tests
# ---------------------------------------------------------------------------


class TestSummarizationLiveMode:
    """Tests for the live_mode flag on SummarizationTrigger."""

    def test_live_mode_skips_summarization(self):
        """When live_mode=True, summarization is completely skipped."""
        from spindl.llm.plugins.summarization import SummarizationTrigger

        provider = MagicMock()
        manager = MagicMock()
        manager.ensure_session = MagicMock()

        trigger = SummarizationTrigger(
            llm_provider=provider,
            manager=manager,
            live_mode=True,
        )

        context = PipelineContext(
            user_input="Hello",
            persona={"id": "test"},
        )
        result = trigger.process(context)
        assert result is context

        # Should NOT have called any token counting
        provider.count_tokens.assert_not_called()

    def test_default_is_not_live_mode(self):
        """Default live_mode is False (backward compatible)."""
        from spindl.llm.plugins.summarization import SummarizationTrigger

        provider = MagicMock()
        manager = MagicMock()

        trigger = SummarizationTrigger(
            llm_provider=provider,
            manager=manager,
        )
        assert trigger._live_mode is False


# ---------------------------------------------------------------------------
# MemoryConfig validation tests (NANO-104)
# ---------------------------------------------------------------------------


class TestMemoryConfigValidation:
    """Tests for MemoryConfig Pydantic validators on NANO-104 fields."""

    def test_reflection_prompt_requires_transcript(self):
        """reflection_prompt without {transcript} raises ValueError."""
        from spindl.orchestrator.config import MemoryConfig

        with pytest.raises(Exception):
            MemoryConfig(reflection_prompt="Extract facts from this conversation.")

    def test_reflection_prompt_with_transcript_valid(self):
        """reflection_prompt with {transcript} is accepted."""
        from spindl.orchestrator.config import MemoryConfig

        config = MemoryConfig(reflection_prompt="Extract facts.\n{transcript}")
        assert "{transcript}" in config.reflection_prompt

    def test_reflection_prompt_none_valid(self):
        """reflection_prompt=None is valid (uses built-in default)."""
        from spindl.orchestrator.config import MemoryConfig

        config = MemoryConfig(reflection_prompt=None)
        assert config.reflection_prompt is None

    def test_reflection_delimiter_empty_rejected(self):
        """Empty reflection_delimiter raises ValueError."""
        from spindl.orchestrator.config import MemoryConfig

        with pytest.raises(Exception):
            MemoryConfig(reflection_delimiter="")

    def test_reflection_delimiter_default(self):
        """Default reflection_delimiter is {qa}."""
        from spindl.orchestrator.config import MemoryConfig

        config = MemoryConfig()
        assert config.reflection_delimiter == "{qa}"

    def test_reflection_system_message_none_valid(self):
        """reflection_system_message=None is valid."""
        from spindl.orchestrator.config import MemoryConfig

        config = MemoryConfig(reflection_system_message=None)
        assert config.reflection_system_message is None
