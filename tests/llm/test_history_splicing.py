"""
NANO-114: Role-array history splicing tests.

Covers:
- Splice path produces [system, ...history, user] array.
- Flattened path unchanged when capability flag absent or False.
- Empty history on splice path leaves [system, user] with orphan header stripped.
- Summary turn placement on splice path.
- Token counting stays honest on splice path (history_formatted metadata populated).
- Multi-turn splice preserves order and role tags.
- Capability default is False (regression guard on LLMProperties).
"""

import pytest

from spindl.llm.base import LLMProperties
from spindl.llm.plugins.base import PipelineContext
from spindl.llm.plugins.conversation_history import (
    ConversationHistoryManager,
    HistoryInjector,
)


# ---------- capability flag defaults ----------


def test_llm_properties_role_history_default_false():
    """Regression guard: any provider not explicitly opting in stays flattened."""
    props = LLMProperties(model_name="test", supports_streaming=True)
    assert props.supports_role_history is False


def test_llm_properties_role_history_explicit_true():
    """Opt-in works when providers want it."""
    props = LLMProperties(
        model_name="test",
        supports_streaming=True,
        supports_role_history=True,
    )
    assert props.supports_role_history is True


# ---------- flatten path (default) ----------


def test_flatten_path_used_when_capability_absent(tmp_path):
    """No metadata flag → legacy flatten path. Messages stay [system, user]."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("first")
    manager.store_turn("first reply")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="second",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n### Conversation\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "second"},
        ],
    )

    result = injector.process(context)

    # Messages stay [system, user]
    assert len(result.messages) == 2
    assert result.messages[0]["role"] == "system"
    assert result.messages[1]["role"] == "user"
    # Flattened content in system
    assert "[user]: first" in result.messages[0]["content"]
    assert "[TestBot]: first reply" in result.messages[0]["content"]
    # Placeholder consumed
    assert "[RECENT_HISTORY]" not in result.messages[0]["content"]


def test_flatten_path_used_when_capability_false(tmp_path):
    """Explicit False on metadata → flatten path. Same behavior as absent."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("first")
    manager.store_turn("first reply")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="second",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "second"},
        ],
    )
    context.metadata["provider_supports_role_history"] = False

    result = injector.process(context)

    assert len(result.messages) == 2
    assert "[user]: first" in result.messages[0]["content"]


# ---------- splice path ----------


def test_splice_path_produces_role_array(tmp_path):
    """Capability True → history spliced between system and user as real roles."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("first")
    manager.store_turn("first reply")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="second",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n### Conversation\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "second"},
        ],
    )
    context.metadata["provider_supports_role_history"] = True

    result = injector.process(context)

    # Array shape: system, user (prior), assistant (prior), user (current)
    assert len(result.messages) == 4
    assert result.messages[0]["role"] == "system"
    assert result.messages[1]["role"] == "user"
    assert result.messages[1]["content"] == "first"
    assert result.messages[2]["role"] == "assistant"
    assert result.messages[2]["content"] == "first reply"
    assert result.messages[3]["role"] == "user"
    assert result.messages[3]["content"] == "second"

    # Placeholder consumed in system
    assert "[RECENT_HISTORY]" not in result.messages[0]["content"]
    # No flattened bracket text in system
    assert "[user]: first" not in result.messages[0]["content"]
    assert "[TestBot]: first reply" not in result.messages[0]["content"]


def test_splice_multi_turn_order_preserved(tmp_path):
    """Three turns splice in correct chronological order."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")

    manager.stash_user_input("q1")
    manager.store_turn("a1")
    manager.stash_user_input("q2")
    manager.store_turn("a2")
    manager.stash_user_input("q3")
    manager.store_turn("a3")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="q4",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "q4"},
        ],
    )
    context.metadata["provider_supports_role_history"] = True

    result = injector.process(context)

    # system + 6 history msgs + current user = 8
    assert len(result.messages) == 8
    expected = [
        ("system", None),
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2"),
        ("user", "q3"),
        ("assistant", "a3"),
        ("user", "q4"),
    ]
    for i, (role, content) in enumerate(expected):
        assert result.messages[i]["role"] == role
        if content is not None:
            assert result.messages[i]["content"] == content


def test_splice_empty_history_strips_orphan_header(tmp_path):
    """Empty history on splice path: messages stay [system, user], orphan header removed."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="first ever",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n### Conversation\n\n[RECENT_HISTORY]\n\n### Input\nRespond."},
            {"role": "user", "content": "first ever"},
        ],
    )
    context.metadata["provider_supports_role_history"] = True

    result = injector.process(context)

    assert len(result.messages) == 2
    assert result.messages[0]["role"] == "system"
    assert result.messages[1]["role"] == "user"
    # Orphan ### Conversation header stripped
    assert "### Conversation" not in result.messages[0]["content"]
    # ### Input header preserved (it has content after it)
    assert "### Input" in result.messages[0]["content"]
    # Placeholder consumed
    assert "[RECENT_HISTORY]" not in result.messages[0]["content"]


def test_splice_preserves_tts_cleaned_content(tmp_path):
    """NANO-109: TTS-cleaned content flows into spliced assistant messages."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("hello")
    # Raw response has action markers; tts_text is the cleaned version
    manager.store_turn("*smiles* Hello there!", tts_text="Hello there!")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="next",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "next"},
        ],
    )
    context.metadata["provider_supports_role_history"] = True

    result = injector.process(context)

    # Assistant message should carry the TTS-cleaned content, not the raw
    assistant_msgs = [m for m in result.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "Hello there!"
    assert "*smiles*" not in assistant_msgs[0]["content"]


# ---------- token accounting honesty ----------


def test_splice_path_populates_history_formatted_for_token_counting(tmp_path):
    """Workshop block token counts stay accurate on splice path."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("first")
    manager.store_turn("first reply")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="second",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "second"},
        ],
    )
    context.metadata["provider_supports_role_history"] = True

    result = injector.process(context)

    # history_formatted should exist and contain the flattened representation
    # (used for Workshop token counting only, NOT injected into system prompt)
    formatted = result.metadata.get("history_formatted", "")
    assert "[user]: first" in formatted
    assert "[TestBot]: first reply" in formatted
    # But the actual system prompt should NOT contain that flattened text
    assert "[user]: first" not in result.messages[0]["content"]


def test_flatten_path_populates_history_formatted(tmp_path):
    """Regression: flatten path token counting behavior unchanged."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("first")
    manager.store_turn("first reply")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="second",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "second"},
        ],
    )
    # No capability flag — flatten path

    result = injector.process(context)

    formatted = result.metadata.get("history_formatted", "")
    assert "[user]: first" in formatted
    assert "[TestBot]: first reply" in formatted


# ---------- summary turn handling ----------


def test_splice_preserves_summary_as_system_message(tmp_path):
    """
    Summary turns (mapped to role=system by get_visible_history) splice in
    as intermediate system messages between the main system prompt and
    first user turn. llama.cpp + jinja handles multi-system arrays.
    """
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")

    # Directly inject a summary turn (simulates SummarizationPlugin output)
    summary_turn = {
        "turn_id": 1,
        "uuid": "summary-uuid",
        "role": "summary",
        "content": "Earlier the user asked about weather and got a forecast.",
        "timestamp": "2026-04-14T00:00:00+00:00",
        "hidden": False,
    }
    manager._history.append(summary_turn)
    manager._next_turn_id = 2

    manager.stash_user_input("q1")
    manager.store_turn("a1")

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="q2",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "q2"},
        ],
    )
    context.metadata["provider_supports_role_history"] = True

    result = injector.process(context)

    # Expect: main system, summary-as-system, user q1, assistant a1, user q2
    assert len(result.messages) == 5
    assert result.messages[0]["role"] == "system"
    assert "You are TestBot" in result.messages[0]["content"]
    assert result.messages[1]["role"] == "system"
    assert "[Previous conversation summary]:" in result.messages[1]["content"]
    assert result.messages[2]["role"] == "user"
    assert result.messages[2]["content"] == "q1"
    assert result.messages[3]["role"] == "assistant"
    assert result.messages[4]["role"] == "user"
    assert result.messages[4]["content"] == "q2"


# ---------- NANO-114 Phase 3: barge-in sanitization (Option D) ----------
#
# Regression: under splice path on strict chat-template models (Gemma-4 via
# llama.cpp + --jinja), a barge-in caused the model to emit structural tokens
# ('### Rules', '### Context', the MODALITY_RULES block, etc.) as response
# content. That polluted content got faithfully stored by
# `amend_last_assistant_content` and then replayed into the next prompt —
# either through the verbatim-quote barge-in injection (now removed) OR through
# the spliced assistant role message. Compounding pollution every turn.
#
# The sanitizer collapses pollution to '[interrupted]' before storage so the
# pollution cannot re-enter prompts via history replay. The pre-sanitization
# form is preserved on `sanitized_from` for inspection.


GEMMA4_POLLUTED_AMEND = (
    "Oh really? That's a bold move, Jubby!\n\n"
    "### Rules\n\n- Just speak naturally, as yourself\n"
    "\n\n### Context\n\nThis is a live voice conversation.\n"
    "Your response will be spoken aloud via TTS.\n\n"
    "The User interrupted you mid-sentence."
)


def test_amend_clean_content_stored_verbatim(tmp_path):
    """Clean amended content is stored unchanged — no false positives."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("Say something bold.")
    manager.store_turn("Oh really? That's a bold move, Jubby! I love that about you.")

    manager.amend_last_assistant_content("Oh really? That's a bold move, Jubby!")

    history = manager.get_history()
    assistant = history[-1]
    assert assistant["content"] == "Oh really? That's a bold move, Jubby!"
    assert "sanitized_from" not in assistant
    # Full generation preserved for display
    assert "display_content" in assistant


def test_amend_polluted_content_collapses_to_placeholder(tmp_path):
    """Structural-token pollution is replaced with [interrupted] on storage."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("Say something bold.")
    manager.store_turn("full original generation")

    manager.amend_last_assistant_content(GEMMA4_POLLUTED_AMEND)

    history = manager.get_history()
    assistant = history[-1]
    # Replayed content is the inert placeholder
    assert assistant["content"] == "[interrupted]"
    # Pre-sanitization form preserved for inspection
    assert assistant["sanitized_from"] == GEMMA4_POLLUTED_AMEND
    # display_content still carries the original pre-barge-in generation
    assert assistant["display_content"] == "full original generation"


def test_amend_polluted_content_prevents_replay_through_splice(tmp_path):
    """End-to-end: pollution cannot re-enter prompts via the splice path."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("Say something bold.")
    manager.store_turn("original clean assistant response")

    # Simulate barge-in storing a polluted truncation
    manager.amend_last_assistant_content(GEMMA4_POLLUTED_AMEND)

    injector = HistoryInjector(manager)
    context = PipelineContext(
        user_input="next turn",
        persona={"id": "testchar", "name": "TestBot"},
        messages=[
            {"role": "system", "content": "You are TestBot.\n\n[RECENT_HISTORY]\n\nRespond."},
            {"role": "user", "content": "next turn"},
        ],
    )
    context.metadata["provider_supports_role_history"] = True

    result = injector.process(context)

    # Find the assistant message in the spliced array
    assistant_msgs = [m for m in result.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "[interrupted]"
    # None of the structural tokens made it into any message content
    combined = "\n".join(m.get("content", "") for m in result.messages)
    assert "### Rules" not in combined
    assert "### Context" not in combined
    assert "spoken aloud via TTS" not in combined


def test_amend_detects_start_of_turn_pollution(tmp_path):
    """<start_of_turn> / <end_of_turn> tokens also trigger sanitization."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("hey")
    manager.store_turn("a1")

    polluted = "OK sure.<end_of_turn>\n<start_of_turn>user\nthanks<end_of_turn>"
    manager.amend_last_assistant_content(polluted)

    assistant = manager.get_history()[-1]
    assert assistant["content"] == "[interrupted]"
    assert assistant["sanitized_from"] == polluted


def test_amend_detects_im_start_pollution(tmp_path):
    """<|im_start|> / <|endoftext|> tokens also trigger sanitization."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("hey")
    manager.store_turn("a1")

    polluted = "Sure thing.<|im_end|>\n<|im_start|>assistant\n"
    manager.amend_last_assistant_content(polluted)

    assistant = manager.get_history()[-1]
    assert assistant["content"] == "[interrupted]"


def test_amend_content_with_hash_but_not_header_is_clean(tmp_path):
    """Inline '#' (e.g. '# 1 priority') without leading '## ' is NOT pollution."""
    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("hey")
    manager.store_turn("a1")

    # Single '#' mid-line or inline hashtags should not trip the sanitizer
    clean = "You're my #1 priority. Obviously."
    manager.amend_last_assistant_content(clean)

    assistant = manager.get_history()[-1]
    assert assistant["content"] == clean
    assert "sanitized_from" not in assistant


def test_amend_polluted_content_persists_sanitized_to_disk(tmp_path):
    """The JSONL file on disk reflects the sanitized content after amend."""
    import json

    manager = ConversationHistoryManager(conversations_dir=str(tmp_path))
    manager.ensure_session("testchar")
    manager.stash_user_input("hey")
    manager.store_turn("original")

    manager.amend_last_assistant_content(GEMMA4_POLLUTED_AMEND)

    # Read raw JSONL
    session_file = manager.session_file
    assert session_file is not None and session_file.exists()
    lines = [json.loads(ln) for ln in session_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Last turn = assistant, post-amend
    assistant = lines[-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "[interrupted]"
    assert assistant["sanitized_from"] == GEMMA4_POLLUTED_AMEND
    assert assistant.get("barge_in_truncated") is True


# ---------- Option A regression coverage: VoiceStateProvider no longer quotes ----------


def test_voice_state_provider_barge_in_does_not_quote_last_message():
    """Option A: barge-in injection is neutral; no verbatim quote of prior output."""
    from spindl.llm.build_context import BuildContext
    from spindl.llm.providers.voice_state_provider import VoiceStateProvider

    context = BuildContext(
        input_content="",
        state_trigger="barge_in",
        last_assistant_message=(
            "### Rules\n- be polite\nOh really? That's a bold move."
        ),
    )
    provider = VoiceStateProvider()
    result = provider.provide(context)

    # Injection exists and conveys the interruption
    assert result is not None
    assert "interrupted" in result.lower()
    # But it does NOT re-inline the polluted prior text
    assert "You were saying" not in result
    assert "### Rules" not in result
    assert "bold move" not in result
