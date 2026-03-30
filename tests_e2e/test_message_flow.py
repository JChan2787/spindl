"""
Message Flow Tests - Core text → LLM → response pipeline.

NANO-029 Phase 2: Verifies that:
1. Text input via send_message() produces LLM response
2. Response triggers TTS audio generation
3. Events are emitted in correct sequence
"""

import pytest

from .helpers.assertions import (
    assert_event_received,
    assert_response_exists,
    assert_transcription_exists,
)


class TestMessageFlow:
    """Core message flow tests with real LLM/TTS services."""

    @pytest.mark.asyncio
    async def test_text_produces_response(self, e2e_harness, response_timeout):
        """
        Send 'hello' → verify response event fires with non-empty content.

        This is the fundamental E2E test: does the full pipeline work?

        Note: orchestrator_ready is already waited for by the e2e_harness fixture
        during auto_start_services (default=True).
        """
        # Clear prior events from harness setup
        e2e_harness.clear_events()

        # Send greeting
        result = await e2e_harness.send_message("Hello, how are you?")
        assert result["success"] is True

        # Wait for response event
        response = await assert_event_received(
            e2e_harness,
            "response",
            timeout=response_timeout,
            message="Expected response event after send_message",
        )

        # Verify non-empty response
        assert_response_exists(response)

    @pytest.mark.asyncio
    async def test_transcription_displayed(self, e2e_harness, response_timeout):
        """
        Send text → verify transcription event contains the sent text.
        """
        e2e_harness.clear_events()

        test_message = "This is a test message for transcription."
        await e2e_harness.send_message(test_message)

        # Wait for transcription event
        transcription = await assert_event_received(
            e2e_harness,
            "transcription",
            timeout=5.0,
            message="Expected transcription event after send_message",
        )

        # Verify text matches
        assert transcription["text"] == test_message
        assert_transcription_exists(transcription)

    @pytest.mark.asyncio
    async def test_tts_audio_generated(self, e2e_harness, response_timeout):
        """
        Send text → verify tts_status event fires (indicating TTS started).
        """
        e2e_harness.clear_events()

        await e2e_harness.send_message("Tell me a short joke.")

        # Wait for TTS status event
        tts_event = await assert_event_received(
            e2e_harness,
            "tts_status",
            timeout=response_timeout,
            predicate=lambda e: e.get("status") in ("started", "playing"),
            message="Expected tts_status event after LLM response",
        )

        # TTS should have started
        assert tts_event["status"] in ("started", "playing")

    @pytest.mark.asyncio
    async def test_token_usage_emitted(self, e2e_harness, response_timeout):
        """
        Verify token_usage event is emitted after LLM response.
        """
        e2e_harness.clear_events()

        await e2e_harness.send_message("What is 2 + 2?")

        # Wait for token usage event
        usage = await assert_event_received(
            e2e_harness,
            "token_usage",
            timeout=response_timeout,
            message="Expected token_usage event after LLM response",
        )

        # Verify structure (event uses 'prompt'/'completion' not 'prompt_tokens'/'completion_tokens')
        assert "prompt" in usage
        assert "completion" in usage
        assert usage["prompt"] > 0
        assert usage["completion"] > 0

    @pytest.mark.asyncio
    async def test_multiple_turns(self, e2e_harness, response_timeout):
        """
        Verify multiple conversation turns work correctly.
        """
        messages = [
            "Hello!",
            "What's your name?",
            "Thanks for answering.",
        ]

        for msg in messages:
            e2e_harness.clear_events()
            await e2e_harness.send_message(msg)

            # Each should produce a response
            response = await assert_event_received(
                e2e_harness,
                "response",
                timeout=response_timeout,
                message=f"Expected response for: {msg}",
            )
            assert_response_exists(response)


class TestEventSequence:
    """Tests for correct event ordering."""

    @pytest.mark.asyncio
    async def test_event_order(self, e2e_harness, response_timeout):
        """
        Verify events fire in expected order:
        transcription → response → tts_status
        """
        e2e_harness.clear_events()

        await e2e_harness.send_message("Hi there!")

        # Collect events in order (with timeout for each)
        events = []

        try:
            transcription = await e2e_harness.wait_for_event("transcription", timeout=5.0)
            events.append(("transcription", transcription))
        except TimeoutError:
            pass

        try:
            response = await e2e_harness.wait_for_event("response", timeout=response_timeout)
            events.append(("response", response))
        except TimeoutError:
            pass

        try:
            tts = await e2e_harness.wait_for_event("tts_status", timeout=5.0)
            events.append(("tts_status", tts))
        except TimeoutError:
            pass

        # Verify order
        event_names = [e[0] for e in events]
        assert "transcription" in event_names, "Missing transcription event"
        assert "response" in event_names, "Missing response event"

        # Transcription should come before response
        trans_idx = event_names.index("transcription")
        resp_idx = event_names.index("response")
        assert trans_idx < resp_idx, "Transcription should fire before response"
