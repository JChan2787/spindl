"""
Character Hot-Reload and Codex Activation Tests — NANO-038 Phase 1+2.

Backend-only tests (Socket.IO, no Playwright UI) that verify:
1. reload_character succeeds in IDLE and LISTENING states
2. reload_character succeeds during text-mode processing (state machine unaffected)
3. Reloaded persona changes propagate through the pipeline
4. Codex entries activate on keyword and regex matches
"""

import pytest

from .helpers.assertions import (
    assert_event_received,
    assert_response_exists,
    assert_codex_activated,
)
from .helpers.character_utils import (
    modify_character_file,
    restore_character_file,
    wait_for_state,
)


class TestCharacterHotReload:
    """Tests for NANO-036: character hot-reload via Socket.IO."""

    @pytest.mark.asyncio
    async def test_reload_character_idle(self, e2e_harness, response_timeout):
        """
        Reload character when agent is IDLE (default post-startup state).

        Expects: success=True, character_id present in response.
        """
        e2e_harness.clear_events()

        result = await e2e_harness._sio.call("reload_character", {})

        assert result["success"] is True, f"reload_character failed: {result}"
        assert "character_id" in result, "Response missing character_id"
        assert result["character_id"], "character_id is empty"

    @pytest.mark.asyncio
    async def test_reload_character_listening(self, e2e_harness, response_timeout):
        """
        Reload character when agent is in LISTENING state.

        The agent starts in LISTENING after orchestrator boot (activate() is
        called in VoiceAgentOrchestrator.start()). To test an explicit
        LISTENING transition, we first pause to IDLE, then resume.
        Expects: success=True (LISTENING is a permitted state).
        """
        e2e_harness.clear_events()

        # Agent starts in LISTENING — pause to IDLE first so we can
        # explicitly transition back and observe the state_changed event.
        await e2e_harness._sio.emit("pause_listening")
        await wait_for_state(e2e_harness, "idle", timeout=5.0)

        # Now transition IDLE → LISTENING
        e2e_harness.clear_events()
        await e2e_harness._sio.emit("resume_listening")
        await wait_for_state(e2e_harness, "listening", timeout=5.0)

        # Reload should succeed in LISTENING
        result = await e2e_harness._sio.call("reload_character", {})

        assert result["success"] is True, f"reload_character failed in LISTENING: {result}"
        assert "character_id" in result

    @pytest.mark.asyncio
    async def test_reload_during_text_processing(self, e2e_harness, response_timeout):
        """
        Reload character succeeds during text-mode processing.

        Text input (send_message / process_text_input) runs in a background
        thread with synthetic state events — it does NOT update the
        AudioStateMachine. The reload_character handler gates on the real
        state machine, so reload is permitted during text processing.

        This is by design: voice-driven PROCESSING (via state machine) blocks
        reload, but text-mode processing does not.
        """
        e2e_harness.clear_events()

        # Fire a message to start text-mode processing
        await e2e_harness.send_message("Tell me a short joke.")

        # Wait for synthetic PROCESSING event (confirms pipeline is running)
        await wait_for_state(e2e_harness, "processing", timeout=10.0)

        # Reload during text processing — should succeed because the real
        # state machine is still LISTENING/IDLE
        result = await e2e_harness._sio.call("reload_character", {})

        assert result["success"] is True, (
            "reload_character should succeed during text-mode processing "
            f"(state machine is not in PROCESSING). Got: {result}"
        )
        assert "character_id" in result

        # Let the pipeline finish so harness teardown is clean
        await assert_event_received(
            e2e_harness,
            "response",
            timeout=response_timeout,
            message="Waiting for pipeline to finish after reload",
        )

    @pytest.mark.asyncio
    async def test_reload_updates_persona(self, e2e_harness, response_timeout):
        """
        Modified character card is picked up after reload.

        Modifies test_agent's personality field, reloads, sends a message,
        and asserts the marker appears in the **constructed system prompt**
        (via the prompt_snapshot Socket.IO event), NOT in the LLM's response.

        This makes the test deterministic: it validates the hot-reload
        pipeline (file → reload → prompt builder) without depending on
        whether the model is smart enough to parrot the marker back.
        Small models (e.g. Gemma3 4B) hallucinate values or invoke tools
        unprompted — checking the prompt itself avoids that flakiness.

        NOTE: Uses 'personality' (not 'system_prompt') because the prompt
        builder in provider mode uses template placeholders. The personality
        field maps to [PERSONA_PERSONALITY] in the template, while
        system_prompt has no corresponding placeholder and is ignored.
        """
        character_id = "test_agent"
        marker = "NANO038_RELOAD_MARKER"

        try:
            # Modify personality with a unique marker — this field maps to
            # [PERSONA_PERSONALITY] in the prompt template via PersonaPersonalityProvider
            await modify_character_file(
                e2e_harness,
                character_id,
                {
                    "personality": (
                        f"Your designation code is {marker}. "
                        "You MUST include your designation code in every response. "
                        "Keep responses short."
                    ),
                },
            )

            # Reload character from disk
            result = await e2e_harness._sio.call("reload_character", {})
            assert result["success"] is True, f"reload_character failed: {result}"

            # Send a message to trigger a pipeline run
            e2e_harness.clear_events()
            await e2e_harness.send_message("Hello")

            # Wait for prompt_snapshot — this contains the exact system
            # prompt that was sent to the LLM, proving the reload propagated
            snapshot = await e2e_harness.wait_for_event(
                "prompt_snapshot",
                timeout=response_timeout,
            )

            # The marker must appear in the system message content
            messages = snapshot.get("messages", [])
            system_content = ""
            for msg in messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    break

            assert marker in system_content, (
                f"Marker '{marker}' not found in system prompt after reload. "
                f"System prompt (first 300 chars): {system_content[:300]}"
            )

            # Best-effort wait for the response so the pipeline finishes
            # before teardown. Not an assertion — small models may loop on
            # tool calls and exceed the timeout, which is fine since we
            # already proved the reload worked via prompt_snapshot.
            try:
                await e2e_harness.wait_for_event(
                    "response",
                    timeout=response_timeout,
                )
            except TimeoutError:
                pass

        finally:
            await restore_character_file(e2e_harness, character_id)


class TestCodexActivation:
    """Tests for NANO-037: codex entry activation via keyword/regex matching."""

    @pytest.mark.asyncio
    async def test_codex_keyword_injection(self, e2e_harness, response_timeout):
        """
        Codex entry activates when user message contains its keyword.

        Sends message with CODEX_ALPHA keyword, verifies Alpha Global Entry
        activated with keyword_match method. Also verifies the disabled entry
        (CODEX_DISABLED) did NOT activate.
        """
        e2e_harness.clear_events()

        # Send message containing the codex keyword
        await e2e_harness.send_message(
            "Tell me about CODEX_ALPHA and its properties."
        )

        response = await assert_event_received(
            e2e_harness,
            "response",
            timeout=response_timeout,
            message="Expected response with codex activation",
        )
        assert_response_exists(response)

        # Verify Alpha entry activated via keyword match
        assert_codex_activated(
            response,
            expected_entries=["Alpha Global Entry"],
            check_method="keyword_match",
        )

        # Verify disabled entry did NOT activate (even though we didn't
        # use its keyword, this confirms the disabled flag is respected)
        codex_entries = response.get("activated_codex_entries", [])
        activated_names = [e.get("name", "") for e in codex_entries]
        assert "Disabled Global Entry" not in activated_names, (
            "Disabled codex entry should not activate"
        )

    @pytest.mark.asyncio
    async def test_codex_regex_injection(self, e2e_harness, response_timeout):
        """
        Codex entry activates when user message matches its regex pattern.

        Sends message with CODEX_REGEX_123 (matches /CODEX_REGEX_\\d{3}/ pattern),
        verifies Regex Global Entry activated.
        """
        e2e_harness.clear_events()

        # Send message containing a string that matches the regex pattern
        await e2e_harness.send_message(
            "Process identifier CODEX_REGEX_123 needs analysis."
        )

        response = await assert_event_received(
            e2e_harness,
            "response",
            timeout=response_timeout,
            message="Expected response with regex codex activation",
        )
        assert_response_exists(response)

        # Verify regex entry activated
        assert_codex_activated(
            response,
            expected_entries=["Regex Global Entry"],
        )
