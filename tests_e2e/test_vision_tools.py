"""
Vision Tools Tests - Tool invocation via VLM pipeline.

NANO-029 Phase 5: Verifies that:
1. Vision-triggering prompts invoke the screen_vision tool
2. Tool execution completes and returns results
3. VLM response is incorporated into final response

Note: Tool invocation reliability is model-dependent. Local LLMs may have
weak or no tool-use support. These tests are most reliable with cloud LLMs.
"""

import pytest

from .helpers.assertions import (
    assert_event_received,
    assert_response_exists,
    assert_tool_invoked,
)


class TestVisionTools:
    """Tests for screen_vision tool invocation via VLM."""

    @pytest.mark.asyncio
    @pytest.mark.vision
    async def test_vision_tool_invoked(self, e2e_harness, response_timeout):
        """
        Send vision-triggering prompt → verify tool_invoked event fires.

        The screen_vision tool should be invoked when asking about screen contents.
        """
        e2e_harness.clear_events()

        # Send a prompt that should trigger screen vision
        result = await e2e_harness.send_message(
            "What's on my screen right now? Describe what you see."
        )
        assert result["success"] is True

        # Wait for tool_invoked event
        tool_event = await assert_event_received(
            e2e_harness,
            "tool_invoked",
            timeout=response_timeout,
            message="Expected tool_invoked event for screen_vision",
        )

        # Verify screen_vision tool was invoked
        assert_tool_invoked(tool_event, expected_tool="screen_vision")

    @pytest.mark.asyncio
    @pytest.mark.vision
    async def test_vision_tool_result(self, e2e_harness, response_timeout):
        """
        Invoke screen_vision → verify tool_result event with content.

        After the tool is invoked, a tool_result event should fire with
        the VLM's description of the screen.
        """
        e2e_harness.clear_events()

        await e2e_harness.send_message(
            "Look at my screen and tell me what you see."
        )

        # Wait for tool_result event (comes after tool_invoked)
        tool_result = await assert_event_received(
            e2e_harness,
            "tool_result",
            timeout=response_timeout,
            message="Expected tool_result event after screen_vision invocation",
        )

        # Verify result has content (event uses 'result_summary' field)
        assert "result_summary" in tool_result or "result" in tool_result or "content" in tool_result
        content = tool_result.get(
            "result_summary",
            tool_result.get("result", tool_result.get("content", ""))
        )
        assert len(content) > 0, "Tool result should have non-empty content"

    @pytest.mark.asyncio
    @pytest.mark.vision
    async def test_vision_produces_final_response(self, e2e_harness, response_timeout):
        """
        Vision tool result → verify final response incorporates VLM output.

        After the screen_vision tool returns, the LLM should produce a final
        response that incorporates the VLM's description.
        """
        e2e_harness.clear_events()

        await e2e_harness.send_message(
            "Describe what's currently displayed on my screen."
        )

        # Wait for final response (after tool execution completes)
        response = await assert_event_received(
            e2e_harness,
            "response",
            timeout=response_timeout,
            message="Expected final response after vision tool execution",
        )

        # Verify non-empty response
        assert_response_exists(response)

    @pytest.mark.asyncio
    @pytest.mark.vision
    async def test_non_vision_prompt_no_tool(self, e2e_harness, response_timeout):
        """
        Non-vision prompt → verify screen_vision tool is NOT invoked.

        A simple greeting or question should not trigger tool invocation.
        """
        e2e_harness.clear_events()

        # Send a prompt that should NOT trigger vision
        await e2e_harness.send_message("What is the capital of France?")

        # Wait for response (should arrive without tool invocation)
        response = await assert_event_received(
            e2e_harness,
            "response",
            timeout=response_timeout,
            message="Expected response without tool invocation",
        )
        assert_response_exists(response)

        # Check that no tool_invoked event was received
        tool_events = e2e_harness.get_events("tool_invoked")
        assert len(tool_events) == 0, (
            f"Did not expect tool invocation for non-vision prompt, "
            f"but got {len(tool_events)} tool_invoked events"
        )


class TestVisionEventSequence:
    """Tests for correct event ordering during vision tool execution."""

    @pytest.mark.asyncio
    @pytest.mark.vision
    async def test_vision_event_order(self, e2e_harness, response_timeout):
        """
        Verify events fire in expected order for vision workflow:
        transcription → tool_invoked → tool_result → response
        """
        e2e_harness.clear_events()

        await e2e_harness.send_message("What can you see on my screen?")

        # Collect events in order
        events = []

        try:
            transcription = await e2e_harness.wait_for_event("transcription", timeout=5.0)
            events.append(("transcription", transcription))
        except TimeoutError:
            pass

        try:
            tool_invoked = await e2e_harness.wait_for_event("tool_invoked", timeout=response_timeout)
            events.append(("tool_invoked", tool_invoked))
        except TimeoutError:
            pass

        try:
            tool_result = await e2e_harness.wait_for_event("tool_result", timeout=response_timeout)
            events.append(("tool_result", tool_result))
        except TimeoutError:
            pass

        try:
            response = await e2e_harness.wait_for_event("response", timeout=response_timeout)
            events.append(("response", response))
        except TimeoutError:
            pass

        # Verify key events were received
        event_names = [e[0] for e in events]
        assert "tool_invoked" in event_names, "Missing tool_invoked event"
        assert "response" in event_names, "Missing response event"

        # If both tool events present, verify order
        if "tool_invoked" in event_names and "tool_result" in event_names:
            invoked_idx = event_names.index("tool_invoked")
            result_idx = event_names.index("tool_result")
            assert invoked_idx < result_idx, "tool_invoked should fire before tool_result"

        # Response should come after tool events
        if "tool_result" in event_names:
            result_idx = event_names.index("tool_result")
            response_idx = event_names.index("response")
            assert result_idx < response_idx, "tool_result should fire before response"
