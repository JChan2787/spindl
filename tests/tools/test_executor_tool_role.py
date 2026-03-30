"""
Tests for tool executor model-aware tool result formatting (NANO-087e).

Tests cover:
- supports_tool_role=True → role: "tool" with tool_call_id
- supports_tool_role=False → role: "user" with [Tool Result: ...] prefix
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from spindl.llm.base import LLMProperties, LLMResponse, ToolCall
from spindl.tools.executor import ToolExecutor
from spindl.tools.registry import ToolRegistry


def _make_provider(supports_tool_role: bool) -> MagicMock:
    """Create a mock LLMProvider with configurable tool role support."""
    provider = MagicMock()
    provider.get_properties.return_value = LLMProperties(
        model_name="test-model",
        supports_streaming=False,
        supports_tools=True,
        supports_tool_role=supports_tool_role,
    )
    return provider


def _make_tool_call_response() -> LLMResponse:
    """Create an LLMResponse that requests a tool call."""
    return LLMResponse(
        content=None,
        input_tokens=10,
        output_tokens=5,
        tool_calls=[
            ToolCall(
                id="call_123",
                name="screen_vision",
                arguments={},
            )
        ],
    )


def _make_final_response() -> LLMResponse:
    """Create a final LLMResponse with no tool calls."""
    return LLMResponse(
        content="Here's what I see on your screen.",
        input_tokens=50,
        output_tokens=20,
        tool_calls=None,
    )


def _make_executor() -> ToolExecutor:
    """Create a ToolExecutor with a mock screen_vision tool."""
    registry = MagicMock(spec=ToolRegistry)
    registry.get_enabled_tools.return_value = {}

    executor = ToolExecutor(registry=registry, max_iterations=3)

    # Mock tool definitions
    executor.get_tool_definitions = MagicMock(return_value=[{
        "type": "function",
        "function": {
            "name": "screen_vision",
            "description": "Capture and describe screen",
            "parameters": {"type": "object", "properties": {}},
        },
    }])

    # Mock tool execution
    async def mock_execute_tools(tool_calls, iteration=1):
        return [{
            "tool_call_id": "call_123",
            "name": "screen_vision",
            "result": "A desktop with a code editor open.",
            "success": True,
        }]

    executor._execute_tools = mock_execute_tools

    return executor


class TestToolResultRoleRouting:
    """Tests for model-aware tool result formatting."""

    def test_supports_tool_role_uses_tool_message(self):
        """When supports_tool_role=True, results use role: 'tool'."""
        provider = _make_provider(supports_tool_role=True)
        # First call returns tool call, second returns final response
        provider.generate = MagicMock(side_effect=[
            _make_tool_call_response(),
            _make_final_response(),
        ])

        executor = _make_executor()
        result = asyncio.get_event_loop().run_until_complete(
            executor.execute(
                provider=provider,
                messages=[{"role": "user", "content": "describe screen"}],
            )
        )

        # Check the second generate call's messages
        second_call_messages = provider.generate.call_args_list[1].kwargs.get(
            "messages"
        ) or provider.generate.call_args_list[1][1].get("messages")
        if second_call_messages is None:
            second_call_messages = provider.generate.call_args_list[1][0][0]

        # Find the tool result message
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_123"
        assert "desktop" in tool_messages[0]["content"]

    def test_no_tool_role_uses_user_message(self):
        """When supports_tool_role=False, results use role: 'user'."""
        provider = _make_provider(supports_tool_role=False)
        provider.generate = MagicMock(side_effect=[
            _make_tool_call_response(),
            _make_final_response(),
        ])

        executor = _make_executor()
        result = asyncio.get_event_loop().run_until_complete(
            executor.execute(
                provider=provider,
                messages=[{"role": "user", "content": "describe screen"}],
            )
        )

        # Check the second generate call's messages
        second_call_messages = provider.generate.call_args_list[1].kwargs.get(
            "messages"
        ) or provider.generate.call_args_list[1][1].get("messages")
        if second_call_messages is None:
            second_call_messages = provider.generate.call_args_list[1][0][0]

        # No tool role messages should exist
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 0

        # Should have a user message with tool result prefix
        user_messages = [m for m in second_call_messages if m.get("role") == "user"]
        tool_result_msgs = [
            m for m in user_messages
            if "[Tool Result: screen_vision]" in m.get("content", "")
        ]
        assert len(tool_result_msgs) == 1
        assert "desktop" in tool_result_msgs[0]["content"]

    def test_no_tool_role_maintains_alternation(self):
        """When supports_tool_role=False, message roles alternate properly."""
        provider = _make_provider(supports_tool_role=False)
        provider.generate = MagicMock(side_effect=[
            _make_tool_call_response(),
            _make_final_response(),
        ])

        executor = _make_executor()
        asyncio.get_event_loop().run_until_complete(
            executor.execute(
                provider=provider,
                messages=[{"role": "user", "content": "describe screen"}],
            )
        )

        second_call_messages = provider.generate.call_args_list[1].kwargs.get(
            "messages"
        ) or provider.generate.call_args_list[1][1].get("messages")
        if second_call_messages is None:
            second_call_messages = provider.generate.call_args_list[1][0][0]

        # Verify alternation: user, assistant, user (tool result)
        roles = [m["role"] for m in second_call_messages]
        assert roles == ["user", "assistant", "user"]
