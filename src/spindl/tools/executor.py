"""
Tool Executor - Handles tool call loop in the LLM pipeline.

When the LLM decides to call tools, this executor:
1. Parses the tool calls from the response
2. Executes each tool
3. Formats results for the next LLM call
4. Loops until the LLM produces a final response (or max iterations hit)

This integrates with the LLMPipeline as a component that wraps the
provider.generate() call with tool execution logic.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..llm.base import LLMProvider, LLMResponse, ToolCall
from .registry import ToolRegistry

if TYPE_CHECKING:
    from ..core.event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """
    Result from the tool execution loop.

    Contains the final LLM response after all tool calls are resolved,
    plus metadata about what tools were executed.
    """

    response: LLMResponse
    """Final LLM response (after tool loop completes)."""

    tools_executed: list[dict]
    """List of tool executions: [{"name": "...", "result": "...", "success": bool}]"""

    iterations: int
    """Number of LLM calls made (including initial + tool result calls)."""


class ToolExecutor:
    """
    Executes tools in a loop until the LLM produces a final response.

    The execution flow:
    1. Call LLM with messages + tool definitions
    2. If LLM returns tool_calls:
       a. Execute each tool via ToolRegistry
       b. Append assistant message (with tool_calls) to messages
       c. Append tool result messages to messages
       d. Call LLM again (goto 2)
    3. If LLM returns content (no tool_calls):
       - Return the final response

    Usage:
        executor = ToolExecutor(tool_registry, max_iterations=5)
        result = await executor.execute(
            provider=llm_provider,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        # result.response is the final LLM response
        # result.tools_executed lists what was called

    Configuration:
        max_iterations: Safety limit on tool call loops (default: 5)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_iterations: int = 5,
        event_bus: Optional["EventBus"] = None,
    ):
        """
        Initialize the tool executor.

        Args:
            registry: ToolRegistry with initialized tools
            max_iterations: Max LLM calls before forcing stop (prevents infinite loops)
            event_bus: Optional EventBus for emitting tool events (NANO-025 Phase 7)
        """
        self._registry = registry
        self._max_iterations = max_iterations
        self._event_bus = event_bus

    def get_tool_definitions(self) -> list[dict]:
        """
        Get OpenAI-format tool definitions for all enabled tools.

        Returns:
            List of tool definitions in OpenAI format:
            [{"type": "function", "function": {"name": "...", ...}}]
        """
        enabled_tools = self._registry.get_enabled_tools()
        if not enabled_tools:
            return []

        definitions = []
        for tool in enabled_tools:
            definitions.append({
                "type": "function",
                "function": tool.get_function_definition(),
            })
        return definitions

    async def execute(
        self,
        provider: LLMProvider,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs,
    ) -> ToolExecutionResult:
        """
        Execute the tool calling loop.

        Args:
            provider: LLMProvider to use for generation
            messages: Initial message list (will be extended with tool interactions)
            temperature: LLM temperature
            max_tokens: Max tokens per LLM call
            **kwargs: Additional provider options

        Returns:
            ToolExecutionResult with final response and execution metadata
        """
        tools_executed: list[dict] = []
        iterations = 0

        # Get tool definitions
        tool_definitions = self.get_tool_definitions()

        # Make a copy of messages to avoid mutating the original
        working_messages = list(messages)

        while iterations < self._max_iterations:
            iterations += 1

            # Call LLM with tools
            logger.debug(f"Tool executor: iteration {iterations}, {len(working_messages)} messages")

            # Run LLM in thread pool (generate is sync)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: provider.generate(
                    messages=working_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tool_definitions if tool_definitions else None,
                    **kwargs,
                )
            )

            # Check if LLM wants to call tools
            if not response.tool_calls:
                # No tool calls - we're done
                logger.debug(f"Tool executor: completed after {iterations} iteration(s)")
                return ToolExecutionResult(
                    response=response,
                    tools_executed=tools_executed,
                    iterations=iterations,
                )

            # LLM wants to call tools - execute them
            logger.info(f"Tool executor: LLM requested {len(response.tool_calls)} tool(s)")

            # Append assistant message with tool calls to conversation
            assistant_message = self._build_assistant_tool_message(response)
            working_messages.append(assistant_message)

            # Execute each tool and collect results
            tool_results = await self._execute_tools(response.tool_calls, iterations)
            tools_executed.extend(tool_results)

            # NANO-087e: format tool results based on model capability
            if provider.get_properties().supports_tool_role:
                # Model supports role: "tool" — use structured format
                for result in tool_results:
                    working_messages.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["result"],
                    })
            else:
                # Model rejects role: "tool" (strict alternation) —
                # collapse results into a user message
                tool_content_parts = []
                for result in tool_results:
                    tool_content_parts.append(
                        f"[Tool Result: {result['name']}]\n{result['result']}"
                    )
                working_messages.append({
                    "role": "user",
                    "content": "\n\n".join(tool_content_parts),
                })

        # Max iterations reached - return last response
        logger.warning(f"Tool executor: max iterations ({self._max_iterations}) reached")

        # Do one final call without tools to get a response
        final_response = await loop.run_in_executor(
            None,
            lambda: provider.generate(
                messages=working_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,  # No tools - force text response
                **kwargs,
            )
        )

        return ToolExecutionResult(
            response=final_response,
            tools_executed=tools_executed,
            iterations=iterations,
        )

    def _build_assistant_tool_message(self, response: LLMResponse) -> dict:
        """
        Build the assistant message containing tool calls.

        This message is appended to the conversation so the LLM
        can see what tools it decided to call.

        Args:
            response: LLM response with tool_calls

        Returns:
            Assistant message dict in OpenAI format
        """
        tool_calls_formatted = []
        for tc in response.tool_calls:
            tool_calls_formatted.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": str(tc.arguments) if tc.arguments else "{}",
                },
            })

        return {
            "role": "assistant",
            "content": response.content or None,
            "tool_calls": tool_calls_formatted,
        }

    async def _execute_tools(self, tool_calls: list[ToolCall], iteration: int = 1) -> list[dict]:
        """
        Execute a list of tool calls.

        Args:
            tool_calls: List of ToolCall objects from LLM response
            iteration: Current iteration in the tool loop (1-based)

        Returns:
            List of execution results with tool_call_id, name, result, success
        """
        results = []

        for tc in tool_calls:
            logger.info(f"Executing tool: {tc.name}")
            print(f"[TOOL] Executing: {tc.name}", flush=True)

            # Emit tool invoked event (NANO-025 Phase 7)
            self._emit_tool_invoked(tc.name, tc.arguments, iteration, tc.id)

            start_time = time.time()
            try:
                tool = self._registry.get_tool(tc.name)
                if tool is None:
                    result_str = f"Error: Tool '{tc.name}' not found or disabled"
                    success = False
                    logger.warning(f"Tool not found: {tc.name}")
                else:
                    # Execute the tool
                    tool_result = await tool.execute(**tc.arguments)
                    result_str = tool_result.output
                    success = tool_result.success

                    if not success and tool_result.error:
                        result_str = f"Error: {tool_result.error}"

                    logger.info(f"Tool {tc.name} completed: success={success}")
                    print(f"[TOOL] {tc.name} completed: {result_str[:100]}...", flush=True)

            except Exception as e:
                result_str = f"Error executing tool: {e}"
                success = False
                logger.error(f"Tool execution error: {tc.name}: {e}")

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Emit tool result event (NANO-025 Phase 7)
            self._emit_tool_result(tc.name, success, result_str, duration_ms, iteration, tc.id)

            results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "result": result_str,
                "success": success,
            })

        return results

    def _emit_tool_invoked(
        self, tool_name: str, arguments: dict, iteration: int, tool_call_id: str
    ) -> None:
        """Emit tool invoked event if EventBus is available."""
        if not self._event_bus:
            return

        from ..core.events import ToolInvokedEvent

        self._event_bus.emit(
            ToolInvokedEvent(
                tool_name=tool_name,
                arguments=arguments or {},
                iteration=iteration,
                tool_call_id=tool_call_id,
            )
        )

    def _emit_tool_result(
        self,
        tool_name: str,
        success: bool,
        result: str,
        duration_ms: int,
        iteration: int,
        tool_call_id: str,
    ) -> None:
        """Emit tool result event if EventBus is available."""
        if not self._event_bus:
            return

        from ..core.events import ToolResultEvent

        # Truncate result for summary (first 200 chars)
        result_summary = result[:200] + "..." if len(result) > 200 else result

        self._event_bus.emit(
            ToolResultEvent(
                tool_name=tool_name,
                success=success,
                result_summary=result_summary,
                duration_ms=duration_ms,
                iteration=iteration,
                tool_call_id=tool_call_id,
            )
        )


def create_tool_executor(
    registry: ToolRegistry,
    max_iterations: int = 5,
    event_bus: Optional["EventBus"] = None,
) -> Optional[ToolExecutor]:
    """
    Create a tool executor if tools are enabled.

    Args:
        registry: ToolRegistry with tools configured
        max_iterations: Max tool call loop iterations
        event_bus: Optional EventBus for emitting tool events (NANO-025 Phase 7)

    Returns:
        ToolExecutor if tools are enabled, None otherwise
    """
    if not registry.get_enabled_tools():
        logger.debug("No tools enabled, tool executor not created")
        return None

    return ToolExecutor(registry, max_iterations=max_iterations, event_bus=event_bus)
