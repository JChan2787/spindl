"""
E2E Test Assertions - Custom wait/assert helpers.

NANO-029: Provides assertion helpers for E2E tests that focus on
existence checks (did we get a response?) rather than content validation.
"""

import asyncio
from typing import Optional, Callable, Any


async def wait_for_event(
    harness,
    event_name: str,
    timeout: float = 5.0,
    predicate: Optional[Callable[[dict], bool]] = None,
) -> dict:
    """
    Wait for a Socket.IO event from the harness.

    Args:
        harness: E2EHarness instance.
        event_name: Name of the event to wait for.
        timeout: Maximum wait time in seconds.
        predicate: Optional filter function.

    Returns:
        Event data dict.

    Raises:
        TimeoutError: If event not received.
    """
    return await harness.wait_for_event(event_name, timeout=timeout, predicate=predicate)


async def assert_event_received(
    harness,
    event_name: str,
    timeout: float = 5.0,
    predicate: Optional[Callable[[dict], bool]] = None,
    message: Optional[str] = None,
) -> dict:
    """
    Assert that an event is received within timeout.

    Args:
        harness: E2EHarness instance.
        event_name: Name of the event to wait for.
        timeout: Maximum wait time in seconds.
        predicate: Optional filter function.
        message: Custom failure message.

    Returns:
        Event data dict.

    Raises:
        AssertionError: If event not received.
    """
    try:
        return await wait_for_event(harness, event_name, timeout, predicate)
    except TimeoutError:
        msg = message or f"Expected event '{event_name}' was not received within {timeout}s"
        raise AssertionError(msg)


def assert_non_empty(value: Any, field_name: str = "value") -> None:
    """
    Assert that a value is non-empty.

    Works for strings, lists, dicts, and other containers.

    Args:
        value: Value to check.
        field_name: Name for error message.

    Raises:
        AssertionError: If value is empty or None.
    """
    if value is None:
        raise AssertionError(f"{field_name} is None")

    if isinstance(value, str):
        if not value.strip():
            raise AssertionError(f"{field_name} is empty string")
    elif hasattr(value, "__len__"):
        if len(value) == 0:
            raise AssertionError(f"{field_name} is empty (length 0)")


def assert_response_exists(event_data: dict) -> None:
    """
    Assert that a response event contains non-empty content.

    Args:
        event_data: Response event data with 'text' field.

    Raises:
        AssertionError: If text is empty.
    """
    text = event_data.get("text", "")
    assert_non_empty(text, "response.text")


def assert_transcription_exists(event_data: dict) -> None:
    """
    Assert that a transcription event contains non-empty text.

    Args:
        event_data: Transcription event data with 'text' field.

    Raises:
        AssertionError: If text is empty.
    """
    text = event_data.get("text", "")
    assert_non_empty(text, "transcription.text")


def assert_tool_invoked(event_data: dict, expected_tool: Optional[str] = None) -> None:
    """
    Assert that a tool was invoked.

    Args:
        event_data: tool_invoked event data.
        expected_tool: Optional specific tool name to check.

    Raises:
        AssertionError: If tool name doesn't match.
    """
    tool_name = event_data.get("tool_name", "")
    assert_non_empty(tool_name, "tool_invoked.tool_name")

    if expected_tool and tool_name != expected_tool:
        raise AssertionError(
            f"Expected tool '{expected_tool}', got '{tool_name}'"
        )


def assert_state_changed(
    event_data: dict,
    expected_to: Optional[str] = None,
    expected_from: Optional[str] = None,
) -> None:
    """
    Assert state change event matches expectations.

    Args:
        event_data: state_changed event data.
        expected_to: Expected destination state.
        expected_from: Expected source state.

    Raises:
        AssertionError: If states don't match.
    """
    to_state = event_data.get("to", "")
    from_state = event_data.get("from", "")

    if expected_to and to_state != expected_to:
        raise AssertionError(
            f"Expected state transition to '{expected_to}', got '{to_state}'"
        )

    if expected_from and from_state != expected_from:
        raise AssertionError(
            f"Expected state transition from '{expected_from}', got '{from_state}'"
        )


def assert_codex_activated(
    event_data: dict,
    expected_entries: list[str],
    check_method: Optional[str] = None,
) -> None:
    """
    Assert that specific codex entries were activated in a response event.

    Verifies the activated_codex_entries field in response event data.

    Args:
        event_data: Response event data containing activated_codex_entries.
        expected_entries: List of expected entry names (e.g., ["Alpha Global Entry"]).
        check_method: Optional activation_method to verify (e.g., "keyword_match").

    Raises:
        AssertionError: If expected entries are not found or method doesn't match.
    """
    codex_entries = event_data.get("activated_codex_entries", [])

    if not codex_entries:
        raise AssertionError(
            f"Expected codex entries {expected_entries} but activated_codex_entries "
            f"is empty or missing in response event"
        )

    activated_names = [e.get("name", "") for e in codex_entries]

    for expected in expected_entries:
        if expected not in activated_names:
            raise AssertionError(
                f"Expected codex entry '{expected}' not found in activated entries. "
                f"Got: {activated_names}"
            )

    if check_method:
        for entry in codex_entries:
            if entry.get("name") in expected_entries:
                actual_method = entry.get("activation_method", "")
                if actual_method != check_method:
                    raise AssertionError(
                        f"Expected activation_method '{check_method}' for "
                        f"'{entry.get('name')}', got '{actual_method}'"
                    )


def assert_no_codex_activated(event_data: dict) -> None:
    """
    Assert that no codex entries were activated in a response event.

    Args:
        event_data: Response event data.

    Raises:
        AssertionError: If any codex entries are present and non-empty.
    """
    codex_entries = event_data.get("activated_codex_entries", [])

    if codex_entries:
        names = [e.get("name", "unnamed") for e in codex_entries]
        raise AssertionError(
            f"Expected no codex entries activated, but got: {names}"
        )
