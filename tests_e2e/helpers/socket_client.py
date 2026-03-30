"""
Socket.IO Test Client - Async client for event verification.

NANO-029: Provides a dedicated Socket.IO client for E2E tests,
separate from Playwright browser connection.
"""

import asyncio
from typing import Optional, Callable

import socketio


class SocketTestClient:
    """
    Async Socket.IO client for test event verification.

    Provides methods for:
    - Emitting events to the server
    - Waiting for specific events with optional predicates
    - Recording all received events for later inspection
    """

    def __init__(self, url: str):
        """
        Initialize the test client.

        Args:
            url: Server URL to connect to.
        """
        self._url = url
        self._sio = socketio.AsyncClient()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._received_events: dict[str, list] = {}
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether the client is connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to the Socket.IO server."""
        # Register handlers before connecting
        self._register_handlers()
        await self._sio.connect(self._url)
        self._connected = True

    def _register_handlers(self) -> None:
        """Register event handlers for common events."""
        events = [
            "state_changed",
            "transcription",
            "response",
            "tts_status",
            "tool_invoked",
            "tool_result",
            "orchestrator_ready",
            "shutdown_complete",
            "health_status",
            "config_loaded",
            "pipeline_error",
            "token_usage",
            "prompt_snapshot",
            "launch_progress",
            "launch_complete",
            "launch_error",
        ]

        for event_name in events:
            self._register_event(event_name)

    def _register_event(self, event_name: str) -> None:
        """Register a single event handler."""
        @self._sio.on(event_name)
        async def handler(data, _event=event_name):
            if _event not in self._received_events:
                self._received_events[_event] = []
            self._received_events[_event].append(data)
            await self._event_queue.put((_event, data))

    async def emit(self, event: str, data: dict) -> None:
        """Emit an event to the server."""
        await self._sio.emit(event, data)

    async def call(self, event: str, data: dict, timeout: float = 5.0) -> dict:
        """
        Emit an event and wait for acknowledgement.

        Args:
            event: Event name.
            data: Event data.
            timeout: Timeout in seconds.

        Returns:
            Acknowledgement data from server.
        """
        return await self._sio.call(event, data, timeout=timeout)

    async def wait_for_event(
        self,
        event_name: str,
        timeout: float = 5.0,
        predicate: Optional[Callable[[dict], bool]] = None,
    ) -> dict:
        """
        Wait for a specific event.

        Args:
            event_name: Name of the event to wait for.
            timeout: Maximum wait time in seconds.
            predicate: Optional filter function.

        Returns:
            Event data.

        Raises:
            TimeoutError: If event not received in time.
        """
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            try:
                event, data = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=remaining,
                )
                if event == event_name:
                    if predicate is None or predicate(data):
                        return data
            except asyncio.TimeoutError:
                break

        raise TimeoutError(f"Event '{event_name}' not received within {timeout}s")

    def get_events(self, event_name: str) -> list:
        """Get all recorded events of a type."""
        return self._received_events.get(event_name, [])

    def clear_events(self) -> None:
        """Clear all recorded events."""
        self._received_events.clear()
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._sio.connected:
            await self._sio.disconnect()
        self._connected = False
