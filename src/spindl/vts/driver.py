"""
VTubeStudio driver — WebSocket command dispatch for VTS (NANO-060a).

Architecture:
- Daemon thread running asyncio event loop
- SimpleQueue for thread-safe command dispatch (any thread → async loop)
- pyvts for WebSocket connection, auth, and VTS API calls
- BaseRequest() passthrough for unwrapped endpoints (expressions)
- Exponential backoff reconnection on WebSocket drop

Thread safety:
- All public methods (trigger_hotkey, trigger_expression, move_model, etc.)
  enqueue commands to a SimpleQueue consumed by the async loop.
- is_connected() and get_status() read atomic fields only.
"""

import asyncio
import logging
import re
import threading
import time
from queue import SimpleQueue, Empty
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Reconnect backoff constants
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 30.0
_BACKOFF_MULTIPLIER = 2.0

# Queue drain interval (seconds)
_QUEUE_POLL_INTERVAL = 0.05

# Emotion bracket pattern: [happy], [sad], etc.
_EMOTION_PATTERN = re.compile(r"\[(\w+)\]")


class VTSDriver:
    """
    Drives VTubeStudio from agent pipeline events.

    Owns a daemon thread with an asyncio event loop. Commands are dispatched
    via SimpleQueue from any thread. pyvts handles WebSocket + auth.
    """

    def __init__(self, config, event_bus=None):
        """
        Args:
            config: VTubeStudioConfig dataclass instance.
            event_bus: Optional EventBus for future Phase 1c wiring.
        """
        self._config = config
        self._event_bus = event_bus

        # State (read from any thread)
        self._connected = False
        self._authenticated = False
        self._running = False
        self._model_name: Optional[str] = ""
        self._cached_hotkeys: list[str] = []
        self._cached_expressions: list[dict] = []

        # Command queue: (command_name, args, kwargs)
        self._queue: SimpleQueue = SimpleQueue()

        # Thread
        self._thread: Optional[threading.Thread] = None

        # EventBus subscription IDs (NANO-060c)
        self._sub_ids: list[int] = []

        # pyvts instance (created in async context)
        self._vts = None

    # ------------------------------------------------------------------ #
    # Public API (thread-safe, enqueue to SimpleQueue)
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Spawn daemon thread, begin connection loop, subscribe to EventBus."""
        if self._running:
            return
        self._running = True
        self._subscribe_events()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="VTSDriver",
            daemon=True,
        )
        self._thread.start()
        logger.info("[VTS] Driver thread started")

    def stop(self) -> None:
        """Signal shutdown, unsubscribe events, join thread."""
        if not self._running:
            return
        self._running = False
        self._unsubscribe_events()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._connected = False
        self._authenticated = False
        logger.info("[VTS] Driver stopped")

    def is_connected(self) -> bool:
        """WebSocket connected and authenticated."""
        return self._connected and self._authenticated

    def get_status(self) -> dict:
        """Current driver status snapshot."""
        return {
            "connected": self._connected,
            "authenticated": self._authenticated,
            "model_name": self._model_name,
            "hotkeys": list(self._cached_hotkeys),
            "expressions": list(self._cached_expressions),
        }

    def trigger_hotkey(self, name: str) -> None:
        """Trigger a VTS hotkey by name. Thread-safe."""
        self._queue.put(("trigger_hotkey", (name,), {}))

    def trigger_expression(self, file: str, active: bool = True) -> None:
        """Activate/deactivate an expression file. Thread-safe."""
        self._queue.put(("trigger_expression", (file, active), {}))

    def move_model(self, preset: str) -> None:
        """Move model to a named position preset. Thread-safe."""
        self._queue.put(("move_model", (preset,), {}))

    def request_hotkey_list(self, callback: Optional[Callable] = None) -> None:
        """Request hotkey list from VTS. Optional callback receives list."""
        self._queue.put(("list_hotkeys", (), {"callback": callback}))

    def request_expression_list(self, callback: Optional[Callable] = None) -> None:
        """Request expression list from VTS. Optional callback receives list."""
        self._queue.put(("list_expressions", (), {"callback": callback}))

    # ------------------------------------------------------------------ #
    # EventBus wiring (NANO-060c)
    # ------------------------------------------------------------------ #

    def _subscribe_events(self) -> None:
        """Subscribe to EventBus events for expression/animation wiring."""
        if self._event_bus is None:
            return
        from ..core.events import EventType

        sub = self._event_bus.subscribe(
            EventType.RESPONSE_READY, self._on_response_ready, priority=5,
        )
        self._sub_ids.append(sub)

        sub = self._event_bus.subscribe(
            EventType.STATE_CHANGED, self._on_state_changed, priority=5,
        )
        self._sub_ids.append(sub)

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from all EventBus events."""
        if self._event_bus is None:
            return
        for sub_id in self._sub_ids:
            self._event_bus.unsubscribe(sub_id)
        self._sub_ids.clear()

    def _on_response_ready(self, event) -> None:
        """Extract emotion bracket from LLM response, trigger matching expression."""
        text = getattr(event, "text", "")
        if not text:
            return

        match = _EMOTION_PATTERN.search(text)
        if match is None:
            return

        emotion = match.group(1).lower()
        expression_file = self._config.expressions.get(emotion)
        if expression_file:
            self.trigger_expression(expression_file)
            logger.debug("[VTS] Emotion '%s' → expression '%s'", emotion, expression_file)

    def _on_state_changed(self, event) -> None:
        """Trigger thinking/idle hotkeys on agent state transitions."""
        to_state = getattr(event, "to_state", "")
        trigger = getattr(event, "trigger", "")

        if to_state == "processing":
            if self._config.thinking_hotkey:
                self.trigger_hotkey(self._config.thinking_hotkey)
        elif to_state == "listening":
            # Voice path: mic reactivated after TTS
            if self._config.idle_hotkey:
                self.trigger_hotkey(self._config.idle_hotkey)
        elif to_state == "idle" and trigger in ("tts_complete", "response_complete"):
            # Text path: return to idle after response delivery
            if self._config.idle_hotkey:
                self.trigger_hotkey(self._config.idle_hotkey)

    # ------------------------------------------------------------------ #
    # Thread entry point
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """Daemon thread entry: run the async main loop."""
        try:
            asyncio.run(self._async_main())
        except Exception as e:
            logger.error("[VTS] Async loop crashed: %s", e)
        finally:
            self._connected = False
            self._authenticated = False

    # ------------------------------------------------------------------ #
    # Async main loop
    # ------------------------------------------------------------------ #

    async def _async_main(self) -> None:
        """Connect, auth, then drain command queue until shutdown."""
        import pyvts

        plugin_info = {
            "developer": self._config.developer,
            "plugin_name": self._config.plugin_name,
            "plugin_icon": None,
            "authentication_token_path": self._config.token_path,
        }
        vts_api_info = {
            "version": "1.0",
            "name": "VTubeStudioPublicAPI",
            "host": self._config.host,
            "port": self._config.port,
        }
        self._vts = pyvts.vts(
            plugin_info=plugin_info,
            vts_api_info=vts_api_info,
        )

        # Initial connection attempt
        connected = await self._connect_and_auth()
        if not connected:
            logger.warning("[VTS] Initial connection failed — driver inert until VTS available")

        # Main loop: drain queue + handle reconnection
        while self._running:
            if not self._connected:
                await self._reconnect_with_backoff()
                if not self._running:
                    break
                continue

            await self._process_queue()

        # Cleanup
        if self._vts and self._connected:
            try:
                await self._vts.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Connection + auth
    # ------------------------------------------------------------------ #

    async def _connect_and_auth(self) -> bool:
        """Connect WebSocket and authenticate. Returns True on success."""
        try:
            await self._vts.connect()
        except Exception as e:
            logger.warning("[VTS] Connection failed: %s", e)
            self._connected = False
            self._authenticated = False
            return False

        if self._vts.get_connection_status() != 1:
            self._connected = False
            self._authenticated = False
            return False

        self._connected = True
        logger.info("[VTS] WebSocket connected to %s:%d", self._config.host, self._config.port)

        # Auth: read token or request new one
        try:
            await self._vts.request_authenticate_token()
            authenticated = await self._vts.request_authenticate()
        except Exception as e:
            logger.warning("[VTS] Auth failed: %s — requesting new token", e)
            authenticated = False

        if not authenticated:
            # Force new token (triggers VTS approval dialog)
            try:
                await self._vts.request_authenticate_token(force=True)
                authenticated = await self._vts.request_authenticate()
            except Exception as e:
                logger.error("[VTS] Auth recovery failed: %s", e)
                authenticated = False

        self._authenticated = authenticated
        if authenticated:
            logger.info("[VTS] Authenticated as '%s'", self._config.plugin_name)
            # Cache available hotkeys and expressions
            await self._refresh_cached_lists()
        else:
            logger.warning("[VTS] Authentication failed — commands will not execute")

        return authenticated

    async def _refresh_cached_lists(self) -> None:
        """Fetch and cache model info, hotkey/expression lists from VTS."""
        # Current model name
        try:
            req = self._vts.vts_request.BaseRequest("CurrentModelRequest", {})
            resp = await self._vts.request(req)
            data = resp.get("data", {})
            if data.get("modelLoaded"):
                self._model_name = data.get("modelName", "")
            else:
                self._model_name = ""
        except Exception as e:
            logger.warning("[VTS] Failed to get model name: %s", e)

        try:
            hotkeys = await self._do_list_hotkeys()
            self._cached_hotkeys = hotkeys
        except Exception as e:
            logger.warning("[VTS] Failed to list hotkeys: %s", e)

        try:
            expressions = await self._do_list_expressions()
            self._cached_expressions = expressions
        except Exception as e:
            logger.warning("[VTS] Failed to list expressions: %s", e)

    # ------------------------------------------------------------------ #
    # Reconnection with exponential backoff
    # ------------------------------------------------------------------ #

    async def _reconnect_with_backoff(self) -> None:
        """Attempt reconnection with exponential backoff."""
        backoff = _INITIAL_BACKOFF
        while self._running and not self._connected:
            logger.info("[VTS] Reconnecting in %.1fs...", backoff)
            await asyncio.sleep(backoff)
            if not self._running:
                return

            connected = await self._connect_and_auth()
            if connected:
                logger.info("[VTS] Reconnected successfully")
                return

            backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)

    # ------------------------------------------------------------------ #
    # Queue processing
    # ------------------------------------------------------------------ #

    async def _process_queue(self) -> None:
        """Drain the command queue. Non-blocking — returns after one pass."""
        try:
            cmd, args, kwargs = self._queue.get(timeout=_QUEUE_POLL_INTERVAL)
        except Empty:
            return

        if not self.is_connected():
            logger.warning("[VTS] Command '%s' dropped — not connected", cmd)
            return

        try:
            if cmd == "trigger_hotkey":
                await self._do_trigger_hotkey(*args)
            elif cmd == "trigger_expression":
                await self._do_trigger_expression(*args)
            elif cmd == "move_model":
                await self._do_move_model(*args)
            elif cmd == "list_hotkeys":
                result = await self._do_list_hotkeys()
                self._cached_hotkeys = result
                cb = kwargs.get("callback")
                if cb:
                    cb(result)
            elif cmd == "list_expressions":
                result = await self._do_list_expressions()
                self._cached_expressions = result
                cb = kwargs.get("callback")
                if cb:
                    cb(result)
            else:
                logger.warning("[VTS] Unknown command: %s", cmd)
        except Exception as e:
            logger.error("[VTS] Command '%s' failed: %s", cmd, e)
            # Check if connection was lost
            if self._vts and self._vts.get_connection_status() != 1:
                self._connected = False
                self._authenticated = False
                logger.warning("[VTS] Connection lost — entering reconnect")

    # ------------------------------------------------------------------ #
    # VTS command handlers (async, called from queue processor)
    # ------------------------------------------------------------------ #

    async def _do_trigger_hotkey(self, name: str) -> None:
        """Trigger a hotkey by name."""
        request_msg = self._vts.vts_request.requestTriggerHotKey(name)
        response = await self._vts.request(request_msg)
        logger.debug("[VTS] Hotkey '%s' triggered: %s", name, response.get("messageType"))

    async def _do_trigger_expression(self, file: str, active: bool = True) -> None:
        """Activate/deactivate an expression file via BaseRequest."""
        request_msg = self._vts.vts_request.BaseRequest(
            "ExpressionActivationRequest",
            {"expressionFile": file, "active": active},
        )
        response = await self._vts.request(request_msg)
        logger.debug("[VTS] Expression '%s' active=%s: %s", file, active, response.get("messageType"))

    async def _do_move_model(self, preset: str) -> None:
        """Move model to a named position preset."""
        from .constants import DEFAULT_POSITIONS

        # Check config positions first, fall back to defaults
        positions = self._config.positions or {}
        pos = positions.get(preset) or DEFAULT_POSITIONS.get(preset)
        if pos is None:
            logger.warning("[VTS] Unknown position preset: '%s'", preset)
            return

        request_msg = self._vts.vts_request.requestMoveModel(
            pos.get("x", 0),
            pos.get("y", 0),
            pos.get("rotation", 0),
            pos.get("size", 0),
            True,   # valuesAreRelativeToModel
            0.5,    # timeInSeconds (smooth transition)
        )
        response = await self._vts.request(request_msg)
        logger.debug("[VTS] Model moved to '%s': %s", preset, response.get("messageType"))

    async def _do_list_hotkeys(self) -> list[str]:
        """Query VTS for available hotkeys. Returns list of hotkey names."""
        request_msg = self._vts.vts_request.requestHotKeyList()
        response = await self._vts.request(request_msg)
        data = response.get("data", {})
        hotkeys = data.get("availableHotkeys", [])
        return [hk.get("name", "") for hk in hotkeys]

    async def _do_list_expressions(self) -> list[dict]:
        """Query VTS for available expressions. Returns list of expression dicts."""
        request_msg = self._vts.vts_request.BaseRequest(
            "ExpressionStateRequest",
            {},
        )
        response = await self._vts.request(request_msg)
        data = response.get("data", {})
        expressions = data.get("expressions", [])
        return [
            {
                "name": expr.get("name", ""),
                "file": expr.get("file", ""),
                "active": expr.get("active", False),
            }
            for expr in expressions
        ]
