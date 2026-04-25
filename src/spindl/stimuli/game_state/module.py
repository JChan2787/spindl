"""
Game-state bridge stimulus module (NANO-116 Phase B.1).

Connects to the SPNDL-001 game-state bridge via TCP, buffers incoming
events, and exposes them as stimuli for the StimuliEngine. Priority 50
— same tier as Twitch (external integration).

Wire protocol: newline-delimited JSON on TCP 127.0.0.1:53817 (default).
First event after connect must be bridge_ready with protocol_version
in payload. Consumer validates against vendored schema version.

Follows the Twitch module pattern: bounded deque buffer, atomic drain
in get_stimulus(), template-formatted output.
"""

import asyncio
import json
import logging
import threading
from collections import deque
from typing import Optional

from ..base import StimulusModule
from ..models import StimulusData, StimulusSource
from .dialogue_buffer import DialogueBuffer
from .models import GameEvent
from .validator import check_protocol_version, load_vendored_version, validate_envelope

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = (
    "**New game events from the bridge.** "
    "These are in-game events — commentate on what's happening, "
    "don't address game characters directly.\n"
    "\n"
    "{events}\n"
)

_RECONNECT_DELAY = 5.0
_READ_BUFFER_SIZE = 65536


class GameStateModule(StimulusModule):
    """
    Game-state bridge stimulus source (NANO-116).

    Connects to the SPNDL-001 bridge TCP channel, validates events
    against the vendored schema, and buffers them for the StimuliEngine.
    Both dialogue and gameplay events enter the same buffer — tier-specific
    filtering happens in later phases (B.2 dialogue pipeline, B.3 gameplay
    window).

    Priority 50 — external integration tier.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 53817,
        buffer_size: int = 20,
        prompt_template: Optional[str] = None,
        enabled: bool = False,
        dialogue_buffer_size: int = 30,
    ):
        self._host = host
        self._port = port
        self._buffer: deque[GameEvent] = deque(maxlen=max(1, buffer_size))
        self._buffer_size = max(1, buffer_size)
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        self._enabled = enabled

        # Dialogue-specific buffer (NANO-116 Phase B.2)
        self._dialogue_buffer = DialogueBuffer(max_size=dialogue_buffer_size)

        self._connected = False
        self._running = False
        self._version_mismatch = False
        self._bridge_protocol_version: str | None = None
        self._schema_version: str | None = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_sequence = -1

    # -- StimulusModule interface ------------------------------------------

    @property
    def name(self) -> str:
        return "game_state"

    @property
    def priority(self) -> int:
        return 50

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self._buffer.clear()
            self._dialogue_buffer.clear()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def version_mismatch(self) -> bool:
        return self._version_mismatch

    @property
    def bridge_protocol_version(self) -> str | None:
        return self._bridge_protocol_version

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    @property
    def dialogue_buffer(self) -> DialogueBuffer:
        """Dialogue-specific buffer with per-line dedup (NANO-116 B.2)."""
        return self._dialogue_buffer

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        self._host = value

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        self._port = value

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self._buffer_size = max(1, value)
        old_events = list(self._buffer)
        self._buffer = deque(old_events, maxlen=self._buffer_size)

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value

    def start(self) -> None:
        if self._running:
            return

        self._schema_version = load_vendored_version()
        if not self._schema_version:
            logger.error(
                "Game-state module cannot start: failed to load vendored schema version"
            )
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_thread, name="GameStateModule", daemon=True
        )
        self._thread.start()
        logger.info(
            "Game-state module started (target=%s:%d, schema_version=%s)",
            self._host,
            self._port,
            self._schema_version,
        )

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self._connected = False
        self._version_mismatch = False
        self._bridge_protocol_version = None
        self._buffer.clear()
        self._dialogue_buffer.clear()
        self._last_sequence = -1
        self._thread = None
        logger.info("Game-state module stopped")

    def has_stimulus(self) -> bool:
        return (
            self._enabled
            and self._running
            and not self._version_mismatch
            and len(self._buffer) > 0
        )

    def get_stimulus(self) -> Optional[StimulusData]:
        if not self.has_stimulus():
            return None

        events = list(self._buffer)
        self._buffer.clear()

        lines: list[str] = []
        for ev in events:
            line = f"[{ev.event_type}] {json.dumps(ev.payload, default=str)}"
            lines.append(line)

        formatted = "\n".join(lines)

        template = self._prompt_template or _DEFAULT_PROMPT_TEMPLATE
        if "{events}" not in template:
            logger.warning(
                "game_state_prompt_template missing {events} placeholder; "
                "falling back to default. Buffered events would otherwise be lost."
            )
            template = _DEFAULT_PROMPT_TEMPLATE

        user_input = template.format(events=formatted)

        logger.info(
            "[GameState] Buffer drained: %d events, user_input_len=%d",
            len(events),
            len(user_input),
        )

        return StimulusData(
            source=StimulusSource.GAME_STATE,
            user_input=user_input,
            metadata={
                "event_count": len(events),
                "event_types": [ev.event_type for ev in events],
                "game_id": events[0].game_id if events else None,
            },
        )

    def health_check(self) -> bool:
        return self._connected and not self._version_mismatch

    # -- Internal async machinery ------------------------------------------

    def _run_thread(self) -> None:
        """Daemon thread entry point — runs the async event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async())
        except Exception:
            logger.exception("Game-state module thread crashed")
        finally:
            self._connected = False
            loop.close()

    async def _run_async(self) -> None:
        """Async main — connect to bridge TCP and listen for events."""
        while self._running and not self._stop_event.is_set():
            try:
                await self._connect_and_consume()
            except (OSError, ConnectionError) as e:
                if self._running:
                    logger.warning(
                        "Game-state bridge connection lost (%s), "
                        "reconnecting in %.1fs",
                        e,
                        _RECONNECT_DELAY,
                    )
            except Exception:
                if self._running:
                    logger.exception(
                        "Unexpected error in game-state consumer, "
                        "reconnecting in %.1fs",
                        _RECONNECT_DELAY,
                    )
            finally:
                self._connected = False
                self._bridge_protocol_version = None
                self._version_mismatch = False

            if self._running and not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self._stop_event.wait, _RECONNECT_DELAY
                        ),
                        timeout=_RECONNECT_DELAY + 1.0,
                    )
                except asyncio.TimeoutError:
                    pass
                if self._stop_event.is_set():
                    break

    async def _connect_and_consume(self) -> None:
        """Single connection lifecycle: connect, validate, consume until disconnect."""
        logger.info(
            "Connecting to game-state bridge at %s:%d", self._host, self._port
        )
        reader, writer = await asyncio.open_connection(self._host, self._port)
        self._connected = True
        logger.info("Connected to game-state bridge at %s:%d", self._host, self._port)

        try:
            # First event must be bridge_ready with protocol version
            first_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not first_line:
                raise ConnectionError("Bridge closed connection before sending bridge_ready")

            first_event = json.loads(first_line.decode("utf-8").strip())
            ok, reason = validate_envelope(first_event)
            if not ok:
                logger.error("Invalid bridge_ready envelope: %s", reason)
                return

            if first_event.get("event_type") != "bridge_ready":
                logger.error(
                    "Expected bridge_ready as first event, got: %s",
                    first_event.get("event_type"),
                )
                return

            # Protocol version check
            payload = first_event.get("payload", {})
            bridge_version = payload.get("protocol_version", first_event.get("protocol_version"))
            self._bridge_protocol_version = bridge_version

            version_ok, level = check_protocol_version(
                bridge_version, self._schema_version
            )
            if level == "major":
                logger.error(
                    "MAJOR protocol version mismatch: bridge=%s, vendored=%s. "
                    "Refusing to consume events. Update vendored schema.",
                    bridge_version,
                    self._schema_version,
                )
                self._version_mismatch = True
                return
            elif level == "minor":
                logger.warning(
                    "Minor protocol version drift: bridge=%s, vendored=%s. "
                    "Continuing with forward-compatible consumption.",
                    bridge_version,
                    self._schema_version,
                )
            elif level == "parse_error":
                logger.error(
                    "Could not parse protocol versions: bridge=%s, vendored=%s",
                    bridge_version,
                    self._schema_version,
                )
                self._version_mismatch = True
                return
            else:
                logger.info(
                    "Protocol version %s (%s): bridge=%s, vendored=%s",
                    level,
                    "ok" if version_ok else "mismatch",
                    bridge_version,
                    self._schema_version,
                )

            # Buffer the bridge_ready event itself
            self._buffer_event(first_event)

            # Steady-state consumption
            while self._running and not self._stop_event.is_set():
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if not line:
                    raise ConnectionError("Bridge closed connection")

                self._process_line(line)

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _process_line(self, raw: bytes) -> None:
        """Parse and validate a single newline-delimited JSON event."""
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return

        try:
            event = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON from bridge: %s (line: %.100s...)", e, text)
            return

        ok, reason = validate_envelope(event)
        if not ok:
            logger.warning("Dropping invalid event: %s", reason)
            return

        # Sequence monotonicity check (detect dropped/reordered events)
        seq = event["sequence"]
        if seq <= self._last_sequence and self._last_sequence >= 0:
            logger.warning(
                "Out-of-order event: seq=%d, last=%d (event_type=%s)",
                seq,
                self._last_sequence,
                event.get("event_type"),
            )
        self._last_sequence = seq

        self._buffer_event(event)

    def _buffer_event(self, event: dict) -> None:
        """Convert validated event dict to GameEvent and append to buffer.

        Also routes dialogue events to the dialogue-specific buffer
        (NANO-116 B.2) and updates the gameplay snapshot from non-dialogue
        events so dialogue lines capture situational context.
        """
        if not self._enabled:
            return

        event_type = event["event_type"]

        # Update gameplay snapshot from non-dialogue events (B.2)
        if event_type == "snapshot":
            payload = event.get("payload", {})
            player = payload.get("player", {})
            self._dialogue_buffer.update_gameplay_snapshot(
                chapter_hash=payload.get("chapter_hash"),
                combat_active=payload.get("combat_active", False),
                enemy_count=payload.get("enemy_count", 0),
                hp_ratio=player.get("hp_ratio"),
                timestamp=event.get("timestamp", ""),
            )
        elif event_type in ("enemy_engaged", "enemy_disengaged"):
            payload = event.get("payload", {})
            self._dialogue_buffer.update_gameplay_snapshot(
                combat_active=event_type == "enemy_engaged",
                enemy_count=payload.get("enemy_count", 0),
                timestamp=event.get("timestamp", ""),
            )

        # Route dialogue events to dialogue buffer (B.2)
        if DialogueBuffer.is_dialogue_event(event_type):
            self._dialogue_buffer.accept_event(event)

        # Generic buffer (B.1 — all events)
        game_event = GameEvent(
            event_type=event_type,
            event_source=event["event_source"],
            timestamp=event["timestamp"],
            sequence=event["sequence"],
            payload=event.get("payload", {}),
            game_id=event.get("game_id"),
            save_slot_hint=event.get("save_slot_hint"),
        )
        self._buffer.append(game_event)
        logger.debug(
            "Buffered event: type=%s, seq=%d, buffer_len=%d, dialogue_len=%d",
            game_event.event_type,
            game_event.sequence,
            len(self._buffer),
            self._dialogue_buffer.count,
        )
