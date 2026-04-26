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
import time
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
        dialogue_min_lines: int = 1,
        dialogue_drain_delay: float = 0.0,
    ):
        self._host = host
        self._port = port
        self._buffer: deque[GameEvent] = deque(maxlen=max(1, buffer_size))
        self._buffer_size = max(1, buffer_size)
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        self._enabled = enabled

        # Dialogue-specific buffer (NANO-116 Phase B.2)
        self._dialogue_buffer = DialogueBuffer(max_size=dialogue_buffer_size)

        # Drain cadence controls (NANO-116 B.5)
        self._dialogue_min_lines = max(1, dialogue_min_lines)
        self._dialogue_drain_delay = max(0.0, dialogue_drain_delay)
        self._first_line_time: float | None = None

        self._dialogue_store = None

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
            self._first_line_time = None

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

    @property
    def dialogue_min_lines(self) -> int:
        return self._dialogue_min_lines

    @dialogue_min_lines.setter
    def dialogue_min_lines(self, value: int) -> None:
        self._dialogue_min_lines = max(1, value)

    @property
    def dialogue_drain_delay(self) -> float:
        return self._dialogue_drain_delay

    @dialogue_drain_delay.setter
    def dialogue_drain_delay(self, value: float) -> None:
        self._dialogue_drain_delay = max(0.0, value)

    def start(self) -> None:
        if self._running:
            print("[GameState] start() called but already running", flush=True)
            return

        self._schema_version = load_vendored_version()
        if not self._schema_version:
            print("[GameState] FATAL: failed to load vendored schema version", flush=True)
            return

        print(f"[GameState] Starting module (target={self._host}:{self._port}, schema={self._schema_version})", flush=True)
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
        self._first_line_time = None
        self._last_sequence = -1
        self._thread = None
        logger.info("Game-state module stopped")

    def has_stimulus(self) -> bool:
        if not (self._enabled and self._running and not self._version_mismatch):
            return False
        count = self._dialogue_buffer.count
        if count < self._dialogue_min_lines:
            return False
        if self._dialogue_drain_delay > 0.0 and self._first_line_time is not None:
            if (time.monotonic() - self._first_line_time) < self._dialogue_drain_delay:
                return False
        return True

    def get_stimulus(self) -> Optional[StimulusData]:
        if not self.has_stimulus():
            return None

        lines = self._dialogue_buffer.drain()
        self._first_line_time = None
        if not lines:
            return None

        if self._dialogue_store:
            for dl in lines:
                self._dialogue_store.record_dialogue_line(dl)

        formatted_lines = []
        for dl in lines:
            if dl.repeat_count > 1:
                formatted_lines.append(f"[{dl.speaker}]: {dl.text} (x{dl.repeat_count})")
            else:
                formatted_lines.append(f"[{dl.speaker}]: {dl.text}")

        dialogue_block = "\n".join(formatted_lines)

        template = getattr(self, "_dialogue_prompt_template", None) or (
            "**The following are in-game character dialogue lines from the game "
            "you're co-hosting.** These characters are not talking to you — "
            "commentate on what they're saying, don't reply to them directly.\n\n"
            "{dialogue}\n"
        )

        user_input = template.format(dialogue=dialogue_block)

        print(f"[GameState] Dialogue drain: {len(lines)} lines, input_len={len(user_input)}", flush=True)

        return StimulusData(
            source=StimulusSource.GAME_STATE,
            user_input=user_input,
            metadata={
                "event_count": len(lines),
                "dialogue_lines": len(lines),
                "game_id": "pragmata",
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
        print(f"[GameState] TCP consumer thread alive, connecting to {self._host}:{self._port}", flush=True)
        while self._running and not self._stop_event.is_set():
            try:
                await self._connect_and_consume()
            except (OSError, ConnectionError) as e:
                if self._running:
                    logger.warning(
                        "Game-state bridge connection lost (type=%s, msg=%s), "
                        "reconnecting in %.1fs",
                        type(e).__name__,
                        e,
                        _RECONNECT_DELAY,
                    )
            except Exception as e:
                if self._running:
                    logger.warning(
                        "Unexpected error in game-state consumer (type=%s, msg=%s), "
                        "reconnecting in %.1fs",
                        type(e).__name__,
                        e,
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
        print(f"[GameState] Attempting TCP connect to {self._host}:{self._port}...", flush=True)
        reader, writer = await asyncio.open_connection(self._host, self._port)
        self._connected = True
        print(f"[GameState] CONNECTED to bridge at {self._host}:{self._port}", flush=True)

        try:
            # No handshake — just read whatever the bridge sends, whenever it sends it.
            while self._running and not self._stop_event.is_set():
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if not line:
                    logger.warning("Bridge returned empty read — connection closed by remote")
                    raise ConnectionError("Bridge closed connection")

                logger.debug("RAW LINE: %s", line[:200])
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

        # Capture protocol version from bridge_ready if we see one
        if event.get("event_type") == "bridge_ready":
            payload = event.get("payload", {})
            self._bridge_protocol_version = payload.get(
                "protocol_version", event.get("protocol_version")
            )
            logger.info(
                "Received bridge_ready (protocol_version=%s)",
                self._bridge_protocol_version,
            )

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
            if self._first_line_time is None and event_type == "dialogue_line":
                self._first_line_time = time.monotonic()
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
