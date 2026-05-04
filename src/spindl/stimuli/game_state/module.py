"""
Game-state bridge stimulus module (NANO-116 Phase B.1, NANO-122 gameplay).

Connects to the SPNDL-001 game-state bridge via TCP, buffers incoming
events, and exposes them as stimuli for the StimuliEngine. Priority 50
— same tier as Twitch (external integration).

Two stimulus sub-paths:
  1. Dialogue — buffered lines with dedup + drain cadence (NANO-116 B.2)
  2. Gameplay — event-driven combat + periodic snapshot aggregate (NANO-122)

Wire protocol: newline-delimited JSON on TCP 127.0.0.1:53817 (default).
First event after connect must be bridge_ready with protocol_version
in payload. Consumer validates against vendored schema version.

Follows the Twitch module pattern: bounded deque buffer, atomic drain
in get_stimulus(), template-formatted output.
"""

import asyncio
import json
import logging
import random
import threading
import time
from collections import deque
from typing import Any, Optional

from ..base import StimulusModule
from ..models import StimulusData, StimulusSource
from ..weighted_rotator import WeightedRotator
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

# NANO-122: Event types that trigger immediate gameplay stimulus
_GAMEPLAY_EVENT_TYPES = frozenset({
    "enemy_engaged_player",
    "enemy_died",
    "boss_battle_started",
    "boss_battle_ended",
    "chapter_status_changed",
})

# Snapshot dirty-check fields (NANO-122 Phase 2)
_SNAPSHOT_DIRTY_FIELDS_BOOL = ("in_combat", "is_dead", "is_boss_battle")
_SNAPSHOT_DIRTY_FIELDS_ENEMY_ROSTER = ("is_hackable", "is_confused")

_DEFAULT_DIALOGUE_PROMPT_TEMPLATES: list[str] = [
    "**The following are in-game character dialogue lines from the game "
    "you're co-hosting.** These characters are not talking to you — "
    "commentate on what they're saying, don't reply to them directly.\n"
    "\n"
    "{dialogue}\n"
]


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
        # NANO-122: Gameplay stimulus config
        gameplay_enabled: bool = False,
        gameplay_base_probability: float = 0.20,
        gameplay_escalation_step: float = 0.15,
        gameplay_probability_ceiling: float = 1.0,
        gameplay_dirty_hp_threshold: float = 0.10,
        gameplay_event_batch_window: float = 2.0,
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

        self._dialogue_prompt_templates: list[str] = list(
            _DEFAULT_DIALOGUE_PROMPT_TEMPLATES
        )
        self._template_rotator = WeightedRotator(self._dialogue_prompt_templates)

        self._dialogue_store = None
        self._summarizing = False
        self._dialogue_dropped_during_summary = 0

        # NANO-122: Gameplay stimulus state
        self._gameplay_enabled = gameplay_enabled
        self._gameplay_base_probability = max(0.05, min(1.0, gameplay_base_probability))
        self._gameplay_escalation_step = max(0.05, min(0.5, gameplay_escalation_step))
        self._gameplay_probability_ceiling = max(0.1, min(1.0, gameplay_probability_ceiling))
        self._gameplay_dirty_hp_threshold = max(0.01, min(0.5, gameplay_dirty_hp_threshold))
        self._gameplay_event_batch_window = max(0.5, min(10.0, gameplay_event_batch_window))

        # Event-based gameplay buffer (Phase 1)
        self._gameplay_event_buffer: list[GameEvent] = []
        self._gameplay_first_event_time: float | None = None
        # Chapter dedupe: last chapter name that fired (zone-level, not status-level)
        self._last_chapter_name: str | None = None

        # Snapshot aggregate (Phase 2)
        self._last_evaluated_snapshot: dict[str, Any] | None = None
        self._current_snapshot: dict[str, Any] | None = None
        self._snapshot_probability: float = self._gameplay_base_probability
        self._snapshot_roll_passed: bool = False

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
            self._gameplay_event_buffer.clear()
            self._gameplay_first_event_time = None

            self._last_evaluated_snapshot = None
            self._current_snapshot = None
            self._snapshot_probability = self._gameplay_base_probability
            self._snapshot_roll_passed = False

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
    def dialogue_prompt_templates(self) -> list[str]:
        return self._dialogue_prompt_templates

    @dialogue_prompt_templates.setter
    def dialogue_prompt_templates(self, value: list[str]) -> None:
        self._dialogue_prompt_templates = value if value else list(
            _DEFAULT_DIALOGUE_PROMPT_TEMPLATES
        )
        self._template_rotator.items = self._dialogue_prompt_templates

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

    # -- NANO-122: Gameplay config properties --------------------------------

    @property
    def gameplay_enabled(self) -> bool:
        return self._gameplay_enabled

    @gameplay_enabled.setter
    def gameplay_enabled(self, value: bool) -> None:
        self._gameplay_enabled = value
        if not value:
            self._gameplay_event_buffer.clear()
            self._gameplay_first_event_time = None

            self._last_evaluated_snapshot = None
            self._current_snapshot = None
            self._snapshot_probability = self._gameplay_base_probability
            self._snapshot_roll_passed = False

    @property
    def gameplay_base_probability(self) -> float:
        return self._gameplay_base_probability

    @gameplay_base_probability.setter
    def gameplay_base_probability(self, value: float) -> None:
        self._gameplay_base_probability = max(0.05, min(1.0, value))

    @property
    def gameplay_escalation_step(self) -> float:
        return self._gameplay_escalation_step

    @gameplay_escalation_step.setter
    def gameplay_escalation_step(self, value: float) -> None:
        self._gameplay_escalation_step = max(0.05, min(0.5, value))

    @property
    def gameplay_probability_ceiling(self) -> float:
        return self._gameplay_probability_ceiling

    @gameplay_probability_ceiling.setter
    def gameplay_probability_ceiling(self, value: float) -> None:
        self._gameplay_probability_ceiling = max(0.1, min(1.0, value))

    @property
    def gameplay_dirty_hp_threshold(self) -> float:
        return self._gameplay_dirty_hp_threshold

    @gameplay_dirty_hp_threshold.setter
    def gameplay_dirty_hp_threshold(self, value: float) -> None:
        self._gameplay_dirty_hp_threshold = max(0.01, min(0.5, value))

    @property
    def gameplay_event_batch_window(self) -> float:
        return self._gameplay_event_batch_window

    @gameplay_event_batch_window.setter
    def gameplay_event_batch_window(self, value: float) -> None:
        self._gameplay_event_batch_window = max(0.5, min(10.0, value))

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
        self._gameplay_event_buffer.clear()
        self._gameplay_first_event_time = None
        self._last_evaluated_snapshot = None
        self._current_snapshot = None
        self._snapshot_probability = self._gameplay_base_probability
        self._snapshot_roll_passed = False
        self._last_chapter_name = None
        self._last_sequence = -1
        self._thread = None
        logger.info("Game-state module stopped")

    def has_stimulus(self) -> bool:
        if not (self._enabled and self._running and not self._version_mismatch):
            return False
        # Gameplay events take priority within the module (NANO-122)
        if self._has_gameplay_event_stimulus():
            return True
        if self._has_gameplay_snapshot_stimulus():
            return True
        return self._has_dialogue_stimulus()

    def get_stimulus(self) -> Optional[StimulusData]:
        if not (self._enabled and self._running and not self._version_mismatch):
            return None
        # Gameplay events first, then snapshot, then dialogue
        stim = self._get_gameplay_event_stimulus()
        if stim:
            return stim
        stim = self._get_gameplay_snapshot_stimulus()
        if stim:
            return stim
        return self._get_dialogue_stimulus()

    def health_check(self) -> bool:
        return self._connected and not self._version_mismatch

    # -- Dialogue stimulus (NANO-116 B.2) ------------------------------------

    def _has_dialogue_stimulus(self) -> bool:
        count = self._dialogue_buffer.count
        if count < self._dialogue_min_lines:
            return False
        if self._dialogue_drain_delay > 0.0 and self._first_line_time is not None:
            if (time.monotonic() - self._first_line_time) < self._dialogue_drain_delay:
                return False
        return True

    @staticmethod
    def _format_dialogue_line(dl) -> str:
        if dl.repeat_count > 1:
            return f"[{dl.speaker}]: {dl.text} (x{dl.repeat_count})"
        return f"[{dl.speaker}]: {dl.text}"

    def _get_dialogue_stimulus(self) -> Optional[StimulusData]:
        if not self._has_dialogue_stimulus():
            return None

        lines = self._dialogue_buffer.drain()
        self._first_line_time = None
        if not lines:
            return None

        if self._dialogue_store:
            for dl in lines:
                self._dialogue_store.record_dialogue_line(dl)

        if len(lines) >= 2:
            context_lines = [self._format_dialogue_line(dl) for dl in lines[:-1]]
            latest_line = self._format_dialogue_line(lines[-1])
            dialogue_block = (
                "\n".join(context_lines)
                + "\n\n**[Latest]**\n"
                + latest_line
            )
        else:
            dialogue_block = self._format_dialogue_line(lines[0])

        template = self._template_rotator.select() or self._dialogue_prompt_templates[0]
        user_input = template.format(dialogue=dialogue_block)

        print(f"[GameState] Dialogue drain: {len(lines)} lines, input_len={len(user_input)}", flush=True)

        return StimulusData(
            source=StimulusSource.GAME_STATE,
            user_input=user_input,
            metadata={
                "stimulus_type": "dialogue",
                "event_count": len(lines),
                "dialogue_lines": len(lines),
                "game_id": "pragmata",
            },
        )

    # -- Gameplay event stimulus (NANO-122 Phase 1) --------------------------

    def _has_gameplay_event_stimulus(self) -> bool:
        if not self._gameplay_enabled:
            return False
        if not self._gameplay_event_buffer:
            return False
        if self._gameplay_first_event_time is None:
            return False
        return (time.monotonic() - self._gameplay_first_event_time) >= self._gameplay_event_batch_window

    def _get_gameplay_event_stimulus(self) -> Optional[StimulusData]:
        if not self._has_gameplay_event_stimulus():
            return None

        events = list(self._gameplay_event_buffer)
        self._gameplay_event_buffer.clear()
        self._gameplay_first_event_time = None
        if not events:
            return None

        lines = []
        for ev in events:
            line = self._format_gameplay_event(ev)
            if line:
                lines.append(line)

        if not lines:
            return None

        if len(lines) >= 2:
            events_block = (
                "\n".join(lines[:-1])
                + "\n\n**[Latest]**\n"
                + lines[-1]
            )
        else:
            events_block = lines[0]

        user_input = self._prompt_template.format(events=events_block)

        print(
            f"[GameState] Gameplay event drain: {len(events)} events, "
            f"input_len={len(user_input)}",
            flush=True,
        )

        return StimulusData(
            source=StimulusSource.GAME_STATE,
            user_input=user_input,
            metadata={
                "stimulus_type": "gameplay_event",
                "event_count": len(events),
                "game_id": "pragmata",
            },
        )

    def _format_gameplay_event(self, event: GameEvent) -> Optional[str]:
        p = event.payload
        etype = event.event_type
        name = p.get("enemy_display_name") or p.get("enemy_name") or p.get("enemy_type") or "unknown enemy"
        mtype = p.get("member_type_name")

        if etype == "enemy_engaged_player":
            label = f"{name} ({mtype})" if mtype else name
            return f"- Combat: {label} engaged!"
        elif etype == "enemy_died":
            label = f"{name} ({mtype})" if mtype else name
            return f"- Kill: {label} eliminated!"
        elif etype == "boss_battle_started":
            return "- Boss battle started!"
        elif etype == "boss_battle_ended":
            return "- Boss battle ended!"
        elif etype == "chapter_status_changed":
            ch_name = p.get("chapter_name") or f"chapter_{p.get('chapter_hash', '?')}"
            return f"- Area transition: entering {ch_name}."
        return None

    def _accept_gameplay_event(self, event: GameEvent) -> None:
        """Route a gameplay event into the batch buffer with dedupe."""
        now = time.monotonic()
        etype = event.event_type
        p = event.payload

        if etype == "chapter_status_changed":
            ch_name = p.get("chapter_name") or ""
            base_name = ch_name.removesuffix(" (sub)").strip()
            if base_name and base_name == self._last_chapter_name:
                logger.debug("Deduped chapter_status_changed name=%s", base_name)
                return
            self._last_chapter_name = base_name or None

        self._gameplay_event_buffer.append(event)
        if self._gameplay_first_event_time is None:
            self._gameplay_first_event_time = now

        logger.debug(
            "Gameplay event buffered: type=%s, batch_size=%d",
            etype, len(self._gameplay_event_buffer),
        )

    # -- Gameplay snapshot stimulus (NANO-122 Phase 2) -----------------------

    def _has_gameplay_snapshot_stimulus(self) -> bool:
        if not self._gameplay_enabled:
            return False
        if self._current_snapshot is None:
            return False
        if not self._is_snapshot_dirty():
            self._snapshot_roll_passed = False
            return False
        if self._roll_snapshot_probability():
            self._snapshot_roll_passed = True
            return True
        self._snapshot_roll_passed = False
        return False

    def _get_gameplay_snapshot_stimulus(self) -> Optional[StimulusData]:
        if not self._snapshot_roll_passed:
            return None
        self._snapshot_roll_passed = False
        if not self._gameplay_enabled or self._current_snapshot is None:
            return None

        snap = self._current_snapshot
        self._last_evaluated_snapshot = dict(snap)
        if snap.get("enemies"):
            self._last_evaluated_snapshot["enemies"] = [dict(e) for e in snap["enemies"]]
        self._snapshot_probability = self._gameplay_base_probability

        report = self._format_snapshot_report(snap)
        if not report:
            return None

        user_input = self._prompt_template.format(events=report)

        print(
            f"[GameState] Snapshot aggregate fired, input_len={len(user_input)}",
            flush=True,
        )

        return StimulusData(
            source=StimulusSource.GAME_STATE,
            user_input=user_input,
            metadata={
                "stimulus_type": "gameplay_snapshot",
                "game_id": "pragmata",
            },
        )

    def _is_snapshot_dirty(self) -> bool:
        cur = self._current_snapshot
        prev = self._last_evaluated_snapshot
        if cur is None:
            return False
        if prev is None:
            return True

        for field in _SNAPSHOT_DIRTY_FIELDS_BOOL:
            if cur.get(field) != prev.get(field):
                return True

        cur_enemies = cur.get("enemies") or []
        prev_enemies = prev.get("enemies") or []
        if len(cur_enemies) != len(prev_enemies):
            return True

        prev_by_id: dict[int, dict] = {}
        for e in prev_enemies:
            eid = e.get("enemy_id", id(e))
            prev_by_id[eid] = e

        for e in cur_enemies:
            eid = e.get("enemy_id", id(e))
            pe = prev_by_id.get(eid)
            if pe is None:
                return True
            for bf in _SNAPSHOT_DIRTY_FIELDS_ENEMY_ROSTER:
                if e.get(bf) != pe.get(bf):
                    return True
            cur_hp = e.get("hp_ratio", 0.0)
            prev_hp = pe.get("hp_ratio", 0.0)
            if abs(cur_hp - prev_hp) > self._gameplay_dirty_hp_threshold:
                return True

        return False

    def _roll_snapshot_probability(self) -> bool:
        if random.random() < self._snapshot_probability:
            return True
        self._snapshot_probability = min(
            self._snapshot_probability + self._gameplay_escalation_step,
            self._gameplay_probability_ceiling,
        )
        logger.debug(
            "Snapshot probability escalated to %.2f", self._snapshot_probability,
        )
        return False

    def _format_snapshot_report(self, snap: dict[str, Any]) -> Optional[str]:
        lines: list[str] = []

        hp_ratio = snap.get("hp_ratio", 1.0)
        hp_pct = round(hp_ratio * 100)
        weapon = snap.get("weapon_name") or "unknown weapon"
        in_combat = snap.get("in_combat", False)
        is_dead = snap.get("is_dead", False)

        status = "DEAD" if is_dead else ("in combat" if in_combat else "exploring")
        lines.append(f"- Hugh: {hp_pct}% HP, {weapon} equipped, {status}")

        enemies = snap.get("enemies") or []
        engaged = [e for e in enemies if not e.get("is_dead", False)]
        if engaged:
            parts: list[str] = []
            for e in engaged:
                ename = e.get("display_name") or e.get("name") or "unknown"
                ehp = round(e.get("hp_ratio", 1.0) * 100)
                tags: list[str] = []
                if e.get("is_hackable"):
                    tags.append("hackable")
                if e.get("is_confused"):
                    tags.append("confused")
                suffix = f", {', '.join(tags)}" if tags else ""
                parts.append(f"{ename} ({ehp}% HP{suffix})")
            lines.append(f"- Enemies ({len(engaged)} active): {', '.join(parts)}")

        is_boss = snap.get("is_boss_battle", False)
        if is_boss:
            lines.append("- **BOSS BATTLE ACTIVE**")

        return "**Current gameplay state:**\n" + "\n".join(lines) if lines else None

    def notify_stimulus_fired(self) -> None:
        """Called by the engine after this module's stimulus reaches the LLM."""
        self._snapshot_probability = self._gameplay_base_probability

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
        """Convert validated event dict to GameEvent and route to buffers.

        Routing:
          - Dialogue events → DialogueBuffer (NANO-116 B.2)
          - Gameplay combat events → gameplay event buffer (NANO-122 Phase 1)
          - Snapshots → dialogue context + full snapshot store (NANO-122 Phase 2)
          - All events → generic buffer (B.1 log/debug)
        """
        if not self._enabled:
            return

        event_type = event["event_type"]
        payload = event.get("payload", {})

        # Update dialogue gameplay snapshot from non-dialogue events (B.2)
        if event_type == "snapshot":
            self._dialogue_buffer.update_gameplay_snapshot(
                chapter_hash=payload.get("chapter_hash"),
                combat_active=payload.get("in_combat", False),
                enemy_count=payload.get("enemy_count_engage", 0),
                hp_ratio=payload.get("hp_ratio"),
                timestamp=event.get("timestamp", ""),
            )
            # NANO-122: Store full snapshot for aggregate stimulus
            if self._gameplay_enabled:
                self._current_snapshot = dict(payload)
        elif event_type in ("enemy_engaged_player", "enemy_disengaged_player"):
            self._dialogue_buffer.update_gameplay_snapshot(
                combat_active=event_type == "enemy_engaged_player",
                enemy_count=payload.get("enemy_count", 0),
                timestamp=event.get("timestamp", ""),
            )

        # Route dialogue events to dialogue buffer (B.2)
        # KD-4: Drop dialogue events while summarizer is running
        if DialogueBuffer.is_dialogue_event(event_type):
            if self._summarizing:
                self._dialogue_dropped_during_summary += 1
                logger.debug(
                    "Dropping dialogue event during summarization (total dropped: %d)",
                    self._dialogue_dropped_during_summary,
                )
            else:
                if self._first_line_time is None and event_type == "dialogue_line":
                    self._first_line_time = time.monotonic()
                self._dialogue_buffer.accept_event(event)

        # NANO-122: Route gameplay combat events to gameplay buffer
        game_event = GameEvent(
            event_type=event_type,
            event_source=event["event_source"],
            timestamp=event["timestamp"],
            sequence=event["sequence"],
            payload=payload,
            game_id=event.get("game_id"),
            save_slot_hint=event.get("save_slot_hint"),
        )

        if self._gameplay_enabled and event_type in _GAMEPLAY_EVENT_TYPES:
            self._accept_gameplay_event(game_event)

        # Generic buffer (B.1 — all events)
        self._buffer.append(game_event)
        logger.debug(
            "Buffered event: type=%s, seq=%d, buffer_len=%d, dialogue_len=%d, gameplay_len=%d",
            game_event.event_type,
            game_event.sequence,
            len(self._buffer),
            self._dialogue_buffer.count,
            len(self._gameplay_event_buffer),
        )
