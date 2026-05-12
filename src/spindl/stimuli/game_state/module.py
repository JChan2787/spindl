"""
Game-state bridge stimulus module (NANO-116 B.1, NANO-122, NANO-123, NANO-124).

Connects to the SPNDL-001 game-state bridge via TCP, buffers incoming
events, and exposes them as stimuli for the StimuliEngine. Priority 60
— above Twitch (50) so game events preempt chat during gameplay.

Stimulus sub-paths (NANO-123 restructure):
  1. Boss/chapter events — independent, deterministic, highest internal priority
  2. Dialogue — fire on 1 line, zero delay, latest-line-only user message.
     During combat, bundles current gameplay snapshot as context.

Self-barge-in (NANO-124): when TTS is active and a dialogue line arrives,
a two-layer probability system (escalating pressure + interrupt fatigue)
decides whether to interrupt the current TTS and react to the new line.
The module triggers TTS stop via callback, reusing the existing barge-in
plumbing. Both layers reset when a TTS output completes without interruption.

Killed paths (NANO-123): independent gameplay event stimulus and snapshot
aggregate stimulus no longer fire. Accumulation logic stays for context.

Wire protocol: newline-delimited JSON on TCP 127.0.0.1:53817 (default).
First event after connect must be bridge_ready with protocol_version
in payload. Consumer validates against vendored schema version.
"""

import asyncio
import json
import logging
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

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

# NANO-122: Event types routed to the gameplay buffer
_GAMEPLAY_EVENT_TYPES = frozenset({
    "enemy_engaged_player",
    "enemy_died",
    "boss_battle_started",
    "boss_battle_ended",
    "chapter_status_changed",
})

# NANO-123: High-signal events that fire independently (no probability, no batch)
_PRIORITY_EVENT_TYPES = frozenset({
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

_DEFAULT_BARGE_IN_PROMPT_TEMPLATES: list[str] = [
    "**Something just happened in the game while you were talking.** "
    "React to this new line instead of continuing your previous thought.\n"
    "\n"
    "{dialogue}\n"
]

_DEFAULT_BARGE_IN_ESCALATION: list[float] = [
    0.01, 0.015, 0.02, 0.025, 0.05,
    0.06, 0.067, 0.07, 0.075, 0.1,
    0.12, 0.15, 0.18, 0.2, 0.23,
    0.25, 0.27, 0.3, 0.33, 0.4,
]
_DEFAULT_BARGE_IN_FATIGUE: list[float] = [1.00, 0.60, 0.30]


class GameStateModule(StimulusModule):
    """
    Game-state bridge stimulus source (NANO-116, NANO-123).

    Connects to the SPNDL-001 bridge TCP channel, validates events
    against the vendored schema, and produces stimuli for the engine.

    Internal priority (NANO-123): boss/chapter events > dialogue.
    Gameplay events and snapshots accumulate for context but do not
    fire independently. Priority 60 — above Twitch (50).
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
        # NANO-123 Phase 2: Engine idle callback for stale-line dropping
        is_engine_idle: Optional[Callable[[], bool]] = None,
        # NANO-124: Self-barge-in probability system
        barge_in_enabled: bool = False,
        barge_in_escalation: Optional[list[float]] = None,
        barge_in_fatigue: Optional[list[float]] = None,
        trigger_barge_in: Optional[Callable[[], None]] = None,
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

        # NANO-123 Phase 2: Stale-line dropping.
        # Only lines arriving while the engine is idle trigger a stimulus.
        # Lines arriving while busy still go to DialogueBuffer/Store for context.
        self._is_engine_idle = is_engine_idle
        self._has_fresh_trigger = False

        # NANO-124: Self-barge-in probability system.
        # Layer 1 (escalating): each dialogue_line during TTS increases chance.
        # Layer 2 (fatigue): each actual barge-in dampens the curve.
        # Both reset when TTS completes without interruption.
        self._barge_in_enabled = barge_in_enabled
        self._barge_in_escalation = list(barge_in_escalation or _DEFAULT_BARGE_IN_ESCALATION)
        self._barge_in_fatigue = list(barge_in_fatigue or _DEFAULT_BARGE_IN_FATIGUE)
        self._trigger_barge_in = trigger_barge_in
        self._barge_in_arrival_count = 0
        self._barge_in_count = 0
        self._barge_in_triggered = False
        self._barge_in_prompt_templates: list[str] = list(_DEFAULT_BARGE_IN_PROMPT_TEMPLATES)
        self._barge_in_template_rotator = WeightedRotator(self._barge_in_prompt_templates)

        # NANO-122/123: Gameplay state
        self._gameplay_enabled = gameplay_enabled
        self._gameplay_base_probability = max(0.05, min(1.0, gameplay_base_probability))
        self._gameplay_escalation_step = max(0.05, min(0.5, gameplay_escalation_step))
        self._gameplay_probability_ceiling = max(0.1, min(1.0, gameplay_probability_ceiling))
        self._gameplay_dirty_hp_threshold = max(0.01, min(0.5, gameplay_dirty_hp_threshold))
        self._gameplay_event_batch_window = max(0.5, min(10.0, gameplay_event_batch_window))

        # Event buffer — accumulates all gameplay events. NANO-123: only
        # boss/chapter events fire independently; the rest are context-only.
        self._gameplay_event_buffer: list[GameEvent] = []
        self._gameplay_first_event_time: float | None = None
        self._last_chapter_name: str | None = None

        # Snapshot — updated from bridge, read-only context for combat chatter bundling
        self._current_snapshot: dict[str, Any] | None = None

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
        return 60

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
            self._current_snapshot = None

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
            self._current_snapshot = None

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

    # -- NANO-124: Self-barge-in config properties ------------------------------

    @property
    def barge_in_enabled(self) -> bool:
        return self._barge_in_enabled

    @barge_in_enabled.setter
    def barge_in_enabled(self, value: bool) -> None:
        self._barge_in_enabled = value
        if not value:
            self._reset_barge_in_state()

    @property
    def barge_in_escalation(self) -> list[float]:
        return self._barge_in_escalation

    @barge_in_escalation.setter
    def barge_in_escalation(self, value: list[float]) -> None:
        self._barge_in_escalation = list(value) if value else list(_DEFAULT_BARGE_IN_ESCALATION)

    @property
    def barge_in_fatigue(self) -> list[float]:
        return self._barge_in_fatigue

    @barge_in_fatigue.setter
    def barge_in_fatigue(self, value: list[float]) -> None:
        self._barge_in_fatigue = list(value) if value else list(_DEFAULT_BARGE_IN_FATIGUE)

    @property
    def barge_in_prompt_templates(self) -> list[str]:
        return self._barge_in_prompt_templates

    @barge_in_prompt_templates.setter
    def barge_in_prompt_templates(self, value: list[str]) -> None:
        self._barge_in_prompt_templates = value if value else list(_DEFAULT_BARGE_IN_PROMPT_TEMPLATES)
        self._barge_in_template_rotator.items = self._barge_in_prompt_templates

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
        self._current_snapshot = None
        self._last_chapter_name = None
        self._last_sequence = -1
        self._thread = None
        logger.info("Game-state module stopped")

    def has_stimulus(self) -> bool:
        if not (self._enabled and self._running and not self._version_mismatch):
            return False
        # NANO-123: Boss/chapter events first, then dialogue. No independent
        # gameplay event or snapshot paths.
        if self._has_priority_event_stimulus():
            return True
        return self._has_dialogue_stimulus()

    def get_stimulus(self) -> Optional[StimulusData]:
        if not (self._enabled and self._running and not self._version_mismatch):
            return None
        # NANO-123: Boss/chapter events first, then dialogue
        stim = self._get_priority_event_stimulus()
        if stim:
            return stim
        return self._get_dialogue_stimulus()

    def health_check(self) -> bool:
        return self._connected and not self._version_mismatch

    # -- Dialogue stimulus (NANO-116 B.2) ------------------------------------

    def _has_dialogue_stimulus(self) -> bool:
        if self._dialogue_buffer.count < 1:
            return False
        # NANO-123 Phase 2: Only fire if a line arrived while the engine was idle.
        # When no idle callback is wired (tests, legacy), fall back to Phase 1
        # behavior (any buffered line triggers).
        if self._is_engine_idle is not None:
            return self._has_fresh_trigger
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
        self._has_fresh_trigger = False
        if not lines:
            return None

        if self._dialogue_store:
            for dl in lines:
                self._dialogue_store.record_dialogue_line(dl)

        # NANO-123: Only the latest line goes into the user message.
        # Prior lines are already recorded to DialogueStore and will
        # appear in CHARACTER_KNOWLEDGE through the normal injection path.
        dialogue_block = self._format_dialogue_line(lines[-1])

        # NANO-124: Use barge-in template if this drain was triggered by self-barge-in
        is_barge_in = self._barge_in_triggered
        if is_barge_in:
            template = self._barge_in_template_rotator.select() or self._barge_in_prompt_templates[0]
            self._barge_in_triggered = False
            self._barge_in_arrival_count = 0
        else:
            template = self._template_rotator.select() or self._dialogue_prompt_templates[0]
        user_input = template.format(dialogue=dialogue_block)

        # NANO-123: Bundle current snapshot during combat
        latest_ctx = lines[-1].gameplay_context
        if latest_ctx.combat_active and self._current_snapshot is not None:
            snapshot_report = self._format_snapshot_report(self._current_snapshot)
            if snapshot_report:
                user_input = user_input.rstrip() + "\n\n" + snapshot_report

        print(f"[GameState] Dialogue drain: {len(lines)} lines, input_len={len(user_input)}", flush=True)

        return StimulusData(
            source=StimulusSource.GAME_STATE,
            user_input=user_input,
            metadata={
                "stimulus_type": "dialogue_barge_in" if is_barge_in else "dialogue",
                "event_count": len(lines),
                "dialogue_lines": len(lines),
                "game_id": "pragmata",
                "combat_snapshot_bundled": latest_ctx.combat_active and self._current_snapshot is not None,
                "barge_in": is_barge_in,
                "weight": 2.0,
            },
        )

    # -- NANO-124: Self-barge-in probability -----------------------------------

    def _roll_barge_in(self) -> bool:
        """Roll the two-layer probability check for self-barge-in.

        Layer 1 (escalating): each arrival during TTS increases chance.
        Layer 2 (fatigue): each actual barge-in dampens the curve.
        Arrival count increments regardless of roll outcome.

        Returns True if the roll succeeds and TTS should be interrupted.
        """
        arrival_idx = min(self._barge_in_arrival_count, len(self._barge_in_escalation) - 1)
        base_prob = self._barge_in_escalation[arrival_idx]

        fatigue_idx = min(self._barge_in_count, len(self._barge_in_fatigue) - 1)
        fatigue_mult = self._barge_in_fatigue[fatigue_idx]

        final_prob = base_prob * fatigue_mult

        self._barge_in_arrival_count += 1

        roll = random.random()
        success = roll < final_prob

        logger.debug(
            "[NANO-124] Barge-in roll: arrival=%d, base=%.0f%%, "
            "fatigue=%d, mult=%.0f%%, final=%.1f%%, roll=%.3f → %s",
            arrival_idx, base_prob * 100,
            fatigue_idx, fatigue_mult * 100,
            final_prob * 100, roll,
            "BARGE-IN" if success else "drop",
        )

        return success

    def _reset_barge_in_state(self) -> None:
        """Reset both probability layers. Called when TTS completes naturally."""
        self._barge_in_arrival_count = 0
        self._barge_in_count = 0
        self._barge_in_triggered = False
        logger.debug("[NANO-124] Barge-in state reset (TTS completed)")

    def on_tts_completed(self) -> None:
        """EventBus callback: TTS finished without interruption."""
        self._reset_barge_in_state()

    # -- Priority event stimulus (NANO-123) -----------------------------------
    # Boss start/end and chapter transitions fire independently with no
    # batch window, no probability gate, and no delay. Checked before dialogue.

    def _has_priority_event_stimulus(self) -> bool:
        if not self._gameplay_enabled:
            return False
        return any(
            ev.event_type in _PRIORITY_EVENT_TYPES
            for ev in self._gameplay_event_buffer
        )

    def _get_priority_event_stimulus(self) -> Optional[StimulusData]:
        if not self._gameplay_enabled:
            return None
        priority_events = [
            ev for ev in self._gameplay_event_buffer
            if ev.event_type in _PRIORITY_EVENT_TYPES
        ]
        if not priority_events:
            return None

        # Pop only priority events; leave non-priority events in buffer
        self._gameplay_event_buffer = [
            ev for ev in self._gameplay_event_buffer
            if ev.event_type not in _PRIORITY_EVENT_TYPES
        ]
        if not self._gameplay_event_buffer:
            self._gameplay_first_event_time = None

        lines = []
        for ev in priority_events:
            line = self._format_gameplay_event(ev)
            if line:
                lines.append(line)

        if not lines:
            return None

        events_block = "\n".join(lines)
        user_input = self._prompt_template.format(events=events_block)

        print(
            f"[GameState] Priority event fired: {len(priority_events)} events, "
            f"input_len={len(user_input)}",
            flush=True,
        )

        return StimulusData(
            source=StimulusSource.GAME_STATE,
            user_input=user_input,
            metadata={
                "stimulus_type": "priority_event",
                "event_count": len(priority_events),
                "game_id": "pragmata",
                "weight": 5.0,
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

    # -- Snapshot formatting (used by combat chatter bundling, NANO-123) ------
    # Independent snapshot stimulus path killed in NANO-123. The accumulation
    # logic (_current_snapshot updates from bridge) stays — _format_snapshot_report
    # is called from _get_dialogue_stimulus when combat_active is True.

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
        pass

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
                # NANO-123 Phase 2: Only mark as trigger if engine is idle.
                # Lines arriving while busy are context-only (DialogueStore
                # gets them on next drain, CHARACTER_KNOWLEDGE stays fed).
                # NANO-124: If barge-in is enabled and engine is busy, roll
                # probability — on success, trigger TTS stop and mark for
                # barge-in template.
                if event_type == "dialogue_line":
                    if self._is_engine_idle is None or self._is_engine_idle():
                        self._has_fresh_trigger = True
                    elif self._barge_in_enabled and self._trigger_barge_in is not None:
                        if self._roll_barge_in():
                            self._barge_in_count += 1
                            self._barge_in_triggered = True
                            self._has_fresh_trigger = True
                            self._trigger_barge_in()
                            logger.info(
                                "[NANO-124] Self-barge-in triggered "
                                "(arrival=%d, total_barges=%d)",
                                self._barge_in_arrival_count,
                                self._barge_in_count,
                            )

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
