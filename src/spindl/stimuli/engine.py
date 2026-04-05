"""
Stimuli Engine — the Prompter equivalent for spindl.

Runs as a daemon thread, monitoring registered StimulusModules
and firing the highest-priority stimulus through the existing
process_text_input() path when the agent is idle.

Event-driven with a periodic 1-second PATIENCE check:
- Subscribes to STATE_CHANGED, RESPONSE_READY, TTS_COMPLETED
  to track activity and wake immediately on state transitions.
- Checks PATIENCE every 1 second via threading.Event timeout.
"""

import threading
import time
from typing import Callable, Optional, TYPE_CHECKING

from ..core.events import (
    EventType,
    Event,
    StateChangedEvent,
    StimulusFiredEvent,
)

from .base import StimulusModule
from .models import StimulusData

if TYPE_CHECKING:
    from ..core import AudioStateMachine, AgentState
    from ..core.event_bus import EventBus
    from ..orchestrator.callbacks import OrchestratorCallbacks

import logging

logger = logging.getLogger(__name__)


class StimuliEngine:
    """
    Central stimuli decision engine.

    Polls registered modules for pending stimuli, gates on agent state
    (must be LISTENING and not processing), and fires the highest-priority
    stimulus through OrchestratorCallbacks.process_text_input().

    Thread model:
        - Runs as a single daemon thread
        - Wakes on EventBus events or 1-second timeout (for PATIENCE)
        - All module access is guarded by _modules_lock
    """

    def __init__(
        self,
        state_machine: "AudioStateMachine",
        callbacks: "OrchestratorCallbacks",
        event_bus: "EventBus",
        enabled: bool = False,
        is_speaking: Optional["Callable[[], bool]"] = None,
    ):
        self._state_machine = state_machine
        self._callbacks = callbacks
        self._event_bus = event_bus
        self._enabled = enabled
        self._is_speaking = is_speaking

        self._modules: list[StimulusModule] = []
        self._modules_lock = threading.Lock()

        self._running = False
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # EventBus subscription IDs for cleanup
        self._sub_ids: list[int] = []

        # Cooldown: minimum seconds between stimulus fires
        self._cooldown_seconds = 2.0
        self._last_fire_time = 0.0

        # Loop interval: how long to wait between checks (1s for production, lower for tests)
        self._loop_interval = 1.0

        # User typing flag: suppresses stimulus firing while user is composing
        self._user_typing = False

        # Track previous playback state for pause/resume transitions
        self._was_speaking = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def user_typing(self) -> bool:
        """Whether the user is actively typing in the text input."""
        return self._user_typing

    @user_typing.setter
    def user_typing(self, value: bool) -> None:
        self._user_typing = value
        # Pause/resume PATIENCE timer directly
        with self._modules_lock:
            for module in self._modules:
                if hasattr(module, "pause") and hasattr(module, "resume"):
                    if value:
                        module.pause()
                    elif not self.is_blocked_by_playback:
                        # Only resume if playback isn't also blocking
                        module.resume()

    @property
    def is_blocked_by_playback(self) -> bool:
        """Whether stimulus firing is currently blocked by active audio playback."""
        return self._is_speaking is not None and self._is_speaking()

    @property
    def is_blocked_by_typing(self) -> bool:
        """Whether stimulus firing is currently blocked by user typing."""
        return self._user_typing

    @property
    def modules(self) -> list[StimulusModule]:
        """Snapshot of registered modules (sorted by priority, descending)."""
        with self._modules_lock:
            return sorted(self._modules, key=lambda m: m.priority, reverse=True)

    def register_module(self, module: StimulusModule) -> None:
        """Register a stimulus module with the engine."""
        with self._modules_lock:
            # Prevent duplicate registration
            for existing in self._modules:
                if existing.name == module.name:
                    logger.warning(
                        "Module '%s' already registered, skipping", module.name
                    )
                    return
            self._modules.append(module)
            logger.info(
                "Registered module '%s' (priority=%d)", module.name, module.priority
            )

    def unregister_module(self, name: str) -> bool:
        """
        Unregister a module by name. Stops it first.

        Returns True if found and removed.
        """
        with self._modules_lock:
            for i, module in enumerate(self._modules):
                if module.name == name:
                    module.stop()
                    self._modules.pop(i)
                    logger.info("Unregistered module '%s'", name)
                    return True
        return False

    def start(self) -> None:
        """Start the engine daemon thread and subscribe to events."""
        if self._running:
            return

        self._subscribe_events()

        # Start all enabled modules
        with self._modules_lock:
            for module in self._modules:
                if module.enabled:
                    try:
                        module.start()
                    except Exception as e:
                        logger.error(
                            "Failed to start module '%s': %s", module.name, e
                        )

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="StimuliEngine", daemon=True
        )
        self._thread.start()
        logger.info("StimuliEngine started (enabled=%s)", self._enabled)

    def stop(self) -> None:
        """Stop the engine and all modules."""
        if not self._running:
            return

        self._running = False
        self._wake_event.set()  # Wake thread so it exits

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Stop all modules
        with self._modules_lock:
            for module in self._modules:
                try:
                    module.stop()
                except Exception as e:
                    logger.error("Failed to stop module '%s': %s", module.name, e)

        self._unsubscribe_events()
        logger.info("StimuliEngine stopped")

    def reset_activity(self) -> None:
        """
        Reset activity timers on all modules.

        Called when any interaction occurs (voice, text, or stimulus response).
        """
        with self._modules_lock:
            for module in self._modules:
                if hasattr(module, "reset_activity"):
                    module.reset_activity()

    def get_module_status(self) -> list[dict]:
        """
        Get status of all registered modules for dashboard display.

        Returns list of dicts with name, priority, enabled, has_stimulus, healthy.
        """
        with self._modules_lock:
            result = []
            for module in sorted(
                self._modules, key=lambda m: m.priority, reverse=True
            ):
                result.append(
                    {
                        "name": module.name,
                        "priority": module.priority,
                        "enabled": module.enabled,
                        "has_stimulus": module.has_stimulus() if module.enabled else False,
                        "healthy": module.health_check(),
                    }
                )
            return result

    # -- Event subscriptions --

    def _subscribe_events(self) -> None:
        """Subscribe to EventBus events for activity tracking."""
        sub = self._event_bus.subscribe(
            EventType.RESPONSE_READY, self._on_activity_event
        )
        self._sub_ids.append(sub)

        sub = self._event_bus.subscribe(
            EventType.TTS_COMPLETED, self._on_activity_event
        )
        self._sub_ids.append(sub)

        sub = self._event_bus.subscribe(
            EventType.STATE_CHANGED, self._on_state_changed
        )
        self._sub_ids.append(sub)

        sub = self._event_bus.subscribe(
            EventType.TRANSCRIPTION_READY, self._on_activity_event
        )
        self._sub_ids.append(sub)

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from all EventBus events."""
        for sub_id in self._sub_ids:
            self._event_bus.unsubscribe(sub_id)
        self._sub_ids.clear()

    def _on_activity_event(self, event: Event) -> None:
        """Any activity resets PATIENCE timers."""
        self.reset_activity()

    def _on_state_changed(self, event: Event) -> None:
        """Wake the engine when transitioning to LISTENING."""
        self.reset_activity()
        if isinstance(event, StateChangedEvent) and event.to_state == "listening":
            self._wake_event.set()

    # -- Main loop --

    def _run_loop(self) -> None:
        """
        Main engine loop.

        Sleeps with a 1-second timeout (for PATIENCE polling).
        Wakes immediately on EventBus events via _wake_event.
        """
        logger.info("StimuliEngine loop started")

        while self._running:
            # Sleep with wake capability
            self._wake_event.wait(timeout=self._loop_interval)
            self._wake_event.clear()

            if not self._running:
                break

            # Detect playback state transitions → pause/resume modules
            self._check_playback_transitions()

            if not self._enabled:
                continue

            if not self._should_fire():
                continue

            stimulus = self._select_stimulus()
            if stimulus:
                self._fire(stimulus)

        logger.info("StimuliEngine loop exited")

    def _check_playback_transitions(self) -> None:
        """
        Detect when audio playback starts/stops and pause/resume modules.

        This mirrors the user_typing setter's pause/resume behavior so that
        get_progress() returns frozen values during playback, not just during
        typing. Without this, the PATIENCE bar label shows "PAUSED" but the
        timer and progress bar keep running.
        """
        speaking_now = self._is_speaking is not None and self._is_speaking()

        if speaking_now and not self._was_speaking:
            # Playback just started — pause modules
            with self._modules_lock:
                for module in self._modules:
                    if hasattr(module, "pause"):
                        module.pause()
        elif not speaking_now and self._was_speaking:
            # Playback just ended — resume modules (only if not typing)
            if not self._user_typing:
                with self._modules_lock:
                    for module in self._modules:
                        if hasattr(module, "resume"):
                            module.resume()

        self._was_speaking = speaking_now

    def _should_fire(self) -> bool:
        """
        Decision function — can we fire a stimulus right now?

        Gates:
            1. Engine must be enabled
            2. Agent must be IDLE or LISTENING (not speaking/processing)
            3. Callbacks must not be processing
            4. Audio playback must not be active (text-path TTS doesn't
               transition the state machine to SYSTEM_SPEAKING, so we
               need an explicit playback check)
            5. User must not be actively typing in the text input
            6. Cooldown period must have elapsed
        """
        # Import here to avoid circular import at module level
        from ..core import AgentState

        current_state = self._state_machine.state
        # Allow firing when IDLE (paused/text-only) or LISTENING (voice mode).
        # Block during USER_SPEAKING, PROCESSING, SYSTEM_SPEAKING.
        if current_state not in (AgentState.IDLE, AgentState.LISTENING):
            return False

        if self._callbacks.is_processing:
            return False

        # Block during audio playback (covers text-path TTS where state
        # machine stays in LISTENING/IDLE instead of SYSTEM_SPEAKING)
        if self._is_speaking is not None and self._is_speaking():
            return False

        # Block while user is composing a message in the text input
        if self._user_typing:
            return False

        # Cooldown check
        if (time.monotonic() - self._last_fire_time) < self._cooldown_seconds:
            return False
        return True

    def _select_stimulus(self) -> Optional[StimulusData]:
        """
        Select the highest-priority stimulus from registered modules.

        Iterates modules sorted by priority (descending). First module
        with a pending stimulus wins.
        """
        with self._modules_lock:
            candidates = sorted(
                self._modules, key=lambda m: m.priority, reverse=True
            )

        for module in candidates:
            if not module.enabled:
                continue
            try:
                if module.has_stimulus():
                    stimulus = module.get_stimulus()
                    if stimulus:
                        logger.info(
                            "[Stimuli] %s fired (priority=%d)",
                            module.name.upper(),
                            module.priority,
                        )
                        return stimulus
            except Exception as e:
                logger.error(
                    "Module '%s' error in has_stimulus/get_stimulus: %s",
                    module.name,
                    e,
                )

        return None

    def _fire(self, stimulus: StimulusData) -> None:
        """
        Fire a stimulus through the existing text input path.

        Emits a STIMULUS_FIRED event and calls process_text_input()
        on the orchestrator callbacks.
        """
        self._last_fire_time = time.monotonic()
        elapsed = stimulus.metadata.get("elapsed_seconds", 0.0)
        print(
            f"[Stimuli] {stimulus.source.value.upper()} fired "
            f"(elapsed={elapsed:.1f}s)"
        )

        # Emit event for GUI/logging
        self._event_bus.emit(
            StimulusFiredEvent(
                source=stimulus.source.value,
                prompt_text=stimulus.user_input[:200],
                elapsed_seconds=stimulus.metadata.get("elapsed_seconds", 0.0),
            )
        )

        # Fire through existing text input path
        twitch_content = stimulus.metadata.get("twitch_content", "")
        print(f"[Stimuli] metadata keys={list(stimulus.metadata.keys())}, twitch_content_len={len(twitch_content)}", flush=True)
        try:
            self._callbacks.process_text_input(
                stimulus.user_input,
                skip_tts=False,
                stimulus_source=stimulus.source.value,
                stimulus_metadata=stimulus.metadata,
            )
        except Exception as e:
            logger.error("[Stimuli] Failed to fire stimulus: %s", e)
