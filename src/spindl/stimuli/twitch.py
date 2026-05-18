"""
Twitch chat + EventSub stimulus module.

Connects to a Twitch channel via IRC (twitchAPI library) and buffers
incoming chat messages as stimuli. Priority 50 — fires above PATIENCE
but below custom injection.

NANO-132: EventSub WebSocket for platform events (follows, etc.).
Runs alongside IRC chat in the same async event loop. Follow events
use a 3-second accumulation window, bypass the selection pass, and
fire at weight 5.0 with deterministic self-barge-in.

Auth: OAuth2 App Flow via twitchAPI UserAuthenticator. First run opens
a browser for authorization; library handles token refresh thereafter.
"""

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .base import StimulusModule
from .models import StimulusData, StimulusSource
from .twitch_selector import TwitchSelector

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = (
    "**A viewer just said something in Twitch chat.**\n"
    "\n"
    "{messages}\n"
)

_LEGACY_BATCH_PROMPT_TEMPLATE = (
    "**You just received new messages in Twitch chat.** "
    "Reply as co-host \u2014 natural, in character, one unified response. "
    "Ignore anything off-topic or spammy.\n"
    "\n"
    "```chat\n"
    "{messages}\n"
    "```"
)

_TRUNCATION_MARKER = "..."


def format_twitch_timestamp(ms: int) -> str:
    """Format a Unix-millis timestamp as mmddyyyy-hh-mm-ss:ms (local time)."""
    if ms <= 0:
        return ""
    seconds, millis = divmod(int(ms), 1000)
    dt = datetime.fromtimestamp(seconds)
    return dt.strftime("%m%d%Y-%H-%M-%S") + f":{millis:03d}"


@dataclass
class TwitchMessage:
    """A single buffered Twitch chat message."""

    username: str
    text: str
    channel: str = ""
    sent_timestamp_ms: int = 0


@dataclass
class TwitchFollowEntry:
    """A buffered follow event from EventSub."""

    user_id: str
    user_login: str
    user_name: str
    timestamp: float = 0.0


_FOLLOW_ACCUMULATION_WINDOW = 3.0
_FOLLOW_BUFFER_CAP = 10
_FOLLOW_WEIGHT = 5.0


def _format_follow_names(names: list[str]) -> str:
    """Format a list of display names for the follow prompt."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    if len(names) == 3:
        return f"{names[0]}, {names[1]}, and {names[2]}"
    rest = len(names) - 3
    return f"{names[0]}, {names[1]}, {names[2]}, and {rest} others"


class TwitchModule(StimulusModule):
    """
    Twitch chat + EventSub stimulus source.

    Connects to a single Twitch channel, buffers incoming chat messages,
    and exposes them as stimuli for the StimuliEngine. Messages are
    filtered by length via _should_accept() — designed to be extensible
    for future spam detection, subscriber priority, keyword filtering, etc.

    NANO-132: EventSub WebSocket for platform events (follows). Runs
    alongside IRC chat. Follow events use a 3-second accumulation window
    and fire at weight 5.0 with deterministic self-barge-in.

    Priority 50 — external integration tier.
    """

    def __init__(
        self,
        channel: str = "",
        app_id: str = "",
        app_secret: str = "",
        buffer_size: int = 10,
        max_message_length: int = 300,
        prompt_template: Optional[str] = None,
        char_cap: int = 150,
        enabled: bool = False,
        on_message_accepted: Optional["Callable[[str, str, str, int], None]"] = None,
        max_message_age_seconds: float = 15.0,
        selection_mode: str = "llm",
        selection_pass_model: str = "",
        selection_pass_api_key: str = "",
        events_enabled: bool = False,
        is_engine_idle: Optional["Callable[[], bool]"] = None,
        trigger_barge_in: Optional["Callable[[], None]"] = None,
    ):
        self._channel = channel
        self._app_id = app_id
        self._app_secret = app_secret
        self._buffer: deque[TwitchMessage] = deque(maxlen=max(1, buffer_size))
        self._buffer_size = max(1, buffer_size)
        self._max_message_length = max_message_length
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        self._char_cap = max(50, min(500, char_cap))
        self._enabled = enabled
        self._on_message_accepted = on_message_accepted

        # NANO-130: Staleness filter + selection pass
        self._max_message_age_seconds = max(1.0, min(120.0, max_message_age_seconds))
        self._selection_mode = selection_mode if selection_mode in ("llm", "heuristic") else "llm"
        self._selector = TwitchSelector(
            api_key=selection_pass_api_key,
            model=selection_pass_model,
        )

        # NANO-132: EventSub follow events
        self._events_enabled = events_enabled
        self._follow_buffer: list[TwitchFollowEntry] = []
        self._first_follow_time: float = 0.0
        self._has_follow_trigger = False
        self._is_engine_idle = is_engine_idle
        self._trigger_barge_in = trigger_barge_in
        self._eventsub = None
        self._events_connected = False
        self._broadcaster_id: Optional[str] = None

        self._connected = False
        self._running = False
        self._twitch = None
        self._chat = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

    # ── StimulusModule interface ─────────────────────────────────────

    @property
    def name(self) -> str:
        return "twitch"

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

    @property
    def channel(self) -> str:
        return self._channel

    @channel.setter
    def channel(self, value: str) -> None:
        self._channel = value

    @property
    def app_id(self) -> str:
        return self._app_id

    @app_id.setter
    def app_id(self, value: str) -> None:
        self._app_id = value

    @property
    def app_secret(self) -> str:
        return self._app_secret

    @app_secret.setter
    def app_secret(self, value: str) -> None:
        self._app_secret = value

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self._buffer_size = max(1, value)
        # Rebuild deque with new maxlen (preserves recent messages)
        old_messages = list(self._buffer)
        self._buffer = deque(old_messages, maxlen=self._buffer_size)

    @property
    def max_message_length(self) -> int:
        return self._max_message_length

    @max_message_length.setter
    def max_message_length(self, value: int) -> None:
        self._max_message_length = value

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value

    @property
    def char_cap(self) -> int:
        return self._char_cap

    @char_cap.setter
    def char_cap(self, value: int) -> None:
        self._char_cap = max(50, min(500, value))

    @property
    def max_message_age_seconds(self) -> float:
        return self._max_message_age_seconds

    @max_message_age_seconds.setter
    def max_message_age_seconds(self, value: float) -> None:
        self._max_message_age_seconds = max(1.0, min(120.0, value))

    @property
    def selection_mode(self) -> str:
        return self._selection_mode

    @selection_mode.setter
    def selection_mode(self, value: str) -> None:
        self._selection_mode = value if value in ("llm", "heuristic") else "llm"

    @property
    def selection_pass_model(self) -> str:
        return self._selector.model

    @selection_pass_model.setter
    def selection_pass_model(self, value: str) -> None:
        self._selector.model = value

    @property
    def selection_pass_api_key(self) -> str:
        return self._selector.api_key

    @selection_pass_api_key.setter
    def selection_pass_api_key(self, value: str) -> None:
        self._selector.api_key = value

    @property
    def resolved_app_id(self) -> str:
        """App ID from config, falling back to TWITCH_APP_ID env var."""
        return self._app_id or os.getenv("TWITCH_APP_ID", "")

    @property
    def resolved_app_secret(self) -> str:
        """App secret from config, falling back to TWITCH_APP_SECRET env var."""
        return self._app_secret or os.getenv("TWITCH_APP_SECRET", "")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def events_enabled(self) -> bool:
        return self._events_enabled

    @events_enabled.setter
    def events_enabled(self, value: bool) -> None:
        self._events_enabled = value
        if not value:
            self._follow_buffer.clear()
            self._first_follow_time = 0.0
            self._has_follow_trigger = False

    @property
    def events_connected(self) -> bool:
        return self._events_connected

    @property
    def recent_messages(self) -> list[str]:
        """Formatted message strings for dashboard display."""
        return [f"{m.username}: {m.text}" for m in self._buffer]

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def start(self) -> None:
        if self._running:
            return

        if not self._channel or not self.resolved_app_id or not self.resolved_app_secret:
            logger.warning(
                "Twitch module cannot start: missing channel, app_id, or app_secret. "
                "Set in config or via TWITCH_APP_ID / TWITCH_APP_SECRET env vars."
            )
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_thread, name="TwitchModule", daemon=True
        )
        self._thread.start()
        logger.info("Twitch module started (channel=%s)", self._channel)

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self._connected = False
        self._events_connected = False
        self._buffer.clear()
        self._follow_buffer.clear()
        self._first_follow_time = 0.0
        self._has_follow_trigger = False
        self._twitch = None
        self._chat = None
        self._eventsub = None
        self._broadcaster_id = None
        self._loop = None
        self._thread = None
        logger.info("Twitch module stopped")

    def _count_fresh_messages(self) -> int:
        """Read-only staleness check — count messages that would survive the filter."""
        if not self._buffer:
            return 0
        now_ms = int(time.time() * 1000)
        threshold_ms = int(self._max_message_age_seconds * 1000)
        return sum(
            1 for m in self._buffer
            if m.sent_timestamp_ms <= 0 or (now_ms - m.sent_timestamp_ms) <= threshold_ms
        )

    def _follow_buffer_ready(self) -> bool:
        """Check if the follow accumulation window has closed and events are pending."""
        if not self._follow_buffer:
            return False
        if self._has_follow_trigger:
            return True
        if len(self._follow_buffer) >= _FOLLOW_BUFFER_CAP:
            return True
        if self._first_follow_time > 0 and (time.monotonic() - self._first_follow_time) >= _FOLLOW_ACCUMULATION_WINDOW:
            return True
        return False

    def has_stimulus(self) -> bool:
        return self._enabled and self._running and (
            self._follow_buffer_ready() or self._count_fresh_messages() > 0
        )

    def get_stimulus(self) -> Optional[StimulusData]:
        if not self._enabled or not self._running:
            return None

        # NANO-132: Follow events take priority over chat messages
        if self._follow_buffer_ready():
            return self._drain_follow_buffer()

        if not self._buffer:
            return None

        # Drain buffer and apply staleness filter
        all_messages = list(self._buffer)
        self._buffer.clear()

        now_ms = int(time.time() * 1000)
        threshold_ms = int(self._max_message_age_seconds * 1000)
        fresh = [
            m for m in all_messages
            if m.sent_timestamp_ms <= 0 or (now_ms - m.sent_timestamp_ms) <= threshold_ms
        ]

        if not fresh:
            logger.debug(
                "[Twitch] All %d messages stale (threshold=%.0fs), skipping",
                len(all_messages), self._max_message_age_seconds,
            )
            return None

        stale_count = len(all_messages) - len(fresh)
        if stale_count:
            logger.debug("[Twitch] Dropped %d stale messages", stale_count)

        # Selection pass — pick the single best message or reject all
        candidates = [
            {"username": m.username, "text": m.text, "sent_timestamp_ms": m.sent_timestamp_ms}
            for m in fresh
        ]
        result = self._selector.select(candidates, mode=self._selection_mode)

        if result.selected_index is None:
            logger.info(
                "[Twitch] Selection pass rejected all %d candidates "
                "(mode=%s, reason=%s)",
                len(fresh), result.mode, result.reason,
            )
            return None

        winner = fresh[result.selected_index]
        logger.info(
            "[Twitch] Selected message %d/%d from %s (mode=%s, reason=%s)",
            result.selected_index + 1, len(fresh),
            winner.username, result.mode, result.reason,
        )

        # Format the single selected message
        cap = self._char_cap
        text = winner.text
        if len(text) > cap:
            text = text[:cap] + _TRUNCATION_MARKER

        formatted = f"{winner.username}: {text}"

        template = self._prompt_template or _DEFAULT_PROMPT_TEMPLATE
        if "{messages}" not in template:
            logger.warning(
                "twitch_prompt_template missing {messages} placeholder; "
                "falling back to default."
            )
            template = _DEFAULT_PROMPT_TEMPLATE

        user_input = template.format(messages=formatted)

        print(
            f"[Twitch] Selection: {winner.username} from {len(fresh)} fresh "
            f"({stale_count} stale dropped), mode={result.mode}",
            flush=True,
        )

        return StimulusData(
            source=StimulusSource.TWITCH,
            user_input=user_input,
            metadata={
                "message_count": 1,
                "channel": self._channel,
                "messages": [
                    {
                        "username": winner.username,
                        "text": winner.text,
                        "sent_timestamp_ms": winner.sent_timestamp_ms,
                    }
                ],
                "selection": {
                    "mode": result.mode,
                    "reason": result.reason,
                    "candidates": len(fresh),
                    "stale_dropped": stale_count,
                },
                "weight": 1.0,
            },
        )

    def health_check(self) -> bool:
        return self._connected or self._events_connected

    # ── NANO-132: Follow event handling ─────────────────────────────

    def _drain_follow_buffer(self) -> Optional[StimulusData]:
        """Drain accumulated follow events into a single stimulus."""
        if not self._follow_buffer:
            return None

        # Deduplicate by user_id, preserving arrival order
        seen: set[str] = set()
        unique: list[TwitchFollowEntry] = []
        for entry in self._follow_buffer:
            if entry.user_id not in seen:
                seen.add(entry.user_id)
                unique.append(entry)

        self._follow_buffer.clear()
        self._first_follow_time = 0.0
        self._has_follow_trigger = False

        if not unique:
            return None

        names = [e.user_name or e.user_login for e in unique]
        formatted = _format_follow_names(names)
        count = len(unique)

        if count <= 3:
            user_input = f"**{formatted} just followed the channel!** Welcome them by name."
        else:
            user_input = (
                f"**{formatted} just followed the channel!** "
                "Welcome them — name the first few, acknowledge the rest."
            )

        overlay_message = f"✦ {formatted} just followed!"

        print(
            f"[Twitch] Follow event: {count} follower(s) — {formatted}",
            flush=True,
        )

        return StimulusData(
            source=StimulusSource.TWITCH_EVENT,
            user_input=user_input,
            metadata={
                "event_type": "follow",
                "weight": _FOLLOW_WEIGHT,
                "follower_count": count,
                "followers": [
                    {"user_id": e.user_id, "user_login": e.user_login, "user_name": e.user_name}
                    for e in unique
                ],
                "overlay_message": overlay_message,
                "usernames": names,
            },
        )

    def _buffer_follow(self, user_id: str, user_login: str, user_name: str) -> None:
        """Buffer a follow event and handle barge-in if engine is busy."""
        if not self._events_enabled:
            return

        entry = TwitchFollowEntry(
            user_id=user_id,
            user_login=user_login,
            user_name=user_name,
            timestamp=time.monotonic(),
        )

        if not self._follow_buffer:
            self._first_follow_time = time.monotonic()

        self._follow_buffer.append(entry)

        print(
            f"[Twitch] Follow buffered: {user_name} ({user_id}), buffer={len(self._follow_buffer)}",
            flush=True,
        )

        # Check if accumulation window has closed and handle barge-in
        if self._follow_buffer_ready():
            if self._is_engine_idle is None or self._is_engine_idle():
                self._has_follow_trigger = True
            elif self._trigger_barge_in is not None:
                self._has_follow_trigger = True
                self._trigger_barge_in()
                print(f"[NANO-132] Follow barge-in triggered (followers={len(self._follow_buffer)})", flush=True)

    # ── Message filtering (extensibility point) ─────────────────────

    def _should_accept(self, username: str, text: str) -> bool:
        """
        Decide whether to accept a chat message into the buffer.

        v1: length cap only. Designed as an isolated method so future
        versions can add spam detection, subscriber priority, keyword
        filtering, or pluggable filter chains without restructuring.
        """
        if len(text) > self._max_message_length:
            return False
        if len(text.strip()) == 0:
            return False
        return True

    # ── Internal async machinery ────────────────────────────────────

    def _run_thread(self) -> None:
        """Daemon thread entry point — runs the async event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        except Exception:
            logger.exception("Twitch module thread crashed")
        finally:
            self._connected = False
            self._loop.close()

    async def _run_async(self) -> None:
        """Async main — connect to Twitch and listen for chat + EventSub."""
        try:
            from twitchAPI.twitch import Twitch
            from twitchAPI.oauth import UserAuthenticator
            from twitchAPI.type import AuthScope, ChatEvent
            from twitchAPI.chat import Chat, EventData, ChatMessage
        except ImportError:
            logger.error(
                "twitchAPI not installed. Install with: pip install twitchAPI>=4.2.0"
            )
            return

        user_scope = [AuthScope.CHAT_READ]
        if self._events_enabled:
            user_scope.append(AuthScope.MODERATOR_READ_FOLLOWERS)

        try:
            twitch = await Twitch(self.resolved_app_id, self.resolved_app_secret)
            auth = UserAuthenticator(twitch, user_scope, force_verify=self._events_enabled)
            token, refresh_token = await auth.authenticate()
            if self._events_enabled:
                print(
                    "[Twitch] Auth completed with EventSub scopes "
                    "(force_verify=True for moderator:read:followers)",
                    flush=True,
                )
            await twitch.set_user_authentication(token, user_scope, refresh_token)

            # NANO-132: Resolve broadcaster user ID for EventSub subscriptions
            if self._events_enabled:
                try:
                    users = [u async for u in twitch.get_users(logins=[self._channel])]
                    if users:
                        self._broadcaster_id = users[0].id
                        print(
                            f"[Twitch] Resolved broadcaster ID: {self._channel} -> {self._broadcaster_id}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Twitch] Could not resolve broadcaster ID for '{self._channel}' — EventSub disabled",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[Twitch] Failed to resolve broadcaster ID — EventSub disabled: {e}", flush=True)

            chat = await Chat(twitch)
            self._twitch = twitch
            self._chat = chat

            async def on_ready(ready_event: EventData) -> None:
                logger.info("Twitch bot ready, joining #%s", self._channel)
                await ready_event.chat.join_room(self._channel)
                self._connected = True

            async def on_message(msg: ChatMessage) -> None:
                if not self._enabled:
                    return
                if not self._should_accept(msg.user.name, msg.text):
                    return

                sent_ms = getattr(msg, "sent_timestamp", None)
                if sent_ms is None:
                    sent_ms = int(time.time() * 1000)
                else:
                    try:
                        sent_ms = int(sent_ms)
                    except (TypeError, ValueError):
                        sent_ms = int(time.time() * 1000)

                self._buffer.append(
                    TwitchMessage(
                        username=msg.user.name,
                        text=msg.text,
                        channel=msg.room.name,
                        sent_timestamp_ms=sent_ms,
                    )
                )
                if self._on_message_accepted:
                    try:
                        self._on_message_accepted(
                            msg.user.name, msg.text, msg.room.name, sent_ms
                        )
                    except Exception:
                        logger.exception("on_message_accepted callback failed")
                logger.debug(
                    "Twitch [#%s] %s: %s", msg.room.name, msg.user.name, msg.text
                )

            chat.register_event(ChatEvent.READY, on_ready)
            chat.register_event(ChatEvent.MESSAGE, on_message)
            chat.start()

            # NANO-132: Start EventSub WebSocket for platform events
            if self._events_enabled and self._broadcaster_id:
                try:
                    from twitchAPI.eventsub.websocket import EventSubWebsocket

                    eventsub = EventSubWebsocket(twitch)
                    eventsub.start()

                    async def on_follow(event) -> None:
                        data = event.event
                        self._buffer_follow(
                            user_id=data.user_id,
                            user_login=data.user_login,
                            user_name=data.user_name,
                        )

                    await eventsub.listen_channel_follow_v2(
                        broadcaster_user_id=self._broadcaster_id,
                        moderator_user_id=self._broadcaster_id,
                        callback=on_follow,
                    )

                    self._eventsub = eventsub
                    self._events_connected = True
                    print(
                        f"[Twitch] EventSub WebSocket started (channel.follow subscribed, broadcaster={self._broadcaster_id})",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[Twitch] EventSub WebSocket failed to start — chat still active: {e}", flush=True)
                    self._events_connected = False
                    self._events_connected = False

            # Block until stop is requested
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)

            # Graceful shutdown inside the async context (before loop closes)
            await self._shutdown_async()

        except Exception:
            logger.exception("Twitch connection failed")
            self._connected = False

    async def _shutdown_async(self) -> None:
        """Graceful shutdown of twitchAPI resources."""
        try:
            if self._eventsub:
                await self._eventsub.stop()
                self._events_connected = False
            if self._chat:
                self._chat.stop()
            if self._twitch:
                await self._twitch.close()
        except Exception:
            logger.exception("Error during Twitch shutdown")
