"""
Twitch chat stimulus module.

Connects to a Twitch channel via IRC (twitchAPI library) and buffers
incoming chat messages as stimuli. Priority 50 — fires above PATIENCE
but below custom injection.

Auth: OAuth2 App Flow via twitchAPI UserAuthenticator. First run opens
a browser for authorization; library handles token refresh thereafter.
"""

import asyncio
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .base import StimulusModule
from .models import StimulusData, StimulusSource

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = (
    "Recent Twitch chat messages:\n"
    "{messages}\n"
    "Pick the most interesting message and respond to it naturally."
)


@dataclass
class TwitchMessage:
    """A single buffered Twitch chat message."""

    username: str
    text: str
    channel: str = ""


class TwitchModule(StimulusModule):
    """
    Twitch chat stimulus source.

    Connects to a single Twitch channel, buffers incoming chat messages,
    and exposes them as stimuli for the StimuliEngine. Messages are
    filtered by length via _should_accept() — designed to be extensible
    for future spam detection, subscriber priority, keyword filtering, etc.

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
        enabled: bool = False,
    ):
        self._channel = channel
        self._app_id = app_id
        self._app_secret = app_secret
        self._buffer: deque[TwitchMessage] = deque(maxlen=max(1, buffer_size))
        self._buffer_size = max(1, buffer_size)
        self._max_message_length = max_message_length
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        self._enabled = enabled

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
        self._buffer.clear()
        self._twitch = None
        self._chat = None
        self._loop = None
        self._thread = None
        logger.info("Twitch module stopped")

    def has_stimulus(self) -> bool:
        return self._enabled and self._running and len(self._buffer) > 0

    def get_stimulus(self) -> Optional[StimulusData]:
        if not self.has_stimulus():
            return None

        # Drain the entire buffer into a single stimulus
        messages = list(self._buffer)
        self._buffer.clear()

        formatted = "\n".join(f"{m.username}: {m.text}" for m in messages)
        twitch_content = self._prompt_template.format(messages=formatted)

        return StimulusData(
            source=StimulusSource.TWITCH,
            user_input="Respond to the Twitch chat messages.",
            metadata={
                "message_count": len(messages),
                "channel": self._channel,
                "messages": [
                    {"username": m.username, "text": m.text} for m in messages
                ],
                "twitch_content": twitch_content,
            },
        )

    def health_check(self) -> bool:
        return self._connected

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
        """Async main — connect to Twitch and listen for chat."""
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

        try:
            twitch = await Twitch(self.resolved_app_id, self.resolved_app_secret)
            auth = UserAuthenticator(twitch, user_scope)
            token, refresh_token = await auth.authenticate()
            await twitch.set_user_authentication(token, user_scope, refresh_token)

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

                self._buffer.append(
                    TwitchMessage(
                        username=msg.user.name,
                        text=msg.text,
                        channel=msg.room.name,
                    )
                )
                logger.debug(
                    "Twitch [#%s] %s: %s", msg.room.name, msg.user.name, msg.text
                )

            chat.register_event(ChatEvent.READY, on_ready)
            chat.register_event(ChatEvent.MESSAGE, on_message)
            chat.start()

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
            if self._chat:
                self._chat.stop()
            if self._twitch:
                await self._twitch.close()
        except Exception:
            logger.exception("Error during Twitch shutdown")
