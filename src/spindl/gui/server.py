"""
Socket.IO server for the spindl GUI.

Provides real-time communication between the orchestrator and web frontend.
Supports two modes:
1. Attached mode: Connected to running orchestrator (normal operation)
2. Standalone mode: No orchestrator, used for GUI-first configuration (NANO-027)

Responsibilities:
- AI Runtime Events: State changes, transcription, response, TTS, tool invocation
- Service Lifecycle: Launch, shutdown, health monitoring
- Session Management: Load, resume, delete conversation sessions
- Configuration: VAD settings, pipeline config, persona switching
- Text Input: Direct message injection (bypassing STT)

NOT handled here (use Next.js API routes instead):
- Character CRUD: /api/characters/* (NANO-035 Phase 1)
- Codex CRUD: /api/codex/* (NANO-035 Phase 2)
- Character Import/Export: /api/characters/import, /api/characters/export
"""

import asyncio
import os
import subprocess
import threading
import time
from typing import Optional, Callable, Awaitable, Union, TYPE_CHECKING
from pathlib import Path

import socketio

from spindl.characters.loader import CharacterLoader
from spindl.launcher.service_runner import kill_process_tree

if TYPE_CHECKING:
    from spindl.orchestrator.voice_agent import VoiceAgentOrchestrator
    from spindl.launcher import ServiceRunner, LogAggregator


class GUIServer:
    """
    Socket.IO server for GUI communication.

    Runs alongside the orchestrator, exposing real-time events
    and accepting commands from the web frontend.

    Note: Character and Codex management has been moved to Next.js API routes
    (NANO-035). This server now focuses exclusively on:
    - AI pipeline events (transcription, response, TTS, tools)
    - Service lifecycle (launch, shutdown, health)
    - Session management (resume, delete)
    - Runtime configuration (VAD, pipeline settings)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        cors_allowed_origins: str = "*",
        config_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self._config_path = config_path  # For YAML persistence

        # Pre-launch config paths (NANO-048): parsed from YAML at init so
        # read-only handlers (_emit_sessions, _emit_personas) can serve data
        # before the orchestrator is attached.
        self._conversations_dir: Optional[str] = None
        self._personas_dir: Optional[str] = None
        self._prompt_blocks_config: Optional[dict] = None  # NANO-064b
        self._tools_config_cache: Optional[dict] = None  # NANO-065a
        self._llm_config_cache: Optional[dict] = None  # NANO-065b
        self._vlm_config_cache: Optional[dict] = None  # NANO-065c
        self._stimuli_config_cache: Optional[dict] = None  # NANO-056b
        if config_path:
            self._parse_config_paths(config_path)

        # Create async Socket.IO server
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors_allowed_origins,
            logger=False,
            engineio_logger=False,
        )

        # ASGI app for serving
        self.app = socketio.ASGIApp(self.sio)

        # Reference to orchestrator (set via attach)
        self._orchestrator: Optional["VoiceAgentOrchestrator"] = None

        # Track connected clients
        self._clients: set[str] = set()

        # Service launcher state (NANO-027 Phase 3)
        self._service_runner: Optional["ServiceRunner"] = None
        self._log_aggregator: Optional["LogAggregator"] = None
        self._launch_in_progress: bool = False
        self._launched_services: set[str] = set()

        # Shutdown state (NANO-028)
        self._shutdown_in_progress: bool = False

        # NANO-097: Avatar process management
        self._avatar_process: Optional[subprocess.Popen] = None
        self._avatar_spawned_by_us: bool = False
        self._avatar_clients: set[str] = set()  # SIDs of avatar renderer clients

        # NANO-100: Subtitle process management
        self._subtitle_process: Optional[subprocess.Popen] = None
        self._subtitle_spawned_by_us: bool = False

        # NANO-110: Stream Deck process management
        self._stream_deck_process: Optional[subprocess.Popen] = None
        self._stream_deck_spawned_by_us: bool = False

        # Callback for when services are ready (standalone mode)
        # Can be sync or async callable
        self._on_services_ready: Optional[Callable[[], Union[None, Awaitable[None]]]] = None

        # Event loop reference for cross-thread async scheduling
        # Must be set from the uvicorn server context before EventBridge can emit
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Uvicorn server reference for graceful shutdown (NANO-028)
        self._uvicorn_server: Optional["uvicorn.Server"] = None

        # Register event handlers
        self._register_handlers()

    def _parse_config_paths(self, config_path: str) -> None:
        """
        Parse directory paths from YAML config for pre-launch data access (NANO-048).

        Extracts conversations_dir and personas_dir so that read-only handlers
        can serve data before the orchestrator is attached. Relative paths are
        resolved against the project root (config file's grandparent directory,
        since config lives at <project>/config/spindl.yaml).
        """
        import yaml

        try:
            cfg_path = Path(config_path)
            if not cfg_path.exists():
                return

            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Project root = config file's grandparent (config/spindl.yaml → project/)
            project_root = cfg_path.parent.parent

            # conversations_dir: pipeline.conversations_dir
            pipeline = data.get("pipeline", {})
            conv_dir = pipeline.get("conversations_dir", "./conversations")
            conv_path = Path(conv_dir)
            if not conv_path.is_absolute():
                conv_path = project_root / conv_path
            self._conversations_dir = str(conv_path)

            # personas_dir: persona.directory (legacy) or character.directory
            character = data.get("character", {})
            persona = data.get("persona", {})
            personas_dir = character.get("directory") or persona.get("directory", "./personas")
            personas_path = Path(personas_dir)
            if not personas_path.is_absolute():
                personas_path = project_root / personas_path
            self._personas_dir = str(personas_path)

            # prompt_blocks: top-level prompt_blocks section (NANO-064b)
            prompt_blocks = data.get("prompt_blocks")
            if prompt_blocks and isinstance(prompt_blocks, dict):
                self._prompt_blocks_config = prompt_blocks

            # tools: top-level tools section (NANO-065a)
            tools_section = data.get("tools")
            if tools_section and isinstance(tools_section, dict):
                self._tools_config_cache = tools_section

            # llm: provider and model info (NANO-065b)
            llm_section = data.get("llm", {})
            if llm_section and isinstance(llm_section, dict):
                provider_name = llm_section.get("provider", "llama")
                providers = llm_section.get("providers", {})
                provider_cfg = providers.get(provider_name, {})
                self._llm_config_cache = {
                    "provider": provider_name,
                    "model": provider_cfg.get("model") or provider_cfg.get("model_path", ""),
                    "context_size": provider_cfg.get("context_size"),
                    "providers": providers,
                }

            # vlm: provider and available providers (NANO-065c)
            vlm_section = data.get("vlm", {})
            if vlm_section and isinstance(vlm_section, dict):
                self._vlm_config_cache = {
                    "provider": vlm_section.get("provider", "llama"),
                    "providers": vlm_section.get("providers", {}),
                }

            # stimuli: pre-launch cache for Settings page hydration (NANO-056b)
            stimuli_section = data.get("stimuli", {})
            if stimuli_section and isinstance(stimuli_section, dict):
                from ..orchestrator.config import StimuliConfig
                parsed = StimuliConfig.from_dict(stimuli_section)
                self._stimuli_config_cache = GUIServer._build_stimuli_hydration(parsed)

            print(f"[GUI] Pre-launch paths: conversations={self._conversations_dir}, personas={self._personas_dir}", flush=True)

        except Exception as e:
            print(f"[GUI] Failed to parse config paths for pre-launch access: {e}", flush=True)

    def _get_block_config_pre_launch(self) -> dict:
        """
        Build block config from YAML + defaults without orchestrator (NANO-064b).

        Mirrors VoiceAgentOrchestrator.get_block_config() serialization.
        """
        from spindl.llm.prompt_block import create_default_blocks, load_block_config

        if self._prompt_blocks_config:
            blocks = load_block_config(self._prompt_blocks_config)
        else:
            blocks = create_default_blocks()

        block_infos = []
        order_list = []
        disabled_list = []
        overrides_dict: dict[str, str | None] = {}

        for block in sorted(blocks, key=lambda b: b.order):
            order_list.append(block.id)
            if not block.enabled:
                disabled_list.append(block.id)
            if block.user_override is not None:
                overrides_dict[block.id] = block.user_override

            block_infos.append({
                "id": block.id,
                "label": block.label,
                "order": block.order,
                "enabled": block.enabled,
                "is_static": block.is_static,
                "section_header": block.section_header,
                "has_override": block.user_override is not None,
                "content_wrapper": block.content_wrapper,
            })

        return {
            "order": order_list,
            "disabled": disabled_list,
            "overrides": overrides_dict,
            "blocks": block_infos,
        }

    def _save_block_config_pre_launch(self, config: dict) -> bool:
        """
        Persist prompt_blocks config to YAML without orchestrator (NANO-064b).

        Uses ruamel.yaml round-trip (NANO-106).

        Returns:
            True if persisted successfully, False otherwise.
        """
        if not self._config_path:
            return False

        try:
            from spindl.orchestrator.config import _make_ruamel_yaml

            config_path = Path(self._config_path)
            if not config_path.exists():
                return False

            has_content = bool(
                config.get("order") or config.get("disabled")
                or config.get("overrides") or config.get("wrappers")
            )

            ry = _make_ruamel_yaml()
            with open(config_path, "r", encoding="utf-8") as f:
                data = ry.load(f)

            if has_content:
                data["prompt_blocks"] = config
            else:
                if "prompt_blocks" in data:
                    del data["prompt_blocks"]

            with open(config_path, "w", encoding="utf-8") as f:
                ry.dump(data, f)

            print(f"[GUI] Block config persisted pre-launch to {config_path.name}", flush=True)
            return True

        except Exception as e:
            print(f"[GUI] Failed to persist block config pre-launch: {e}", flush=True)
            return False

    @staticmethod
    def _validate_local_llm_config(config: dict) -> list[str]:
        """Validate local LLM config dict before persisting (NANO-089).

        Checks required fields for a local llama launch config.
        Returns list of error strings (empty if valid).
        """
        errors = []
        has_url = bool(config.get("url"))
        has_local = bool(config.get("executable_path") or config.get("model_path"))
        if not has_url and not has_local:
            errors.append("Local LLM config requires executable_path + model_path, or url")
        if has_local:
            if not config.get("executable_path"):
                errors.append("executable_path is required for local LLM launch")
            if not config.get("model_path"):
                errors.append("model_path is required for local LLM launch")
        port = config.get("port")
        if port is not None and (not isinstance(port, int) or port < 1 or port > 65535):
            errors.append(f"port must be 1-65535, got {port}")
        return errors

    def _persist_local_llm_config(self, config: dict) -> bool:
        """
        Persist llama provider config to YAML after successful launch (NANO-065b).

        Updates llm.providers.llama section and sets llm.provider to "llama".
        Works both with orchestrator (delegates to save_to_yaml) and pre-launch
        (direct YAML surgery).

        Returns:
            True if persisted successfully.
        """
        # NANO-089: validate before persisting
        errors = self._validate_local_llm_config(config)
        if errors:
            print(f"[GUI] LLM config validation failed: {errors}", flush=True)
            return False

        if self._orchestrator:
            try:
                # Update orchestrator config in-memory, then persist.
                # Merge onto existing to preserve dashboard-only keys
                # (e.g. repeat_penalty from NANO-108).
                llm_cfg = self._orchestrator._config.llm_config
                if "llama" not in llm_cfg.providers:
                    llm_cfg.providers["llama"] = {}
                llm_cfg.providers["llama"].update(config)
                self._orchestrator._config.save_to_yaml(self._config_path)
                print(
                    f"[GUI] Local LLM config persisted via orchestrator to "
                    f"{Path(self._config_path).name}",
                    flush=True,
                )
                return True
            except Exception as e:
                print(f"[GUI] Failed to persist local LLM config: {e}", flush=True)
                return False

        # Pre-launch: ruamel.yaml round-trip (NANO-106)
        if not self._config_path:
            return False

        try:
            from spindl.orchestrator.config import _make_ruamel_yaml

            config_path = Path(self._config_path)
            if not config_path.exists():
                return False

            ry = _make_ruamel_yaml()
            with open(config_path, "r", encoding="utf-8") as f:
                data = ry.load(f)

            if "llm" not in data:
                data["llm"] = {}
            if "providers" not in data["llm"]:
                data["llm"]["providers"] = {}
            # Merge launcher config onto existing section to preserve
            # dashboard-only keys (e.g. repeat_penalty from NANO-108)
            if "llama" not in data["llm"]["providers"]:
                data["llm"]["providers"]["llama"] = {}
            data["llm"]["providers"]["llama"].update(config)

            with open(config_path, "w", encoding="utf-8") as f:
                ry.dump(data, f)

            # Update cache
            self._llm_config_cache = self._llm_config_cache or {}
            if "providers" not in self._llm_config_cache:
                self._llm_config_cache["providers"] = {}
            self._llm_config_cache["providers"]["llama"] = config

            print(
                f"[GUI] Local LLM config persisted pre-launch to {config_path.name}",
                flush=True,
            )
            return True

        except Exception as e:
            print(f"[GUI] Failed to persist local LLM config pre-launch: {e}", flush=True)
            return False

    @staticmethod
    def _validate_local_vlm_config(config: dict) -> list[str]:
        """Validate local VLM config dict before persisting (NANO-089).

        Checks required fields for a local llama VLM launch config.
        Returns list of error strings (empty if valid).
        """
        errors = []
        if not config.get("executable_path"):
            errors.append("executable_path is required for local VLM launch")
        if not config.get("model_path"):
            errors.append("model_path is required for local VLM launch")
        if not config.get("model_type"):
            errors.append("model_type is required for local VLM launch")
        port = config.get("port")
        if port is not None and (not isinstance(port, int) or port < 1 or port > 65535):
            errors.append(f"port must be 1-65535, got {port}")
        return errors

    def _persist_local_vlm_config(self, config: dict) -> bool:
        """
        Persist llama VLM provider config to YAML after successful launch (NANO-079).

        Updates vision.providers.llama section and sets vision.provider to "llama".
        """
        # NANO-089: validate before persisting
        errors = self._validate_local_vlm_config(config)
        if errors:
            print(f"[GUI] VLM config validation failed: {errors}", flush=True)
            return False

        if self._orchestrator:
            try:
                vis_cfg = self._orchestrator._config.vlm_config
                vis_cfg.provider = "llama"
                if "llama" not in vis_cfg.providers:
                    vis_cfg.providers["llama"] = {}
                vis_cfg.providers["llama"].update(config)
                self._orchestrator._config.save_to_yaml(self._config_path)
                print(
                    f"[GUI] Local VLM config persisted via orchestrator to "
                    f"{Path(self._config_path).name}",
                    flush=True,
                )
                return True
            except Exception as e:
                print(f"[GUI] Failed to persist local VLM config: {e}", flush=True)
                return False

        # Pre-launch: ruamel.yaml round-trip (NANO-106)
        if not self._config_path:
            return False

        try:
            from spindl.orchestrator.config import _make_ruamel_yaml

            config_path = Path(self._config_path)
            if not config_path.exists():
                return False

            ry = _make_ruamel_yaml()
            with open(config_path, "r", encoding="utf-8") as f:
                data = ry.load(f)

            if "vlm" not in data:
                data["vlm"] = {}
            if "providers" not in data["vlm"]:
                data["vlm"]["providers"] = {}
            if "llama" not in data["vlm"]["providers"]:
                data["vlm"]["providers"]["llama"] = {}
            data["vlm"]["providers"]["llama"].update(config)

            with open(config_path, "w", encoding="utf-8") as f:
                ry.dump(data, f)

            # Update cache
            self._vlm_config_cache = self._vlm_config_cache or {}
            if "providers" not in self._vlm_config_cache:
                self._vlm_config_cache["providers"] = {}
            self._vlm_config_cache["providers"]["llama"] = config

            print(
                f"[GUI] Local VLM config persisted pre-launch to {config_path.name}",
                flush=True,
            )
            return True

        except Exception as e:
            print(f"[GUI] Failed to persist local VLM config pre-launch: {e}", flush=True)
            return False

    def attach(self, orchestrator: "VoiceAgentOrchestrator") -> None:
        """Attach the orchestrator for event bridging.

        Also schedules a hydration broadcast on the uvicorn event loop so that
        any clients that connected before the orchestrator was ready (and whose
        connect-time request_config was silently dropped) receive config/health/state.
        """
        self._orchestrator = orchestrator
        print("[GUI] Attached to orchestrator", flush=True)

        # NANO-108: Wire mic health change callback so dashboard updates on stream death/recovery.
        # Callback fires from AudioCapture's watchdog thread → schedule async _emit_health()
        # on the uvicorn event loop for socket.io delivery.
        def _on_mic_health():
            if self._event_loop:
                asyncio.run_coroutine_threadsafe(
                    self._emit_health(), self._event_loop
                )

        orchestrator._on_health_change_callback = _on_mic_health

        # NANO-068: Hydrate clients that connected before orchestrator was ready.
        # attach() is called from the launcher thread (asyncio.run() in a daemon
        # thread), NOT the uvicorn event loop — so we must schedule the broadcast
        # on the correct loop for socket.io to actually deliver the emissions.
        if self._clients and self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._hydrate_connected_clients(), self._event_loop
            )

        # NANO-097: Auto-spawn avatar if enabled in config at startup
        if orchestrator._config.avatar_config.enabled and self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._avatar_spawn(), self._event_loop
            )

        # NANO-100: Auto-spawn subtitle if enabled in config at startup
        if orchestrator._config.avatar_config.subtitles_enabled and self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._subtitle_spawn(), self._event_loop
            )

        # NANO-110: Auto-spawn stream deck if enabled in config at startup
        if orchestrator._config.avatar_config.stream_deck_enabled and self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._stream_deck_spawn(), self._event_loop
            )

    async def _hydrate_connected_clients(self) -> None:
        """Broadcast config/health/state to all connected clients.

        Called from attach() to close the reconnect hydration gap: clients that
        connected before the orchestrator was available had their connect-time
        request_config/request_health/request_state silently dropped.
        """
        print(f"[GUI] Hydrating {len(self._clients)} connected client(s)", flush=True)
        await self._emit_config()
        await self._emit_health()
        await self._emit_state()

        # NANO-097: Push active character's VRM + config to connected avatar clients
        if self._orchestrator and self._orchestrator._config.avatar_config.enabled:
            try:
                config = self._orchestrator._config
                loader = CharacterLoader(config.characters_dir)
                vrm_path = loader.get_vrm_path(config.character_id)
                expressions = loader.get_avatar_expressions(config.character_id)
                animations = loader.get_avatar_animations(config.character_id)
                char_anim_dir = str(loader.get_character_animations_dir(config.character_id))
                await self.emit_avatar_load_model(
                    str(vrm_path) if vrm_path else "",
                    expressions=expressions,
                    animations=animations,
                    character_animations_dir=char_anim_dir,
                )
            except Exception:
                pass  # Non-critical: avatar will use its default model

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Set the event loop reference for cross-thread async scheduling.

        Must be called from the uvicorn server context (inside asyncio.run())
        before any EventBridge can emit events. This allows the EventBridge
        to schedule coroutines on the correct event loop even when the
        orchestrator is initialized from a different thread.
        """
        self._event_loop = loop
        print("[GUI] Event loop captured for cross-thread scheduling", flush=True)

    @property
    def event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the uvicorn event loop reference for EventBridge."""
        return self._event_loop

    def set_uvicorn_server(self, server: "uvicorn.Server") -> None:
        """
        Set the uvicorn server reference for graceful shutdown.

        Must be called after creating the uvicorn.Server instance so that
        shutdown_backend can trigger server.should_exit to stop the process.
        """
        self._uvicorn_server = server
        print("[GUI] Uvicorn server reference captured for shutdown control", flush=True)

    def _register_handlers(self) -> None:
        """Register Socket.IO event handlers."""

        @self.sio.event
        async def connect(sid: str, environ: dict) -> None:
            self._clients.add(sid)
            print(f"[GUI] Client connected (total: {len(self._clients)})", flush=True)

            # Send initial config on connect
            if self._orchestrator:
                await self._emit_config(sid)
                await self._emit_health(sid)
                await self._emit_state(sid)
            else:
                # Pre-launch: hydrate cached configs so Settings page has data
                if self._stimuli_config_cache:
                    await self.sio.emit(
                        "stimuli_config_updated",
                        self._stimuli_config_cache,
                        to=sid,
                    )

        @self.sio.event
        async def disconnect(sid: str) -> None:
            self._clients.discard(sid)
            # NANO-097: Track avatar client disconnections
            was_avatar = sid in self._avatar_clients
            self._avatar_clients.discard(sid)
            if was_avatar:
                print(
                    f"[GUI] Avatar client disconnected "
                    f"(avatar clients: {len(self._avatar_clients)})",
                    flush=True,
                )
                await self.sio.emit(
                    "avatar_connection_status",
                    {"connected": self.has_avatar_client},
                )
            print(f"[GUI] Client disconnected (total: {len(self._clients)})", flush=True)

        @self.sio.event
        async def register_avatar_client(sid: str, data: dict) -> None:
            """Avatar renderer identifies itself on connect (NANO-097)."""
            self._avatar_clients.add(sid)
            print(
                f"[GUI] Avatar client registered "
                f"(avatar clients: {len(self._avatar_clients)})",
                flush=True,
            )
            await self.sio.emit(
                "avatar_connection_status",
                {"connected": True},
            )
            # Push active character's VRM + config to the newly connected avatar
            if self._orchestrator:
                try:
                    config = self._orchestrator._config
                    loader = CharacterLoader(config.characters_dir)
                    vrm_path = loader.get_vrm_path(config.character_id)
                    expressions = loader.get_avatar_expressions(config.character_id)
                    animations = loader.get_avatar_animations(config.character_id)
                    char_anim_dir = str(loader.get_character_animations_dir(config.character_id))
                    payload: dict = {"path": str(vrm_path) if vrm_path else ""}
                    if expressions:
                        payload["expressions"] = expressions
                    if animations:
                        payload["animations"] = animations
                    if char_anim_dir:
                        payload["character_animations_dir"] = char_anim_dir
                    await self.sio.emit(
                        "avatar_load_model",
                        payload,
                        to=sid,
                    )
                    print(
                        f"[GUI] Avatar load_model pushed to new client: {vrm_path or '(default)'}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[GUI] Avatar load_model on register failed: {e}", flush=True)

        @self.sio.event
        async def preview_avatar_expressions(sid: str, data: dict) -> None:
            """Live preview of expression composites from character editor (NANO-098).

            Relays to all connected avatar clients without persisting.
            Editor is responsible for reverting on cancel/navigate-away.
            """
            expressions = data.get("expressions", {})
            preview_mood = data.get("previewMood")
            payload: dict = {"expressions": expressions}
            if preview_mood is not None:
                payload["previewMood"] = preview_mood
            await self.sio.emit("avatar_preview_expressions", payload)

        @self.sio.event
        async def update_avatar_animation_config(sid: str, data: dict) -> None:
            """Push updated animation config to renderer on save (NANO-098 Session 3).

            Sent directly from editor with the config data — no disk read needed.
            Avoids race condition with card.json write.
            """
            animations = data.get("animations")
            await self.sio.emit("avatar_update_animation_config", {"animations": animations})

        @self.sio.event
        async def preview_avatar_animation(sid: str, data: dict) -> None:
            """Live preview of animation clip from character editor (NANO-098 Session 3).

            Relays clip name to all connected avatar clients.
            clip=null means stop playing (revert to procedural idle).
            """
            clip = data.get("clip")
            await self.sio.emit("avatar_preview_animation", {"clip": clip})

        @self.sio.event
        async def reload_avatar_model(sid: str, data: dict) -> None:
            """Dashboard requests avatar model reload after VRM change (NANO-097)."""
            character_id = data.get("character_id")
            if not character_id or not self._orchestrator:
                return
            # Only push if this is the active character
            if character_id != self._orchestrator._config.character_id:
                return
            try:
                config = self._orchestrator._config
                loader = CharacterLoader(config.characters_dir)
                vrm_path = loader.get_vrm_path(character_id)
                if vrm_path:
                    expressions = loader.get_avatar_expressions(character_id)
                    animations = loader.get_avatar_animations(character_id)
                    char_anim_dir = str(loader.get_character_animations_dir(character_id))
                    await self.emit_avatar_load_model(
                        str(vrm_path), expressions=expressions,
                        animations=animations, character_animations_dir=char_anim_dir,
                    )
                    print(
                        f"[GUI] Avatar model reloaded: {vrm_path}",
                        flush=True,
                    )
                else:
                    # No VRM — tell avatar to load default model, but still send config
                    expressions = loader.get_avatar_expressions(config.character_id)
                    animations = loader.get_avatar_animations(config.character_id)
                    char_anim_dir = str(loader.get_character_animations_dir(config.character_id))
                    payload: dict = {"path": ""}
                    if expressions:
                        payload["expressions"] = expressions
                    if animations:
                        payload["animations"] = animations
                    if char_anim_dir:
                        payload["character_animations_dir"] = char_anim_dir
                    await self.sio.emit("avatar_load_model", payload)
                    print(
                        "[GUI] Avatar model reloaded: (default)",
                        flush=True,
                    )
            except Exception as e:
                print(f"[GUI] Avatar model reload failed: {e}", flush=True)

        @self.sio.event
        async def request_state(sid: str, data: dict) -> None:
            """Client requests current state snapshot."""
            await self._emit_state(sid)

        @self.sio.event
        async def request_health(sid: str, data: dict) -> None:
            """Client requests health check."""
            await self._emit_health(sid)

        @self.sio.event
        async def request_config(sid: str, data: dict) -> None:
            """Client requests full config."""
            await self._emit_config(sid)

        @self.sio.event
        async def request_sessions(sid: str, data: dict) -> None:
            """Client requests session list."""
            await self._emit_sessions(sid, data.get("persona"))

        @self.sio.event
        async def request_session_detail(sid: str, data: dict) -> None:
            """Client requests session details."""
            filepath = data.get("filepath")
            if filepath:
                await self._emit_session_detail(sid, filepath)

        @self.sio.event
        async def request_chat_history(sid: str, data: dict) -> None:
            """Client requests chat history for the active session (NANO-073a)."""
            from spindl.history import jsonl_store

            if not self._orchestrator or not self._orchestrator.session_file:
                await self.sio.emit(
                    "chat_history",
                    {"turns": []},
                    to=sid,
                )
                return

            try:
                filepath = Path(self._orchestrator.session_file)
                if not filepath.exists():
                    await self.sio.emit(
                        "chat_history",
                        {"turns": []},
                        to=sid,
                    )
                    return

                visible = jsonl_store.read_visible_turns(filepath)
                # Map to frontend-friendly format with metadata (NANO-075)
                turns = []
                for t in visible:
                    role = t.get("role")
                    if role in ("user", "assistant"):
                        # NANO-109: display_content holds raw LLM output (with
                        # formatting); content holds cleaned text for LLM replay.
                        # Chat display should show the raw version when available.
                        display_text = t.get("display_content") or t.get("content", "")
                        turn = {
                            "role": role,
                            "text": display_text,
                            "timestamp": t.get("timestamp", ""),
                        }
                        # NANO-075: Forward metadata for hydration survival
                        if role == "user":
                            input_mod = t.get("input_modality")
                            if input_mod:
                                turn["input_modality"] = input_mod
                        elif role == "assistant":
                            reasoning = t.get("reasoning")
                            if reasoning:
                                turn["reasoning"] = reasoning
                            stimulus = t.get("stimulus_source")
                            if stimulus:
                                turn["stimulus_source"] = stimulus
                            codex = t.get("activated_codex_entries")
                            if codex:
                                turn["activated_codex_entries"] = codex
                            memories = t.get("retrieved_memories")
                            if memories:
                                turn["retrieved_memories"] = memories
                            # NANO-094: Emotion classifier metadata
                            emotion = t.get("emotion")
                            if emotion:
                                turn["emotion"] = emotion
                                turn["emotion_confidence"] = t.get("emotion_confidence")
                        turns.append(turn)

                # Cap at 200 most recent to avoid DOM bloat
                if len(turns) > 200:
                    turns = turns[-200:]

                await self.sio.emit(
                    "chat_history",
                    {"turns": turns},
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Error reading chat history: {e}", flush=True)
                await self.sio.emit(
                    "chat_history",
                    {"turns": []},
                    to=sid,
                )

        @self.sio.event
        async def request_prompt_snapshot(sid: str, data: dict | None = None) -> None:
            """Client requests current prompt snapshot (NANO-076)."""
            if not self._orchestrator:
                # No orchestrator — frontend stays on currentSnapshot: null
                return

            try:
                snapshot = self._orchestrator.get_prompt_snapshot()
                if snapshot:
                    await self.sio.emit("prompt_snapshot", snapshot, to=sid)
                # else: no snapshot available — frontend stays on null (cleared on session switch)
            except Exception as e:
                print(f"[GUI] Error building prompt snapshot: {e}", flush=True)

        @self.sio.event
        async def resume_session(sid: str, data: dict) -> None:
            """Client requests to resume a specific session by filename."""
            filename = data.get("filename")
            if not filename or not self._orchestrator:
                return

            # Resolve filename to full path in conversations directory
            conv_dir_str = self._orchestrator._config.conversations_dir
            if not conv_dir_str:
                return
            resolved = Path(conv_dir_str) / filename
            if not resolved.exists():
                print(f"[GUI] Session resume failed: {filename} not found", flush=True)
                await self.sio.emit(
                    "session_resumed",
                    {"filepath": str(resolved), "success": False, "error": "Session file not found"},
                    to=sid,
                )
                return

            success = self._orchestrator.load_session(str(resolved))
            if success:
                print(f"[GUI] Session resumed: {filename}", flush=True)
                await self.sio.emit(
                    "session_resumed",
                    {"filepath": str(resolved), "success": True},
                    to=sid,
                )
                # Refresh session list so active_session badge updates
                await self._emit_sessions()
            else:
                print(f"[GUI] Session resume failed: {filename}", flush=True)
                await self.sio.emit(
                    "session_resumed",
                    {"filepath": str(resolved), "success": False, "error": "Failed to load session"},
                    to=sid,
                )

        @self.sio.event
        async def create_session(sid: str, data: dict | None = None) -> None:
            """Client requests a new session for the current persona (NANO-071)."""
            if not self._orchestrator:
                await self.sio.emit(
                    "session_created",
                    {"success": False, "error": "Services not running"},
                    to=sid,
                )
                return

            success = self._orchestrator.create_new_session()
            if success:
                new_filepath = str(self._orchestrator.session_file) if self._orchestrator.session_file else None
                # Touch the file so it exists on disk for _emit_sessions glob
                if new_filepath:
                    Path(new_filepath).parent.mkdir(parents=True, exist_ok=True)
                    Path(new_filepath).touch()
                print(f"[GUI] New session created: {Path(new_filepath).name if new_filepath else 'unknown'}", flush=True)
                await self.sio.emit(
                    "session_created",
                    {"success": True, "filepath": new_filepath},
                    to=sid,
                )
                # Refresh session list for all clients
                await self._emit_sessions()
            else:
                await self.sio.emit(
                    "session_created",
                    {"success": False, "error": "Failed to create new session"},
                    to=sid,
                )

        @self.sio.event
        async def delete_session(sid: str, data: dict) -> None:
            """Client requests to delete a session file (NANO-064a)."""
            filepath = data.get("filepath")
            if not filepath:
                return

            try:
                path = Path(filepath)
                if not path.exists():
                    await self.sio.emit(
                        "session_deleted",
                        {"filepath": filepath, "success": False, "error": "Session not found"},
                        to=sid,
                    )
                    return

                # Don't allow deleting the active session (only when services are running)
                if (
                    self._orchestrator
                    and self._orchestrator.session_file
                    and path == self._orchestrator.session_file
                ):
                    await self.sio.emit(
                        "session_deleted",
                        {"filepath": filepath, "success": False, "error": "Cannot delete active session"},
                        to=sid,
                    )
                    return

                path.unlink()
                # NANO-076: Clean up snapshot sidecar if it exists
                from spindl.history.snapshot_store import delete_sidecar
                delete_sidecar(path)
                print(f"[GUI] Session deleted: {path.name}", flush=True)
                await self.sio.emit(
                    "session_deleted",
                    {"filepath": filepath, "success": True},
                    to=sid,
                )
                # Refresh session list for all clients
                await self._emit_sessions()

            except Exception as e:
                print(f"[GUI] Session delete error: {e}", flush=True)
                await self.sio.emit(
                    "session_deleted",
                    {"filepath": filepath, "success": False, "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def generate_session_summary(sid: str, data: dict) -> None:
            """Client requests session summary generation (NANO-043 Phase 4)."""
            filepath = data.get("filepath")
            if filepath and self._orchestrator:
                try:
                    # Run in executor to avoid blocking event loop (LLM call)
                    loop = asyncio.get_event_loop()
                    summary = await loop.run_in_executor(
                        None,
                        self._orchestrator.generate_session_summary,
                        filepath,
                    )
                    session_name = Path(filepath).name
                    if summary:
                        print(f"[GUI] Session summary generated: {session_name}", flush=True)
                    else:
                        print(f"[GUI] Session summary failed: {session_name}", flush=True)
                    await self.sio.emit(
                        "session_summary_generated",
                        {
                            "filepath": filepath,
                            "success": summary is not None,
                            "summary_preview": summary[:200] if summary else None,
                            "error": None if summary else "Failed to generate summary",
                        },
                        to=sid,
                    )
                except Exception as e:
                    print(f"[GUI] Session summary error: {e}", flush=True)
                    await self.sio.emit(
                        "session_summary_generated",
                        {
                            "filepath": filepath,
                            "success": False,
                            "summary_preview": None,
                            "error": str(e),
                        },
                        to=sid,
                    )

        @self.sio.event
        async def request_personas(sid: str, data: dict) -> None:
            """Client requests available personas list."""
            await self._emit_personas(sid)

        @self.sio.event
        async def set_persona(sid: str, data: dict) -> None:
            """Client requests persona change (NANO-077: runtime hot-swap)."""
            persona_id = data.get("persona_id")
            if not persona_id or not self._orchestrator:
                await self.sio.emit(
                    "persona_change_failed",
                    {"error": "Missing persona_id or orchestrator not ready"},
                    to=sid,
                )
                return

            try:
                success = self._orchestrator.switch_persona(persona_id)
            except Exception as e:
                print(f"[GUI] Persona switch failed: {e}", flush=True)
                await self.sio.emit(
                    "persona_change_failed",
                    {"error": str(e)},
                    to=sid,
                )
                return

            if success:
                print(f"[GUI] Persona switched to: {persona_id}", flush=True)

                # Persist to spindl.yaml
                if self._config_path:
                    try:
                        self._orchestrator._config.save_to_yaml(self._config_path)
                    except Exception as e:
                        print(f"[GUI] Failed to persist character.default: {e}", flush=True)

                # Broadcast to ALL clients
                await self.sio.emit(
                    "persona_changed",
                    {"persona_id": persona_id, "restart_required": False},
                )
                # Push updated session list (filtered to new character)
                await self._emit_sessions(sid, persona_id)
                # Push updated config (persona name/id changed)
                await self._emit_config(sid)

                # NANO-097: Notify avatar renderer of new character's VRM model
                if self._orchestrator._config.avatar_config.enabled and self.has_avatar_client:
                    try:
                        loader = CharacterLoader(
                            self._orchestrator._config.characters_dir
                        )
                        vrm_path = loader.get_vrm_path(persona_id)
                        if vrm_path:
                            expressions = loader.get_avatar_expressions(persona_id)
                            animations = loader.get_avatar_animations(persona_id)
                            char_anim_dir = str(loader.get_character_animations_dir(persona_id))
                            await self.emit_avatar_load_model(
                                str(vrm_path), expressions=expressions,
                                animations=animations, character_animations_dir=char_anim_dir,
                            )
                            print(
                                f"[GUI] Avatar load_model emitted: {vrm_path}",
                                flush=True,
                            )
                        else:
                            # Character has no VRM — tell avatar to load default, but still send config
                            expressions = loader.get_avatar_expressions(persona_id)
                            animations = loader.get_avatar_animations(persona_id)
                            char_anim_dir = str(loader.get_character_animations_dir(persona_id))
                            payload: dict = {"path": ""}
                            if expressions:
                                payload["expressions"] = expressions
                            if animations:
                                payload["animations"] = animations
                            if char_anim_dir:
                                payload["character_animations_dir"] = char_anim_dir
                            await self.sio.emit(
                                "avatar_load_model", payload
                            )
                            print(
                                "[GUI] Avatar load_model emitted: (default)",
                                flush=True,
                            )
                    except Exception as e:
                        print(
                            f"[GUI] Avatar load_model failed: {e}",
                            flush=True,
                        )
            else:
                current_state = "unknown"
                if self._orchestrator._state_machine:
                    current_state = self._orchestrator._state_machine.state.value
                await self.sio.emit(
                    "persona_change_failed",
                    {"error": f"Cannot switch while {current_state}"},
                    to=sid,
                )

        @self.sio.event
        async def set_vad_config(sid: str, data: dict) -> None:
            """Client updates VAD configuration."""
            if self._orchestrator:
                config = self._orchestrator._config
                updated = False

                if "threshold" in data:
                    config.vad_threshold = float(data["threshold"])
                    updated = True
                if "min_speech_ms" in data:
                    config.min_speech_ms = int(data["min_speech_ms"])
                    updated = True
                if "min_silence_ms" in data:
                    config.min_silence_ms = int(data["min_silence_ms"])
                    updated = True
                if "speech_pad_ms" in data:
                    config.speech_pad_ms = int(data["speech_pad_ms"])
                    updated = True

                if updated:
                    # Propagate to live VAD instance
                    sm = getattr(self._orchestrator, "_state_machine", None)
                    if sm is not None:
                        sm.update_vad_params(
                            threshold=config.vad_threshold,
                            min_speech_ms=config.min_speech_ms,
                            min_silence_ms=config.min_silence_ms,
                            speech_pad_ms=config.speech_pad_ms,
                        )

                    print(
                        f"[GUI] VAD: threshold={config.vad_threshold:.2f}, "
                        f"min_speech={config.min_speech_ms}ms, "
                        f"min_silence={config.min_silence_ms}ms, "
                        f"speech_pad={config.speech_pad_ms}ms",
                        flush=True,
                    )

                    # Persist to YAML if config path is available
                    persisted = False
                    if self._config_path:
                        try:
                            config.save_to_yaml(self._config_path)
                            persisted = True
                            print(f"[GUI] VAD config persisted to {Path(self._config_path).name}", flush=True)
                        except Exception as e:
                            print(f"[GUI] Failed to persist VAD config: {e}", flush=True)

                    # Emit updated config to confirm
                    await self.sio.emit(
                        "vad_config_updated",
                        {
                            "threshold": config.vad_threshold,
                            "min_speech_ms": config.min_speech_ms,
                            "min_silence_ms": config.min_silence_ms,
                            "speech_pad_ms": config.speech_pad_ms,
                            "persisted": persisted,
                        },
                        to=sid,
                    )

        @self.sio.event
        async def set_pipeline_config(sid: str, data: dict) -> None:
            """Client updates pipeline configuration."""
            if self._orchestrator:
                config = self._orchestrator._config
                updated = False

                if "summarization_threshold" in data:
                    config.summarization_threshold = float(data["summarization_threshold"])
                    updated = True

                if updated:
                    print(
                        f"[GUI] Pipeline: summarization={config.summarization_threshold:.0%}",
                        flush=True,
                    )

                    # Persist to YAML if config path is available
                    persisted = False
                    if self._config_path:
                        try:
                            config.save_to_yaml(self._config_path)
                            persisted = True
                            print(f"[GUI] Pipeline config persisted to {Path(self._config_path).name}", flush=True)
                        except Exception as e:
                            print(f"[GUI] Failed to persist Pipeline config: {e}", flush=True)

                    await self.sio.emit(
                        "pipeline_config_updated",
                        {
                            "summarization_threshold": config.summarization_threshold,
                            "budget_strategy": config.budget_strategy,
                            "persisted": persisted,
                        },
                        to=sid,
                    )

        @self.sio.event
        async def set_memory_config(sid: str, data: dict) -> None:
            """Client updates memory/RAG configuration."""
            if self._orchestrator:
                updated = False

                # Use ellipsis as sentinel to distinguish "not provided" from "set to None"
                top_k = None
                relevance_threshold = ...
                dedup_threshold = ...
                reflection_interval = None
                reflection_prompt = ...
                reflection_system_message = ...
                reflection_delimiter = None

                if "top_k" in data:
                    top_k = int(data["top_k"])
                    updated = True
                if "relevance_threshold" in data:
                    val = data["relevance_threshold"]
                    relevance_threshold = float(val) if val is not None else None
                    updated = True
                if "dedup_threshold" in data:
                    val = data["dedup_threshold"]
                    dedup_threshold = float(val) if val is not None else None
                    updated = True
                if "reflection_interval" in data:
                    reflection_interval = int(data["reflection_interval"])
                    updated = True
                if "reflection_prompt" in data:
                    val = data["reflection_prompt"]
                    # Empty string or null → None (use built-in default)
                    reflection_prompt = val if val else None
                    updated = True
                if "reflection_system_message" in data:
                    val = data["reflection_system_message"]
                    reflection_system_message = val if val else None
                    updated = True
                if "reflection_delimiter" in data:
                    val = data["reflection_delimiter"]
                    if val:  # Don't accept empty delimiter
                        reflection_delimiter = str(val)
                        updated = True

                if updated:
                    self._orchestrator.update_memory_config(
                        top_k=top_k,
                        relevance_threshold=relevance_threshold,
                        dedup_threshold=dedup_threshold,
                        reflection_interval=reflection_interval,
                        reflection_prompt=reflection_prompt,
                        reflection_system_message=reflection_system_message,
                        reflection_delimiter=reflection_delimiter,
                    )

                    config = self._orchestrator._config
                    print(
                        f"[GUI] Memory: top_k={config.memory_config.rag_top_k}, "
                        f"relevance_threshold={config.memory_config.relevance_threshold}, "
                        f"dedup_threshold={config.memory_config.dedup_threshold}, "
                        f"reflection_interval={config.memory_config.reflection_interval}",
                        flush=True,
                    )

                    # Persist to YAML
                    persisted = False
                    if self._config_path:
                        try:
                            config.save_to_yaml(self._config_path)
                            persisted = True
                            print(f"[GUI] Memory config persisted to {Path(self._config_path).name}", flush=True)
                        except Exception as e:
                            print(f"[GUI] Failed to persist Memory config: {e}", flush=True)

                    await self.sio.emit(
                        "memory_config_updated",
                        {
                            "top_k": config.memory_config.rag_top_k,
                            "relevance_threshold": config.memory_config.relevance_threshold,
                            "dedup_threshold": config.memory_config.dedup_threshold,
                            "reflection_interval": config.memory_config.reflection_interval,
                            "reflection_prompt": config.memory_config.reflection_prompt,
                            "reflection_system_message": config.memory_config.reflection_system_message,
                            "reflection_delimiter": config.memory_config.reflection_delimiter,
                            "enabled": config.memory_config.enabled,
                            "persisted": persisted,
                        },
                        to=sid,
                    )

        # ============================================================
        # NANO-102: Memory Curation Config — Socket Handler
        # ============================================================

        @self.sio.event
        async def set_curation_config(sid: str, data: dict) -> None:
            """Client updates memory curation configuration (NANO-102)."""
            if self._orchestrator:
                config = self._orchestrator._config
                curation = config.memory_config.curation
                updated = False

                if "enabled" in data:
                    curation.enabled = bool(data["enabled"])
                    updated = True
                if "api_key" in data:
                    curation.api_key = data["api_key"]
                    updated = True
                if "model" in data:
                    curation.model = str(data["model"])
                    updated = True
                if "prompt" in data:
                    curation.prompt = data["prompt"]
                    updated = True
                if "timeout" in data:
                    curation.timeout = float(data["timeout"])
                    updated = True

                if updated:
                    print(
                        f"[GUI] Curation: enabled={curation.enabled}, "
                        f"model={curation.model}",
                        flush=True,
                    )

                    # Persist to YAML
                    persisted = False
                    if self._config_path:
                        try:
                            config.save_to_yaml(self._config_path)
                            persisted = True
                            print(f"[GUI] Curation config persisted to {Path(self._config_path).name}", flush=True)
                        except Exception as e:
                            print(f"[GUI] Failed to persist Curation config: {e}", flush=True)

                    await self.sio.emit(
                        "curation_config_updated",
                        {
                            "enabled": curation.enabled,
                            "api_key": curation.api_key,
                            "model": curation.model,
                            "prompt": curation.prompt,
                            "timeout": curation.timeout,
                            "persisted": persisted,
                        },
                        to=sid,
                    )

        # ============================================================
        # NANO-045d: Prompt Injection Wrappers — Socket Handler
        # ============================================================

        @self.sio.event
        async def set_prompt_config(sid: str, data: dict) -> None:
            """Client updates prompt injection wrapper strings (NANO-045d)."""
            if self._orchestrator:
                # Use ellipsis sentinel for "not provided"
                rag_prefix = ...
                rag_suffix = ...
                codex_prefix = ...
                codex_suffix = ...
                example_dialogue_prefix = ...
                example_dialogue_suffix = ...
                updated = False

                if "rag_prefix" in data:
                    rag_prefix = str(data["rag_prefix"])
                    updated = True
                if "rag_suffix" in data:
                    rag_suffix = str(data["rag_suffix"])
                    updated = True
                if "codex_prefix" in data:
                    codex_prefix = str(data["codex_prefix"])
                    updated = True
                if "codex_suffix" in data:
                    codex_suffix = str(data["codex_suffix"])
                    updated = True
                if "example_dialogue_prefix" in data:
                    example_dialogue_prefix = str(data["example_dialogue_prefix"])
                    updated = True
                if "example_dialogue_suffix" in data:
                    example_dialogue_suffix = str(data["example_dialogue_suffix"])
                    updated = True

                if updated:
                    self._orchestrator.update_prompt_config(
                        rag_prefix=rag_prefix,
                        rag_suffix=rag_suffix,
                        codex_prefix=codex_prefix,
                        codex_suffix=codex_suffix,
                        example_dialogue_prefix=example_dialogue_prefix,
                        example_dialogue_suffix=example_dialogue_suffix,
                    )

                    pc = self._orchestrator._config.prompt_config
                    print(
                        f"[GUI] Prompt wrappers updated: "
                        f"rag_prefix={pc.rag_prefix[:40]}..., "
                        f"codex_prefix={pc.codex_prefix[:40]}...",
                        flush=True,
                    )

                    # Persist to YAML
                    persisted = False
                    if self._config_path:
                        try:
                            self._orchestrator._config.save_to_yaml(self._config_path)
                            persisted = True
                            print(f"[GUI] Prompt config persisted to {Path(self._config_path).name}", flush=True)
                        except Exception as e:
                            print(f"[GUI] Failed to persist prompt config: {e}", flush=True)

                    await self.sio.emit(
                        "prompt_config_updated",
                        {
                            "rag_prefix": pc.rag_prefix,
                            "rag_suffix": pc.rag_suffix,
                            "codex_prefix": pc.codex_prefix,
                            "codex_suffix": pc.codex_suffix,
                            "example_dialogue_prefix": pc.example_dialogue_prefix,
                            "example_dialogue_suffix": pc.example_dialogue_suffix,
                            "persisted": persisted,
                        },
                        to=sid,
                    )

        # ============================================================
        # NANO-045c-1: Block Config — Socket Handlers
        # ============================================================

        @self.sio.event
        async def request_block_config(sid: str, data: dict) -> None:
            """Client requests current block configuration (NANO-045c-1, NANO-064b)."""
            try:
                if self._orchestrator:
                    config = self._orchestrator.get_block_config()
                else:
                    config = self._get_block_config_pre_launch()
                await self.sio.emit("block_config_loaded", config, to=sid)
            except Exception as e:
                print(f"[GUI] Error getting block config: {e}", flush=True)

        @self.sio.event
        async def set_block_config(sid: str, data: dict) -> None:
            """Client updates block configuration (NANO-045c-1, NANO-064b)."""
            try:
                # Build config dict from provided fields
                config: dict = {}
                if "order" in data:
                    config["order"] = list(data["order"])
                if "disabled" in data:
                    config["disabled"] = list(data["disabled"])
                if "overrides" in data:
                    config["overrides"] = dict(data["overrides"])

                if self._orchestrator:
                    # Live path: merge + hot-reload + YAML save
                    existing = self._orchestrator._config.prompt_blocks
                    if existing and isinstance(existing, dict):
                        merged = {**existing, **config}
                    else:
                        merged = config

                    self._orchestrator.update_block_config(merged)

                    print(
                        f"[GUI] Block config updated: "
                        f"order={len(merged.get('order', []))} blocks, "
                        f"disabled={len(merged.get('disabled', []))} blocks",
                        flush=True,
                    )

                    persisted = False
                    if self._config_path:
                        try:
                            self._orchestrator._config.save_block_config_to_yaml(
                                self._config_path
                            )
                            persisted = True
                            print(
                                f"[GUI] Block config persisted to {Path(self._config_path).name}",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"[GUI] Failed to persist block config: {e}", flush=True)

                    current = self._orchestrator.get_block_config()
                else:
                    # Pre-launch path: YAML-only save (NANO-064b)
                    existing = self._prompt_blocks_config
                    if existing and isinstance(existing, dict):
                        merged = {**existing, **config}
                    else:
                        merged = config

                    self._prompt_blocks_config = merged
                    persisted = self._save_block_config_pre_launch(merged)

                    print(
                        f"[GUI] Block config updated pre-launch: "
                        f"order={len(merged.get('order', []))} blocks, "
                        f"disabled={len(merged.get('disabled', []))} blocks",
                        flush=True,
                    )

                    current = self._get_block_config_pre_launch()

                await self.sio.emit(
                    "block_config_updated",
                    {
                        "success": True,
                        "persisted": persisted,
                        "order": current["order"],
                        "disabled": current["disabled"],
                        "overrides": current["overrides"],
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Error updating block config: {e}", flush=True)
                await self.sio.emit(
                    "block_config_updated",
                    {"success": False, "persisted": False, "order": [], "disabled": [], "overrides": {}},
                    to=sid,
                )

        @self.sio.event
        async def reset_block_config(sid: str, data: dict) -> None:
            """Client resets block configuration to defaults (NANO-045c-1, NANO-064b)."""
            try:
                if self._orchestrator:
                    self._orchestrator.reset_block_config()

                    persisted = False
                    if self._config_path:
                        try:
                            self._orchestrator._config.save_block_config_to_yaml(
                                self._config_path
                            )
                            persisted = True
                        except Exception as e:
                            print(f"[GUI] Failed to persist block config reset: {e}", flush=True)

                    current = self._orchestrator.get_block_config()
                else:
                    # Pre-launch: clear cached config, remove section from YAML
                    self._prompt_blocks_config = None
                    persisted = self._save_block_config_pre_launch({})
                    current = self._get_block_config_pre_launch()

                print("[GUI] Block config reset to defaults", flush=True)

                await self.sio.emit(
                    "block_config_updated",
                    {
                        "success": True,
                        "persisted": persisted,
                        "order": current["order"],
                        "disabled": current["disabled"],
                        "overrides": current["overrides"],
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Error resetting block config: {e}", flush=True)
                await self.sio.emit(
                    "block_config_updated",
                    {"success": False, "persisted": False, "order": [], "disabled": [], "overrides": {}},
                    to=sid,
                )

        # ============================================================
        # NANO-053: Generation Parameters — Socket Handler
        # ============================================================

        @self.sio.event
        async def set_generation_params(sid: str, data: dict) -> None:
            """Client updates generation parameters at runtime (NANO-053, NANO-108)."""
            if self._orchestrator:
                temperature = ...
                max_tokens = ...
                top_p = ...
                repeat_penalty = ...
                repeat_last_n = ...
                frequency_penalty = ...
                presence_penalty = ...
                updated = False

                if "temperature" in data:
                    val = float(data["temperature"])
                    if 0.0 <= val <= 2.0:
                        temperature = val
                        updated = True
                if "max_tokens" in data:
                    val = int(data["max_tokens"])
                    if 64 <= val <= 8192:
                        max_tokens = val
                        updated = True
                if "top_p" in data:
                    val = float(data["top_p"])
                    if 0.0 <= val <= 1.0:
                        top_p = val
                        updated = True
                if "repeat_penalty" in data:
                    val = float(data["repeat_penalty"])
                    if 0.0 <= val <= 2.0:
                        repeat_penalty = val
                        updated = True
                if "repeat_last_n" in data:
                    val = int(data["repeat_last_n"])
                    if 0 <= val <= 2048:
                        repeat_last_n = val
                        updated = True
                if "frequency_penalty" in data:
                    val = float(data["frequency_penalty"])
                    if -2.0 <= val <= 2.0:
                        frequency_penalty = val
                        updated = True
                if "presence_penalty" in data:
                    val = float(data["presence_penalty"])
                    if -2.0 <= val <= 2.0:
                        presence_penalty = val
                        updated = True

                if updated:
                    self._orchestrator.update_generation_params(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        repeat_last_n=repeat_last_n,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )

                    pc = self._orchestrator._config.llm_config.provider_config
                    print(
                        f"[GUI] Generation params: "
                        f"temp={pc.get('temperature', 0.7)}, "
                        f"max_tokens={pc.get('max_tokens', 256)}, "
                        f"top_p={pc.get('top_p', 0.95)}, "
                        f"repeat_penalty={pc.get('repeat_penalty', 1.1)}, "
                        f"repeat_last_n={pc.get('repeat_last_n', 64)}, "
                        f"freq_penalty={pc.get('frequency_penalty', 0.0)}, "
                        f"pres_penalty={pc.get('presence_penalty', 0.0)}",
                        flush=True,
                    )

                    # Persist to YAML
                    persisted = False
                    if self._config_path:
                        try:
                            self._orchestrator._config.save_to_yaml(self._config_path)
                            persisted = True
                            print(f"[GUI] Generation params persisted to {Path(self._config_path).name}", flush=True)
                        except Exception as e:
                            print(f"[GUI] Failed to persist generation params: {e}", flush=True)

                    await self.sio.emit(
                        "generation_params_updated",
                        {
                            "temperature": pc.get("temperature", 0.7),
                            "max_tokens": pc.get("max_tokens", 256),
                            "top_p": pc.get("top_p", 0.95),
                            "repeat_penalty": pc.get("repeat_penalty", 1.1),
                            "repeat_last_n": pc.get("repeat_last_n", 64),
                            "frequency_penalty": pc.get("frequency_penalty", 0.0),
                            "presence_penalty": pc.get("presence_penalty", 0.0),
                            "persisted": persisted,
                        },
                        to=sid,
                    )

        # ============================================================
        # NANO-065a: Runtime Tools Toggle — Socket Handlers
        # ============================================================

        @self.sio.event
        async def request_tools_config(sid: str, data: dict) -> None:
            """Client requests current tools state (NANO-065a)."""
            if self._orchestrator:
                state = self._orchestrator.get_tools_state()
            else:
                # Pre-launch fallback: read from cached YAML config
                cache = self._tools_config_cache or {}
                master = cache.get("enabled", False)
                tools_section = cache.get("tools", {})
                tools = {}
                for name, cfg in tools_section.items():
                    label = name.replace("_", " ").title()
                    tools[name] = {
                        "enabled": cfg.get("enabled", True),
                        "label": label,
                    }
                state = {"master_enabled": master, "tools": tools}

            await self.sio.emit("tools_config_updated", state, to=sid)

        @self.sio.event
        async def set_tools_config(sid: str, data: dict) -> None:
            """Client updates tools enable/disable state at runtime (NANO-065a)."""
            if self._orchestrator:
                master_enabled = ...
                tools_changes = None

                if "master_enabled" in data:
                    master_enabled = bool(data["master_enabled"])

                if "tools" in data and isinstance(data["tools"], dict):
                    tools_changes = data["tools"]

                result = self._orchestrator.update_tools_config(
                    master_enabled=master_enabled,
                    tools=tools_changes,
                )

                # NANO-089 Phase 4: relay errors to frontend
                if not result.get("success", True):
                    state = self._orchestrator.get_tools_state()
                    state["error"] = result.get("error", "Tools toggle failed")
                    await self.sio.emit("tools_config_updated", state, to=sid)
                    return

                print(
                    f"[GUI] Tools config updated: "
                    f"master={data.get('master_enabled', '(unchanged)')}, "
                    f"tools={list(data.get('tools', {}).keys()) or '(unchanged)'}",
                    flush=True,
                )

                # Persist to YAML
                persisted = False
                if self._config_path:
                    try:
                        self._orchestrator._config.save_to_yaml(self._config_path)
                        persisted = True
                        print(f"[GUI] Tools config persisted to {Path(self._config_path).name}", flush=True)
                    except Exception as e:
                        print(f"[GUI] Failed to persist tools config: {e}", flush=True)

                # Get current state and emit
                state = self._orchestrator.get_tools_state()
                state["persisted"] = persisted
                await self.sio.emit("tools_config_updated", state, to=sid)
                await self._emit_health()

        # ============================================================
        # NANO-065b: Runtime LLM Provider/Model Swap — Socket Handlers
        # ============================================================

        @self.sio.event
        async def request_llm_config(sid: str, data: dict) -> None:
            """Client requests current LLM provider state (NANO-065b)."""
            if self._orchestrator:
                state = self._orchestrator.get_llm_state()
            else:
                # Pre-launch fallback: read from cached YAML config
                cache = self._llm_config_cache or {}
                state = {
                    "provider": cache.get("provider", "llama"),
                    "model": cache.get("model", ""),
                    "context_size": cache.get("context_size"),
                    "available_providers": list(
                        (cache.get("providers") or {}).keys()
                    ),
                }

            await self.sio.emit("llm_config_updated", state, to=sid)

        @self.sio.event
        async def set_llm_provider(sid: str, data: dict) -> None:
            """Client requests LLM provider/model swap at runtime (NANO-065b)."""
            if not self._orchestrator:
                await self.sio.emit(
                    "llm_config_updated",
                    {"error": "Services not launched"},
                    to=sid,
                )
                return

            provider_name = data.get("provider")
            provider_config = data.get("config")

            if not provider_name or not isinstance(provider_config, dict):
                await self.sio.emit(
                    "llm_config_updated",
                    {"error": "Missing provider name or config"},
                    to=sid,
                )
                return

            result = self._orchestrator.swap_llm_provider(
                provider_name, provider_config
            )

            if not result.get("success"):
                await self.sio.emit(
                    "llm_config_updated",
                    {"error": result.get("error", "Unknown error")},
                    to=sid,
                )
                return

            print(
                f"[GUI] LLM provider swapped to {provider_name} "
                f"(model: {result.get('model', 'unknown')})",
                flush=True,
            )

            # Persist to YAML
            persisted = False
            if self._config_path:
                try:
                    self._orchestrator._config.save_to_yaml(self._config_path)
                    persisted = True
                    print(
                        f"[GUI] LLM config persisted to "
                        f"{Path(self._config_path).name}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[GUI] Failed to persist LLM config: {e}", flush=True)

            result["persisted"] = persisted
            await self.sio.emit("llm_config_updated", result, to=sid)
            await self._emit_health()

        @self.sio.event
        async def request_local_llm_config(sid: str, data: dict) -> None:
            """Return stored llama provider config for Dashboard hydration (NANO-065b)."""
            if self._orchestrator:
                llm_cfg = self._orchestrator._config.llm_config
                llama_cfg = dict(llm_cfg.providers.get("llama", {}))
            elif self._llm_config_cache:
                llama_cfg = dict((self._llm_config_cache.get("providers") or {}).get("llama", {}))
            else:
                llama_cfg = {}

            # Convert extra_args list → string for frontend display
            if isinstance(llama_cfg.get("extra_args"), list):
                llama_cfg["extra_args"] = " ".join(str(a) for a in llama_cfg["extra_args"])

            # Convert tensor_split list → string for frontend display
            if isinstance(llama_cfg.get("tensor_split"), list):
                llama_cfg["tensor_split"] = ",".join(str(x) for x in llama_cfg["tensor_split"])

            # Check if LLM service is currently running
            server_running = False
            if self._service_runner and self._service_runner.is_service_running("llm"):
                server_running = True

            await self.sio.emit(
                "local_llm_config",
                {"config": llama_cfg, "server_running": server_running},
                to=sid,
            )

        @self.sio.event
        async def launch_llm_server(sid: str, data: dict) -> None:
            """Launch (or relaunch) a local llama-server from the Dashboard (NANO-065b)."""
            from spindl.llm.builtin.llama.provider import LlamaProvider
            from spindl.launcher.config import (
                ServiceConfig, HealthCheckConfig, LLMProviderConfig,
            )
            from spindl.launcher.service_runner import ServiceRunner
            from spindl.launcher.log_aggregator import LogAggregator

            config = data.get("config", {})

            # Convert extra_args string → list for backend (frontend sends string)
            if isinstance(config.get("extra_args"), str):
                raw = config["extra_args"].strip()
                config["extra_args"] = raw.split() if raw else []

            # Convert tensor_split string → list[float] for backend
            if isinstance(config.get("tensor_split"), str):
                raw_ts = config["tensor_split"].strip()
                if raw_ts:
                    config["tensor_split"] = [
                        float(v.strip()) for v in raw_ts.split(",")
                        if v.strip()
                    ]
                else:
                    config.pop("tensor_split", None)

            # Validate
            errors = LlamaProvider.validate_config(config)
            if errors:
                await self.sio.emit(
                    "llm_server_launched",
                    {"success": False, "error": f"Validation failed: {'; '.join(errors)}"},
                    to=sid,
                )
                return

            host = config.get("host", "127.0.0.1")
            port = config.get("port", 5557)

            # NANO-079: Extract unified_vision flag before passing to provider
            unified_vision = config.pop("unified_vision", False)

            # Quick check: is a server already running at this address?
            # Skip if unified_vision changed — the user is requesting a config
            # change (e.g. adding mmproj + -np 2) that requires a real restart.
            import requests
            try:
                resp = requests.get(f"http://{host}:{port}/health", timeout=3)
                if resp.status_code == 200 and not unified_vision:
                    await self.sio.emit(
                        "llm_server_launched",
                        {"success": True, "already_running": True},
                        to=sid,
                    )
                    return
            except Exception:
                pass  # Not running — proceed with launch

            # Build provider config
            llm_provider_config = LLMProviderConfig(
                provider="llama",
                provider_config=config,
            )

            # NANO-079: When unified vision is active, set VLM provider config
            # so the ServiceRunner injects -np 2 for dual-slot operation
            vision_provider_config = None
            if unified_vision:
                from spindl.launcher.config import VisionProviderConfig
                vision_provider_config = VisionProviderConfig(
                    enabled=True,
                    provider="llm",
                    provider_config={},
                )

            context_size = config.get("context_size", 8192)

            # Stop existing LLM service if running (model hot-swap)
            if self._service_runner and self._service_runner.is_service_running("llm"):
                # Distinguish cloud (just unregister) vs local (actual process stop)
                is_cloud = "llm" in self._service_runner._cloud_services
                if is_cloud:
                    print("[GUI] Unregistering cloud LLM provider before local launch...", flush=True)
                else:
                    print("[GUI] Stopping existing LLM service for model swap...", flush=True)
                self._service_runner.stop_service("llm")
                self._launched_services.discard("llm")

            # Create or update ServiceRunner
            if self._service_runner is None:
                if self._log_aggregator is None:
                    self._log_aggregator = LogAggregator()
                kwargs = dict(
                    logger=self._log_aggregator,
                    llm_provider_config=llm_provider_config,
                    llm_context_size=context_size,
                )
                if vision_provider_config:
                    kwargs["vision_provider_config"] = vision_provider_config
                self._service_runner = ServiceRunner(**kwargs)
            else:
                # Update existing runner's LLM config + bust cache
                self._service_runner._llm_provider_config = llm_provider_config
                self._service_runner._llm_provider_class = None
                self._service_runner._llm_context_size = context_size
                # NANO-079: Update vision config for -np 2 injection
                if vision_provider_config:
                    self._service_runner._vision_provider_config = vision_provider_config

            # Build synthetic ServiceConfig
            svc_config = ServiceConfig(
                name="llm",
                platform="native",
                command=None,
                health_check=HealthCheckConfig(type="provider", timeout=90),
            )

            # Preview the command for diagnostics
            try:
                preview_cmd = self._service_runner._build_command(svc_config)
                print(f"[GUI] LLM launch command: {preview_cmd}", flush=True)
            except Exception as e:
                print(f"[GUI] LLM command preview failed: {e}", flush=True)

            # Launch in background thread
            def _launch():
                try:
                    success = self._service_runner.start_service(svc_config)
                except Exception as e:
                    success = False
                    print(f"[GUI] LLM launch exception: {e}", flush=True)

                # Schedule async emit back on event loop
                async def _emit_result():
                    if success:
                        self._launched_services.add("llm")
                        print(
                            f"[GUI] LLM server launched at {host}:{port}",
                            flush=True,
                        )

                        # Persist config to YAML
                        persisted = self._persist_local_llm_config(config)

                        await self.sio.emit(
                            "llm_server_launched",
                            {"success": True, "persisted": persisted},
                        )
                    else:
                        await self.sio.emit(
                            "llm_server_launched",
                            {
                                "success": False,
                                "error": f"LLM server failed to start at {host}:{port}",
                            },
                        )

                loop = self._event_loop
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(_emit_result(), loop)

            thread = threading.Thread(target=_launch, daemon=True)
            thread.start()

            # Acknowledge receipt immediately
            await self.sio.emit(
                "llm_server_launched",
                {"success": None, "status": "launching"},
                to=sid,
            )

        @self.sio.event
        async def request_openrouter_models(sid: str, data: dict) -> None:
            """Fetch available models from OpenRouter API (NANO-065b)."""
            import aiohttp

            try:
                # Resolve base URL from config
                base_url = "https://openrouter.ai/api/v1"
                if self._orchestrator:
                    cfg = self._orchestrator._config.llm_config
                    or_cfg = cfg.providers.get("openrouter", {})
                    base_url = or_cfg.get("url", base_url).rstrip("/")
                elif self._llm_config_cache:
                    providers = self._llm_config_cache.get("providers", {})
                    or_cfg = providers.get("openrouter", {})
                    base_url = or_cfg.get("url", base_url).rstrip("/")

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{base_url}/models", timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status != 200:
                            await self.sio.emit(
                                "openrouter_models",
                                {"error": f"OpenRouter API returned {resp.status}"},
                                to=sid,
                            )
                            return
                        body = await resp.json()

                models = []
                for m in body.get("data", []):
                    models.append({
                        "id": m.get("id", ""),
                        "name": m.get("name", ""),
                        "context_length": m.get("context_length"),
                    })

                # Sort by name for display
                models.sort(key=lambda x: x["name"].lower())

                await self.sio.emit(
                    "openrouter_models", {"models": models}, to=sid
                )

            except Exception as e:
                print(f"[GUI] Failed to fetch OpenRouter models: {e}", flush=True)
                await self.sio.emit(
                    "openrouter_models",
                    {"error": f"Failed to fetch models: {e}"},
                    to=sid,
                )

        # ============================================================
        # NANO-065c: Runtime VLM Provider Swap — Socket Handlers
        # ============================================================

        @self.sio.event
        async def request_vlm_config(sid: str, data: dict) -> None:
            """Client requests current VLM provider state (NANO-065c)."""
            if self._orchestrator:
                state = self._orchestrator.get_vlm_state()
            else:
                # Pre-launch fallback: read from cached YAML config
                cache = self._vlm_config_cache or {}
                provider = cache.get("provider", "llama")
                cloud_config = {}
                if provider == "openai":
                    stored = (cache.get("providers") or {}).get("openai", {})
                    api_key = stored.get("api_key", "")
                    if api_key and not api_key.startswith("${"):
                        masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "••••"
                    else:
                        masked = api_key
                    cloud_config = {
                        "api_key": masked,
                        "model": stored.get("model", ""),
                        "base_url": stored.get("base_url", ""),
                    }
                state = {
                    "provider": provider,
                    "available_providers": list(
                        (cache.get("providers") or {}).keys()
                    ) + ["llm"],
                    "healthy": False,
                }
                if cloud_config:
                    state["cloud_config"] = cloud_config

            await self.sio.emit("vlm_config_updated", state, to=sid)

        @self.sio.event
        async def set_vlm_provider(sid: str, data: dict) -> None:
            """Client requests VLM provider swap at runtime (NANO-065c)."""
            if not self._orchestrator:
                await self.sio.emit(
                    "vlm_config_updated",
                    {"error": "Services not launched"},
                    to=sid,
                )
                return

            provider_name = data.get("provider")
            provider_config = data.get("config")

            if not provider_name or not isinstance(provider_config, dict):
                await self.sio.emit(
                    "vlm_config_updated",
                    {"error": "Missing provider name or config"},
                    to=sid,
                )
                return

            result = self._orchestrator.swap_vlm_provider(
                provider_name, provider_config
            )

            if not result.get("success"):
                await self.sio.emit(
                    "vlm_config_updated",
                    {"error": result.get("error", "Unknown error")},
                    to=sid,
                )
                return

            # Stop orphaned VLM server when switching to unified mode —
            # the LLM handles vision now, no need for a dedicated VLM server
            if provider_name == "llm" and self._service_runner:
                if self._service_runner.is_service_running("vlm"):
                    print("[GUI] Stopping dedicated VLM server (switching to unified mode)...", flush=True)
                    self._service_runner.stop_service("vlm")
                    self._launched_services.discard("vlm")

            print(
                f"[GUI] VLM provider swapped to {provider_name}",
                flush=True,
            )

            # Persist to YAML
            persisted = False
            if self._config_path:
                try:
                    self._orchestrator._config.save_to_yaml(self._config_path)
                    persisted = True
                    print(
                        f"[GUI] VLM config persisted to "
                        f"{Path(self._config_path).name}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[GUI] Failed to persist VLM config: {e}", flush=True)

            result["persisted"] = persisted
            await self.sio.emit("vlm_config_updated", result, to=sid)
            await self._emit_health()

        # ============================================================
        # NANO-079: Dashboard VLM Launch — Socket Handlers
        # ============================================================

        @self.sio.event
        async def request_local_vlm_config(sid: str, data: dict) -> None:
            """Client requests stored local VLM config for hydration (NANO-079)."""
            vlm_cfg: dict = {}
            if self._vlm_config_cache:
                providers = self._vlm_config_cache.get("providers", {})
                vlm_cfg = dict(providers.get("llama", {}))

            # Convert extra_args list → string for frontend display
            if isinstance(vlm_cfg.get("extra_args"), list):
                vlm_cfg["extra_args"] = " ".join(str(a) for a in vlm_cfg["extra_args"])

            # Convert tensor_split list → string for frontend display
            if isinstance(vlm_cfg.get("tensor_split"), list):
                vlm_cfg["tensor_split"] = ",".join(str(x) for x in vlm_cfg["tensor_split"])

            # Check if VLM service is currently running
            server_running = False
            if self._service_runner and self._service_runner.is_service_running("vlm"):
                server_running = True

            await self.sio.emit(
                "local_vlm_config",
                {"config": vlm_cfg, "server_running": server_running},
                to=sid,
            )

        @self.sio.event
        async def launch_vlm_server(sid: str, data: dict) -> None:
            """Launch (or relaunch) a local VLM llama-server from the Dashboard (NANO-079)."""
            from spindl.vision.builtin.llama.provider import LlamaVLMProvider
            from spindl.launcher.config import (
                ServiceConfig, HealthCheckConfig, VisionProviderConfig,
            )
            from spindl.launcher.service_runner import ServiceRunner
            from spindl.launcher.log_aggregator import LogAggregator

            config = data.get("config", {})

            # Convert extra_args string → list for backend (frontend sends string)
            if isinstance(config.get("extra_args"), str):
                raw = config["extra_args"].strip()
                config["extra_args"] = raw.split() if raw else []

            # Convert tensor_split string → list[float] for backend
            if isinstance(config.get("tensor_split"), str):
                raw_ts = config["tensor_split"].strip()
                if raw_ts:
                    config["tensor_split"] = [
                        float(v.strip()) for v in raw_ts.split(",")
                        if v.strip()
                    ]
                else:
                    config.pop("tensor_split", None)

            # Validate using VLM provider validator (not LLM)
            errors = LlamaVLMProvider.validate_config(config)
            if errors:
                await self.sio.emit(
                    "vlm_server_launched",
                    {"success": False, "error": f"Validation failed: {'; '.join(errors)}"},
                    to=sid,
                )
                return

            host = config.get("host", "127.0.0.1")
            port = config.get("port", 5558)

            # Quick check: is a server already running at this address?
            import requests
            try:
                resp = requests.get(f"http://{host}:{port}/health", timeout=3)
                if resp.status_code == 200:
                    await self.sio.emit(
                        "vlm_server_launched",
                        {"success": True, "already_running": True},
                        to=sid,
                    )
                    return
            except Exception:
                pass  # Not running — proceed with launch

            # Build vision provider config
            vision_provider_config = VisionProviderConfig(
                enabled=True,
                provider="llama",
                provider_config=config,
            )

            # Stop existing VLM service if running
            if self._service_runner and self._service_runner.is_service_running("vlm"):
                is_cloud = "vlm" in self._service_runner._cloud_services
                if is_cloud:
                    print("[GUI] Unregistering cloud VLM provider before local launch...", flush=True)
                else:
                    print("[GUI] Stopping existing VLM service for model swap...", flush=True)
                self._service_runner.stop_service("vlm")
                self._launched_services.discard("vlm")

            # Create or update ServiceRunner
            if self._service_runner is None:
                if self._log_aggregator is None:
                    self._log_aggregator = LogAggregator()
                self._service_runner = ServiceRunner(
                    logger=self._log_aggregator,
                    vision_provider_config=vision_provider_config,
                )
            else:
                # Update existing runner's VLM config + bust cache
                self._service_runner._vision_provider_config = vision_provider_config
                self._service_runner._vlm_provider_class = None

            # Build synthetic ServiceConfig for VLM
            svc_config = ServiceConfig(
                name="vlm",
                platform="native",
                command=None,
                health_check=HealthCheckConfig(type="provider", timeout=90),
            )

            # Preview the command for diagnostics
            try:
                preview_cmd = self._service_runner._build_command(svc_config)
                print(f"[GUI] VLM launch command: {preview_cmd}", flush=True)
            except Exception as e:
                print(f"[GUI] VLM command preview failed: {e}", flush=True)

            # Launch in background thread
            def _launch():
                try:
                    success = self._service_runner.start_service(svc_config)
                except Exception as e:
                    success = False
                    print(f"[GUI] VLM launch exception: {e}", flush=True)

                # Schedule async emit back on event loop
                async def _emit_result():
                    if success:
                        self._launched_services.add("vlm")
                        print(
                            f"[GUI] VLM server launched at {host}:{port}",
                            flush=True,
                        )

                        # Persist config to YAML
                        persisted = self._persist_local_vlm_config(config)

                        await self.sio.emit(
                            "vlm_server_launched",
                            {"success": True, "persisted": persisted},
                        )
                    else:
                        await self.sio.emit(
                            "vlm_server_launched",
                            {
                                "success": False,
                                "error": f"VLM server failed to start at {host}:{port}",
                            },
                        )

                loop = self._event_loop
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(_emit_result(), loop)

            thread = threading.Thread(target=_launch, daemon=True)
            thread.start()

            # Acknowledge receipt immediately
            await self.sio.emit(
                "vlm_server_launched",
                {"success": None, "status": "launching"},
                to=sid,
            )

        # ============================================================
        # NANO-056: Stimuli System — Socket Handlers
        # ============================================================

        @self.sio.event
        async def set_stimuli_config(sid: str, data: dict) -> None:
            """Client updates stimuli configuration at runtime (NANO-056)."""
            if self._orchestrator:
                enabled = data.get("enabled")
                patience_enabled = data.get("patience_enabled")
                patience_seconds = data.get("patience_seconds")
                patience_prompt = data.get("patience_prompt")

                # Twitch fields (NANO-056b)
                twitch_enabled = data.get("twitch_enabled")
                twitch_channel = data.get("twitch_channel")
                twitch_app_id = data.get("twitch_app_id")
                twitch_app_secret = data.get("twitch_app_secret")
                twitch_buffer_size = data.get("twitch_buffer_size")
                twitch_max_message_length = data.get("twitch_max_message_length")
                twitch_prompt_template = data.get("twitch_prompt_template")

                # Type coerce
                if enabled is not None:
                    enabled = bool(enabled)
                if patience_enabled is not None:
                    patience_enabled = bool(patience_enabled)
                if patience_seconds is not None:
                    patience_seconds = float(patience_seconds)
                    if patience_seconds < 5:
                        patience_seconds = 5.0
                    elif patience_seconds > 600:
                        patience_seconds = 600.0
                if patience_prompt is not None:
                    patience_prompt = str(patience_prompt).strip()
                    if not patience_prompt:
                        patience_prompt = None  # Don't set empty prompts

                # Twitch type coercion (NANO-056b)
                if twitch_enabled is not None:
                    twitch_enabled = bool(twitch_enabled)
                if twitch_channel is not None:
                    twitch_channel = str(twitch_channel).strip()
                if twitch_app_id is not None:
                    twitch_app_id = str(twitch_app_id).strip()
                if twitch_app_secret is not None:
                    twitch_app_secret = str(twitch_app_secret).strip()
                if twitch_buffer_size is not None:
                    twitch_buffer_size = int(twitch_buffer_size)
                    twitch_buffer_size = max(1, min(50, twitch_buffer_size))
                if twitch_max_message_length is not None:
                    twitch_max_message_length = int(twitch_max_message_length)
                    twitch_max_message_length = max(50, min(1000, twitch_max_message_length))
                if twitch_prompt_template is not None:
                    twitch_prompt_template = str(twitch_prompt_template).strip()
                    if not twitch_prompt_template:
                        twitch_prompt_template = None

                # Addressing-others contexts (NANO-110)
                addressing_others_contexts = data.get("addressing_others_contexts")
                if addressing_others_contexts is not None:
                    if isinstance(addressing_others_contexts, list):
                        # Validate: must have at least one entry
                        if not addressing_others_contexts:
                            addressing_others_contexts = None
                        else:
                            # Sanitize each context entry
                            addressing_others_contexts = [
                                {
                                    "id": str(ctx.get("id", f"ctx_{i}")),
                                    "label": str(ctx.get("label", "Others")).strip(),
                                    "prompt": str(ctx.get("prompt", "")).strip(),
                                }
                                for i, ctx in enumerate(addressing_others_contexts)
                                if isinstance(ctx, dict)
                            ]
                    else:
                        addressing_others_contexts = None

                self._orchestrator.update_stimuli_config(
                    enabled=enabled,
                    patience_enabled=patience_enabled,
                    patience_seconds=patience_seconds,
                    patience_prompt=patience_prompt,
                    twitch_enabled=twitch_enabled,
                    twitch_channel=twitch_channel,
                    twitch_app_id=twitch_app_id,
                    twitch_app_secret=twitch_app_secret,
                    twitch_buffer_size=twitch_buffer_size,
                    twitch_max_message_length=twitch_max_message_length,
                    twitch_prompt_template=twitch_prompt_template,
                    addressing_others_contexts=addressing_others_contexts,
                )

                cfg = self._orchestrator._config.stimuli_config
                print(
                    f"[GUI] Stimuli: enabled={cfg.enabled}, "
                    f"patience_enabled={cfg.patience_enabled}, "
                    f"patience_seconds={cfg.patience_seconds}, "
                    f"twitch_enabled={cfg.twitch_enabled}",
                    flush=True,
                )

                # Persist to YAML
                persisted = False
                if self._config_path:
                    try:
                        self._orchestrator._config.save_to_yaml(
                            self._config_path
                        )
                        persisted = True
                        print(
                            f"[GUI] Stimuli config persisted to "
                            f"{Path(self._config_path).name}",
                            flush=True,
                        )
                    except Exception as e:
                        print(
                            f"[GUI] Failed to persist stimuli config: {e}",
                            flush=True,
                        )

                stimuli_data = self._build_stimuli_hydration(cfg)
                stimuli_data["persisted"] = persisted

                await self.sio.emit(
                    "stimuli_config_updated",
                    stimuli_data,
                    to=sid,
                )

            if not self._orchestrator:
                # Pre-launch: no persist, no emit. Credentials persist
                # via test_twitch_credentials handler on Test Connection click.
                # Orchestrator path handles persistence after launch.
                pass

        # ============================================================
        # NANO-110: Addressing Others — Socket Handlers
        # ============================================================

        @self.sio.event
        async def addressing_others_start(sid: str, data: dict) -> None:
            """Stream Deck / hotkey activates addressing-others mode (NANO-110)."""
            if not self._orchestrator:
                return
            context_id = data.get("context_id", "ctx_0")
            self._orchestrator.set_addressing_others(str(context_id))
            await self.sio.emit(
                "addressing_others_state",
                {"active": True, "context_id": context_id},
            )

        @self.sio.event
        async def addressing_others_stop(sid: str, data: dict) -> None:
            """Stream Deck / hotkey deactivates addressing-others mode (NANO-110)."""
            if not self._orchestrator:
                return
            self._orchestrator.clear_addressing_others()
            await self.sio.emit(
                "addressing_others_state",
                {"active": False, "context_id": None},
            )

        @self.sio.event
        async def request_patience_progress(sid: str, data: dict) -> None:
            """Client requests current PATIENCE progress (NANO-056)."""
            if (
                not self._orchestrator
                or not self._orchestrator.stimuli_engine
            ):
                await self.sio.emit(
                    "patience_progress",
                    {"elapsed": 0, "total": 0, "progress": 0},
                    to=sid,
                )
                return

            engine = self._orchestrator.stimuli_engine
            for module in engine.modules:
                if module.name == "patience":
                    progress_data = module.get_progress()
                    progress_data["blocked"] = engine.is_blocked_by_playback or engine.is_blocked_by_typing
                    progress_data["blocked_reason"] = (
                        "typing" if engine.is_blocked_by_typing
                        else "playback" if engine.is_blocked_by_playback
                        else None
                    )
                    await self.sio.emit(
                        "patience_progress",
                        progress_data,
                        to=sid,
                    )
                    return

            # No PATIENCE module found
            await self.sio.emit(
                "patience_progress",
                {"elapsed": 0, "total": 0, "progress": 0},
                to=sid,
            )

        @self.sio.event
        async def request_twitch_status(sid: str, data: dict) -> None:
            """Client requests current Twitch module status (NANO-056b)."""
            if (
                not self._orchestrator
                or not self._orchestrator.stimuli_engine
            ):
                await self.sio.emit(
                    "twitch_status",
                    {
                        "connected": False,
                        "channel": "",
                        "buffer_count": 0,
                        "recent_messages": [],
                    },
                    to=sid,
                )
                return

            engine = self._orchestrator.stimuli_engine
            for module in engine.modules:
                if module.name == "twitch":
                    await self.sio.emit(
                        "twitch_status",
                        {
                            "connected": module.connected,
                            "channel": module.channel,
                            "buffer_count": module.buffer_count,
                            "recent_messages": module.recent_messages,
                        },
                        to=sid,
                    )
                    return

            # No Twitch module registered
            await self.sio.emit(
                "twitch_status",
                {
                    "connected": False,
                    "channel": "",
                    "buffer_count": 0,
                    "recent_messages": [],
                },
                to=sid,
            )

        @self.sio.event
        async def test_twitch_credentials(sid: str, data: dict) -> None:
            """Validate Twitch app credentials without launching the module."""
            import os
            import re
            env_pattern = re.compile(r"\$\{([^}]+)\}")

            def resolve(value: str) -> str:
                """Resolve ${ENV_VAR} patterns and fall back to env vars."""
                value = value.strip()
                if env_pattern.search(value):
                    value = env_pattern.sub(
                        lambda m: os.getenv(m.group(1), ""), value
                    )
                return value

            app_id = resolve(str(data.get("app_id", ""))) or os.getenv("TWITCH_APP_ID", "")
            app_secret = resolve(str(data.get("app_secret", ""))) or os.getenv("TWITCH_APP_SECRET", "")
            channel = str(data.get("channel", "")).strip()

            if not app_id or not app_secret:
                await self.sio.emit(
                    "twitch_credentials_result",
                    {"success": False, "error": "App ID and App Secret are required."},
                    to=sid,
                )
                return

            try:
                from twitchAPI.twitch import Twitch

                twitch = await Twitch(app_id, app_secret)

                # Validate channel exists if provided
                if channel:
                    user_list = [u async for u in twitch.get_users(logins=[channel])]
                    if not user_list:
                        await twitch.close()
                        await self.sio.emit(
                            "twitch_credentials_result",
                            {"success": False, "error": f"Channel '{channel}' not found on Twitch."},
                            to=sid,
                        )
                        return

                await twitch.close()

                # Persist validated credentials to YAML via regex surgery
                if self._config_path:
                    self._persist_twitch_credentials(
                        data.get("channel", ""),
                        data.get("app_id", ""),
                        data.get("app_secret", ""),
                    )

                # Update stimuli cache (for next connect hydration)
                # but do NOT emit stimuli_config_updated — that would
                # overwrite the frontend's local input state with YAML values
                if self._config_path:
                    import yaml as _yaml
                    with open(Path(self._config_path), "r", encoding="utf-8") as _f:
                        _yd = _yaml.safe_load(_f) or {}
                    from ..orchestrator.config import StimuliConfig
                    _parsed = StimuliConfig.from_dict(_yd.get("stimuli", {}))
                    self._stimuli_config_cache = GUIServer._build_stimuli_hydration(_parsed)

                # If orchestrator is running, update the live module and bounce it
                if self._orchestrator and self._orchestrator.stimuli_engine:
                    cfg = self._orchestrator._config.stimuli_config
                    cfg.twitch_channel = data.get("channel", "") or cfg.twitch_channel
                    cfg.twitch_app_id = data.get("app_id", "") or cfg.twitch_app_id
                    cfg.twitch_app_secret = data.get("app_secret", "") or cfg.twitch_app_secret

                    engine = self._orchestrator.stimuli_engine
                    for module in engine.modules:
                        if module.name == "twitch":
                            module.stop()
                            module.channel = cfg.twitch_channel
                            module.app_id = cfg.twitch_app_id
                            module.app_secret = cfg.twitch_app_secret
                            module.start()
                            print("[GUI] Twitch module bounced with new credentials", flush=True)
                            break

                has_creds = bool(channel and app_id and app_secret)
                await self.sio.emit(
                    "twitch_credentials_result",
                    {"success": True, "error": None, "has_credentials": has_creds},
                    to=sid,
                )
                print(f"[GUI] Twitch credentials validated and persisted (channel={channel or 'N/A'})", flush=True)

            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "invalid" in error_msg.lower():
                    error_msg = "Invalid App ID or App Secret."
                await self.sio.emit(
                    "twitch_credentials_result",
                    {"success": False, "error": error_msg},
                    to=sid,
                )
                print(f"[GUI] Twitch credential test failed: {e}", flush=True)

        @self.sio.event
        async def typing_active(sid: str, data: dict) -> None:
            """Client signals user is typing (focus) or done typing (blur)."""
            if not self._orchestrator or not self._orchestrator.stimuli_engine:
                return
            active = data.get("active", False)
            self._orchestrator.stimuli_engine.user_typing = active

        # ============================================================
        # NANO-060b: VTubeStudio — Socket Handlers
        # ============================================================

        @self.sio.event
        async def set_vts_config(sid: str, data: dict) -> None:
            """Client updates VTubeStudio configuration at runtime (NANO-060b)."""
            if self._orchestrator:
                enabled = data.get("enabled")
                host = data.get("host")
                port = data.get("port")

                # Type coerce
                if enabled is not None:
                    enabled = bool(enabled)
                if host is not None:
                    host = str(host).strip()
                    if not host:
                        host = None
                if port is not None:
                    port = int(port)
                    if port < 1 or port > 65535:
                        port = None

                self._orchestrator.update_vts_config(
                    enabled=enabled,
                    host=host,
                    port=port,
                )

                cfg = self._orchestrator._config.vtubestudio_config
                print(
                    f"[GUI] VTS: enabled={cfg.enabled}, "
                    f"host={cfg.host}, port={cfg.port}",
                    flush=True,
                )

                # Persist to YAML
                persisted = False
                if self._config_path:
                    try:
                        self._orchestrator._config.save_to_yaml(
                            self._config_path
                        )
                        persisted = True
                        print(
                            f"[GUI] VTS config persisted to "
                            f"{Path(self._config_path).name}",
                            flush=True,
                        )
                    except Exception as e:
                        print(
                            f"[GUI] Failed to persist VTS config: {e}",
                            flush=True,
                        )

                await self.sio.emit(
                    "vts_config_updated",
                    {
                        "enabled": cfg.enabled,
                        "host": cfg.host,
                        "port": cfg.port,
                        "persisted": persisted,
                    },
                    to=sid,
                )

                # If enabling, schedule a delayed status push after driver connects
                if enabled:
                    async def _push_vts_status_after_connect(target_sid: str) -> None:
                        """Poll driver status and push to client once connected."""
                        import asyncio as _asyncio
                        for _ in range(20):  # Up to 10 seconds
                            await _asyncio.sleep(0.5)
                            driver = (
                                self._orchestrator.vts_driver
                                if self._orchestrator
                                else None
                            )
                            if driver and driver.is_connected():
                                status = driver.get_status()
                                status["enabled"] = True
                                await self.sio.emit("vts_status", status, to=target_sid)
                                return
                    asyncio.ensure_future(_push_vts_status_after_connect(sid))

        @self.sio.event
        async def request_vts_status(sid: str, data: dict) -> None:
            """Client requests VTubeStudio connection status (NANO-060b)."""
            if (
                not self._orchestrator
                or not self._orchestrator.vts_driver
            ):
                cfg_enabled = False
                if self._orchestrator:
                    cfg_enabled = self._orchestrator._config.vtubestudio_config.enabled
                await self.sio.emit(
                    "vts_status",
                    {
                        "connected": False,
                        "authenticated": False,
                        "enabled": cfg_enabled,
                        "model_name": None,
                        "hotkeys": [],
                        "expressions": [],
                    },
                    to=sid,
                )
                return

            status = self._orchestrator.vts_driver.get_status()
            status["enabled"] = self._orchestrator._config.vtubestudio_config.enabled
            await self.sio.emit("vts_status", status, to=sid)

        @self.sio.event
        async def request_vts_hotkeys(sid: str, data: dict) -> None:
            """Client requests available VTS hotkeys (NANO-060b).

            Default: serve cached list for instant render.
            With {refresh: true}: live query VTS then emit updated list.
            """
            driver = (
                self._orchestrator.vts_driver
                if self._orchestrator
                else None
            )
            if not driver:
                await self.sio.emit(
                    "vts_hotkeys", {"hotkeys": []}, to=sid,
                )
                return

            if data.get("refresh"):
                # Live query — callback fires from driver thread
                loop = self._event_loop or asyncio.get_event_loop()

                def _on_hotkeys(result: list) -> None:
                    asyncio.run_coroutine_threadsafe(
                        self.sio.emit(
                            "vts_hotkeys", {"hotkeys": result}, to=sid,
                        ),
                        loop,
                    )

                driver.request_hotkey_list(callback=_on_hotkeys)
            else:
                # Serve cached list
                status = driver.get_status()
                await self.sio.emit(
                    "vts_hotkeys",
                    {"hotkeys": status.get("hotkeys", [])},
                    to=sid,
                )

        @self.sio.event
        async def request_vts_expressions(sid: str, data: dict) -> None:
            """Client requests available VTS expressions (NANO-060b).

            Default: serve cached list for instant render.
            With {refresh: true}: live query VTS then emit updated list.
            """
            driver = (
                self._orchestrator.vts_driver
                if self._orchestrator
                else None
            )
            if not driver:
                await self.sio.emit(
                    "vts_expressions", {"expressions": []}, to=sid,
                )
                return

            if data.get("refresh"):
                # Live query — callback fires from driver thread
                loop = self._event_loop or asyncio.get_event_loop()

                def _on_expressions(result: list) -> None:
                    asyncio.run_coroutine_threadsafe(
                        self.sio.emit(
                            "vts_expressions",
                            {"expressions": result},
                            to=sid,
                        ),
                        loop,
                    )

                driver.request_expression_list(callback=_on_expressions)
            else:
                # Serve cached list
                status = driver.get_status()
                await self.sio.emit(
                    "vts_expressions",
                    {"expressions": status.get("expressions", [])},
                    to=sid,
                )

        @self.sio.event
        async def send_vts_hotkey(sid: str, data: dict) -> None:
            """Client triggers a VTS hotkey (NANO-060b)."""
            driver = (
                self._orchestrator.vts_driver
                if self._orchestrator
                else None
            )
            if not driver:
                return

            name = data.get("name")
            if name:
                driver.trigger_hotkey(str(name))
                print(f"[GUI] VTS hotkey triggered: {name}", flush=True)
                await self.sio.emit(
                    "vts_hotkey_triggered",
                    {"name": name},
                    to=sid,
                )

        @self.sio.event
        async def send_vts_expression(sid: str, data: dict) -> None:
            """Client activates/deactivates a VTS expression (NANO-060b)."""
            driver = (
                self._orchestrator.vts_driver
                if self._orchestrator
                else None
            )
            if not driver:
                return

            file = data.get("file")
            if file:
                active = bool(data.get("active", True))
                driver.trigger_expression(str(file), active)
                print(
                    f"[GUI] VTS expression {'activated' if active else 'deactivated'}: {file}",
                    flush=True,
                )
                await self.sio.emit(
                    "vts_expression_triggered",
                    {"file": file, "active": active},
                    to=sid,
                )

        @self.sio.event
        async def send_vts_move(sid: str, data: dict) -> None:
            """Client moves VTS model to a preset position (NANO-060b)."""
            driver = (
                self._orchestrator.vts_driver
                if self._orchestrator
                else None
            )
            if not driver:
                return

            preset = data.get("preset")
            if preset:
                driver.move_model(str(preset))
                print(f"[GUI] VTS move: {preset}", flush=True)
                await self.sio.emit(
                    "vts_move_triggered",
                    {"preset": preset},
                    to=sid,
                )

        # ============================================================
        # NANO-093: Avatar Config — Socket Handler
        # ============================================================

        @self.sio.event
        async def set_avatar_config(sid: str, data: dict) -> None:
            """Client updates avatar configuration at runtime (NANO-093, NANO-094)."""
            if self._orchestrator:
                config = self._orchestrator._config
                updated = False

                enabled = data.get("enabled")
                emotion_classifier = data.get("emotion_classifier")
                show_emotion_in_chat = data.get("show_emotion_in_chat")
                emotion_confidence_threshold = data.get("emotion_confidence_threshold")
                expression_fade_delay = data.get("expression_fade_delay")
                subtitles_enabled = data.get("subtitles_enabled")
                subtitle_fade_delay = data.get("subtitle_fade_delay")
                avatar_always_on_top = data.get("avatar_always_on_top")
                subtitle_always_on_top = data.get("subtitle_always_on_top")
                stream_deck_enabled = data.get("stream_deck_enabled")

                if enabled is not None:
                    config.avatar_config.enabled = bool(enabled)
                    updated = True
                if emotion_classifier is not None:
                    ec = str(emotion_classifier)
                    if ec in ("classifier", "off"):
                        config.avatar_config.emotion_classifier = ec
                        updated = True
                if show_emotion_in_chat is not None:
                    config.avatar_config.show_emotion_in_chat = bool(show_emotion_in_chat)
                    updated = True
                if emotion_confidence_threshold is not None:
                    try:
                        threshold = float(emotion_confidence_threshold)
                        if 0.0 <= threshold <= 1.0:
                            config.avatar_config.emotion_confidence_threshold = threshold
                            updated = True
                    except (ValueError, TypeError):
                        pass
                if expression_fade_delay is not None:
                    try:
                        delay = float(expression_fade_delay)
                        if 0.0 <= delay <= 10.0:
                            config.avatar_config.expression_fade_delay = delay
                            updated = True
                    except (ValueError, TypeError):
                        pass
                if subtitles_enabled is not None:
                    config.avatar_config.subtitles_enabled = bool(subtitles_enabled)
                    updated = True
                if subtitle_fade_delay is not None:
                    try:
                        delay = float(subtitle_fade_delay)
                        if 0.0 <= delay <= 10.0:
                            config.avatar_config.subtitle_fade_delay = delay
                            updated = True
                    except (ValueError, TypeError):
                        pass
                if avatar_always_on_top is not None:
                    config.avatar_config.avatar_always_on_top = bool(avatar_always_on_top)
                    updated = True
                if subtitle_always_on_top is not None:
                    config.avatar_config.subtitle_always_on_top = bool(subtitle_always_on_top)
                    updated = True
                if stream_deck_enabled is not None:
                    config.avatar_config.stream_deck_enabled = bool(stream_deck_enabled)
                    updated = True

                if updated:
                    cfg = config.avatar_config
                    print(
                        f"[GUI] Avatar: enabled={cfg.enabled}, "
                        f"classifier={cfg.emotion_classifier}, "
                        f"show_in_chat={cfg.show_emotion_in_chat}",
                        flush=True,
                    )

                    persisted = False
                    if self._config_path:
                        try:
                            config.save_to_yaml(self._config_path)
                            persisted = True
                            print(
                                f"[GUI] Avatar config persisted to "
                                f"{Path(self._config_path).name}",
                                flush=True,
                            )
                        except Exception as e:
                            print(
                                f"[GUI] Failed to persist avatar config: {e}",
                                flush=True,
                            )

                    await self.sio.emit(
                        "avatar_config_updated",
                        {
                            "enabled": cfg.enabled,
                            "emotion_classifier": cfg.emotion_classifier,
                            "show_emotion_in_chat": cfg.show_emotion_in_chat,
                            "emotion_confidence_threshold": cfg.emotion_confidence_threshold,
                            "expression_fade_delay": cfg.expression_fade_delay,
                            "subtitles_enabled": cfg.subtitles_enabled,
                            "subtitle_fade_delay": cfg.subtitle_fade_delay,
                            "avatar_always_on_top": cfg.avatar_always_on_top,
                            "subtitle_always_on_top": cfg.subtitle_always_on_top,
                            "stream_deck_enabled": cfg.stream_deck_enabled,
                            "persisted": persisted,
                        },
                    )

                # NANO-097: Process management on enabled transition
                if enabled is not None:
                    if bool(enabled):
                        await self._avatar_spawn()
                    else:
                        self._avatar_kill()

                # NANO-100: Subtitle process management on toggle
                if subtitles_enabled is not None:
                    if bool(subtitles_enabled):
                        await self._subtitle_spawn()
                    else:
                        self._subtitle_kill()

                # NANO-110: Stream Deck process management on toggle
                if stream_deck_enabled is not None:
                    if bool(stream_deck_enabled):
                        await self._stream_deck_spawn()
                    else:
                        self._stream_deck_kill()

        # ============================================================
        # NANO-099: Avatar Rescan Animations — Socket Handler
        # ============================================================

        @self.sio.event
        async def avatar_rescan_animations(sid: str) -> None:
            """Notify avatar renderer to rescan animation files (NANO-099)."""
            print("[GUI] Avatar rescan animations requested", flush=True)
            # Reload base_animations from YAML into in-memory config
            if self._orchestrator and self._config_path:
                try:
                    import yaml as _yaml
                    with open(self._config_path, "r", encoding="utf-8") as f:
                        raw = _yaml.safe_load(f)
                    ba = (raw.get("avatar") or {}).get("base_animations") or {}
                    self._orchestrator._config.avatar_config.base_animations = {
                        "idle": ba.get("idle"),
                        "happy": ba.get("happy"),
                        "sad": ba.get("sad"),
                    }
                except Exception as e:
                    print(f"[GUI] Failed to reload base_animations: {e}", flush=True)
            # Forward to all connected clients with base_animations config
            ba = {}
            if self._orchestrator:
                ba = self._orchestrator._config.avatar_config.base_animations
            await self.sio.emit("avatar_rescan_animations", {"base_animations": ba})

        # ============================================================
        # NANO-042: Reasoning Config — Socket Handler
        # ============================================================

        @self.sio.event
        async def set_reasoning_config(sid: str, data: dict) -> None:
            """Client updates reasoning/thinking budget (persisted to YAML)."""
            reasoning_budget = data.get("reasoning_budget")
            if reasoning_budget is None:
                return

            reasoning_budget = int(reasoning_budget)
            persisted = False

            if self._config_path:
                try:
                    from spindl.orchestrator.config import _make_ruamel_yaml

                    config_path = Path(self._config_path)
                    ry = _make_ruamel_yaml()
                    with open(config_path, "r", encoding="utf-8") as f:
                        yaml_data = ry.load(f)

                    llama = yaml_data.get("llm", {}).get("providers", {}).get("llama")
                    if llama is not None:
                        llama["reasoning_budget"] = reasoning_budget
                        with open(config_path, "w", encoding="utf-8") as f:
                            ry.dump(yaml_data, f)
                        persisted = True

                    print(
                        f"[GUI] Reasoning: budget={reasoning_budget} persisted to {config_path.name}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[GUI] Failed to persist reasoning config: {e}", flush=True)

            await self.sio.emit(
                "reasoning_config_updated",
                {
                    "reasoning_budget": reasoning_budget,
                    "persisted": persisted,
                },
                to=sid,
            )

        # ============================================================
        # NANO-043 Phase 6: Memory Curation GUI — Socket Handlers
        # ============================================================

        @self.sio.event
        async def request_memory_counts(sid: str, data: dict) -> None:
            """Client requests memory collection counts."""
            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_counts",
                    {"global": 0, "general": 0, "flashcards": 0, "summaries": 0, "enabled": False},
                    to=sid,
                )
                return

            try:
                counts = self._orchestrator.memory_store.counts
                await self.sio.emit(
                    "memory_counts",
                    {**counts, "enabled": True},
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Memory counts error: {e}", flush=True)
                await self.sio.emit(
                    "memory_counts",
                    {"global": 0, "general": 0, "flashcards": 0, "summaries": 0, "enabled": True, "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def request_memories(sid: str, data: dict) -> None:
            """Client requests all memories from a specific collection."""
            collection = data.get("collection")
            if not collection or not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_list",
                    {
                        "collection": collection or "unknown",
                        "memories": [],
                        "error": "Memory system not available" if not (self._orchestrator and self._orchestrator.memory_store) else None,
                    },
                    to=sid,
                )
                return

            try:
                memories = self._orchestrator.memory_store.get_all(collection)
                await self.sio.emit(
                    "memory_list",
                    {"collection": collection, "memories": memories},
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Memory list error ({collection}): {e}", flush=True)
                await self.sio.emit(
                    "memory_list",
                    {"collection": collection, "memories": [], "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def add_general_memory(sid: str, data: dict) -> None:
            """Client adds a new general memory."""
            content = data.get("content", "").strip()
            if not content:
                await self.sio.emit(
                    "memory_added",
                    {"success": False, "error": "Content is required", "collection": "general"},
                    to=sid,
                )
                return

            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_added",
                    {"success": False, "error": "Memory system not available", "collection": "general"},
                    to=sid,
                )
                return

            try:
                metadata = {"type": "general", "source": "gui_manual"}
                loop = asyncio.get_event_loop()
                doc_id = await loop.run_in_executor(
                    None,
                    self._orchestrator.memory_store.add_general,
                    content,
                    metadata,
                )
                print(f"[GUI] General memory added: {doc_id}", flush=True)
                await self.sio.emit(
                    "memory_added",
                    {
                        "success": True,
                        "collection": "general",
                        "memory": {"id": doc_id, "content": content, "metadata": metadata},
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Add memory error: {e}", flush=True)
                await self.sio.emit(
                    "memory_added",
                    {"success": False, "error": str(e), "collection": "general"},
                    to=sid,
                )

        @self.sio.event
        async def edit_general_memory(sid: str, data: dict) -> None:
            """Client edits a general memory (delete + re-add with new embedding)."""
            doc_id = data.get("id")
            new_content = data.get("content", "").strip()

            if not doc_id or not new_content:
                await self.sio.emit(
                    "memory_edited",
                    {"success": False, "error": "ID and content are required"},
                    to=sid,
                )
                return

            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_edited",
                    {"success": False, "error": "Memory system not available"},
                    to=sid,
                )
                return

            store = self._orchestrator.memory_store
            try:
                # Get original metadata before deleting
                originals = store.get_all("general")
                original = next((m for m in originals if m["id"] == doc_id), None)
                original_meta = original["metadata"] if original else {}

                # Delete old
                store.delete("general", doc_id)

                # Re-add with new content, preserving original timestamp
                metadata = {
                    **original_meta,
                    "type": "general",
                    "source": "gui_manual",
                    "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                loop = asyncio.get_event_loop()
                new_id = await loop.run_in_executor(
                    None,
                    store.add_general,
                    new_content,
                    metadata,
                )

                print(f"[GUI] General memory edited: {doc_id} -> {new_id}", flush=True)
                await self.sio.emit(
                    "memory_edited",
                    {
                        "success": True,
                        "old_id": doc_id,
                        "memory": {"id": new_id, "content": new_content, "metadata": metadata},
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Edit memory error: {e}", flush=True)
                await self.sio.emit(
                    "memory_edited",
                    {"success": False, "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def add_global_memory(sid: str, data: dict) -> None:
            """Client adds a new global memory (cross-character, NANO-105)."""
            content = data.get("content", "").strip()
            if not content:
                await self.sio.emit(
                    "memory_added",
                    {"success": False, "error": "Content is required", "collection": "global"},
                    to=sid,
                )
                return

            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_added",
                    {"success": False, "error": "Memory system not available", "collection": "global"},
                    to=sid,
                )
                return

            try:
                metadata = {"type": "global", "source": "gui_manual"}
                loop = asyncio.get_event_loop()
                doc_id = await loop.run_in_executor(
                    None,
                    self._orchestrator.memory_store.add_global,
                    content,
                    metadata,
                )
                print(f"[GUI] Global memory added: {doc_id}", flush=True)
                await self.sio.emit(
                    "memory_added",
                    {
                        "success": True,
                        "collection": "global",
                        "memory": {"id": doc_id, "content": content, "metadata": metadata},
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Add global memory error: {e}", flush=True)
                await self.sio.emit(
                    "memory_added",
                    {"success": False, "error": str(e), "collection": "global"},
                    to=sid,
                )

        @self.sio.event
        async def edit_global_memory(sid: str, data: dict) -> None:
            """Client edits a global memory (NANO-105)."""
            doc_id = data.get("id")
            new_content = data.get("content", "").strip()

            if not doc_id or not new_content:
                await self.sio.emit(
                    "memory_edited",
                    {"success": False, "error": "ID and content are required"},
                    to=sid,
                )
                return

            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_edited",
                    {"success": False, "error": "Memory system not available"},
                    to=sid,
                )
                return

            store = self._orchestrator.memory_store
            try:
                loop = asyncio.get_event_loop()
                new_id = await loop.run_in_executor(
                    None,
                    store.edit_global,
                    doc_id,
                    new_content,
                )

                print(f"[GUI] Global memory edited: {doc_id} -> {new_id}", flush=True)
                metadata = {"type": "global", "source": "gui_manual", "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                await self.sio.emit(
                    "memory_edited",
                    {
                        "success": True,
                        "old_id": doc_id,
                        "memory": {"id": new_id, "content": new_content, "metadata": metadata},
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Edit global memory error: {e}", flush=True)
                await self.sio.emit(
                    "memory_edited",
                    {"success": False, "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def delete_memory(sid: str, data: dict) -> None:
            """Client deletes a memory from any collection."""
            collection = data.get("collection")
            doc_id = data.get("id")

            if not collection or not doc_id:
                await self.sio.emit(
                    "memory_deleted",
                    {"success": False, "error": "Collection and ID are required"},
                    to=sid,
                )
                return

            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_deleted",
                    {"success": False, "error": "Memory system not available"},
                    to=sid,
                )
                return

            try:
                success = self._orchestrator.memory_store.delete(collection, doc_id)
                print(f"[GUI] Memory deleted: {collection}/{doc_id} (success={success})", flush=True)
                await self.sio.emit(
                    "memory_deleted",
                    {"success": success, "collection": collection, "id": doc_id},
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Delete memory error: {e}", flush=True)
                await self.sio.emit(
                    "memory_deleted",
                    {"success": False, "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def promote_memory(sid: str, data: dict) -> None:
            """Promote a flash card or summary to general memory."""
            source_collection = data.get("source_collection")
            doc_id = data.get("id")
            delete_source = data.get("delete_source", False)

            if not source_collection or not doc_id:
                await self.sio.emit(
                    "memory_promoted",
                    {"success": False, "error": "Source collection and ID required"},
                    to=sid,
                )
                return

            if source_collection not in ("flashcards", "summaries"):
                await self.sio.emit(
                    "memory_promoted",
                    {"success": False, "error": "Can only promote from flashcards or summaries"},
                    to=sid,
                )
                return

            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_promoted",
                    {"success": False, "error": "Memory system not available"},
                    to=sid,
                )
                return

            store = self._orchestrator.memory_store
            try:
                # Get the source document
                all_docs = store.get_all(source_collection)
                source_doc = next((m for m in all_docs if m["id"] == doc_id), None)
                if not source_doc:
                    await self.sio.emit(
                        "memory_promoted",
                        {"success": False, "error": "Source memory not found"},
                        to=sid,
                    )
                    return

                # Add to general with promotion metadata
                metadata = {
                    **source_doc.get("metadata", {}),
                    "type": "general",
                    "source": f"promoted_from_{source_collection}",
                    "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                loop = asyncio.get_event_loop()
                new_id = await loop.run_in_executor(
                    None,
                    store.add_general,
                    source_doc["content"],
                    metadata,
                )

                # Optionally delete from source
                if delete_source:
                    store.delete(source_collection, doc_id)

                print(
                    f"[GUI] Memory promoted: {source_collection}/{doc_id} -> general/{new_id}"
                    f" (source {'deleted' if delete_source else 'kept'})",
                    flush=True,
                )
                await self.sio.emit(
                    "memory_promoted",
                    {
                        "success": True,
                        "source_collection": source_collection,
                        "source_id": doc_id,
                        "new_id": new_id,
                        "deleted_source": delete_source,
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Promote memory error: {e}", flush=True)
                await self.sio.emit(
                    "memory_promoted",
                    {"success": False, "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def search_memories(sid: str, data: dict) -> None:
            """Semantic search across all memory collections."""
            query_text = data.get("query", "").strip()
            top_k = data.get("top_k", 10)

            if not query_text:
                await self.sio.emit(
                    "memory_search_results",
                    {"results": [], "query": ""},
                    to=sid,
                )
                return

            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "memory_search_results",
                    {"results": [], "query": query_text, "error": "Memory system not available"},
                    to=sid,
                )
                return

            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    self._orchestrator.memory_store.query,
                    query_text,
                    top_k,
                )
                await self.sio.emit(
                    "memory_search_results",
                    {"results": results, "query": query_text},
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Memory search error: {e}", flush=True)
                await self.sio.emit(
                    "memory_search_results",
                    {"results": [], "query": query_text, "error": str(e)},
                    to=sid,
                )

        @self.sio.event
        async def clear_flashcards(sid: str, data: dict) -> None:
            """Clear all flash cards."""
            if not self._orchestrator or not self._orchestrator.memory_store:
                await self.sio.emit(
                    "flashcards_cleared",
                    {"success": False, "error": "Memory system not available"},
                    to=sid,
                )
                return

            try:
                self._orchestrator.memory_store.clear_flash_cards()
                print("[GUI] Flash cards cleared", flush=True)
                await self.sio.emit(
                    "flashcards_cleared",
                    {"success": True},
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Clear flashcards error: {e}", flush=True)
                await self.sio.emit(
                    "flashcards_cleared",
                    {"success": False, "error": str(e)},
                    to=sid,
                )

        # ============================================================
        # End NANO-043 Phase 6 Memory Handlers
        # ============================================================

        @self.sio.event
        async def pause_listening(sid: str) -> None:
            """Client requests to pause listening."""
            if self._orchestrator:
                success = self._orchestrator.pause_listening()
                if success:
                    print("[GUI] Listening paused", flush=True)
                # Emit updated state so frontend syncs
                await self._emit_state(sid)

        @self.sio.event
        async def resume_listening(sid: str) -> None:
            """Client requests to resume listening."""
            if self._orchestrator:
                success = self._orchestrator.resume_listening()
                if success:
                    print("[GUI] Listening resumed", flush=True)
                # Emit updated state so frontend syncs
                await self._emit_state(sid)

        # === NANO-027 Phase 3: Service Launch Events ===

        @self.sio.event
        async def start_services(sid: str, data: dict) -> None:
            """
            Client requests to start services (GUI-first mode).

            This triggers the service launcher to boot configured services.
            Progress is reported back via launch_progress events.

            Args:
                data: Optional dict with:
                    - services: list of service names to start (optional, defaults to all)
                    - skip_orchestrator: bool, if True don't start orchestrator
            """
            # Guard: reject if services are already running (NANO-070)
            if self._orchestrator is not None:
                await self.sio.emit(
                    "launch_error",
                    {"error": "Services already running. Shut down first.", "service": None},
                    to=sid,
                )
                return

            if self._launch_in_progress:
                await self.sio.emit(
                    "launch_error",
                    {"error": "Launch already in progress", "service": None},
                    to=sid,
                )
                return

            self._launch_in_progress = True
            print("[GUI] Service launch requested", flush=True)

            # Acknowledge receipt
            await self.sio.emit(
                "launch_progress",
                {"status": "starting", "service": None, "message": "Initializing launcher..."},
            )

            # Run launch in background thread to not block event loop
            def launch_thread():
                asyncio.run(self._launch_services_async(data))

            thread = threading.Thread(target=launch_thread, daemon=True)
            thread.start()

        @self.sio.event
        async def request_launch_status(sid: str, data: dict) -> None:
            """Client requests current launch status."""
            await self.sio.emit(
                "launch_status",
                {
                    "in_progress": self._launch_in_progress,
                    "launched_services": list(self._launched_services),
                    "has_orchestrator": self._orchestrator is not None,
                },
                to=sid,
            )

        # === NANO-028: Graceful Shutdown Events ===

        @self.sio.event
        async def shutdown_backend(sid: str) -> None:
            """
            Client requests graceful shutdown of orchestrator and services.

            This stops the orchestrator, event bridge, and all launched services,
            then notifies the client to redirect to the launcher page.
            """
            if self._shutdown_in_progress:
                await self.sio.emit(
                    "shutdown_error",
                    {"error": "Shutdown already in progress"},
                    to=sid,
                )
                return

            if self._launch_in_progress:
                await self.sio.emit(
                    "shutdown_error",
                    {"error": "Cannot shutdown while launch is in progress"},
                    to=sid,
                )
                return

            self._shutdown_in_progress = True
            print("[GUI] Backend shutdown requested", flush=True)

            # Run shutdown in background thread to not block event loop
            def shutdown_thread():
                asyncio.run(self._shutdown_backend_async())

            thread = threading.Thread(target=shutdown_thread, daemon=True)
            thread.start()

        # === NANO-029: E2E Test Injection ===

        @self.sio.event
        async def test_inject_transcription(sid: str, data: dict) -> dict:
            """
            Test-only: Inject text as if STT transcribed it.

            NANO-029: E2E test automation endpoint.
            Gated behind SPINDL_TEST_MODE environment variable.

            Args:
                data: Dict with:
                    - text: Transcription text to inject

            Returns:
                Dict with success status or error message.
            """
            if not os.environ.get("SPINDL_TEST_MODE"):
                return {"success": False, "error": "Not in test mode"}

            if not self._orchestrator:
                return {"success": False, "error": "No orchestrator attached"}

            text = data.get("text", "")
            if not text:
                return {"success": False, "error": "No text provided"}

            # Inject text through callbacks
            callbacks = self._orchestrator.callbacks
            if not callbacks:
                return {"success": False, "error": "Orchestrator callbacks not available"}

            # Process the text input (runs in background thread)
            callbacks.process_text_input(text)

            print(f"[GUI] Test injection: '{text[:50]}...' " if len(text) > 50 else f"[GUI] Test injection: '{text}'", flush=True)
            return {"success": True}

        # === NANO-031: User-Facing Text Input ===

        @self.sio.event
        async def send_message(sid: str, data: dict) -> dict:
            """
            Send a text message through the pipeline.

            NANO-031: User-facing endpoint for text input mode.
            Bypasses STT/VAD and processes text directly through LLM → TTS.
            Works in both GUI-first and backend+GUI workflows.

            Args:
                data: Dict with:
                    - text: Message text
                    - skip_tts: Optional bool, if True skip TTS synthesis

            Returns:
                Dict with success status or error message.
            """
            if not self._orchestrator:
                return {"success": False, "error": "Orchestrator not ready"}

            text = data.get("text", "").strip()
            if not text:
                return {"success": False, "error": "No text provided"}

            skip_tts = data.get("skip_tts", False)

            # Get callbacks from orchestrator
            callbacks = self._orchestrator.callbacks
            if not callbacks:
                return {"success": False, "error": "Callbacks not available"}

            # Process text input (runs in background thread)
            callbacks.process_text_input(text, skip_tts=skip_tts)

            print(f"[GUI] Text input: '{text[:50]}...'" if len(text) > 50 else f"[GUI] Text input: '{text}'", flush=True)
            return {"success": True}

        # === NANO-036: Character Hot-Reload ===

        @self.sio.event
        async def reload_character(sid: str, data: dict) -> dict:
            """
            Reload current character from disk after GUI save.

            NANO-036: Hot-reload character data without restarting the backend.
            Only permitted when agent state is IDLE or LISTENING (not mid-conversation).

            Args:
                data: Optional dict (currently unused, reserved for future options)

            Returns:
                Dict with:
                    - success: bool
                    - character_id: str (on success)
                    - error: str (on failure)
                    - current_state: str (on state-gated failure)
            """
            if not self._orchestrator:
                return {"success": False, "error": "Orchestrator not ready"}

            # Import AgentState at runtime to avoid circular imports
            from spindl.core import AgentState

            # Check current state
            current_state = self._orchestrator.state
            if current_state not in (AgentState.IDLE, AgentState.LISTENING):
                return {
                    "success": False,
                    "error": f"Cannot reload while {current_state.value}",
                    "current_state": current_state.value,
                }

            # Attempt reload
            try:
                success = self._orchestrator.reload_persona()
                if success:
                    character_id = self._orchestrator.persona.get("id") if self._orchestrator.persona else "unknown"
                    print(f"[GUI] Character reloaded: {character_id}", flush=True)
                    return {"success": True, "character_id": character_id}
                else:
                    return {"success": False, "error": "Reload failed (state changed during reload)"}
            except Exception as e:
                print(f"[GUI] Character reload error: {e}", flush=True)
                return {"success": False, "error": str(e)}

    def _get_characters_dir(self) -> Optional[str]:
        """Get characters directory from config or orchestrator."""
        if self._orchestrator and hasattr(self._orchestrator._config, "characters_dir"):
            return self._orchestrator._config.characters_dir

        # Try to read from config file
        if self._config_path:
            try:
                import yaml
                with open(self._config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config and "character" in config:
                        return config["character"].get("characters_dir", "./characters")
                    # Fallback to default
                    return "./characters"
            except Exception:
                pass

        return "./characters"

    async def _launch_services_async(self, data: dict) -> None:
        """
        Launch services asynchronously (NANO-027 Phase 3).

        This is run in a background thread to not block the Socket.IO event loop.
        """
        try:
            from spindl.launcher import (
                load_launcher_config,
                ServiceRunner,
                LogAggregator,
            )

            # Load launcher configuration
            if not self._config_path:
                await self._emit_launch_error("No config path set")
                return

            await self._emit_launch_progress("loading_config", None, "Loading configuration...")

            try:
                config = load_launcher_config(self._config_path)
            except Exception as e:
                await self._emit_launch_error(f"Failed to load config: {e}")
                return

            await self._emit_launch_progress(
                "config_loaded", None,
                f"Found {len(config.services)} services"
            )

            # Get startup order
            try:
                startup_order = config.get_startup_order()
            except ValueError as e:
                await self._emit_launch_error(f"Dependency error: {e}")
                return

            # Filter services if requested
            requested_services = data.get("services")
            skip_orchestrator = data.get("skip_orchestrator", False)

            if requested_services:
                startup_order = [s for s in startup_order if s in requested_services]

            # Auto-skip orchestrator in standalone mode (GUI-first)
            # Standalone mode initializes orchestrator in-process via initialize_orchestrator()
            # rather than spawning it as a separate service
            if self._on_services_ready is not None and "orchestrator" in startup_order:
                print("[GUI] Standalone mode: skipping orchestrator service (will init in-process)", flush=True)
                startup_order.remove("orchestrator")
            elif skip_orchestrator and "orchestrator" in startup_order:
                startup_order.remove("orchestrator")

            # Set up logging
            self._log_aggregator = LogAggregator(
                log_file=config.log_file,
                log_level=config.log_level,
                service_levels=config.service_levels,
                suppress_patterns=config.suppress_patterns,
            )

            # Create service runner
            self._service_runner = ServiceRunner(
                logger=self._log_aggregator,
                default_health_timeout=config.health_check_timeout,
                debug_mode=False,
                tts_provider_config=config.tts_provider_config,
                llm_provider_config=config.llm_provider_config,
                vision_provider_config=config.vision_provider_config,
                llm_context_size=config.llm_context_size,
            )

            # Start services in order
            for name in startup_order:
                svc_config = config.services[name]

                if not svc_config.enabled:
                    await self._emit_launch_progress(
                        "skipped", name, f"Skipping disabled service: {name}"
                    )
                    continue

                await self._emit_launch_progress(
                    "starting", name, f"Starting {name}..."
                )

                success = self._service_runner.start_service(svc_config)

                if success:
                    self._launched_services.add(name)
                    await self._emit_launch_progress(
                        "started", name, f"{name} started successfully"
                    )
                else:
                    await self._emit_launch_error(
                        f"Failed to start {name}", service=name
                    )
                    # Shutdown already-started services
                    self._service_runner.shutdown_all()
                    self._launched_services.clear()
                    return

            # All services started successfully
            await self._emit_launch_progress(
                "complete", None, "All services started"
            )

            # Emit launch complete event
            await self.sio.emit(
                "launch_complete",
                {"services": list(self._launched_services)},
            )

            # Call callback if set (for standalone mode orchestrator initialization)
            if self._on_services_ready:
                result = self._on_services_ready()
                # Await if callback is async
                if asyncio.iscoroutine(result):
                    await result

        except Exception as e:
            import traceback
            traceback.print_exc()
            await self._emit_launch_error(f"Unexpected error: {e}")
        finally:
            self._launch_in_progress = False

    async def _emit_launch_progress(
        self, status: str, service: Optional[str], message: str
    ) -> None:
        """Emit launch progress event to all clients."""
        await self.sio.emit(
            "launch_progress",
            {"status": status, "service": service, "message": message},
        )

    async def _emit_launch_error(
        self, error: str, service: Optional[str] = None
    ) -> None:
        """Emit launch error event to all clients."""
        print(f"[GUI] Launch error: {error}", flush=True)
        await self.sio.emit(
            "launch_error",
            {"error": error, "service": service},
        )
        self._launch_in_progress = False

    # === NANO-028: Graceful Shutdown Implementation ===

    async def _shutdown_backend_async(self) -> None:
        """
        Shutdown backend asynchronously (NANO-028).

        This is run in a background thread to not block the Socket.IO event loop.
        Stops orchestrator, event bridge, and all launched services in order.
        """
        try:
            # Step 1: Stop orchestrator
            if self._orchestrator:
                await self._emit_shutdown_progress("Stopping orchestrator...")
                try:
                    self._orchestrator.stop()
                    self._orchestrator = None
                    print("[GUI] Orchestrator stopped", flush=True)
                except Exception as e:
                    print(f"[GUI] Error stopping orchestrator: {e}", flush=True)
                    # Continue with shutdown even if orchestrator stop fails

            # Step 1b: Stop avatar and subtitle processes
            self._avatar_kill()
            self._subtitle_kill()

            # Step 2: Stop services
            if self._service_runner:
                await self._emit_shutdown_progress("Stopping services...")
                try:
                    self._service_runner.shutdown_all()
                    self._launched_services.clear()
                    print("[GUI] Services stopped", flush=True)
                except Exception as e:
                    print(f"[GUI] Error stopping services: {e}", flush=True)

            # Step 3: Close log aggregator
            if self._log_aggregator:
                await self._emit_shutdown_progress("Closing logs...")
                try:
                    self._log_aggregator.close()
                    self._log_aggregator = None
                except Exception as e:
                    print(f"[GUI] Error closing log aggregator: {e}", flush=True)

            # Shutdown complete
            await self._emit_shutdown_complete()
            print("[GUI] Backend shutdown complete", flush=True)

            # Step 4: Stop the uvicorn server to exit the process (NANO-028)
            if self._uvicorn_server:
                print("[GUI] Triggering server exit...", flush=True)
                self._uvicorn_server.should_exit = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            await self._emit_shutdown_error(f"Unexpected error: {e}")
        finally:
            self._shutdown_in_progress = False

    async def _emit_shutdown_progress(self, message: str) -> None:
        """Emit shutdown progress event to all clients."""
        await self.sio.emit(
            "shutdown_progress",
            {"message": message},
        )

    async def _emit_shutdown_complete(self) -> None:
        """Emit shutdown complete event to all clients."""
        from datetime import datetime
        await self.sio.emit(
            "shutdown_complete",
            {"timestamp": datetime.now().isoformat()},
        )

    async def _emit_shutdown_error(self, error: str) -> None:
        """Emit shutdown error event to all clients."""
        print(f"[GUI] Shutdown error: {error}", flush=True)
        await self.sio.emit(
            "shutdown_error",
            {"error": error},
        )
        self._shutdown_in_progress = False

    def set_services_ready_callback(
        self, callback: Callable[[], Union[None, Awaitable[None]]]
    ) -> None:
        """Set callback for when services are ready (standalone mode).

        Callback can be sync or async. If async, it will be awaited.
        """
        self._on_services_ready = callback

    def shutdown_services(self) -> None:
        """Shutdown all launched services, avatar, and subtitle processes."""
        self._avatar_kill()
        self._subtitle_kill()
        if self._service_runner:
            print("[GUI] Shutting down services...", flush=True)
            self._service_runner.shutdown_all()
            self._launched_services.clear()
        if self._log_aggregator:
            self._log_aggregator.close()

    # === NANO-027 Phase 4: Orchestrator Ready Events ===

    async def emit_orchestrator_ready(self, persona_name: str) -> None:
        """
        Emit event when orchestrator has initialized successfully.

        Called from run_gui_standalone.py after services launch and
        orchestrator startup completes.

        Also broadcasts initial state (config, health, state) to all connected
        clients, since they may have connected before the orchestrator existed
        and missed the initial state emissions from the connect handler.
        """
        print(f"[GUI] Orchestrator ready, persona: {persona_name}", flush=True)
        await self.sio.emit(
            "orchestrator_ready",
            {
                "persona": persona_name,
                "has_orchestrator": True,
            },
        )

        # Broadcast initial state to all connected clients
        # Clients connected during launcher phase missed the connect-time emissions
        if self._orchestrator:
            await self._emit_config()
            await self._emit_health()
            await self._emit_state()

    async def emit_orchestrator_error(self, error: str) -> None:
        """
        Emit event when orchestrator initialization fails.

        Called from run_gui_standalone.py if orchestrator startup fails.
        """
        print(f"[GUI] Orchestrator init failed: {error}", flush=True)
        await self.sio.emit(
            "orchestrator_error",
            {"error": error},
        )

    async def _emit_state(self, sid: Optional[str] = None) -> None:
        """Emit current agent state."""
        if not self._orchestrator:
            return

        state_data = {
            "state": self._orchestrator.state.value,
        }

        if sid:
            await self.sio.emit("state_snapshot", state_data, to=sid)
        else:
            await self.sio.emit("state_snapshot", state_data)

    async def _emit_health(self, sid: Optional[str] = None) -> None:
        """Emit service health status."""
        if not self._orchestrator:
            return

        health = self._orchestrator.health_check()
        health_data = {
            "stt": health.get("stt", False),
            "tts": health.get("tts", False),
            "llm": health.get("llm", False),
            "vlm": health.get("vlm", False),
            "embedding": health.get("embedding", False),
            "mic": health.get("mic", "ok"),
        }

        if sid:
            await self.sio.emit("health_status", health_data, to=sid)
        else:
            await self.sio.emit("health_status", health_data)

    async def _emit_config(self, sid: Optional[str] = None) -> None:
        """Emit current configuration."""
        if not self._orchestrator:
            return

        config = self._orchestrator._config
        persona = self._orchestrator.persona or {}

        config_data = {
            "persona": {
                "id": persona.get("id", "unknown"),
                "name": persona.get("name", "Unknown"),
                "voice": persona.get("voice"),
            },
            "providers": {
                "llm": self._get_llm_provider_info(config),
                "tts": {
                    "name": config.tts_config.provider,
                    "config": {},
                },
                "stt": {
                    "name": config.stt_config.provider,
                    "config": config.stt_config.provider_config,
                },
                "vlm": self._get_vlm_provider_info(config),
                "embedding": {
                    "base_url": config.memory_config.embedding_base_url,
                    "enabled": config.memory_config.enabled,
                },
            },
            "settings": {
                "vad": {
                    "threshold": config.vad_threshold,
                    "min_speech_ms": config.min_speech_ms,
                    "min_silence_ms": config.min_silence_ms,
                },
                "pipeline": {
                    "summarization_threshold": config.summarization_threshold,
                    "budget_strategy": config.budget_strategy,
                },
                "memory": {
                    "top_k": config.memory_config.rag_top_k,
                    "relevance_threshold": config.memory_config.relevance_threshold,
                    "dedup_threshold": config.memory_config.dedup_threshold,
                    "reflection_interval": config.memory_config.reflection_interval,
                    "reflection_prompt": config.memory_config.reflection_prompt,
                    "reflection_system_message": config.memory_config.reflection_system_message,
                    "reflection_delimiter": config.memory_config.reflection_delimiter,
                    "enabled": config.memory_config.enabled,
                    "curation": {
                        "enabled": config.memory_config.curation.enabled,
                        "api_key": config.memory_config.curation.api_key,
                        "model": config.memory_config.curation.model,
                        "prompt": config.memory_config.curation.prompt,
                        "timeout": config.memory_config.curation.timeout,
                    },
                },
                "prompt": {
                    "rag_prefix": config.prompt_config.rag_prefix,
                    "rag_suffix": config.prompt_config.rag_suffix,
                    "codex_prefix": config.prompt_config.codex_prefix,
                    "codex_suffix": config.prompt_config.codex_suffix,
                    "example_dialogue_prefix": config.prompt_config.example_dialogue_prefix,
                    "example_dialogue_suffix": config.prompt_config.example_dialogue_suffix,
                },
                "generation": {
                    "temperature": config.llm_config.provider_config.get("temperature", 0.7),
                    "max_tokens": config.llm_config.provider_config.get("max_tokens", 256),
                    "top_p": config.llm_config.provider_config.get("top_p", 0.95),
                    "repeat_penalty": config.llm_config.provider_config.get("repeat_penalty", 1.1),
                    "repeat_last_n": config.llm_config.provider_config.get("repeat_last_n", 64),
                    "frequency_penalty": config.llm_config.provider_config.get("frequency_penalty", 0.0),
                    "presence_penalty": config.llm_config.provider_config.get("presence_penalty", 0.0),
                },
                "stimuli": self._build_stimuli_hydration(config.stimuli_config),
                # NANO-065a: Tools runtime state
                "tools": {
                    "master_enabled": config.tools_config.enabled,
                    "tools": {
                        name: {"enabled": tool_cfg.get("enabled", True)}
                        for name, tool_cfg in config.tools_config.tools.items()
                    },
                },
                # NANO-093/094: Avatar config
                "avatar": {
                    "enabled": config.avatar_config.enabled,
                    "emotion_classifier": config.avatar_config.emotion_classifier,
                    "show_emotion_in_chat": config.avatar_config.show_emotion_in_chat,
                    "emotion_confidence_threshold": config.avatar_config.emotion_confidence_threshold,
                    "expression_fade_delay": config.avatar_config.expression_fade_delay,
                    "subtitles_enabled": config.avatar_config.subtitles_enabled,
                    "subtitle_fade_delay": config.avatar_config.subtitle_fade_delay,
                    "avatar_always_on_top": config.avatar_config.avatar_always_on_top,
                    "subtitle_always_on_top": config.avatar_config.subtitle_always_on_top,
                    "stream_deck_enabled": config.avatar_config.stream_deck_enabled,
                },
                # NANO-065b: LLM provider runtime state
                "llm": self._orchestrator.get_llm_state(),
            },
        }

        if sid:
            await self.sio.emit("config_loaded", config_data, to=sid)
        else:
            await self.sio.emit("config_loaded", config_data)

    @staticmethod
    def _extract_model_name(model_path: str) -> str:
        """
        Extract clean model name from model path.

        Removes:
        - Directory path (keeps filename only)
        - .gguf extension
        - Common quantization suffixes (Q8_0, Q4_K_M, IQ4_XS, etc.)

        Args:
            model_path: Full path to model file

        Returns:
            Clean model name for display
        """
        # Handle both forward and backslash paths
        path = Path(model_path)
        model_name = path.stem  # filename without extension

        # Quantization suffixes to strip (ordered by specificity)
        # Standard quantization: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_K_S, Q4_0, Q3_K_M, Q2_K
        # Importance matrix: IQ4_XS, IQ4_NL, IQ3_XXS, IQ3_S, IQ2_XXS, IQ1_S
        # Float formats: f16, f32, bf16
        suffixes = [
            # K-quants with size variants (most specific first)
            "-Q6_K_L", "-Q5_K_L", "-Q4_K_L", "-Q3_K_L",
            "-Q5_K_M", "-Q4_K_M", "-Q3_K_M", "-Q2_K_M",
            "-Q5_K_S", "-Q4_K_S", "-Q3_K_S", "-Q2_K_S",
            "-Q6_K", "-Q5_K", "-Q4_K", "-Q3_K", "-Q2_K",
            # Standard quants
            "-Q8_0", "-Q5_0", "-Q4_0", "-Q3_0", "-Q2_0",
            "-Q5_1", "-Q4_1",
            # Importance matrix quants
            "-IQ4_XS", "-IQ4_NL", "-IQ3_XXS", "-IQ3_XS", "-IQ3_S", "-IQ3_M",
            "-IQ2_XXS", "-IQ2_XS", "-IQ2_S", "-IQ2_M",
            "-IQ1_S", "-IQ1_M",
            # Float formats
            "-f16", "-f32", "-bf16",
        ]

        for suffix in suffixes:
            if model_name.endswith(suffix):
                model_name = model_name[:-len(suffix)]
                break

        return model_name

    def _persist_twitch_credentials(
        self, channel: str, app_id: str, app_secret: str
    ) -> None:
        """Persist Twitch credentials to YAML via ruamel.yaml round-trip (NANO-106)."""
        cfg_path = Path(self._config_path)
        if not cfg_path.exists():
            return

        fields = {}
        if channel:
            fields["channel"] = channel
        if app_id:
            fields["app_id"] = app_id
        if app_secret:
            fields["app_secret"] = app_secret

        if not fields:
            return

        from spindl.orchestrator.config import _make_ruamel_yaml

        ry = _make_ruamel_yaml()
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = ry.load(f)

        if "stimuli" not in data:
            data["stimuli"] = {}
        if "twitch" not in data["stimuli"]:
            data["stimuli"]["twitch"] = {}

        tw = data["stimuli"]["twitch"]
        for key, value in fields.items():
            tw[key] = value

        with open(cfg_path, "w", encoding="utf-8") as f:
            ry.dump(data, f)

        print(f"[GUI] Twitch credentials persisted to {cfg_path.name} "
              f"(fields={set(fields.keys())})",
              flush=True)

    @staticmethod
    def _build_stimuli_hydration(cfg) -> dict:
        """Build stimuli config dict with resolved twitch_has_credentials flag."""
        import os
        import re
        _env_pat = re.compile(r"\$\{([^}]+)\}")
        def _resolve(v: str) -> str:
            v = v or ""
            if _env_pat.search(v):
                v = _env_pat.sub(lambda m: os.getenv(m.group(1), ""), v)
            return v
        resolved_channel = _resolve(cfg.twitch_channel)
        resolved_app_id = _resolve(cfg.twitch_app_id) or os.getenv("TWITCH_APP_ID", "")
        resolved_app_secret = _resolve(cfg.twitch_app_secret) or os.getenv("TWITCH_APP_SECRET", "")

        return {
            "enabled": cfg.enabled,
            "patience_enabled": cfg.patience_enabled,
            "patience_seconds": cfg.patience_seconds,
            "patience_prompt": cfg.patience_prompt,
            "twitch_enabled": cfg.twitch_enabled,
            "twitch_channel": cfg.twitch_channel or "",
            "twitch_app_id": cfg.twitch_app_id or "",
            "twitch_app_secret": cfg.twitch_app_secret or "",
            "twitch_buffer_size": cfg.twitch_buffer_size,
            "twitch_max_message_length": cfg.twitch_max_message_length,
            "twitch_prompt_template": cfg.twitch_prompt_template,
            "twitch_has_credentials": bool(
                resolved_channel and resolved_app_id and resolved_app_secret
            ),
            # NANO-110: Addressing-others contexts
            "addressing_others_contexts": [
                {"id": ctx.id, "label": ctx.label, "prompt": ctx.prompt}
                for ctx in cfg.addressing_others_contexts
            ],
        }

    def _get_llm_provider_info(self, config) -> dict:
        """Get LLM provider info, extracting model name for local/cloud providers."""
        provider = config.llm_config.provider
        provider_config = config.llm_config.provider_config

        # Priority 1: Local providers (llama) - extract from model_path
        if provider == "llama" and "model_path" in provider_config:
            model_name = self._extract_model_name(provider_config["model_path"])
            config_info: dict = {}
            if provider_config.get("reasoning_format"):
                config_info["reasoning_format"] = provider_config["reasoning_format"]
            if "reasoning_budget" in provider_config:
                config_info["reasoning_budget"] = provider_config["reasoning_budget"]
            return {"name": model_name, "config": config_info}

        # Priority 2: Cloud providers - check for explicit model field
        if "model" in provider_config:
            return {"name": provider_config["model"], "config": {}}

        # Priority 3: Fallback to provider name
        return {"name": provider, "config": {}}

    def _get_vlm_provider_info(self, config) -> dict:
        """Get VLM provider info, extracting model name for local/cloud providers."""
        provider = config.vlm_config.provider
        provider_config = config.vlm_config.providers.get(provider, {})

        # Priority 1: Local providers (llama) - extract from model_path
        if provider == "llama" and "model_path" in provider_config:
            model_name = self._extract_model_name(provider_config["model_path"])
            return {"name": model_name, "config": {}}

        # Priority 2: Cloud providers - check for explicit model field
        if "model" in provider_config:
            return {"name": provider_config["model"], "config": {}}

        # Priority 3: Fallback to provider name
        return {"name": provider, "config": {}}

    async def _emit_personas(self, sid: Optional[str] = None) -> None:
        """Emit list of available personas."""
        # NANO-048: Resolve personas_dir from orchestrator (preferred) or pre-launch config
        personas_dir_str = None
        if self._orchestrator:
            personas_dir_str = self._orchestrator._config.personas_dir
        elif self._personas_dir:
            personas_dir_str = self._personas_dir

        if not personas_dir_str:
            return

        personas_dir = Path(personas_dir_str)
        personas = []
        active_persona = self._orchestrator.persona.get("id", "unknown") if self._orchestrator and self._orchestrator.persona else "unknown"

        if personas_dir.exists():
            import yaml

            for filepath in sorted(personas_dir.glob("*.yaml")):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if data:
                            personas.append({
                                "id": filepath.stem,
                                "name": data.get("name", filepath.stem),
                                "description": data.get("description", ""),
                            })
                except Exception as e:
                    print(f"[GUI] Failed to read persona {filepath.name}: {e}", flush=True)

        if sid:
            await self.sio.emit(
                "persona_list",
                {"personas": personas, "active": active_persona},
                to=sid,
            )
        else:
            await self.sio.emit(
                "persona_list",
                {"personas": personas, "active": active_persona},
            )

    async def _emit_sessions(
        self, sid: Optional[str] = None, persona_filter: Optional[str] = None
    ) -> None:
        """Emit list of conversation sessions."""
        from spindl.history import jsonl_store

        # NANO-048: Resolve conversations_dir from orchestrator (preferred) or pre-launch config
        conv_dir_str = None
        if self._orchestrator:
            conv_dir_str = self._orchestrator._config.conversations_dir
        elif self._conversations_dir:
            conv_dir_str = self._conversations_dir

        if not conv_dir_str:
            return

        conversations_dir = Path(conv_dir_str)
        sessions = []

        if conversations_dir.exists():
            for filepath in sorted(
                (p for p in conversations_dir.glob("*.jsonl") if ".snapshot." not in p.name),
                key=lambda p: p.stat().st_mtime, reverse=True,
            ):
                # Parse filename: {persona}_{timestamp}.jsonl
                parts = filepath.stem.rsplit("_", 2)
                if len(parts) >= 2:
                    persona = parts[0]
                    if persona_filter and persona != persona_filter:
                        continue

                    stat = filepath.stat()

                    # Count turns (read file to get actual count)
                    try:
                        turns = jsonl_store.read_turns(filepath)
                        turn_count = len(turns)
                        visible_count = len([t for t in turns if not t.get("hidden", False)])
                    except Exception:
                        turn_count = 0
                        visible_count = 0

                    sessions.append({
                        "filepath": str(filepath),
                        "persona": persona,
                        "timestamp": filepath.stem.split("_", 1)[1] if "_" in filepath.stem else "",
                        "turn_count": turn_count,
                        "visible_count": visible_count,
                        "file_size": stat.st_size,
                    })

        # Include the orchestrator's current session so the frontend can badge it
        active_session = None
        if self._orchestrator and self._orchestrator.session_file:
            active_session = str(self._orchestrator.session_file)

        payload = {"sessions": sessions, "active_session": active_session}
        if sid:
            await self.sio.emit("session_list", payload, to=sid)
        else:
            await self.sio.emit("session_list", payload)

    async def _emit_session_detail(self, sid: str, filepath: str) -> None:
        """Emit detailed session data."""
        from spindl.history import jsonl_store

        try:
            path = Path(filepath)
            if not path.exists():
                await self.sio.emit("error", {"message": "Session not found"}, to=sid)
                return

            turns = jsonl_store.read_turns(path)
            await self.sio.emit(
                "session_detail",
                {"filepath": filepath, "turns": turns},
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Error reading session: {e}", flush=True)
            await self.sio.emit("error", {"message": str(e)}, to=sid)

    # === Public emit methods for EventBridge ===

    async def emit_state_changed(
        self, from_state: str, to_state: str, trigger: str, timestamp: str
    ) -> None:
        """Emit state change event to all clients."""
        await self.sio.emit(
            "state_changed",
            {
                "from": from_state,
                "to": to_state,
                "trigger": trigger,
                "timestamp": timestamp,
            },
        )

    async def emit_transcription(
        self, text: str, duration: float, is_final: bool, input_modality: str = "voice"
    ) -> None:
        """Emit transcription event to all clients."""
        await self.sio.emit(
            "transcription",
            {"text": text, "duration": duration, "is_final": is_final, "input_modality": input_modality},
        )

    async def emit_response(
        self,
        text: str,
        is_final: bool,
        activated_codex_entries: Optional[list] = None,
        retrieved_memories: Optional[list] = None,
        reasoning: Optional[str] = None,
        stimulus_source: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        tts_text: Optional[str] = None,
    ) -> None:
        """Emit response event to all clients (NANO-037: codex, NANO-042: reasoning, NANO-044: memories, NANO-056: stimulus, NANO-094: emotion, NANO-109: tts_text)."""
        data = {"text": text, "is_final": is_final}
        if activated_codex_entries:
            data["activated_codex_entries"] = activated_codex_entries
        if retrieved_memories:
            data["retrieved_memories"] = retrieved_memories
        if reasoning:
            data["reasoning"] = reasoning
        if stimulus_source:
            data["stimulus_source"] = stimulus_source
        if emotion is not None:
            data["emotion"] = emotion
            data["emotion_confidence"] = emotion_confidence
        if tts_text is not None:
            data["tts_text"] = tts_text
        await self.sio.emit("response", data)

    async def emit_stimulus_fired(
        self,
        source: str,
        prompt_text: str,
        elapsed_seconds: float,
    ) -> None:
        """Emit stimulus fired event to all clients (NANO-056)."""
        await self.sio.emit(
            "stimulus_fired",
            {
                "source": source,
                "prompt_text": prompt_text,
                "elapsed_seconds": elapsed_seconds,
            },
        )

    async def emit_tts_status(
        self, status: str, duration: Optional[float] = None
    ) -> None:
        """Emit TTS status event to all clients."""
        data = {"status": status}
        if duration is not None:
            data["duration"] = duration
        await self.sio.emit("tts_status", data)

    async def emit_audio_level(self, level: float) -> None:
        """Emit real-time audio output level for portrait visualization (NANO-069)."""
        await self.sio.emit("audio_level", {"level": level})

    async def emit_mic_level(self, level: float) -> None:
        """Emit real-time mic input level for voice overlay visualization (NANO-073b)."""
        await self.sio.emit("mic_level", {"level": level})

    async def emit_avatar_mood(self, mood: str, confidence: float = 0.0) -> None:
        """Emit avatar mood event from emotion classifier (NANO-093, NANO-098)."""
        await self.sio.emit("avatar_mood", {"mood": mood, "confidence": confidence})

    async def emit_avatar_tool_mood(self, tool_mood: str) -> None:
        """Emit avatar tool mood event from tool invocation (NANO-093)."""
        await self.sio.emit("avatar_tool_mood", {"tool_mood": tool_mood})

    async def emit_avatar_load_model(
        self,
        vrm_path: str,
        expressions: dict[str, dict[str, float]] | None = None,
        animations: dict | None = None,
        character_animations_dir: str | None = None,
    ) -> None:
        """Emit avatar model load event after character switch (NANO-097, NANO-098)."""
        payload: dict = {"path": vrm_path}
        if expressions:
            payload["expressions"] = expressions
        if animations:
            payload["animations"] = animations
        if character_animations_dir:
            payload["character_animations_dir"] = character_animations_dir
        # NANO-099: Include base animations for fallback
        if self._orchestrator:
            payload["base_animations"] = self._orchestrator._config.avatar_config.base_animations
        await self.sio.emit("avatar_load_model", payload)

    async def emit_token_usage(
        self,
        prompt: int,
        completion: int,
        total: int,
        max_tokens: int,
        percent: float,
    ) -> None:
        """Emit token usage event to all clients."""
        await self.sio.emit(
            "token_usage",
            {
                "prompt": prompt,
                "completion": completion,
                "total": total,
                "max": max_tokens,
                "percent": percent,
            },
        )

    async def emit_pipeline_error(
        self, stage: str, error_type: str, message: str
    ) -> None:
        """Emit pipeline error event to all clients."""
        await self.sio.emit(
            "pipeline_error",
            {"stage": stage, "error_type": error_type, "message": message},
        )

    async def emit_context_updated(self, sources: list[str]) -> None:
        """Emit context update event to all clients."""
        await self.sio.emit("context_updated", {"sources": sources})

    async def emit_prompt_snapshot(
        self,
        messages: list[dict],
        token_breakdown: dict,
        input_modality: str,
        state_trigger: Optional[str],
        timestamp: str,
    ) -> None:
        """Emit prompt snapshot event to all clients (NANO-025 Phase 3)."""
        await self.sio.emit(
            "prompt_snapshot",
            {
                "messages": messages,
                "token_breakdown": token_breakdown,
                "input_modality": input_modality,
                "state_trigger": state_trigger,
                "timestamp": timestamp,
            },
        )

    async def emit_tool_invoked(
        self,
        tool_name: str,
        arguments: dict,
        iteration: int,
        tool_call_id: str,
        timestamp: str,
    ) -> None:
        """Emit tool invoked event to all clients (NANO-025 Phase 7)."""
        await self.sio.emit(
            "tool_invoked",
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "iteration": iteration,
                "tool_call_id": tool_call_id,
                "timestamp": timestamp,
            },
        )

    async def emit_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_summary: str,
        duration_ms: int,
        iteration: int,
        tool_call_id: str,
    ) -> None:
        """Emit tool result event to all clients (NANO-025 Phase 7)."""
        await self.sio.emit(
            "tool_result",
            {
                "tool_name": tool_name,
                "success": success,
                "result_summary": result_summary,
                "duration_ms": duration_ms,
                "iteration": iteration,
                "tool_call_id": tool_call_id,
            },
        )

    # ============================================================
    # NANO-097: Avatar Process Management
    # ============================================================

    @staticmethod
    def _ensure_tauri_built(app_dir: Path, app_name: str) -> bool:
        """
        Ensure a Tauri app binary exists, building if necessary (NANO-110).

        First-time users cloning the repo won't have compiled binaries.
        This runs `cargo build` with visible output so they see progress
        instead of a silent hang.

        Args:
            app_dir: Root of the Tauri app (e.g., spindl-avatar/).
            app_name: Human-readable name for log messages.

        Returns:
            True if binary is ready, False if build failed.
        """
        import platform
        ext = ".exe" if platform.system() == "Windows" else ""
        cargo_name = app_dir.name  # e.g., "spindl-avatar"
        binary = app_dir / "src-tauri" / "target" / "debug" / f"{cargo_name}{ext}"

        if binary.exists():
            return True

        # Need to build — run cargo build with visible output
        print(
            f"[GUI] {app_name} binary not found — building (first-time only)...",
            flush=True,
        )
        try:
            result = subprocess.run(
                ["cargo", "build"],
                cwd=str(app_dir / "src-tauri"),
                timeout=600,  # 10 minutes max
            )
            if result.returncode != 0:
                print(f"[GUI] {app_name} cargo build failed (exit {result.returncode})", flush=True)
                return False
            print(f"[GUI] {app_name} build complete", flush=True)
            return True
        except subprocess.TimeoutExpired:
            print(f"[GUI] {app_name} cargo build timed out", flush=True)
            return False
        except Exception as e:
            print(f"[GUI] {app_name} cargo build error: {e}", flush=True)
            return False

    async def _avatar_spawn(self) -> None:
        """Spawn the avatar renderer if no avatar client is already connected."""
        if self.has_avatar_client:
            # External launch detected — don't spawn, don't adopt
            self._avatar_process = None
            self._avatar_spawned_by_us = False
            print("[GUI] Avatar client already connected — skipping spawn", flush=True)
            return

        if self._avatar_process and self._avatar_process.poll() is None:
            # Already running and alive
            print("[GUI] Avatar process already running", flush=True)
            return

        # Resolve spindl-avatar directory relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        avatar_dir = project_root / "spindl-avatar"
        if not avatar_dir.exists():
            print(f"[GUI] Avatar directory not found: {avatar_dir}", flush=True)
            return

        # Ensure binary is built (first-time users)
        if not self._ensure_tauri_built(avatar_dir, "Avatar"):
            return

        try:
            self._avatar_process = subprocess.Popen(
                ["npm", "run", "tauri", "dev"],
                cwd=str(avatar_dir),
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._avatar_spawned_by_us = True
            print(
                f"[GUI] Avatar process spawned (PID: {self._avatar_process.pid})",
                flush=True,
            )
        except Exception as e:
            print(f"[GUI] Failed to spawn avatar: {e}", flush=True)
            self._avatar_process = None
            self._avatar_spawned_by_us = False

    def _avatar_kill(self) -> None:
        """Kill the avatar process if we spawned it."""
        if not self._avatar_spawned_by_us or not self._avatar_process:
            self._avatar_process = None
            self._avatar_spawned_by_us = False
            return

        if self._avatar_process.poll() is not None:
            # Already dead
            self._avatar_process = None
            self._avatar_spawned_by_us = False
            return

        try:
            terminated, force_killed = kill_process_tree(
                self._avatar_process.pid, timeout=5.0
            )
            print(
                f"[GUI] Avatar process killed "
                f"(terminated: {len(terminated)}, force-killed: {len(force_killed)})",
                flush=True,
            )
        except Exception as e:
            print(f"[GUI] Failed to kill avatar process: {e}", flush=True)
        finally:
            self._avatar_process = None
            self._avatar_spawned_by_us = False

    # ============================================================
    # NANO-100: Subtitle Process Management
    # ============================================================

    async def _subtitle_spawn(self) -> None:
        """Spawn the subtitle overlay if not already running."""
        if self._subtitle_process and self._subtitle_process.poll() is None:
            print("[GUI] Subtitle process already running", flush=True)
            return

        project_root = Path(__file__).parent.parent.parent.parent
        subtitle_dir = project_root / "spindl-subtitles"
        if not subtitle_dir.exists():
            print(f"[GUI] Subtitle directory not found: {subtitle_dir}", flush=True)
            return

        # Ensure binary is built (first-time users)
        if not self._ensure_tauri_built(subtitle_dir, "Subtitle"):
            return

        try:
            self._subtitle_process = subprocess.Popen(
                ["npm", "run", "tauri", "dev"],
                cwd=str(subtitle_dir),
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._subtitle_spawned_by_us = True
            print(
                f"[GUI] Subtitle process spawned (PID: {self._subtitle_process.pid})",
                flush=True,
            )
        except Exception as e:
            print(f"[GUI] Failed to spawn subtitle: {e}", flush=True)
            self._subtitle_process = None
            self._subtitle_spawned_by_us = False

    def _subtitle_kill(self) -> None:
        """Kill the subtitle process if we spawned it."""
        if not self._subtitle_spawned_by_us or not self._subtitle_process:
            self._subtitle_process = None
            self._subtitle_spawned_by_us = False
            return

        if self._subtitle_process.poll() is not None:
            self._subtitle_process = None
            self._subtitle_spawned_by_us = False
            return

        try:
            terminated, force_killed = kill_process_tree(
                self._subtitle_process.pid, timeout=5.0
            )
            print(
                f"[GUI] Subtitle process killed "
                f"(terminated: {len(terminated)}, force-killed: {len(force_killed)})",
                flush=True,
            )
        except Exception as e:
            print(f"[GUI] Failed to kill subtitle process: {e}", flush=True)
        finally:
            self._subtitle_process = None
            self._subtitle_spawned_by_us = False

    # ============================================================
    # NANO-110: Stream Deck Process Management
    # ============================================================

    async def _stream_deck_spawn(self) -> None:
        """Spawn the stream deck overlay if not already running."""
        if self._stream_deck_process and self._stream_deck_process.poll() is None:
            print("[GUI] Stream Deck process already running", flush=True)
            return

        project_root = Path(__file__).parent.parent.parent.parent
        deck_dir = project_root / "spindl-stream-deck"
        if not deck_dir.exists():
            print(f"[GUI] Stream Deck directory not found: {deck_dir}", flush=True)
            return

        # Ensure binary is built (first-time users)
        if not self._ensure_tauri_built(deck_dir, "Stream Deck"):
            return

        try:
            self._stream_deck_process = subprocess.Popen(
                ["npm", "run", "tauri", "dev"],
                cwd=str(deck_dir),
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._stream_deck_spawned_by_us = True
            print(
                f"[GUI] Stream Deck process spawned (PID: {self._stream_deck_process.pid})",
                flush=True,
            )
        except Exception as e:
            print(f"[GUI] Failed to spawn stream deck: {e}", flush=True)
            self._stream_deck_process = None
            self._stream_deck_spawned_by_us = False

    def _stream_deck_kill(self) -> None:
        """Kill the stream deck process if we spawned it."""
        if not self._stream_deck_spawned_by_us or not self._stream_deck_process:
            self._stream_deck_process = None
            self._stream_deck_spawned_by_us = False
            return

        if self._stream_deck_process.poll() is not None:
            self._stream_deck_process = None
            self._stream_deck_spawned_by_us = False
            return

        try:
            terminated, force_killed = kill_process_tree(
                self._stream_deck_process.pid, timeout=5.0
            )
            print(
                f"[GUI] Stream Deck process killed "
                f"(terminated: {len(terminated)}, force-killed: {len(force_killed)})",
                flush=True,
            )
        except Exception as e:
            print(f"[GUI] Failed to kill stream deck process: {e}", flush=True)
        finally:
            self._stream_deck_process = None
            self._stream_deck_spawned_by_us = False

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    @property
    def has_clients(self) -> bool:
        """Whether any clients are connected."""
        return len(self._clients) > 0

    @property
    def has_avatar_client(self) -> bool:
        """Whether any avatar renderer clients are connected (NANO-097)."""
        return len(self._avatar_clients) > 0
