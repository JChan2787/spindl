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
from spindl.gui.server_memory import register_memory_handlers
from spindl.gui.server_sessions import register_session_handlers, emit_sessions
from spindl.gui.server_vts import register_vts_handlers
from spindl.gui.server_stimuli import register_stimuli_handlers, build_stimuli_hydration
from spindl.gui.server_config import register_config_handlers, emit_personas
from spindl.gui.server_providers import (
    register_provider_handlers, get_llm_provider_info, get_vlm_provider_info,
)
from spindl.gui.server_avatar import (
    register_avatar_handlers, tauri_binary_path, tauri_binary_exists,
)

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
                self._stimuli_config_cache = build_stimuli_hydration(parsed)

            print(f"[GUI] Pre-launch paths: conversations={self._conversations_dir}, personas={self._personas_dir}", flush=True)

        except Exception as e:
            print(f"[GUI] Failed to parse config paths for pre-launch access: {e}", flush=True)


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
        # NANO-112: Skip when STT is disabled (Stream Deck controls voice input suppression)
        stt_enabled = orchestrator._config.stt_config.enabled
        if orchestrator._config.avatar_config.stream_deck_enabled and self._event_loop and stt_enabled:
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

        # Domain-specific handler modules (NANO-113)
        register_memory_handlers(self)
        register_session_handlers(self)
        register_config_handlers(self)
        register_provider_handlers(self)
        register_stimuli_handlers(self)
        register_vts_handlers(self)
        register_avatar_handlers(self)

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
        async def typing_active(sid: str, data: dict) -> None:
            """Client signals user is typing (focus) or done typing (blur)."""
            if not self._orchestrator or not self._orchestrator.stimuli_engine:
                return
            active = data.get("active", False)
            self._orchestrator.stimuli_engine.user_typing = active

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

            # Step 1b: Stop avatar, subtitle, and stream deck processes
            self._avatar_kill()
            self._subtitle_kill()
            self._stream_deck_kill()

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
        """Shutdown all launched services, avatar, subtitle, and stream deck processes."""
        self._avatar_kill()
        self._subtitle_kill()
        self._stream_deck_kill()
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
                "llm": get_llm_provider_info(self, config),
                "tts": {
                    "name": config.tts_config.provider,
                    "config": {},
                },
                "stt": {
                    "name": config.stt_config.provider,
                    "config": config.stt_config.provider_config,
                },
                "vlm": get_vlm_provider_info(self, config),
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
                    "top_k": config.llm_config.provider_config.get("top_k", 40),
                    "min_p": config.llm_config.provider_config.get("min_p", 0.05),
                    "repeat_penalty": config.llm_config.provider_config.get("repeat_penalty", 1.1),
                    "repeat_last_n": config.llm_config.provider_config.get("repeat_last_n", 64),
                    "frequency_penalty": config.llm_config.provider_config.get("frequency_penalty", 0.0),
                    "presence_penalty": config.llm_config.provider_config.get("presence_penalty", 0.0),
                },
                "stimuli": build_stimuli_hydration(config.stimuli_config),
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
                    "avatar_connected": self.has_avatar_client,
                },
                # NANO-065b: LLM provider runtime state
                "llm": self._orchestrator.get_llm_state(),
            },
        }

        if sid:
            await self.sio.emit("config_loaded", config_data, to=sid)
        else:
            await self.sio.emit("config_loaded", config_data)

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
        chunks: Optional[list] = None,
    ) -> None:
        """Emit response event to all clients (NANO-037: codex, NANO-042: reasoning, NANO-044: memories, NANO-056: stimulus, NANO-094: emotion, NANO-109: tts_text, NANO-111: chunks)."""
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
        if chunks is not None:
            data["chunks"] = chunks
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

    async def emit_llm_chunk(
        self,
        text: str,
        is_final: bool,
        emotion: str | None = None,
        emotion_confidence: float | None = None,
    ) -> None:
        """Emit streaming LLM sentence chunk for real-time dashboard text (NANO-111)."""
        data: dict = {"text": text, "is_final": is_final}
        if emotion is not None:
            data["emotion"] = emotion
            data["emotion_confidence"] = emotion_confidence
        await self.sio.emit("llm_chunk", data)

    async def emit_llm_token(self, token: str, is_final: bool) -> None:
        """Emit token-level LLM text for real-time dashboard display (NANO-111)."""
        await self.sio.emit("llm_token", {"token": token, "is_final": is_final})

    async def emit_barge_in_truncated(self, truncated_text: str, delivered_sentences: int) -> None:
        """Emit barge-in truncation event for frontend bubble update (NANO-111 Phase 2.5)."""
        await self.sio.emit("barge_in_truncated", {
            "truncated_text": truncated_text,
            "delivered_sentences": delivered_sentences,
        })

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

        # Check binary exists — install must happen first via Install button
        if not tauri_binary_exists(avatar_dir):
            print("[GUI] Avatar binary not installed — use Install button", flush=True)
            return

        # Avatar needs npm run tauri dev (Vite serves VRM/FBX assets at runtime)
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

        # Check binary exists — install must happen first via Install button
        if not tauri_binary_exists(subtitle_dir):
            print("[GUI] Subtitle binary not installed — use Install button", flush=True)
            return

        # Subtitle needs npm run tauri dev (Vite serves assets at runtime)
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

        # Check binary exists — install must happen first via Install button
        binary = tauri_binary_path(deck_dir)
        if not binary:
            print("[GUI] Stream Deck binary not installed — use Install button", flush=True)
            return

        try:
            self._stream_deck_process = subprocess.Popen(
                [str(binary)],
                cwd=str(deck_dir),
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
