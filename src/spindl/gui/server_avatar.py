"""
Avatar-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- Avatar client registration (register_avatar_client)
- Expression preview (preview_avatar_expressions)
- Animation config push (update_avatar_animation_config)
- Animation preview (preview_avatar_animation)
- Avatar model reload (reload_avatar_model)
- Avatar config (set_avatar_config)
- Animation rescan (avatar_rescan_animations)
- Tauri install check/trigger (check_tauri_install, install_tauri_apps)

Also exposes Tauri binary helpers as standalone functions:
tauri_binary_path(), tauri_binary_exists(),
tauri_install_in_background(), tauri_install_all_in_background().

NOTE: The spawn/kill methods (_avatar_spawn, _avatar_kill, etc.)
remain on GUIServer because they are called from connect, shutdown,
and _launch_services_async.
"""

import asyncio
import platform
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .server import GUIServer


def tauri_binary_path(app_dir: Path) -> Optional[Path]:
    """Return the Tauri app binary path if it exists, else None."""
    ext = ".exe" if platform.system() == "Windows" else ""
    cargo_name = app_dir.name
    project_root = app_dir.parent
    # Check release first (installed via Install button), then debug (dev builds)
    for profile in ("release", "debug"):
        workspace = project_root / "target" / profile / f"{cargo_name}{ext}"
        if workspace.exists():
            return workspace
        legacy = app_dir / "src-tauri" / "target" / profile / f"{cargo_name}{ext}"
        if legacy.exists():
            return legacy
    return None


def tauri_binary_exists(app_dir: Path) -> bool:
    """Check if a Tauri app binary exists (workspace or legacy path)."""
    return tauri_binary_path(app_dir) is not None


def tauri_install_in_background(server: "GUIServer", app_dir: Path, app_name: str) -> None:
    """
    Build a Tauri app binary in a background thread (NANO-110).

    Streams cargo compilation progress to the frontend via socket events.
    Non-blocking — returns immediately, build runs in a daemon thread.
    On completion, emits 'ready' or 'failed' status.
    """
    ext = ".exe" if platform.system() == "Windows" else ""
    cargo_name = app_dir.name
    project_root = app_dir.parent
    app_key = app_name.lower().replace(" ", "_")

    def _emit(status: str, message: str, progress: int = 0, total: int = 0) -> None:
        if server._event_loop and server.sio:
            asyncio.run_coroutine_threadsafe(
                server.sio.emit("tauri_build_status", {
                    "app": app_key,
                    "status": status,
                    "message": message,
                    "progress": progress,
                    "total": total,
                }),
                server._event_loop,
            )

    def _build() -> None:
        _emit("building", f"Installing {app_name} (first time only — this may take a few minutes)...")
        print(f"[GUI] {app_name} binary not found — building (first-time only)...", flush=True)

        building_re = re.compile(r"\[=*>?\s*\]\s+(\d+)/(\d+)")
        compiling_count = 0

        try:
            proc = subprocess.Popen(
                ["cargo", "build", "--release", "-p", cargo_name],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue

                m = building_re.search(line)
                if m:
                    current, total = int(m.group(1)), int(m.group(2))
                    _emit("building", f"Compiling crate {current}/{total}...", current, total)
                elif "Compiling" in line:
                    compiling_count += 1
                    parts = line.split("Compiling")
                    crate = parts[-1].strip().split(" ")[0] if len(parts) > 1 else "..."
                    _emit("building", f"Compiling {crate}...", compiling_count, 0)

                print(f"[GUI] {app_name} build: {line}", flush=True)

            proc.stdout.close()
            ret = proc.wait(timeout=600)

            if ret != 0:
                print(f"[GUI] {app_name} cargo build failed (exit {ret})", flush=True)
                _emit("failed", f"{app_name} build failed.")
                return

            print(f"[GUI] {app_name} build complete", flush=True)
            _emit("ready", f"{app_name} installed successfully.")

        except subprocess.TimeoutExpired:
            print(f"[GUI] {app_name} cargo build timed out", flush=True)
            _emit("failed", f"{app_name} build timed out.")
            try:
                proc.kill()
            except Exception:
                pass
        except Exception as e:
            print(f"[GUI] {app_name} cargo build error: {e}", flush=True)
            _emit("failed", f"{app_name} build error: {e}")

    thread = threading.Thread(target=_build, daemon=True, name=f"tauri-build-{app_key}")
    thread.start()


def tauri_install_all_in_background(server: "GUIServer", apps: list[tuple[Path, str]]) -> None:
    """
    Build multiple Tauri apps sequentially in a background thread (NANO-110).

    First app compiles all shared deps (~2-3 min). Subsequent apps reuse
    the cached deps and compile in seconds. Emits 'ready' only after ALL
    apps are built.
    """
    ext = ".exe" if platform.system() == "Windows" else ""

    def _emit(status: str, message: str, progress: int = 0, total: int = 0) -> None:
        if server._event_loop and server.sio:
            asyncio.run_coroutine_threadsafe(
                server.sio.emit("tauri_build_status", {
                    "app": "all",
                    "status": status,
                    "message": message,
                    "progress": progress,
                    "total": total,
                }),
                server._event_loop,
            )

    def _build_all() -> None:
        building_re = re.compile(r"\[=*>?\s*\]\s+(\d+)/(\d+)")

        npm = "npm.cmd" if platform.system() == "Windows" else "npm"

        for i, (app_dir, app_name) in enumerate(apps):
            cargo_name = app_dir.name
            project_root = app_dir.parent

            label = f"({i + 1}/{len(apps)}) {app_name}"

            # Step 1: Build frontend dist (vite build — skip tsc for resilience)
            dist_dir = app_dir / "dist"
            npx = "npx.cmd" if platform.system() == "Windows" else "npx"
            if not dist_dir.exists():
                _emit("building", f"{label}: building frontend...")
                print(f"[GUI] {app_name}: building frontend dist...", flush=True)
                try:
                    # Ensure node_modules exist
                    node_modules = app_dir / "node_modules"
                    if not node_modules.exists():
                        subprocess.run(
                            [npm, "install"],
                            cwd=str(app_dir),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=120,
                        )
                    # vite build directly — more resilient than tsc + vite
                    vite_result = subprocess.run(
                        [npx, "vite", "build"],
                        cwd=str(app_dir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120,
                    )
                    if vite_result.returncode != 0:
                        print(f"[GUI] {app_name} frontend build failed", flush=True)
                        _emit("failed", f"{app_name} frontend build failed.")
                        return
                except Exception as e:
                    print(f"[GUI] {app_name} frontend build error: {e}", flush=True)
                    _emit("failed", f"{app_name} frontend build error.")
                    return

            # Step 2: Build Rust binary
            _emit("building", f"Installing {label}...")
            print(f"[GUI] Building {label}...", flush=True)

            compiling_count = 0
            try:
                proc = subprocess.Popen(
                    ["cargo", "build", "--release", "-p", cargo_name],
                    cwd=str(project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in iter(proc.stdout.readline, ""):
                    line = line.strip()
                    if not line:
                        continue

                    m = building_re.search(line)
                    if m:
                        current, total = int(m.group(1)), int(m.group(2))
                        _emit("building", f"{label}: crate {current}/{total}", current, total)
                    elif "Compiling" in line:
                        compiling_count += 1
                        parts = line.split("Compiling")
                        crate = parts[-1].strip().split(" ")[0] if len(parts) > 1 else "..."
                        _emit("building", f"{label}: {crate}", compiling_count, 0)

                    print(f"[GUI] {app_name} build: {line}", flush=True)

                proc.stdout.close()
                ret = proc.wait(timeout=600)

                if ret != 0:
                    print(f"[GUI] {app_name} build failed (exit {ret})", flush=True)
                    _emit("failed", f"{app_name} build failed.")
                    return

                print(f"[GUI] {app_name} build complete", flush=True)

            except subprocess.TimeoutExpired:
                print(f"[GUI] {app_name} build timed out", flush=True)
                _emit("failed", f"{app_name} build timed out.")
                try:
                    proc.kill()
                except Exception:
                    pass
                return
            except Exception as e:
                print(f"[GUI] {app_name} build error: {e}", flush=True)
                _emit("failed", f"{app_name} build error: {e}")
                return

        # All done
        _emit("ready", "All overlay apps installed successfully.")
        print("[GUI] All Tauri apps built successfully", flush=True)

    thread = threading.Thread(target=_build_all, daemon=True, name="tauri-install-all")
    thread.start()


def register_avatar_handlers(server: "GUIServer") -> None:
    """Register avatar-domain Socket.IO event handlers."""
    sio = server.sio

    @sio.event
    async def register_avatar_client(sid: str, data: dict) -> None:
        """Avatar renderer identifies itself on connect (NANO-097)."""
        server._avatar_clients.add(sid)
        print(
            f"[GUI] Avatar client registered "
            f"(avatar clients: {len(server._avatar_clients)})",
            flush=True,
        )
        await sio.emit(
            "avatar_connection_status",
            {"connected": True},
        )
        # Push active character's VRM + config to the newly connected avatar
        if server._orchestrator:
            try:
                from spindl.characters.loader import CharacterLoader

                config = server._orchestrator._config
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
                await sio.emit(
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

    @sio.event
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
        await sio.emit("avatar_preview_expressions", payload)

    @sio.event
    async def update_avatar_animation_config(sid: str, data: dict) -> None:
        """Push updated animation config to renderer on save (NANO-098 Session 3).

        Sent directly from editor with the config data — no disk read needed.
        Avoids race condition with card.json write.
        """
        animations = data.get("animations")
        await sio.emit("avatar_update_animation_config", {"animations": animations})

    @sio.event
    async def preview_avatar_animation(sid: str, data: dict) -> None:
        """Live preview of animation clip from character editor (NANO-098 Session 3).

        Relays clip name to all connected avatar clients.
        clip=null means stop playing (revert to procedural idle).
        """
        clip = data.get("clip")
        await sio.emit("avatar_preview_animation", {"clip": clip})

    @sio.event
    async def reload_avatar_model(sid: str, data: dict) -> None:
        """Dashboard requests avatar model reload after VRM change (NANO-097)."""
        from spindl.characters.loader import CharacterLoader

        character_id = data.get("character_id")
        if not character_id or not server._orchestrator:
            return
        # Only push if this is the active character
        if character_id != server._orchestrator._config.character_id:
            return
        try:
            config = server._orchestrator._config
            loader = CharacterLoader(config.characters_dir)
            vrm_path = loader.get_vrm_path(character_id)
            if vrm_path:
                expressions = loader.get_avatar_expressions(character_id)
                animations = loader.get_avatar_animations(character_id)
                char_anim_dir = str(loader.get_character_animations_dir(character_id))
                await server.emit_avatar_load_model(
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
                await sio.emit("avatar_load_model", payload)
                print(
                    "[GUI] Avatar model reloaded: (default)",
                    flush=True,
                )
        except Exception as e:
            print(f"[GUI] Avatar model reload failed: {e}", flush=True)

    # ============================================================
    # NANO-093: Avatar Config — Socket Handler
    # ============================================================

    @sio.event
    async def set_avatar_config(sid: str, data: dict) -> None:
        """Client updates avatar configuration at runtime (NANO-093, NANO-094)."""
        if server._orchestrator:
            config = server._orchestrator._config
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
                if server._config_path:
                    try:
                        config.save_to_yaml(server._config_path)
                        persisted = True
                        print(
                            f"[GUI] Avatar config persisted to "
                            f"{Path(server._config_path).name}",
                            flush=True,
                        )
                    except Exception as e:
                        print(
                            f"[GUI] Failed to persist avatar config: {e}",
                            flush=True,
                        )

                await sio.emit(
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
                    await server._avatar_spawn()
                else:
                    server._avatar_kill()

            # NANO-100: Subtitle process management on toggle
            if subtitles_enabled is not None:
                if bool(subtitles_enabled):
                    await server._subtitle_spawn()
                else:
                    server._subtitle_kill()

            # NANO-110: Stream Deck process management on toggle
            if stream_deck_enabled is not None:
                if bool(stream_deck_enabled):
                    await server._stream_deck_spawn()
                else:
                    server._stream_deck_kill()

        else:
            # Pre-launch: orchestrator not running yet, but Tauri app
            # builds and spawns don't need services. Handle toggle-only
            # fields so users can build/launch windows from settings page
            # before going through the launcher flow.
            enabled = data.get("enabled")
            subtitles_enabled = data.get("subtitles_enabled")
            stream_deck_enabled = data.get("stream_deck_enabled")

            if enabled is not None:
                if bool(enabled):
                    await server._avatar_spawn()
                else:
                    server._avatar_kill()
            if subtitles_enabled is not None:
                if bool(subtitles_enabled):
                    await server._subtitle_spawn()
                else:
                    server._subtitle_kill()
            if stream_deck_enabled is not None:
                if bool(stream_deck_enabled):
                    await server._stream_deck_spawn()
                else:
                    server._stream_deck_kill()

    # ============================================================
    # NANO-099: Avatar Rescan Animations — Socket Handler
    # ============================================================

    @sio.event
    async def avatar_rescan_animations(sid: str) -> None:
        """Notify avatar renderer to rescan animation files (NANO-099)."""
        print("[GUI] Avatar rescan animations requested", flush=True)
        # Reload base_animations from YAML into in-memory config
        if server._orchestrator and server._config_path:
            try:
                import yaml as _yaml
                with open(server._config_path, "r", encoding="utf-8") as f:
                    raw = _yaml.safe_load(f)
                ba = (raw.get("avatar") or {}).get("base_animations") or {}
                server._orchestrator._config.avatar_config.base_animations = {
                    "idle": ba.get("idle"),
                    "happy": ba.get("happy"),
                    "sad": ba.get("sad"),
                }
            except Exception as e:
                print(f"[GUI] Failed to reload base_animations: {e}", flush=True)
        # Forward to all connected clients with base_animations config
        ba = {}
        if server._orchestrator:
            ba = server._orchestrator._config.avatar_config.base_animations
        await sio.emit("avatar_rescan_animations", {"base_animations": ba})

    # ============================================================
    # NANO-110: Tauri App Install — Socket Handlers
    # ============================================================

    @sio.event
    async def check_tauri_install(sid: str, data: dict) -> None:
        """Check which Tauri app binaries are installed (NANO-110)."""
        project_root = Path(__file__).parent.parent.parent.parent
        status = {
            "avatar": tauri_binary_exists(project_root / "spindl-avatar"),
            "subtitle": tauri_binary_exists(project_root / "spindl-subtitles"),
            "stream_deck": tauri_binary_exists(project_root / "spindl-stream-deck"),
        }
        await sio.emit("tauri_install_status", status, to=sid)

    @sio.event
    async def install_tauri_apps(sid: str, data: dict) -> None:
        """Trigger background build of all Tauri app binaries (NANO-110)."""
        project_root = Path(__file__).parent.parent.parent.parent

        apps = [
            (project_root / "spindl-avatar", "Avatar"),
            (project_root / "spindl-subtitles", "Subtitle"),
            (project_root / "spindl-stream-deck", "Stream Deck"),
        ]

        missing = [
            (d, n) for d, n in apps
            if d.exists() and not tauri_binary_exists(d)
        ]

        if not missing:
            await sio.emit("tauri_build_status", {
                "app": "all",
                "status": "ready",
                "message": "All overlay apps are already installed.",
                "progress": 0,
                "total": 0,
            })
            return

        # Build all missing apps sequentially in one background thread.
        # Shared workspace means first build compiles all deps,
        # subsequent builds are near-instant (seconds).
        tauri_install_all_in_background(server, missing)
