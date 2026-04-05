"""
Stimuli-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- Stimuli config (set_stimuli_config)
- Addressing others start/stop (addressing_others_start, addressing_others_stop)
- Patience progress (request_patience_progress)
- Twitch status (request_twitch_status)
- Twitch credential testing (test_twitch_credentials)

Also exposes persist_twitch_credentials() and build_stimuli_hydration()
as standalone helpers.
"""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import GUIServer


def persist_twitch_credentials(
    server: "GUIServer", channel: str, app_id: str, app_secret: str
) -> None:
    """Persist Twitch credentials to YAML via ruamel.yaml round-trip (NANO-106)."""
    cfg_path = Path(server._config_path)
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


def build_stimuli_hydration(cfg) -> dict:
    """Build stimuli config dict with resolved twitch_has_credentials flag."""
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


def register_stimuli_handlers(server: "GUIServer") -> None:
    """Register stimuli-domain Socket.IO event handlers."""
    sio = server.sio

    # ============================================================
    # NANO-056: Stimuli System — Socket Handlers
    # ============================================================

    @sio.event
    async def set_stimuli_config(sid: str, data: dict) -> None:
        """Client updates stimuli configuration at runtime (NANO-056)."""
        if server._orchestrator:
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

            server._orchestrator.update_stimuli_config(
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

            cfg = server._orchestrator._config.stimuli_config
            print(
                f"[GUI] Stimuli: enabled={cfg.enabled}, "
                f"patience_enabled={cfg.patience_enabled}, "
                f"patience_seconds={cfg.patience_seconds}, "
                f"twitch_enabled={cfg.twitch_enabled}",
                flush=True,
            )

            # Persist to YAML
            persisted = False
            if server._config_path:
                try:
                    server._orchestrator._config.save_to_yaml(
                        server._config_path
                    )
                    persisted = True
                    print(
                        f"[GUI] Stimuli config persisted to "
                        f"{Path(server._config_path).name}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[GUI] Failed to persist stimuli config: {e}",
                        flush=True,
                    )

            stimuli_data = build_stimuli_hydration(cfg)
            stimuli_data["persisted"] = persisted

            # Broadcast to all clients (including Stream Deck) so
            # context button grids rebuild when contexts are added/removed
            await sio.emit(
                "stimuli_config_updated",
                stimuli_data,
            )

        if not server._orchestrator:
            # Pre-launch: no persist, no emit. Credentials persist
            # via test_twitch_credentials handler on Test Connection click.
            # Orchestrator path handles persistence after launch.
            pass

    # ============================================================
    # NANO-110: Addressing Others — Socket Handlers
    # ============================================================

    @sio.event
    async def addressing_others_start(sid: str, data: dict) -> None:
        """Stream Deck / hotkey activates addressing-others mode (NANO-110)."""
        print(f"[GUI] addressing_others_start received: {data}", flush=True)
        if not server._orchestrator:
            print("[GUI] addressing_others_start: no orchestrator", flush=True)
            return
        context_id = data.get("context_id", "ctx_0")
        server._orchestrator.set_addressing_others(str(context_id))
        await sio.emit(
            "addressing_others_state",
            {"active": True, "context_id": context_id},
        )

    @sio.event
    async def addressing_others_stop(sid: str, data: dict) -> None:
        """Stream Deck / hotkey deactivates addressing-others mode (NANO-110)."""
        print(f"[GUI] addressing_others_stop received", flush=True)
        if not server._orchestrator:
            return
        server._orchestrator.clear_addressing_others()
        await sio.emit(
            "addressing_others_state",
            {"active": False, "context_id": None},
        )

    @sio.event
    async def request_patience_progress(sid: str, data: dict) -> None:
        """Client requests current PATIENCE progress (NANO-056)."""
        if (
            not server._orchestrator
            or not server._orchestrator.stimuli_engine
        ):
            await sio.emit(
                "patience_progress",
                {"elapsed": 0, "total": 0, "progress": 0},
                to=sid,
            )
            return

        engine = server._orchestrator.stimuli_engine
        for module in engine.modules:
            if module.name == "patience":
                progress_data = module.get_progress()
                progress_data["blocked"] = engine.is_blocked_by_playback or engine.is_blocked_by_typing
                progress_data["blocked_reason"] = (
                    "typing" if engine.is_blocked_by_typing
                    else "playback" if engine.is_blocked_by_playback
                    else None
                )
                await sio.emit(
                    "patience_progress",
                    progress_data,
                    to=sid,
                )
                return

        # No PATIENCE module found
        await sio.emit(
            "patience_progress",
            {"elapsed": 0, "total": 0, "progress": 0},
            to=sid,
        )

    @sio.event
    async def request_twitch_status(sid: str, data: dict) -> None:
        """Client requests current Twitch module status (NANO-056b)."""
        if (
            not server._orchestrator
            or not server._orchestrator.stimuli_engine
        ):
            await sio.emit(
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

        engine = server._orchestrator.stimuli_engine
        for module in engine.modules:
            if module.name == "twitch":
                await sio.emit(
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
        await sio.emit(
            "twitch_status",
            {
                "connected": False,
                "channel": "",
                "buffer_count": 0,
                "recent_messages": [],
            },
            to=sid,
        )

    @sio.event
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
            await sio.emit(
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
                    await sio.emit(
                        "twitch_credentials_result",
                        {"success": False, "error": f"Channel '{channel}' not found on Twitch."},
                        to=sid,
                    )
                    return

            await twitch.close()

            # Persist validated credentials to YAML via ruamel.yaml round-trip
            if server._config_path:
                persist_twitch_credentials(
                    server,
                    data.get("channel", ""),
                    data.get("app_id", ""),
                    data.get("app_secret", ""),
                )

            # Update stimuli cache (for next connect hydration)
            # but do NOT emit stimuli_config_updated — that would
            # overwrite the frontend's local input state with YAML values
            if server._config_path:
                import yaml as _yaml
                with open(Path(server._config_path), "r", encoding="utf-8") as _f:
                    _yd = _yaml.safe_load(_f) or {}
                from spindl.orchestrator.config import StimuliConfig
                _parsed = StimuliConfig.from_dict(_yd.get("stimuli", {}))
                server._stimuli_config_cache = build_stimuli_hydration(_parsed)

            # If orchestrator is running, update the live module and bounce it
            if server._orchestrator and server._orchestrator.stimuli_engine:
                cfg = server._orchestrator._config.stimuli_config
                cfg.twitch_channel = data.get("channel", "") or cfg.twitch_channel
                cfg.twitch_app_id = data.get("app_id", "") or cfg.twitch_app_id
                cfg.twitch_app_secret = data.get("app_secret", "") or cfg.twitch_app_secret

                engine = server._orchestrator.stimuli_engine
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
            await sio.emit(
                "twitch_credentials_result",
                {"success": True, "error": None, "has_credentials": has_creds},
                to=sid,
            )
            print(f"[GUI] Twitch credentials validated and persisted (channel={channel or 'N/A'})", flush=True)

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid" in error_msg.lower():
                error_msg = "Invalid App ID or App Secret."
            await sio.emit(
                "twitch_credentials_result",
                {"success": False, "error": error_msg},
                to=sid,
            )
            print(f"[GUI] Twitch credential test failed: {e}", flush=True)
