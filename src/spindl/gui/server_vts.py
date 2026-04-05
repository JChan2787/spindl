"""
VTubeStudio-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- VTS config (set_vts_config)
- VTS status (request_vts_status)
- VTS hotkeys (request_vts_hotkeys)
- VTS expressions (request_vts_expressions)
- VTS hotkey trigger (send_vts_hotkey)
- VTS expression trigger (send_vts_expression)
- VTS model move (send_vts_move)
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import GUIServer


def register_vts_handlers(server: "GUIServer") -> None:
    """Register VTubeStudio-domain Socket.IO event handlers."""
    sio = server.sio

    # ============================================================
    # NANO-060b: VTubeStudio — Socket Handlers
    # ============================================================

    @sio.event
    async def set_vts_config(sid: str, data: dict) -> None:
        """Client updates VTubeStudio configuration at runtime (NANO-060b)."""
        if server._orchestrator:
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

            server._orchestrator.update_vts_config(
                enabled=enabled,
                host=host,
                port=port,
            )

            cfg = server._orchestrator._config.vtubestudio_config
            print(
                f"[GUI] VTS: enabled={cfg.enabled}, "
                f"host={cfg.host}, port={cfg.port}",
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
                        f"[GUI] VTS config persisted to "
                        f"{Path(server._config_path).name}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[GUI] Failed to persist VTS config: {e}",
                        flush=True,
                    )

            await sio.emit(
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
                            server._orchestrator.vts_driver
                            if server._orchestrator
                            else None
                        )
                        if driver and driver.is_connected():
                            status = driver.get_status()
                            status["enabled"] = True
                            await sio.emit("vts_status", status, to=target_sid)
                            return
                asyncio.ensure_future(_push_vts_status_after_connect(sid))

    @sio.event
    async def request_vts_status(sid: str, data: dict) -> None:
        """Client requests VTubeStudio connection status (NANO-060b)."""
        if (
            not server._orchestrator
            or not server._orchestrator.vts_driver
        ):
            cfg_enabled = False
            if server._orchestrator:
                cfg_enabled = server._orchestrator._config.vtubestudio_config.enabled
            await sio.emit(
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

        status = server._orchestrator.vts_driver.get_status()
        status["enabled"] = server._orchestrator._config.vtubestudio_config.enabled
        await sio.emit("vts_status", status, to=sid)

    @sio.event
    async def request_vts_hotkeys(sid: str, data: dict) -> None:
        """Client requests available VTS hotkeys (NANO-060b).

        Default: serve cached list for instant render.
        With {refresh: true}: live query VTS then emit updated list.
        """
        driver = (
            server._orchestrator.vts_driver
            if server._orchestrator
            else None
        )
        if not driver:
            await sio.emit(
                "vts_hotkeys", {"hotkeys": []}, to=sid,
            )
            return

        if data.get("refresh"):
            # Live query — callback fires from driver thread
            loop = server._event_loop or asyncio.get_event_loop()

            def _on_hotkeys(result: list) -> None:
                asyncio.run_coroutine_threadsafe(
                    sio.emit(
                        "vts_hotkeys", {"hotkeys": result}, to=sid,
                    ),
                    loop,
                )

            driver.request_hotkey_list(callback=_on_hotkeys)
        else:
            # Serve cached list
            status = driver.get_status()
            await sio.emit(
                "vts_hotkeys",
                {"hotkeys": status.get("hotkeys", [])},
                to=sid,
            )

    @sio.event
    async def request_vts_expressions(sid: str, data: dict) -> None:
        """Client requests available VTS expressions (NANO-060b).

        Default: serve cached list for instant render.
        With {refresh: true}: live query VTS then emit updated list.
        """
        driver = (
            server._orchestrator.vts_driver
            if server._orchestrator
            else None
        )
        if not driver:
            await sio.emit(
                "vts_expressions", {"expressions": []}, to=sid,
            )
            return

        if data.get("refresh"):
            # Live query — callback fires from driver thread
            loop = server._event_loop or asyncio.get_event_loop()

            def _on_expressions(result: list) -> None:
                asyncio.run_coroutine_threadsafe(
                    sio.emit(
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
            await sio.emit(
                "vts_expressions",
                {"expressions": status.get("expressions", [])},
                to=sid,
            )

    @sio.event
    async def send_vts_hotkey(sid: str, data: dict) -> None:
        """Client triggers a VTS hotkey (NANO-060b)."""
        driver = (
            server._orchestrator.vts_driver
            if server._orchestrator
            else None
        )
        if not driver:
            return

        name = data.get("name")
        if name:
            driver.trigger_hotkey(str(name))
            print(f"[GUI] VTS hotkey triggered: {name}", flush=True)
            await sio.emit(
                "vts_hotkey_triggered",
                {"name": name},
                to=sid,
            )

    @sio.event
    async def send_vts_expression(sid: str, data: dict) -> None:
        """Client activates/deactivates a VTS expression (NANO-060b)."""
        driver = (
            server._orchestrator.vts_driver
            if server._orchestrator
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
            await sio.emit(
                "vts_expression_triggered",
                {"file": file, "active": active},
                to=sid,
            )

    @sio.event
    async def send_vts_move(sid: str, data: dict) -> None:
        """Client moves VTS model to a preset position (NANO-060b)."""
        driver = (
            server._orchestrator.vts_driver
            if server._orchestrator
            else None
        )
        if not driver:
            return

        preset = data.get("preset")
        if preset:
            driver.move_model(str(preset))
            print(f"[GUI] VTS move: {preset}", flush=True)
            await sio.emit(
                "vts_move_triggered",
                {"preset": preset},
                to=sid,
            )
