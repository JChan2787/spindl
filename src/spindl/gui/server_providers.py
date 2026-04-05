"""
Provider-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- LLM config/provider (request_llm_config, set_llm_provider)
- Local LLM launch (request_local_llm_config, launch_llm_server)
- OpenRouter models (request_openrouter_models)
- VLM config/provider (request_vlm_config, set_vlm_provider)
- Local VLM launch (request_local_vlm_config, launch_vlm_server)
- Tools config (request_tools_config, set_tools_config)

Also exposes validation, persistence, and provider info helpers
as standalone functions.
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import GUIServer


def validate_local_llm_config(config: dict) -> list[str]:
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

    # mmproj_path: if set, file must exist (Session 606: stale path = silent launch failure)
    mmproj_path = config.get("mmproj_path")
    if mmproj_path and isinstance(mmproj_path, str) and not os.path.isfile(mmproj_path):
        errors.append(
            f"mmproj_path does not exist: {mmproj_path} — "
            "stale path from a previous model? Clear it if VLM is disabled"
        )

    return errors


def strip_empty_optional_fields(config: dict) -> None:
    """Remove keys whose values are empty strings — prevents zombie fields in YAML.

    Fields like mmproj_path, reasoning_format, extra_args can be "" when the
    user clears them in the GUI. Leaving "" in YAML means the next .update()
    merge won't delete them, and the backend may interpret their presence
    as intentional config even though the value is blank.
    """
    zombie_keys = [
        k for k, v in config.items()
        if isinstance(v, str) and v == ""
    ]
    for k in zombie_keys:
        del config[k]


def persist_local_llm_config(server: "GUIServer", config: dict) -> bool:
    """
    Persist llama provider config to YAML after successful launch (NANO-065b).

    Updates llm.providers.llama section and sets llm.provider to "llama".
    Works both with orchestrator (delegates to save_to_yaml) and pre-launch
    (direct YAML surgery).

    Returns:
        True if persisted successfully.
    """
    # NANO-089: validate before persisting
    errors = validate_local_llm_config(config)
    if errors:
        print(f"[GUI] LLM config validation failed: {errors}", flush=True)
        return False

    # Strip empty-string fields before merge — prevents zombie keys in YAML
    strip_empty_optional_fields(config)

    if server._orchestrator:
        try:
            # Update orchestrator config in-memory, then persist.
            # Merge onto existing to preserve dashboard-only keys
            # (e.g. repeat_penalty from NANO-108).
            llm_cfg = server._orchestrator._config.llm_config
            if "llama" not in llm_cfg.providers:
                llm_cfg.providers["llama"] = {}
            existing = llm_cfg.providers["llama"]
            # Delete keys that the incoming config explicitly cleared
            for k in list(existing.keys()):
                if k in ("mmproj_path", "reasoning_format", "extra_args", "device", "tensor_split") and k not in config:
                    del existing[k]
            existing.update(config)
            server._orchestrator._config.save_to_yaml(server._config_path)
            print(
                f"[GUI] Local LLM config persisted via orchestrator to "
                f"{Path(server._config_path).name}",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[GUI] Failed to persist local LLM config: {e}", flush=True)
            return False

    # Pre-launch: ruamel.yaml round-trip (NANO-106)
    if not server._config_path:
        return False

    try:
        from spindl.orchestrator.config import _make_ruamel_yaml

        config_path = Path(server._config_path)
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
        existing_yaml = data["llm"]["providers"]["llama"]
        # Delete zombie keys that were cleared (not in incoming config)
        for k in list(existing_yaml.keys()):
            if k in ("mmproj_path", "reasoning_format", "extra_args", "device", "tensor_split") and k not in config:
                del existing_yaml[k]
        existing_yaml.update(config)

        with open(config_path, "w", encoding="utf-8") as f:
            ry.dump(data, f)

        # Update cache
        server._llm_config_cache = server._llm_config_cache or {}
        if "providers" not in server._llm_config_cache:
            server._llm_config_cache["providers"] = {}
        server._llm_config_cache["providers"]["llama"] = config

        print(
            f"[GUI] Local LLM config persisted pre-launch to {config_path.name}",
            flush=True,
        )
        return True

    except Exception as e:
        print(f"[GUI] Failed to persist local LLM config pre-launch: {e}", flush=True)
        return False


def validate_local_vlm_config(config: dict) -> list[str]:
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

    # mmproj_path: if set, file must exist (Session 606: stale path = silent launch failure)
    mmproj_path = config.get("mmproj_path")
    if mmproj_path and isinstance(mmproj_path, str) and not os.path.isfile(mmproj_path):
        errors.append(
            f"mmproj_path does not exist: {mmproj_path} — "
            "stale path from a previous model? Clear it if VLM is disabled"
        )

    return errors


def persist_local_vlm_config(server: "GUIServer", config: dict) -> bool:
    """
    Persist llama VLM provider config to YAML after successful launch (NANO-079).

    Updates vision.providers.llama section and sets vision.provider to "llama".
    """
    # NANO-089: validate before persisting
    errors = validate_local_vlm_config(config)
    if errors:
        print(f"[GUI] VLM config validation failed: {errors}", flush=True)
        return False

    # Strip empty-string fields before merge — prevents zombie keys in YAML
    strip_empty_optional_fields(config)

    if server._orchestrator:
        try:
            vis_cfg = server._orchestrator._config.vlm_config
            vis_cfg.provider = "llama"
            if "llama" not in vis_cfg.providers:
                vis_cfg.providers["llama"] = {}
            existing = vis_cfg.providers["llama"]
            # Delete zombie keys that were cleared
            for k in list(existing.keys()):
                if k in ("mmproj_path", "extra_args", "device", "tensor_split") and k not in config:
                    del existing[k]
            existing.update(config)
            server._orchestrator._config.save_to_yaml(server._config_path)
            print(
                f"[GUI] Local VLM config persisted via orchestrator to "
                f"{Path(server._config_path).name}",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[GUI] Failed to persist local VLM config: {e}", flush=True)
            return False

    # Pre-launch: ruamel.yaml round-trip (NANO-106)
    if not server._config_path:
        return False

    try:
        from spindl.orchestrator.config import _make_ruamel_yaml

        config_path = Path(server._config_path)
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
        existing_yaml = data["vlm"]["providers"]["llama"]
        # Delete zombie keys that were cleared
        for k in list(existing_yaml.keys()):
            if k in ("mmproj_path", "extra_args", "device", "tensor_split") and k not in config:
                del existing_yaml[k]
        existing_yaml.update(config)

        with open(config_path, "w", encoding="utf-8") as f:
            ry.dump(data, f)

        # Update cache
        server._vlm_config_cache = server._vlm_config_cache or {}
        if "providers" not in server._vlm_config_cache:
            server._vlm_config_cache["providers"] = {}
        server._vlm_config_cache["providers"]["llama"] = config

        print(
            f"[GUI] Local VLM config persisted pre-launch to {config_path.name}",
            flush=True,
        )
        return True

    except Exception as e:
        print(f"[GUI] Failed to persist local VLM config pre-launch: {e}", flush=True)
        return False


def get_llm_provider_info(server: "GUIServer", config) -> dict:
    """Get LLM provider info, extracting model name for local/cloud providers."""
    provider = config.llm_config.provider
    provider_config = config.llm_config.provider_config

    # Priority 1: Local providers (llama) - extract from model_path
    if provider == "llama" and "model_path" in provider_config:
        model_name = extract_model_name(provider_config["model_path"])
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


def get_vlm_provider_info(server: "GUIServer", config) -> dict:
    """Get VLM provider info, extracting model name for local/cloud providers."""
    provider = config.vlm_config.provider
    provider_config = config.vlm_config.providers.get(provider, {})

    # Priority 1: Local providers (llama) - extract from model_path
    if provider == "llama" and "model_path" in provider_config:
        model_name = extract_model_name(provider_config["model_path"])
        return {"name": model_name, "config": {}}

    # Priority 2: Cloud providers - check for explicit model field
    if "model" in provider_config:
        return {"name": provider_config["model"], "config": {}}

    # Priority 3: Fallback to provider name
    return {"name": provider, "config": {}}


def extract_model_name(model_path: str) -> str:
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


def register_provider_handlers(server: "GUIServer") -> None:
    """Register provider-domain Socket.IO event handlers."""
    sio = server.sio

    # ============================================================
    # NANO-065a: Runtime Tools Toggle — Socket Handlers
    # ============================================================

    @sio.event
    async def request_tools_config(sid: str, data: dict) -> None:
        """Client requests current tools state (NANO-065a)."""
        if server._orchestrator:
            state = server._orchestrator.get_tools_state()
        else:
            # Pre-launch fallback: read from cached YAML config
            cache = server._tools_config_cache or {}
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

        await sio.emit("tools_config_updated", state, to=sid)

    @sio.event
    async def set_tools_config(sid: str, data: dict) -> None:
        """Client updates tools enable/disable state at runtime (NANO-065a)."""
        if server._orchestrator:
            master_enabled = ...
            tools_changes = None

            if "master_enabled" in data:
                master_enabled = bool(data["master_enabled"])

            if "tools" in data and isinstance(data["tools"], dict):
                tools_changes = data["tools"]

            result = server._orchestrator.update_tools_config(
                master_enabled=master_enabled,
                tools=tools_changes,
            )

            # NANO-089 Phase 4: relay errors to frontend
            if not result.get("success", True):
                state = server._orchestrator.get_tools_state()
                state["error"] = result.get("error", "Tools toggle failed")
                await sio.emit("tools_config_updated", state, to=sid)
                return

            print(
                f"[GUI] Tools config updated: "
                f"master={data.get('master_enabled', '(unchanged)')}, "
                f"tools={list(data.get('tools', {}).keys()) or '(unchanged)'}",
                flush=True,
            )

            # Persist to YAML
            persisted = False
            if server._config_path:
                try:
                    server._orchestrator._config.save_to_yaml(server._config_path)
                    persisted = True
                    print(f"[GUI] Tools config persisted to {Path(server._config_path).name}", flush=True)
                except Exception as e:
                    print(f"[GUI] Failed to persist tools config: {e}", flush=True)

            # Get current state and emit
            state = server._orchestrator.get_tools_state()
            state["persisted"] = persisted
            await sio.emit("tools_config_updated", state, to=sid)
            await server._emit_health()

    # ============================================================
    # NANO-065b: Runtime LLM Provider/Model Swap — Socket Handlers
    # ============================================================

    @sio.event
    async def request_llm_config(sid: str, data: dict) -> None:
        """Client requests current LLM provider state (NANO-065b)."""
        if server._orchestrator:
            state = server._orchestrator.get_llm_state()
        else:
            # Pre-launch fallback: read from cached YAML config
            cache = server._llm_config_cache or {}
            state = {
                "provider": cache.get("provider", "llama"),
                "model": cache.get("model", ""),
                "context_size": cache.get("context_size"),
                "available_providers": list(
                    (cache.get("providers") or {}).keys()
                ),
            }

        await sio.emit("llm_config_updated", state, to=sid)

    @sio.event
    async def set_llm_provider(sid: str, data: dict) -> None:
        """Client requests LLM provider/model swap at runtime (NANO-065b)."""
        if not server._orchestrator:
            await sio.emit(
                "llm_config_updated",
                {"error": "Services not launched"},
                to=sid,
            )
            return

        provider_name = data.get("provider")
        provider_config = data.get("config")

        if not provider_name or not isinstance(provider_config, dict):
            await sio.emit(
                "llm_config_updated",
                {"error": "Missing provider name or config"},
                to=sid,
            )
            return

        result = server._orchestrator.swap_llm_provider(
            provider_name, provider_config
        )

        if not result.get("success"):
            await sio.emit(
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
        if server._config_path:
            try:
                server._orchestrator._config.save_to_yaml(server._config_path)
                persisted = True
                print(
                    f"[GUI] LLM config persisted to "
                    f"{Path(server._config_path).name}",
                    flush=True,
                )
            except Exception as e:
                print(f"[GUI] Failed to persist LLM config: {e}", flush=True)

        result["persisted"] = persisted
        await sio.emit("llm_config_updated", result, to=sid)
        await server._emit_health()

    @sio.event
    async def request_local_llm_config(sid: str, data: dict) -> None:
        """Return stored llama provider config for Dashboard hydration (NANO-065b)."""
        if server._orchestrator:
            llm_cfg = server._orchestrator._config.llm_config
            llama_cfg = dict(llm_cfg.providers.get("llama", {}))
        elif server._llm_config_cache:
            llama_cfg = dict((server._llm_config_cache.get("providers") or {}).get("llama", {}))
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
        if server._service_runner and server._service_runner.is_service_running("llm"):
            server_running = True

        await sio.emit(
            "local_llm_config",
            {"config": llama_cfg, "server_running": server_running},
            to=sid,
        )

    @sio.event
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
            await sio.emit(
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
                await sio.emit(
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
        if server._service_runner and server._service_runner.is_service_running("llm"):
            # Distinguish cloud (just unregister) vs local (actual process stop)
            is_cloud = "llm" in server._service_runner._cloud_services
            if is_cloud:
                print("[GUI] Unregistering cloud LLM provider before local launch...", flush=True)
            else:
                print("[GUI] Stopping existing LLM service for model swap...", flush=True)
            server._service_runner.stop_service("llm")
            server._launched_services.discard("llm")

        # Create or update ServiceRunner
        if server._service_runner is None:
            if server._log_aggregator is None:
                server._log_aggregator = LogAggregator()
            kwargs = dict(
                logger=server._log_aggregator,
                llm_provider_config=llm_provider_config,
                llm_context_size=context_size,
            )
            if vision_provider_config:
                kwargs["vision_provider_config"] = vision_provider_config
            server._service_runner = ServiceRunner(**kwargs)
        else:
            # Update existing runner's LLM config + bust cache
            server._service_runner._llm_provider_config = llm_provider_config
            server._service_runner._llm_provider_class = None
            server._service_runner._llm_context_size = context_size
            # NANO-079: Update vision config for -np 2 injection
            if vision_provider_config:
                server._service_runner._vision_provider_config = vision_provider_config

        # Build synthetic ServiceConfig
        svc_config = ServiceConfig(
            name="llm",
            platform="native",
            command=None,
            health_check=HealthCheckConfig(type="provider", timeout=90),
        )

        # Preview the command for diagnostics
        try:
            preview_cmd = server._service_runner._build_command(svc_config)
            print(f"[GUI] LLM launch command: {preview_cmd}", flush=True)
        except Exception as e:
            print(f"[GUI] LLM command preview failed: {e}", flush=True)

        # Launch in background thread
        def _launch():
            try:
                success = server._service_runner.start_service(svc_config)
            except Exception as e:
                success = False
                print(f"[GUI] LLM launch exception: {e}", flush=True)

            # Schedule async emit back on event loop
            async def _emit_result():
                if success:
                    server._launched_services.add("llm")
                    print(
                        f"[GUI] LLM server launched at {host}:{port}",
                        flush=True,
                    )

                    # Persist config to YAML
                    persisted = persist_local_llm_config(server, config)

                    await sio.emit(
                        "llm_server_launched",
                        {"success": True, "persisted": persisted},
                    )
                else:
                    await sio.emit(
                        "llm_server_launched",
                        {
                            "success": False,
                            "error": f"LLM server failed to start at {host}:{port}",
                        },
                    )

            loop = server._event_loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(_emit_result(), loop)

        thread = threading.Thread(target=_launch, daemon=True)
        thread.start()

        # Acknowledge receipt immediately
        await sio.emit(
            "llm_server_launched",
            {"success": None, "status": "launching"},
            to=sid,
        )

    @sio.event
    async def request_openrouter_models(sid: str, data: dict) -> None:
        """Fetch available models from OpenRouter API (NANO-065b)."""
        import aiohttp

        try:
            # Resolve base URL from config
            base_url = "https://openrouter.ai/api/v1"
            if server._orchestrator:
                cfg = server._orchestrator._config.llm_config
                or_cfg = cfg.providers.get("openrouter", {})
                base_url = or_cfg.get("url", base_url).rstrip("/")
            elif server._llm_config_cache:
                providers = server._llm_config_cache.get("providers", {})
                or_cfg = providers.get("openrouter", {})
                base_url = or_cfg.get("url", base_url).rstrip("/")

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/models", timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        await sio.emit(
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

            await sio.emit(
                "openrouter_models", {"models": models}, to=sid
            )

        except Exception as e:
            print(f"[GUI] Failed to fetch OpenRouter models: {e}", flush=True)
            await sio.emit(
                "openrouter_models",
                {"error": f"Failed to fetch models: {e}"},
                to=sid,
            )

    # ============================================================
    # NANO-065c: Runtime VLM Provider Swap — Socket Handlers
    # ============================================================

    @sio.event
    async def request_vlm_config(sid: str, data: dict) -> None:
        """Client requests current VLM provider state (NANO-065c)."""
        if server._orchestrator:
            state = server._orchestrator.get_vlm_state()
        else:
            # Pre-launch fallback: read from cached YAML config
            cache = server._vlm_config_cache or {}
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

        await sio.emit("vlm_config_updated", state, to=sid)

    @sio.event
    async def set_vlm_provider(sid: str, data: dict) -> None:
        """Client requests VLM provider swap at runtime (NANO-065c)."""
        if not server._orchestrator:
            await sio.emit(
                "vlm_config_updated",
                {"error": "Services not launched"},
                to=sid,
            )
            return

        provider_name = data.get("provider")
        provider_config = data.get("config")

        if not provider_name or not isinstance(provider_config, dict):
            await sio.emit(
                "vlm_config_updated",
                {"error": "Missing provider name or config"},
                to=sid,
            )
            return

        result = server._orchestrator.swap_vlm_provider(
            provider_name, provider_config
        )

        if not result.get("success"):
            await sio.emit(
                "vlm_config_updated",
                {"error": result.get("error", "Unknown error")},
                to=sid,
            )
            return

        # Stop orphaned VLM server when switching to unified mode —
        # the LLM handles vision now, no need for a dedicated VLM server
        if provider_name == "llm" and server._service_runner:
            if server._service_runner.is_service_running("vlm"):
                print("[GUI] Stopping dedicated VLM server (switching to unified mode)...", flush=True)
                server._service_runner.stop_service("vlm")
                server._launched_services.discard("vlm")

        print(
            f"[GUI] VLM provider swapped to {provider_name}",
            flush=True,
        )

        # Persist to YAML
        persisted = False
        if server._config_path:
            try:
                server._orchestrator._config.save_to_yaml(server._config_path)
                persisted = True
                print(
                    f"[GUI] VLM config persisted to "
                    f"{Path(server._config_path).name}",
                    flush=True,
                )
            except Exception as e:
                print(f"[GUI] Failed to persist VLM config: {e}", flush=True)

        result["persisted"] = persisted
        await sio.emit("vlm_config_updated", result, to=sid)
        await server._emit_health()

    # ============================================================
    # NANO-079: Dashboard VLM Launch — Socket Handlers
    # ============================================================

    @sio.event
    async def request_local_vlm_config(sid: str, data: dict) -> None:
        """Client requests stored local VLM config for hydration (NANO-079)."""
        vlm_cfg: dict = {}
        if server._vlm_config_cache:
            providers = server._vlm_config_cache.get("providers", {})
            vlm_cfg = dict(providers.get("llama", {}))

        # Convert extra_args list → string for frontend display
        if isinstance(vlm_cfg.get("extra_args"), list):
            vlm_cfg["extra_args"] = " ".join(str(a) for a in vlm_cfg["extra_args"])

        # Convert tensor_split list → string for frontend display
        if isinstance(vlm_cfg.get("tensor_split"), list):
            vlm_cfg["tensor_split"] = ",".join(str(x) for x in vlm_cfg["tensor_split"])

        # Check if VLM service is currently running
        server_running = False
        if server._service_runner and server._service_runner.is_service_running("vlm"):
            server_running = True

        await sio.emit(
            "local_vlm_config",
            {"config": vlm_cfg, "server_running": server_running},
            to=sid,
        )

    @sio.event
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
            await sio.emit(
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
                await sio.emit(
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
        if server._service_runner and server._service_runner.is_service_running("vlm"):
            is_cloud = "vlm" in server._service_runner._cloud_services
            if is_cloud:
                print("[GUI] Unregistering cloud VLM provider before local launch...", flush=True)
            else:
                print("[GUI] Stopping existing VLM service for model swap...", flush=True)
            server._service_runner.stop_service("vlm")
            server._launched_services.discard("vlm")

        # Create or update ServiceRunner
        if server._service_runner is None:
            if server._log_aggregator is None:
                server._log_aggregator = LogAggregator()
            server._service_runner = ServiceRunner(
                logger=server._log_aggregator,
                vision_provider_config=vision_provider_config,
            )
        else:
            # Update existing runner's VLM config + bust cache
            server._service_runner._vision_provider_config = vision_provider_config
            server._service_runner._vlm_provider_class = None

        # Build synthetic ServiceConfig for VLM
        svc_config = ServiceConfig(
            name="vlm",
            platform="native",
            command=None,
            health_check=HealthCheckConfig(type="provider", timeout=90),
        )

        # Preview the command for diagnostics
        try:
            preview_cmd = server._service_runner._build_command(svc_config)
            print(f"[GUI] VLM launch command: {preview_cmd}", flush=True)
        except Exception as e:
            print(f"[GUI] VLM command preview failed: {e}", flush=True)

        # Launch in background thread
        def _launch():
            try:
                success = server._service_runner.start_service(svc_config)
            except Exception as e:
                success = False
                print(f"[GUI] VLM launch exception: {e}", flush=True)

            # Schedule async emit back on event loop
            async def _emit_result():
                if success:
                    server._launched_services.add("vlm")
                    print(
                        f"[GUI] VLM server launched at {host}:{port}",
                        flush=True,
                    )

                    # Persist config to YAML
                    persisted = persist_local_vlm_config(server, config)

                    await sio.emit(
                        "vlm_server_launched",
                        {"success": True, "persisted": persisted},
                    )
                else:
                    await sio.emit(
                        "vlm_server_launched",
                        {
                            "success": False,
                            "error": f"VLM server failed to start at {host}:{port}",
                        },
                    )

            loop = server._event_loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(_emit_result(), loop)

        thread = threading.Thread(target=_launch, daemon=True)
        thread.start()

        # Acknowledge receipt immediately
        await sio.emit(
            "vlm_server_launched",
            {"success": None, "status": "launching"},
            to=sid,
        )
