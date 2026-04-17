"""
Config-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- Persona list/switch (request_personas, set_persona)
- VAD config (set_vad_config)
- Pipeline config (set_pipeline_config)
- Prompt injection wrappers (set_prompt_config)
- Reasoning config (set_reasoning_config)
- Generation params (set_generation_params)
- Block config CRUD (request_block_config, set_block_config, reset_block_config)
- Prompt snapshot (request_prompt_snapshot)

Also exposes emit_personas() and block config helpers as standalone functions.
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .server import GUIServer


async def emit_personas(
    server: "GUIServer", sid: Optional[str] = None
) -> None:
    """Emit list of available personas."""
    sio = server.sio

    # NANO-048: Resolve personas_dir from orchestrator (preferred) or pre-launch config
    personas_dir_str = None
    if server._orchestrator:
        personas_dir_str = server._orchestrator._config.personas_dir
    elif server._personas_dir:
        personas_dir_str = server._personas_dir

    if not personas_dir_str:
        return

    personas_dir = Path(personas_dir_str)
    personas = []
    active_persona = server._orchestrator.persona.get("id", "unknown") if server._orchestrator and server._orchestrator.persona else "unknown"

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
        await sio.emit(
            "persona_list",
            {"personas": personas, "active": active_persona},
            to=sid,
        )
    else:
        await sio.emit(
            "persona_list",
            {"personas": personas, "active": active_persona},
        )


def get_block_config_pre_launch(server: "GUIServer") -> dict:
    """
    Build block config from YAML + defaults without orchestrator (NANO-064b).

    Mirrors VoiceAgentOrchestrator.get_block_config() serialization.
    """
    from spindl.llm.prompt_block import create_default_blocks, load_block_config

    if server._prompt_blocks_config:
        blocks = load_block_config(server._prompt_blocks_config)
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


def save_block_config_pre_launch(server: "GUIServer", config: dict) -> bool:
    """
    Persist prompt_blocks config to YAML without orchestrator (NANO-064b).

    Uses ruamel.yaml round-trip (NANO-106).

    Returns:
        True if persisted successfully, False otherwise.
    """
    if not server._config_path:
        return False

    try:
        from spindl.orchestrator.config import _make_ruamel_yaml

        config_path = Path(server._config_path)
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


def register_config_handlers(server: "GUIServer") -> None:
    """Register config-domain Socket.IO event handlers."""
    sio = server.sio

    @sio.event
    async def request_personas(sid: str, data: dict) -> None:
        """Client requests available personas list."""
        await emit_personas(server, sid)

    @sio.event
    async def set_persona(sid: str, data: dict) -> None:
        """Client requests persona change (NANO-077: runtime hot-swap)."""
        from spindl.characters.loader import CharacterLoader

        persona_id = data.get("persona_id")
        if not persona_id or not server._orchestrator:
            await sio.emit(
                "persona_change_failed",
                {"error": "Missing persona_id or orchestrator not ready"},
                to=sid,
            )
            return

        try:
            success = server._orchestrator.switch_persona(persona_id)
        except Exception as e:
            print(f"[GUI] Persona switch failed: {e}", flush=True)
            await sio.emit(
                "persona_change_failed",
                {"error": str(e)},
                to=sid,
            )
            return

        if success:
            print(f"[GUI] Persona switched to: {persona_id}", flush=True)

            # Persist to spindl.yaml
            if server._config_path:
                try:
                    server._orchestrator._config.save_to_yaml(server._config_path)
                except Exception as e:
                    print(f"[GUI] Failed to persist character.default: {e}", flush=True)

            # Broadcast to ALL clients
            await sio.emit(
                "persona_changed",
                {"persona_id": persona_id, "restart_required": False},
            )
            # Push updated session list (filtered to new character)
            from .server_sessions import emit_sessions
            await emit_sessions(server, sid, persona_id)
            # Push updated config (persona name/id changed)
            await server._emit_config(sid)

            # NANO-097: Notify avatar renderer of new character's VRM model
            if server._orchestrator._config.avatar_config.enabled and server.has_avatar_client:
                try:
                    loader = CharacterLoader(
                        server._orchestrator._config.characters_dir
                    )
                    vrm_path = loader.get_vrm_path(persona_id)
                    if vrm_path:
                        expressions = loader.get_avatar_expressions(persona_id)
                        animations = loader.get_avatar_animations(persona_id)
                        char_anim_dir = str(loader.get_character_animations_dir(persona_id))
                        await server.emit_avatar_load_model(
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
                        await sio.emit(
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
            if server._orchestrator._state_machine:
                current_state = server._orchestrator._state_machine.state.value
            await sio.emit(
                "persona_change_failed",
                {"error": f"Cannot switch while {current_state}"},
                to=sid,
            )

    @sio.event
    async def set_vad_config(sid: str, data: dict) -> None:
        """Client updates VAD configuration."""
        if server._orchestrator:
            config = server._orchestrator._config
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
                sm = getattr(server._orchestrator, "_state_machine", None)
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
                if server._config_path:
                    try:
                        config.save_to_yaml(server._config_path)
                        persisted = True
                        print(f"[GUI] VAD config persisted to {Path(server._config_path).name}", flush=True)
                    except Exception as e:
                        print(f"[GUI] Failed to persist VAD config: {e}", flush=True)

                # Emit updated config to confirm
                await sio.emit(
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

    @sio.event
    async def set_pipeline_config(sid: str, data: dict) -> None:
        """Client updates pipeline configuration."""
        if server._orchestrator:
            config = server._orchestrator._config
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
                if server._config_path:
                    try:
                        config.save_to_yaml(server._config_path)
                        persisted = True
                        print(f"[GUI] Pipeline config persisted to {Path(server._config_path).name}", flush=True)
                    except Exception as e:
                        print(f"[GUI] Failed to persist Pipeline config: {e}", flush=True)

                await sio.emit(
                    "pipeline_config_updated",
                    {
                        "summarization_threshold": config.summarization_threshold,
                        "budget_strategy": config.budget_strategy,
                        "persisted": persisted,
                    },
                    to=sid,
                )

    # ============================================================
    # NANO-045d: Prompt Injection Wrappers — Socket Handler
    # ============================================================

    @sio.event
    async def set_prompt_config(sid: str, data: dict) -> None:
        """Client updates prompt injection wrapper strings (NANO-045d)."""
        if server._orchestrator:
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
                server._orchestrator.update_prompt_config(
                    rag_prefix=rag_prefix,
                    rag_suffix=rag_suffix,
                    codex_prefix=codex_prefix,
                    codex_suffix=codex_suffix,
                    example_dialogue_prefix=example_dialogue_prefix,
                    example_dialogue_suffix=example_dialogue_suffix,
                )

                pc = server._orchestrator._config.prompt_config
                print(
                    f"[GUI] Prompt wrappers updated: "
                    f"rag_prefix={pc.rag_prefix[:40]}..., "
                    f"codex_prefix={pc.codex_prefix[:40]}...",
                    flush=True,
                )

                # Persist to YAML
                persisted = False
                if server._config_path:
                    try:
                        server._orchestrator._config.save_to_yaml(server._config_path)
                        persisted = True
                        print(f"[GUI] Prompt config persisted to {Path(server._config_path).name}", flush=True)
                    except Exception as e:
                        print(f"[GUI] Failed to persist prompt config: {e}", flush=True)

                await sio.emit(
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
    # NANO-042: Reasoning Config — Socket Handler
    # ============================================================

    @sio.event
    async def set_reasoning_config(sid: str, data: dict) -> None:
        """Client updates reasoning/thinking budget (persisted to YAML)."""
        reasoning_budget = data.get("reasoning_budget")
        if reasoning_budget is None:
            return

        reasoning_budget = int(reasoning_budget)
        persisted = False

        if server._config_path:
            try:
                from spindl.orchestrator.config import _make_ruamel_yaml

                config_path = Path(server._config_path)
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

        await sio.emit(
            "reasoning_config_updated",
            {
                "reasoning_budget": reasoning_budget,
                "persisted": persisted,
            },
            to=sid,
        )

    # ============================================================
    # NANO-053: Generation Parameters — Socket Handler
    # ============================================================

    @sio.event
    async def set_generation_params(sid: str, data: dict) -> None:
        """Client updates generation parameters at runtime (NANO-053, NANO-108)."""
        if server._orchestrator:
            temperature = ...
            max_tokens = ...
            top_p = ...
            top_k = ...
            min_p = ...
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
            if "top_k" in data:
                val = int(data["top_k"])
                # top_k=0 disables the k-cap (llama.cpp semantics).
                if 0 <= val <= 1000:
                    top_k = val
                    updated = True
            if "min_p" in data:
                val = float(data["min_p"])
                if 0.0 <= val <= 1.0:
                    min_p = val
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

            # NANO-115: Handle force_role_history override (separate from gen params).
            # Session 645 removed "auto"; coerce legacy payloads for safety.
            if "force_role_history" in data:
                val = data["force_role_history"]
                if val == "auto":
                    val = "flatten"
                if val in ("splice", "flatten"):
                    server._orchestrator._config.force_role_history = val
                    server._orchestrator._pipeline._force_role_history = val
                    updated = True

            if updated:
                server._orchestrator.update_generation_params(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repeat_penalty=repeat_penalty,
                    repeat_last_n=repeat_last_n,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )

                pc = server._orchestrator._config.llm_config.provider_config
                print(
                    f"[GUI] Generation params: "
                    f"temp={pc.get('temperature', 0.7)}, "
                    f"max_tokens={pc.get('max_tokens', 256)}, "
                    f"top_p={pc.get('top_p', 0.95)}, "
                    f"top_k={pc.get('top_k', 40)}, "
                    f"min_p={pc.get('min_p', 0.05)}, "
                    f"repeat_penalty={pc.get('repeat_penalty', 1.1)}, "
                    f"repeat_last_n={pc.get('repeat_last_n', 64)}, "
                    f"freq_penalty={pc.get('frequency_penalty', 0.0)}, "
                    f"pres_penalty={pc.get('presence_penalty', 0.0)}",
                    flush=True,
                )

                # Persist to YAML
                persisted = False
                if server._config_path:
                    try:
                        server._orchestrator._config.save_to_yaml(server._config_path)
                        persisted = True
                        print(f"[GUI] Generation params persisted to {Path(server._config_path).name}", flush=True)
                    except Exception as e:
                        print(f"[GUI] Failed to persist generation params: {e}", flush=True)

                await sio.emit(
                    "generation_params_updated",
                    {
                        "temperature": pc.get("temperature", 0.7),
                        "max_tokens": pc.get("max_tokens", 256),
                        "top_p": pc.get("top_p", 0.95),
                        "top_k": pc.get("top_k", 40),
                        "min_p": pc.get("min_p", 0.05),
                        "repeat_penalty": pc.get("repeat_penalty", 1.1),
                        "repeat_last_n": pc.get("repeat_last_n", 64),
                        "frequency_penalty": pc.get("frequency_penalty", 0.0),
                        "presence_penalty": pc.get("presence_penalty", 0.0),
                        "force_role_history": server._orchestrator._config.force_role_history,
                        "persisted": persisted,
                    },
                    to=sid,
                )

    # ============================================================
    # NANO-045c-1: Block Config — Socket Handlers
    # ============================================================

    @sio.event
    async def request_block_config(sid: str, data: dict) -> None:
        """Client requests current block configuration (NANO-045c-1, NANO-064b)."""
        try:
            if server._orchestrator:
                config = server._orchestrator.get_block_config()
            else:
                config = get_block_config_pre_launch(server)
            await sio.emit("block_config_loaded", config, to=sid)
        except Exception as e:
            print(f"[GUI] Error getting block config: {e}", flush=True)

    @sio.event
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

            if server._orchestrator:
                # Live path: merge + hot-reload + YAML save
                existing = server._orchestrator._config.prompt_blocks
                if existing and isinstance(existing, dict):
                    merged = {**existing, **config}
                else:
                    merged = config

                server._orchestrator.update_block_config(merged)

                print(
                    f"[GUI] Block config updated: "
                    f"order={len(merged.get('order', []))} blocks, "
                    f"disabled={len(merged.get('disabled', []))} blocks",
                    flush=True,
                )

                persisted = False
                if server._config_path:
                    try:
                        server._orchestrator._config.save_block_config_to_yaml(
                            server._config_path
                        )
                        persisted = True
                        print(
                            f"[GUI] Block config persisted to {Path(server._config_path).name}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GUI] Failed to persist block config: {e}", flush=True)

                current = server._orchestrator.get_block_config()
            else:
                # Pre-launch path: YAML-only save (NANO-064b)
                existing = server._prompt_blocks_config
                if existing and isinstance(existing, dict):
                    merged = {**existing, **config}
                else:
                    merged = config

                server._prompt_blocks_config = merged
                persisted = save_block_config_pre_launch(server, merged)

                print(
                    f"[GUI] Block config updated pre-launch: "
                    f"order={len(merged.get('order', []))} blocks, "
                    f"disabled={len(merged.get('disabled', []))} blocks",
                    flush=True,
                )

                current = get_block_config_pre_launch(server)

            await sio.emit(
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
            await sio.emit(
                "block_config_updated",
                {"success": False, "persisted": False, "order": [], "disabled": [], "overrides": {}},
                to=sid,
            )

    @sio.event
    async def reset_block_config(sid: str, data: dict) -> None:
        """Client resets block configuration to defaults (NANO-045c-1, NANO-064b)."""
        try:
            if server._orchestrator:
                server._orchestrator.reset_block_config()

                persisted = False
                if server._config_path:
                    try:
                        server._orchestrator._config.save_block_config_to_yaml(
                            server._config_path
                        )
                        persisted = True
                    except Exception as e:
                        print(f"[GUI] Failed to persist block config reset: {e}", flush=True)

                current = server._orchestrator.get_block_config()
            else:
                # Pre-launch: clear cached config, remove section from YAML
                server._prompt_blocks_config = None
                persisted = save_block_config_pre_launch(server, {})
                current = get_block_config_pre_launch(server)

            print("[GUI] Block config reset to defaults", flush=True)

            await sio.emit(
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
            await sio.emit(
                "block_config_updated",
                {"success": False, "persisted": False, "order": [], "disabled": [], "overrides": {}},
                to=sid,
            )

    @sio.event
    async def request_prompt_snapshot(sid: str, data: dict | None = None) -> None:
        """Client requests current prompt snapshot (NANO-076)."""
        if not server._orchestrator:
            # No orchestrator — frontend stays on currentSnapshot: null
            return

        try:
            snapshot = server._orchestrator.get_prompt_snapshot()
            if snapshot:
                await sio.emit("prompt_snapshot", snapshot, to=sid)
            # else: no snapshot available — frontend stays on null (cleared on session switch)
        except Exception as e:
            print(f"[GUI] Error building prompt snapshot: {e}", flush=True)
