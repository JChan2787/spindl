"""
Stimuli-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- Stimuli config (set_stimuli_config)
- Addressing others start/stop (addressing_others_start, addressing_others_stop)
- Patience progress (request_patience_progress)
- Twitch status (request_twitch_status)
- Twitch credential testing (test_twitch_credentials)
- Game-state bridge status (request_game_state_status) (NANO-116)
- Game-state bridge connection testing (test_game_state_connection) (NANO-116)
- Chat-TTS server launch/stop/status (NANO-130 Phase 2)

Also exposes persist_twitch_credentials() and build_stimuli_hydration()
as standalone helpers.
"""

import os
import re
import socket
import subprocess
import threading
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
        "patience_prompts": cfg.patience_prompts,
        "twitch_enabled": cfg.twitch_enabled,
        "twitch_channel": cfg.twitch_channel or "",
        "twitch_app_id": cfg.twitch_app_id or "",
        "twitch_app_secret": cfg.twitch_app_secret or "",
        "twitch_buffer_size": cfg.twitch_buffer_size,
        "twitch_max_message_length": cfg.twitch_max_message_length,
        "twitch_prompt_template": cfg.twitch_prompt_template,
        "twitch_audience_window": cfg.twitch_audience_window,
        "twitch_audience_char_cap": cfg.twitch_audience_char_cap,
        # NANO-130: Selection pass + staleness filter
        "twitch_max_message_age_seconds": cfg.twitch_max_message_age_seconds,
        "twitch_selection_mode": cfg.twitch_selection_mode,
        "twitch_selection_pass_model": cfg.twitch_selection_pass_model,
        "twitch_selection_pass_api_key": cfg.twitch_selection_pass_api_key or "",
        # NANO-130 Phase 2: Chat-TTS
        "twitch_chat_tts_enabled": cfg.twitch_chat_tts_enabled,
        "twitch_chat_tts_host": cfg.twitch_chat_tts_host,
        "twitch_chat_tts_port": cfg.twitch_chat_tts_port,
        "twitch_chat_tts_device": cfg.twitch_chat_tts_device,
        "twitch_chat_tts_voice": cfg.twitch_chat_tts_voice,
        "twitch_chat_tts_speed": cfg.twitch_chat_tts_speed,
        "twitch_chat_tts_format": cfg.twitch_chat_tts_format,
        "twitch_chat_tts_max_length": cfg.twitch_chat_tts_max_length,
        "twitch_has_credentials": bool(
            resolved_channel and resolved_app_id and resolved_app_secret
        ),
        # NANO-116: Game-state bridge integration
        "game_state_enabled": cfg.game_state_enabled,
        "game_state_host": cfg.game_state_host,
        "game_state_port": cfg.game_state_port,
        "game_state_buffer_size": cfg.game_state_buffer_size,
        "game_state_prompt_template": cfg.game_state_prompt_template,
        # NANO-116 B.2: Dialogue pipeline
        "game_state_dialogue_enabled": cfg.game_state_dialogue_enabled,
        "game_state_dialogue_buffer_size": cfg.game_state_dialogue_buffer_size,
        "game_state_dialogue_prompt_templates": cfg.game_state_dialogue_prompt_templates,
        "game_state_dialogue_token_budget": cfg.game_state_dialogue_token_budget,
        "game_state_dialogue_summary_max_tokens": cfg.game_state_dialogue_summary_max_tokens,
        "game_state_dialogue_min_lines": cfg.game_state_dialogue_min_lines,
        "game_state_dialogue_drain_delay": cfg.game_state_dialogue_drain_delay,
        "game_state_dialogue_summarizer_model": cfg.game_state_dialogue_summarizer_model,
        "game_state_dialogue_summarizer_api_key": cfg.game_state_dialogue_summarizer_api_key or "",
        "game_state_dialogue_summarizer_persona": cfg.game_state_dialogue_summarizer_persona or "",
        # NANO-122: Gameplay stimulus
        "game_state_gameplay_enabled": cfg.game_state_gameplay_enabled,
        "game_state_gameplay_base_probability": cfg.game_state_gameplay_base_probability,
        "game_state_gameplay_escalation_step": cfg.game_state_gameplay_escalation_step,
        "game_state_gameplay_probability_ceiling": cfg.game_state_gameplay_probability_ceiling,
        "game_state_gameplay_dirty_hp_threshold": cfg.game_state_gameplay_dirty_hp_threshold,
        "game_state_gameplay_event_batch_window": cfg.game_state_gameplay_event_batch_window,
        # NANO-124: Self-barge-in
        "game_state_barge_in_enabled": cfg.game_state_barge_in_enabled,
        "game_state_barge_in_escalation": cfg.game_state_barge_in_escalation,
        "game_state_barge_in_fatigue": cfg.game_state_barge_in_fatigue,
        "game_state_barge_in_prompt_templates": cfg.game_state_barge_in_prompt_templates,
        # NANO-110: Addressing-others contexts
        "addressing_others_contexts": [
            {"id": ctx.id, "label": ctx.label, "prompt": ctx.prompt}
            for ctx in cfg.addressing_others_contexts
        ],
        # NANO-121: Model cycling
        "model_rotation_enabled": cfg.model_rotation_enabled,
        "model_rotation_models": cfg.model_rotation_models,
        "model_rotation_api_key": cfg.model_rotation_api_key or "",
        # NANO-117: Weighted arbitration
        "arbitration_decay_multiplier": cfg.arbitration_decay_multiplier,
        "arbitration_recovery_per_cycle": cfg.arbitration_recovery_per_cycle,
        "arbitration_weight_overrides": dict(cfg.arbitration_weight_overrides),
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
            patience_prompts = data.get("patience_prompts")

            # Twitch fields (NANO-056b)
            twitch_enabled = data.get("twitch_enabled")
            twitch_channel = data.get("twitch_channel")
            twitch_app_id = data.get("twitch_app_id")
            twitch_app_secret = data.get("twitch_app_secret")
            twitch_buffer_size = data.get("twitch_buffer_size")
            twitch_max_message_length = data.get("twitch_max_message_length")
            twitch_prompt_template = data.get("twitch_prompt_template")
            twitch_audience_window = data.get("twitch_audience_window")
            twitch_audience_char_cap = data.get("twitch_audience_char_cap")

            # NANO-130: Twitch selection pass + staleness filter
            twitch_max_message_age_seconds = data.get("twitch_max_message_age_seconds")
            twitch_selection_mode = data.get("twitch_selection_mode")
            twitch_selection_pass_model = data.get("twitch_selection_pass_model")
            twitch_selection_pass_api_key = data.get("twitch_selection_pass_api_key")

            # NANO-130 Phase 2: Chat-TTS
            twitch_chat_tts_enabled = data.get("twitch_chat_tts_enabled")
            twitch_chat_tts_host = data.get("twitch_chat_tts_host")
            twitch_chat_tts_port = data.get("twitch_chat_tts_port")
            twitch_chat_tts_device = data.get("twitch_chat_tts_device")
            twitch_chat_tts_voice = data.get("twitch_chat_tts_voice")
            twitch_chat_tts_speed = data.get("twitch_chat_tts_speed")
            twitch_chat_tts_format = data.get("twitch_chat_tts_format")
            twitch_chat_tts_max_length = data.get("twitch_chat_tts_max_length")

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
            if patience_prompts is not None:
                if not isinstance(patience_prompts, list):
                    patience_prompts = None
                else:
                    cleaned_prompts: list[str] = []
                    for p in patience_prompts:
                        s = str(p).strip()
                        if s:
                            cleaned_prompts.append(s)
                    if not cleaned_prompts:
                        patience_prompts = None
                    else:
                        patience_prompts = cleaned_prompts

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
                elif "{messages}" not in twitch_prompt_template:
                    await sio.emit(
                        "stimuli_config_error",
                        {
                            "field": "twitch_prompt_template",
                            "message": (
                                "twitch_prompt_template must contain the "
                                "{messages} placeholder. Without it, buffered "
                                "Twitch messages have nowhere to render and "
                                "the model receives only the directive text."
                            ),
                        },
                        to=sid,
                    )
                    return
            if twitch_audience_window is not None:
                twitch_audience_window = int(twitch_audience_window)
                twitch_audience_window = max(25, min(300, twitch_audience_window))
            if twitch_audience_char_cap is not None:
                twitch_audience_char_cap = int(twitch_audience_char_cap)
                twitch_audience_char_cap = max(50, min(500, twitch_audience_char_cap))

            # NANO-130: Twitch selection pass type coercion
            if twitch_max_message_age_seconds is not None:
                twitch_max_message_age_seconds = float(twitch_max_message_age_seconds)
                twitch_max_message_age_seconds = max(1.0, min(120.0, twitch_max_message_age_seconds))
            if twitch_selection_mode is not None:
                twitch_selection_mode = str(twitch_selection_mode).strip()
                if twitch_selection_mode not in ("llm", "heuristic"):
                    twitch_selection_mode = "llm"
            if twitch_selection_pass_model is not None:
                twitch_selection_pass_model = str(twitch_selection_pass_model).strip()
            if twitch_selection_pass_api_key is not None:
                twitch_selection_pass_api_key = str(twitch_selection_pass_api_key).strip()

            # NANO-130 Phase 2: Chat-TTS type coercion
            if twitch_chat_tts_enabled is not None:
                twitch_chat_tts_enabled = bool(twitch_chat_tts_enabled)
            if twitch_chat_tts_host is not None:
                twitch_chat_tts_host = str(twitch_chat_tts_host).strip() or "127.0.0.1"
            if twitch_chat_tts_port is not None:
                twitch_chat_tts_port = int(twitch_chat_tts_port)
                twitch_chat_tts_port = max(1, min(65535, twitch_chat_tts_port))
            if twitch_chat_tts_device is not None:
                twitch_chat_tts_device = str(twitch_chat_tts_device).strip() or "cpu"
            if twitch_chat_tts_voice is not None:
                twitch_chat_tts_voice = str(twitch_chat_tts_voice).strip()
            if twitch_chat_tts_speed is not None:
                twitch_chat_tts_speed = float(twitch_chat_tts_speed)
                twitch_chat_tts_speed = max(0.5, min(2.0, twitch_chat_tts_speed))
            if twitch_chat_tts_format is not None:
                twitch_chat_tts_format = str(twitch_chat_tts_format).strip()
                if not twitch_chat_tts_format:
                    twitch_chat_tts_format = "{username} says: {message}"
            if twitch_chat_tts_max_length is not None:
                twitch_chat_tts_max_length = int(twitch_chat_tts_max_length)
                twitch_chat_tts_max_length = max(20, min(500, twitch_chat_tts_max_length))

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

            # Game-state bridge fields (NANO-116)
            game_state_enabled = data.get("game_state_enabled")
            game_state_host = data.get("game_state_host")
            game_state_port = data.get("game_state_port")
            game_state_buffer_size = data.get("game_state_buffer_size")
            game_state_prompt_template = data.get("game_state_prompt_template")

            # Game-state dialogue pipeline fields (NANO-116 B.2)
            game_state_dialogue_enabled = data.get("game_state_dialogue_enabled")
            game_state_dialogue_buffer_size = data.get("game_state_dialogue_buffer_size")
            game_state_dialogue_prompt_templates = data.get("game_state_dialogue_prompt_templates")
            game_state_dialogue_token_budget = data.get("game_state_dialogue_token_budget")
            game_state_dialogue_summary_max_tokens = data.get("game_state_dialogue_summary_max_tokens")
            game_state_dialogue_min_lines = data.get("game_state_dialogue_min_lines")
            game_state_dialogue_drain_delay = data.get("game_state_dialogue_drain_delay")
            game_state_dialogue_summarizer_model = data.get("game_state_dialogue_summarizer_model")
            game_state_dialogue_summarizer_api_key = data.get("game_state_dialogue_summarizer_api_key")
            game_state_dialogue_summarizer_persona = data.get("game_state_dialogue_summarizer_persona")

            # NANO-122: Gameplay stimulus fields
            game_state_gameplay_enabled = data.get("game_state_gameplay_enabled")
            game_state_gameplay_base_probability = data.get("game_state_gameplay_base_probability")
            game_state_gameplay_escalation_step = data.get("game_state_gameplay_escalation_step")
            game_state_gameplay_probability_ceiling = data.get("game_state_gameplay_probability_ceiling")
            game_state_gameplay_dirty_hp_threshold = data.get("game_state_gameplay_dirty_hp_threshold")
            game_state_gameplay_event_batch_window = data.get("game_state_gameplay_event_batch_window")

            # NANO-124: Self-barge-in fields
            game_state_barge_in_enabled = data.get("game_state_barge_in_enabled")
            game_state_barge_in_escalation = data.get("game_state_barge_in_escalation")
            game_state_barge_in_fatigue = data.get("game_state_barge_in_fatigue")
            game_state_barge_in_prompt_templates = data.get("game_state_barge_in_prompt_templates")

            # NANO-121: Model cycling fields
            model_rotation_enabled = data.get("model_rotation_enabled")
            model_rotation_models = data.get("model_rotation_models")
            model_rotation_api_key = data.get("model_rotation_api_key")

            if model_rotation_enabled is not None:
                model_rotation_enabled = bool(model_rotation_enabled)
            if model_rotation_models is not None:
                if not isinstance(model_rotation_models, list):
                    model_rotation_models = None
                else:
                    model_rotation_models = [
                        str(m).strip() for m in model_rotation_models if str(m).strip()
                    ]
            if model_rotation_api_key is not None:
                model_rotation_api_key = str(model_rotation_api_key).strip()

            # NANO-117: Weighted arbitration fields
            arbitration_decay_multiplier = data.get("arbitration_decay_multiplier")
            arbitration_recovery_per_cycle = data.get("arbitration_recovery_per_cycle")

            if arbitration_decay_multiplier is not None:
                arbitration_decay_multiplier = max(0.1, min(1.0, float(arbitration_decay_multiplier)))
            if arbitration_recovery_per_cycle is not None:
                arbitration_recovery_per_cycle = max(0.05, min(0.5, float(arbitration_recovery_per_cycle)))

            arbitration_weight_overrides = data.get("arbitration_weight_overrides")
            if arbitration_weight_overrides is not None:
                if isinstance(arbitration_weight_overrides, dict):
                    arbitration_weight_overrides = {
                        str(k): max(0.1, min(10.0, float(v)))
                        for k, v in arbitration_weight_overrides.items()
                    }
                else:
                    arbitration_weight_overrides = None

            # Game-state type coercion (NANO-116)
            if game_state_enabled is not None:
                game_state_enabled = bool(game_state_enabled)
                print(f"[GUI] game_state_enabled={game_state_enabled}", flush=True)
            if game_state_host is not None:
                game_state_host = str(game_state_host).strip()
            if game_state_port is not None:
                game_state_port = int(game_state_port)
                game_state_port = max(1, min(65535, game_state_port))
            if game_state_buffer_size is not None:
                game_state_buffer_size = int(game_state_buffer_size)
                game_state_buffer_size = max(1, min(100, game_state_buffer_size))
            if game_state_prompt_template is not None:
                game_state_prompt_template = str(game_state_prompt_template).strip()
                if not game_state_prompt_template:
                    game_state_prompt_template = None
                elif "{events}" not in game_state_prompt_template:
                    await sio.emit(
                        "stimuli_config_error",
                        {
                            "field": "game_state_prompt_template",
                            "message": (
                                "game_state_prompt_template must contain the "
                                "{events} placeholder."
                            ),
                        },
                        to=sid,
                    )
                    return

            # Game-state dialogue type coercion (NANO-116 B.2)
            if game_state_dialogue_enabled is not None:
                game_state_dialogue_enabled = bool(game_state_dialogue_enabled)
            if game_state_dialogue_buffer_size is not None:
                game_state_dialogue_buffer_size = int(game_state_dialogue_buffer_size)
                game_state_dialogue_buffer_size = max(1, min(200, game_state_dialogue_buffer_size))
            if game_state_dialogue_prompt_templates is not None:
                if not isinstance(game_state_dialogue_prompt_templates, list):
                    game_state_dialogue_prompt_templates = None
                else:
                    cleaned: list[str] = []
                    for t in game_state_dialogue_prompt_templates:
                        s = str(t).strip()
                        if not s:
                            continue
                        if "{dialogue}" not in s:
                            await sio.emit(
                                "stimuli_config_error",
                                {
                                    "field": "game_state_dialogue_prompt_templates",
                                    "message": (
                                        "Every dialogue prompt template must "
                                        "contain the {dialogue} placeholder."
                                    ),
                                },
                                to=sid,
                            )
                            return
                        cleaned.append(s)
                    if not cleaned:
                        game_state_dialogue_prompt_templates = None
                    else:
                        game_state_dialogue_prompt_templates = cleaned
            if game_state_dialogue_token_budget is not None:
                game_state_dialogue_token_budget = int(game_state_dialogue_token_budget)
                game_state_dialogue_token_budget = max(200, min(4000, game_state_dialogue_token_budget))
            if game_state_dialogue_summary_max_tokens is not None:
                game_state_dialogue_summary_max_tokens = int(game_state_dialogue_summary_max_tokens)
                game_state_dialogue_summary_max_tokens = max(64, min(2048, game_state_dialogue_summary_max_tokens))
            if game_state_dialogue_min_lines is not None:
                game_state_dialogue_min_lines = int(game_state_dialogue_min_lines)
                game_state_dialogue_min_lines = max(1, min(50, game_state_dialogue_min_lines))
            if game_state_dialogue_drain_delay is not None:
                game_state_dialogue_drain_delay = float(game_state_dialogue_drain_delay)
                game_state_dialogue_drain_delay = max(0.0, min(30.0, game_state_dialogue_drain_delay))
            if game_state_dialogue_summarizer_model is not None:
                game_state_dialogue_summarizer_model = str(game_state_dialogue_summarizer_model).strip()
            if game_state_dialogue_summarizer_api_key is not None:
                game_state_dialogue_summarizer_api_key = str(game_state_dialogue_summarizer_api_key).strip()
            if game_state_dialogue_summarizer_persona is not None:
                game_state_dialogue_summarizer_persona = str(game_state_dialogue_summarizer_persona).strip()

            # NANO-122: Gameplay stimulus type coercion
            if game_state_gameplay_enabled is not None:
                game_state_gameplay_enabled = bool(game_state_gameplay_enabled)
            if game_state_gameplay_base_probability is not None:
                game_state_gameplay_base_probability = float(game_state_gameplay_base_probability)
                game_state_gameplay_base_probability = max(0.05, min(1.0, game_state_gameplay_base_probability))
            if game_state_gameplay_escalation_step is not None:
                game_state_gameplay_escalation_step = float(game_state_gameplay_escalation_step)
                game_state_gameplay_escalation_step = max(0.05, min(0.5, game_state_gameplay_escalation_step))
            if game_state_gameplay_probability_ceiling is not None:
                game_state_gameplay_probability_ceiling = float(game_state_gameplay_probability_ceiling)
                game_state_gameplay_probability_ceiling = max(0.1, min(1.0, game_state_gameplay_probability_ceiling))
            if game_state_gameplay_dirty_hp_threshold is not None:
                game_state_gameplay_dirty_hp_threshold = float(game_state_gameplay_dirty_hp_threshold)
                game_state_gameplay_dirty_hp_threshold = max(0.01, min(0.5, game_state_gameplay_dirty_hp_threshold))
            if game_state_gameplay_event_batch_window is not None:
                game_state_gameplay_event_batch_window = float(game_state_gameplay_event_batch_window)
                game_state_gameplay_event_batch_window = max(0.5, min(10.0, game_state_gameplay_event_batch_window))

            # NANO-124: Self-barge-in type coercion
            if game_state_barge_in_enabled is not None:
                game_state_barge_in_enabled = bool(game_state_barge_in_enabled)
            if game_state_barge_in_escalation is not None:
                if not isinstance(game_state_barge_in_escalation, list):
                    game_state_barge_in_escalation = None
                else:
                    game_state_barge_in_escalation = [
                        max(0.0, min(1.0, float(v))) for v in game_state_barge_in_escalation
                    ]
            if game_state_barge_in_fatigue is not None:
                if not isinstance(game_state_barge_in_fatigue, list):
                    game_state_barge_in_fatigue = None
                else:
                    game_state_barge_in_fatigue = [
                        max(0.0, min(1.0, float(v))) for v in game_state_barge_in_fatigue
                    ]
            if game_state_barge_in_prompt_templates is not None:
                if not isinstance(game_state_barge_in_prompt_templates, list):
                    game_state_barge_in_prompt_templates = None
                else:
                    cleaned_bi: list[str] = []
                    for t in game_state_barge_in_prompt_templates:
                        s = str(t).strip()
                        if not s:
                            continue
                        if "{dialogue}" not in s:
                            await sio.emit(
                                "stimuli_config_error",
                                {
                                    "field": "game_state_barge_in_prompt_templates",
                                    "message": (
                                        "Every barge-in prompt template must "
                                        "contain the {dialogue} placeholder."
                                    ),
                                },
                                to=sid,
                            )
                            return
                        cleaned_bi.append(s)
                    if not cleaned_bi:
                        game_state_barge_in_prompt_templates = None
                    else:
                        game_state_barge_in_prompt_templates = cleaned_bi

            server._orchestrator.update_stimuli_config(
                enabled=enabled,
                patience_enabled=patience_enabled,
                patience_seconds=patience_seconds,
                patience_prompts=patience_prompts,
                twitch_enabled=twitch_enabled,
                twitch_channel=twitch_channel,
                twitch_app_id=twitch_app_id,
                twitch_app_secret=twitch_app_secret,
                twitch_buffer_size=twitch_buffer_size,
                twitch_max_message_length=twitch_max_message_length,
                twitch_prompt_template=twitch_prompt_template,
                twitch_audience_window=twitch_audience_window,
                twitch_audience_char_cap=twitch_audience_char_cap,
                twitch_max_message_age_seconds=twitch_max_message_age_seconds,
                twitch_selection_mode=twitch_selection_mode,
                twitch_selection_pass_model=twitch_selection_pass_model,
                twitch_selection_pass_api_key=twitch_selection_pass_api_key,
                twitch_chat_tts_enabled=twitch_chat_tts_enabled,
                twitch_chat_tts_host=twitch_chat_tts_host,
                twitch_chat_tts_port=twitch_chat_tts_port,
                twitch_chat_tts_device=twitch_chat_tts_device,
                twitch_chat_tts_voice=twitch_chat_tts_voice,
                twitch_chat_tts_speed=twitch_chat_tts_speed,
                twitch_chat_tts_format=twitch_chat_tts_format,
                twitch_chat_tts_max_length=twitch_chat_tts_max_length,
                addressing_others_contexts=addressing_others_contexts,
                game_state_enabled=game_state_enabled,
                game_state_host=game_state_host,
                game_state_port=game_state_port,
                game_state_buffer_size=game_state_buffer_size,
                game_state_prompt_template=game_state_prompt_template,
                game_state_dialogue_enabled=game_state_dialogue_enabled,
                game_state_dialogue_buffer_size=game_state_dialogue_buffer_size,
                game_state_dialogue_prompt_templates=game_state_dialogue_prompt_templates,
                game_state_dialogue_token_budget=game_state_dialogue_token_budget,
                game_state_dialogue_summary_max_tokens=game_state_dialogue_summary_max_tokens,
                game_state_dialogue_min_lines=game_state_dialogue_min_lines,
                game_state_dialogue_drain_delay=game_state_dialogue_drain_delay,
                game_state_dialogue_summarizer_model=game_state_dialogue_summarizer_model,
                game_state_dialogue_summarizer_api_key=game_state_dialogue_summarizer_api_key,
                game_state_dialogue_summarizer_persona=game_state_dialogue_summarizer_persona,
                game_state_gameplay_enabled=game_state_gameplay_enabled,
                game_state_gameplay_base_probability=game_state_gameplay_base_probability,
                game_state_gameplay_escalation_step=game_state_gameplay_escalation_step,
                game_state_gameplay_probability_ceiling=game_state_gameplay_probability_ceiling,
                game_state_gameplay_dirty_hp_threshold=game_state_gameplay_dirty_hp_threshold,
                game_state_gameplay_event_batch_window=game_state_gameplay_event_batch_window,
                game_state_barge_in_enabled=game_state_barge_in_enabled,
                game_state_barge_in_escalation=game_state_barge_in_escalation,
                game_state_barge_in_fatigue=game_state_barge_in_fatigue,
                game_state_barge_in_prompt_templates=game_state_barge_in_prompt_templates,
                model_rotation_enabled=model_rotation_enabled,
                model_rotation_models=model_rotation_models,
                model_rotation_api_key=model_rotation_api_key,
                arbitration_decay_multiplier=arbitration_decay_multiplier,
                arbitration_recovery_per_cycle=arbitration_recovery_per_cycle,
                arbitration_weight_overrides=arbitration_weight_overrides,
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

    # ============================================================
    # NANO-125: Mic Passthrough — Socket Handlers
    # ============================================================

    @sio.event
    async def mic_passthrough_on(sid: str, data: dict) -> None:
        """Stream Deck toggles mic passthrough ON (NANO-125)."""
        if not server._orchestrator:
            return
        server._orchestrator.set_mic_passthrough(True)
        await sio.emit("mic_passthrough_state", {"active": True})

    @sio.event
    async def mic_passthrough_off(sid: str, data: dict) -> None:
        """Stream Deck toggles mic passthrough OFF (NANO-125)."""
        if not server._orchestrator:
            return
        server._orchestrator.set_mic_passthrough(False)
        await sio.emit("mic_passthrough_state", {"active": False})

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

    # ============================================================
    # NANO-116: Game-State Bridge — Socket Handlers
    # ============================================================

    @sio.event
    async def request_game_state_status(sid: str, data: dict) -> None:
        """Client requests current game-state bridge status (NANO-116)."""
        default_status = {
            "connected": False,
            "protocol_version": None,
            "buffer_count": 0,
            "recent_lines": [],
            "enabled": False,
            "dialogue_enabled": False,
            "current_summary": "",
            "gameplay_enabled": False,
            "gameplay_event_buffer_count": 0,
            "gameplay_snapshot_probability": 0.0,
        }

        if (
            not server._orchestrator
            or not server._orchestrator.stimuli_engine
        ):
            await sio.emit("game_state_status", default_status, to=sid)
            return

        engine = server._orchestrator.stimuli_engine
        for module in engine.modules:
            if module.name == "game_state":
                recent_lines = []
                if hasattr(module, "dialogue_buffer") and module.dialogue_buffer:
                    recent_lines = module.dialogue_buffer.get_recent_lines(10)

                current_summary = ""
                if hasattr(server._orchestrator, "_dialogue_store") and server._orchestrator._dialogue_store:
                    current_summary = server._orchestrator._dialogue_store.summary_blob or ""

                await sio.emit(
                    "game_state_status",
                    {
                        "connected": module.connected,
                        "protocol_version": module.bridge_protocol_version,
                        "buffer_count": module.buffer_count,
                        "recent_lines": recent_lines,
                        "enabled": module.enabled,
                        "dialogue_enabled": server._orchestrator._config.stimuli_config.game_state_dialogue_enabled,
                        "current_summary": current_summary,
                        "gameplay_enabled": module.gameplay_enabled,
                        "gameplay_event_buffer_count": len(module._gameplay_event_buffer),
                        "gameplay_snapshot_probability": 0.0,
                    },
                    to=sid,
                )
                return

        await sio.emit("game_state_status", default_status, to=sid)

    @sio.event
    async def test_game_state_connection(sid: str, data: dict) -> None:
        """Test TCP connection to game-state bridge without launching module (NANO-116)."""
        import asyncio

        host = str(data.get("host", "127.0.0.1")).strip()
        port = int(data.get("port", 53817))

        if not host:
            await sio.emit(
                "game_state_connection_result",
                {"success": False, "error": "Host is required."},
                to=sid,
            )
            return

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=3.0,
            )

            # Don't read anything — just verify the port is listening.
            # Reading would consume bridge_ready from the ring buffer,
            # leaving the module with no banner on connect.
            protocol_version = None

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

            await sio.emit(
                "game_state_connection_result",
                {
                    "success": True,
                    "error": None,
                    "protocol_version": protocol_version,
                },
                to=sid,
            )
            print(
                f"[GUI] Game-state bridge connection test succeeded "
                f"(host={host}, port={port}, version={protocol_version})",
                flush=True,
            )

        except asyncio.TimeoutError:
            await sio.emit(
                "game_state_connection_result",
                {
                    "success": False,
                    "error": f"Connection timed out ({host}:{port}). Is the bridge running?",
                },
                to=sid,
            )
            print(f"[GUI] Game-state bridge connection test timed out ({host}:{port})", flush=True)

        except OSError as e:
            await sio.emit(
                "game_state_connection_result",
                {
                    "success": False,
                    "error": f"Connection refused ({host}:{port}): {e}",
                },
                to=sid,
            )
            print(f"[GUI] Game-state bridge connection test failed: {e}", flush=True)

    # ============================================================
    # NANO-130: Chat-TTS Kokoro Server Launch / Stop / Status
    # ============================================================

    @sio.event
    async def launch_chat_tts(sid: str, data: dict) -> None:
        """Launch the dedicated chat-TTS Kokoro server (NANO-130)."""
        cfg = server._config.stimuli_config if server._config else None
        if cfg is None:
            await sio.emit("chat_tts_launched", {"success": False, "error": "No config loaded"}, to=sid)
            return

        host = data.get("host", cfg.twitch_chat_tts_host)
        port = int(data.get("port", cfg.twitch_chat_tts_port))
        device = data.get("device", cfg.twitch_chat_tts_device)

        # Check if already running
        if server._chat_tts_process and server._chat_tts_process.poll() is None:
            # Verify via TCP
            if _tcp_check(host, port):
                await sio.emit("chat_tts_launched", {"success": True, "already_running": True}, to=sid)
                return
            else:
                server._chat_tts_process = None

        # Check for port conflict
        if _tcp_check(host, port):
            await sio.emit(
                "chat_tts_launched",
                {"success": False, "error": f"Port {port} already in use. Choose a different port or stop the existing process."},
                to=sid,
            )
            return

        # Resolve server script and models_dir
        from spindl.tts.builtin.kokoro.provider import KokoroTTSProvider
        kokoro_cfg = {}
        if server._config and server._config.tts_config:
            kokoro_cfg = server._config.tts_config.provider_config or {}

        server_script = str(Path(KokoroTTSProvider.__module__.replace(".", "/")).resolve().parent / "server.py")
        # Fallback: find relative to provider.py source file
        import spindl.tts.builtin.kokoro.provider as _kprov
        server_script = str(Path(_kprov.__file__).resolve().parent / "server.py")

        models_dir = kokoro_cfg.get("models_dir", "tts/models")
        from spindl.utils.paths import resolve_relative_path
        models_dir = resolve_relative_path(models_dir)

        conda_env = kokoro_cfg.get("conda_env", "pixl")

        # Quote paths with spaces
        script_q = f'"{server_script}"' if " " in server_script else server_script
        models_q = f'"{models_dir}"' if " " in models_dir else models_dir

        cmd = (
            f"conda run -n {conda_env} --no-capture-output "
            f"python {script_q} "
            f"--port {port} "
            f"--models-dir {models_q} "
            f"--device {device}"
        )

        print(f"[GUI] Chat-TTS launch command: {cmd}", flush=True)

        def _launch():
            import asyncio as _asyncio
            try:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                server._chat_tts_process = process
                print(f"[GUI] Chat-TTS process spawned (PID: {process.pid})", flush=True)

                # Wait for server to become available (health check)
                for attempt in range(30):
                    if process.poll() is not None:
                        break
                    if _tcp_check(host, port):
                        break
                    import time
                    time.sleep(2.0)

                alive = process.poll() is None
                reachable = _tcp_check(host, port)
                success = alive and reachable

                async def _emit():
                    if success:
                        await sio.emit("chat_tts_launched", {"success": True})
                    else:
                        error = "Process died during startup" if not alive else f"Server not reachable at {host}:{port}"
                        server._chat_tts_process = None
                        await sio.emit("chat_tts_launched", {"success": False, "error": error})

                loop = server._event_loop
                if loop and loop.is_running():
                    _asyncio.run_coroutine_threadsafe(_emit(), loop)

            except Exception as e:
                print(f"[GUI] Chat-TTS launch exception: {e}", flush=True)
                server._chat_tts_process = None

                async def _emit_err():
                    await sio.emit("chat_tts_launched", {"success": False, "error": str(e)})

                loop = server._event_loop
                if loop and loop.is_running():
                    _asyncio.run_coroutine_threadsafe(_emit_err(), loop)

        thread = threading.Thread(target=_launch, daemon=True)
        thread.start()

    @sio.event
    async def stop_chat_tts(sid: str, data: dict) -> None:
        """Stop the chat-TTS Kokoro server (NANO-130)."""
        server._chat_tts_kill()
        await sio.emit("chat_tts_stopped", {"success": True}, to=sid)

    @sio.event
    async def request_chat_tts_status(sid: str, data: dict) -> None:
        """Report chat-TTS server status (NANO-130)."""
        cfg = server._config.stimuli_config if server._config else None
        host = cfg.twitch_chat_tts_host if cfg else "127.0.0.1"
        port = cfg.twitch_chat_tts_port if cfg else 5560

        process_alive = (
            server._chat_tts_process is not None
            and server._chat_tts_process.poll() is None
        )
        reachable = _tcp_check(host, port) if process_alive else False

        await sio.emit(
            "chat_tts_status",
            {
                "running": process_alive and reachable,
                "process_alive": process_alive,
                "reachable": reachable,
                "host": host,
                "port": port,
            },
            to=sid,
        )


def _tcp_check(host: str, port: int, timeout: float = 2.0) -> bool:
    """Quick TCP connect check to see if something is listening."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except (ConnectionError, socket.error, socket.timeout, OSError):
        return False
