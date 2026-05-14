/**
 * NANO-089 Phase 2: Zod schemas for runtime socket events.
 *
 * Defense-in-depth validation for the 4 config-carrying socket events
 * that directly update persistent state. Backend Pydantic is the primary
 * gate; these schemas catch shape mismatches before store updates.
 */
import { z } from "zod";

// ============================================
// Runtime Config Event Schemas
// ============================================

/** Matches backend LLMConfigResponse (response_models.py).
 *  provider is optional to allow error-only payloads (e.g. {"error": "Services not launched"}).
 *  Socket handlers already gate on !validated.error before updating store. */
export const LLMConfigUpdatedSchema = z.object({
  provider: z.string().min(1).optional(),
  model: z.string().nullable().optional().default(null),
  context_size: z.number().nullable().optional().default(null),
  available_providers: z.array(z.string()).default([]),
  // NANO-114: True when active provider's chat template benefits from
  // role-array history (Gemma-3/Gemma-4 via llama.cpp --jinja).
  supports_role_history: z.boolean().optional().default(false),
  // Session 645: removed "auto"; legacy values are coerced to "flatten".
  force_role_history: z
    .enum(["splice", "flatten"])
    .or(z.literal("auto").transform(() => "flatten" as const))
    .optional()
    .default("flatten"),
  persisted: z.boolean().optional(),
  success: z.boolean().optional(),
  error: z.string().optional(),
});

/** Matches backend VLMConfigResponse (response_models.py).
 *  provider is optional to allow error-only payloads (e.g. {"error": "Services not launched"}).
 *  Socket handlers already gate on !validated.error before updating store. */
export const VLMConfigUpdatedSchema = z.object({
  provider: z.string().min(1).optional(),
  available_providers: z.array(z.string()).default([]),
  healthy: z.boolean().default(false),
  unified_vlm: z.boolean().optional(),
  persisted: z.boolean().optional(),
  success: z.boolean().optional(),
  error: z.string().optional(),
  cloud_config: z
    .object({
      api_key: z.string(),
      model: z.string(),
      base_url: z.string(),
    })
    .optional(),
});

/** Matches LocalLLMConfigEvent (events.ts) */
export const LocalLLMConfigEventSchema = z.object({
  config: z.object({
    executable_path: z.string().optional(),
    model_path: z.string().optional(),
    mmproj_path: z.string().optional(),
    host: z.string().optional(),
    port: z.number().optional(),
    gpu_layers: z.number().optional(),
    context_size: z.number().optional(),
    device: z.string().optional(),
    tensor_split: z.string().optional(),
    extra_args: z.string().optional(),
    timeout: z.number().optional(),
    temperature: z.number().optional(),
    max_tokens: z.number().optional(),
    top_p: z.number().optional(),
    top_k: z.number().optional(),
    min_p: z.number().optional(),
    repeat_penalty: z.number().optional(),
    repeat_last_n: z.number().optional(),
    frequency_penalty: z.number().optional(),
    presence_penalty: z.number().optional(),
    reasoning_format: z.string().optional(),
    reasoning_budget: z.number().optional(),
  }),
  server_running: z.boolean(),
});

/** Matches LocalVLMConfigEvent (events.ts) */
export const LocalVLMConfigEventSchema = z.object({
  config: z.object({
    model_type: z.string().optional(),
    executable_path: z.string().optional(),
    model_path: z.string().optional(),
    mmproj_path: z.string().optional(),
    host: z.string().optional(),
    port: z.number().optional(),
    gpu_layers: z.number().optional(),
    context_size: z.number().optional(),
    device: z.string().optional(),
    tensor_split: z.string().optional(),
    extra_args: z.string().optional(),
    timeout: z.number().optional(),
    max_tokens: z.number().optional(),
  }),
  server_running: z.boolean(),
});

/** Matches backend ToolsConfigResponse (response_models.py) */
export const ToolsConfigUpdatedSchema = z.object({
  master_enabled: z.boolean(),
  tools: z
    .record(
      z.string(),
      z.object({
        enabled: z.boolean(),
        label: z.string(),
      }),
    )
    .default({}),
  persisted: z.boolean().optional(),
  error: z.string().optional(),
});

/** Matches backend AvatarConfig (NANO-093, NANO-094) */
export const AvatarConfigSchema = z.object({
  enabled: z.boolean().default(false),
  emotion_classifier: z.enum(["classifier", "off"]).default("off"),
  show_emotion_in_chat: z.boolean().default(true),
  emotion_confidence_threshold: z.number().min(0).max(1).default(0.3),
  expression_fade_delay: z.number().min(0).max(10).default(1.0),
  curious_hold_duration: z.number().min(0).max(30).default(8.0),
  angry_hold_duration: z.number().min(0).max(30).default(8.0),
  subtitles_enabled: z.boolean().default(false),
  subtitle_fade_delay: z.number().min(0).max(10).default(1.5),
  stream_deck_enabled: z.boolean().default(false), // NANO-110
  avatar_always_on_top: z.boolean().default(true),
  subtitle_always_on_top: z.boolean().default(true),
});

// ============================================
// VAD Config Schema
// ============================================

/** Matches backend vad_config_updated emission */
export const VADConfigUpdatedSchema = z.object({
  threshold: z.number().min(0).max(1),
  min_speech_ms: z.number().int().min(1),
  min_silence_ms: z.number().int().min(1),
  speech_pad_ms: z.number().int().min(0),
  persisted: z.boolean().optional(),
});

// ============================================
// Avatar Schemas (NANO-098)
// ============================================

/** Per-character expression composites (e.g. curious built from aa + oh + happy) */
export const AvatarExpressionsSchema = z
  .record(z.string(), z.record(z.string(), z.number()))
  .optional();

/** Per-character emotion-to-animation threshold map */
export const AvatarAnimationsSchema = z
  .object({
    default: z.string().optional(),
    emotions: z
      .record(
        z.string(),
        z.object({
          threshold: z.number().min(0).max(1),
          clip: z.string().min(1),
        }),
      )
      .optional(),
  })
  .optional();

// ============================================
// Stimuli Schemas (NANO-056b)
// ============================================

// NANO-110: Addressing-others context entry schema
const AddressingContextSchema = z.object({
  id: z.string(),
  label: z.string(),
  prompt: z.string(),
});

/** Matches backend stimuli_config_updated emission */
export const StimuliConfigUpdatedSchema = z.object({
  enabled: z.boolean(),
  patience_enabled: z.boolean(),
  patience_seconds: z.number().min(1),
  patience_prompts: z.array(z.string()).min(1),
  twitch_enabled: z.boolean(),
  twitch_channel: z.string(),
  twitch_app_id: z.string(),
  twitch_app_secret: z.string(),
  twitch_buffer_size: z.number().int().min(1).max(50),
  twitch_max_message_length: z.number().int().min(50).max(1000),
  twitch_prompt_template: z.string(),
  twitch_audience_window: z.number().int().min(25).max(300).default(25),
  twitch_audience_char_cap: z.number().int().min(50).max(500).default(150),
  // NANO-130: Selection pass + staleness filter
  twitch_max_message_age_seconds: z.number().min(1).max(120).default(15),
  twitch_selection_mode: z.string().default("llm"),
  twitch_selection_pass_model: z.string().default(""),
  twitch_selection_pass_api_key: z.string().default(""),
  // NANO-130 Phase 2: Chat-TTS
  twitch_chat_tts_enabled: z.boolean().default(false),
  twitch_chat_tts_host: z.string().default("127.0.0.1"),
  twitch_chat_tts_port: z.number().int().min(1).max(65535).default(5560),
  twitch_chat_tts_device: z.string().default("cpu"),
  twitch_chat_tts_voice: z.string().default("af_sarah"),
  twitch_chat_tts_speed: z.number().min(0.5).max(2.0).default(1.1),
  twitch_chat_tts_format: z.string().default("{username} says: {message}"),
  twitch_chat_tts_max_length: z.number().int().min(20).max(500).default(100),
  twitch_has_credentials: z.boolean(),
  // NANO-116: Game-state bridge integration
  game_state_enabled: z.boolean(),
  game_state_host: z.string(),
  game_state_port: z.number().int().min(1).max(65535),
  game_state_buffer_size: z.number().int().min(1).max(100),
  game_state_prompt_template: z.string(),
  // NANO-116 Phase B.2: Dialogue pipeline
  game_state_dialogue_enabled: z.boolean(),
  game_state_dialogue_buffer_size: z.number().int().min(1).max(200),
  game_state_dialogue_prompt_templates: z.array(z.string()).min(1),
  game_state_dialogue_token_budget: z.number().int().min(200).max(4000),
  game_state_dialogue_summary_max_tokens: z.number().int().min(64).max(2048),
  game_state_dialogue_min_lines: z.number().int().min(1).max(50),
  game_state_dialogue_drain_delay: z.number().min(0).max(30),
  game_state_dialogue_summarizer_model: z.string(),
  game_state_dialogue_summarizer_api_key: z.string(),
  game_state_dialogue_summarizer_persona: z.string(),
  // NANO-122: Gameplay stimulus
  game_state_gameplay_enabled: z.boolean(),
  game_state_gameplay_base_probability: z.number().min(0.05).max(1.0),
  game_state_gameplay_escalation_step: z.number().min(0.05).max(0.5),
  game_state_gameplay_probability_ceiling: z.number().min(0.1).max(1.0),
  game_state_gameplay_dirty_hp_threshold: z.number().min(0.01).max(0.5),
  game_state_gameplay_event_batch_window: z.number().min(0.5).max(10.0),
  // NANO-124: Self-barge-in
  game_state_barge_in_enabled: z.boolean(),
  game_state_barge_in_escalation: z.array(z.number().min(0).max(1)),
  game_state_barge_in_fatigue: z.array(z.number().min(0).max(1)),
  game_state_barge_in_prompt_templates: z.array(z.string()).min(1),
  // NANO-110: Addressing-others contexts
  addressing_others_contexts: z.array(AddressingContextSchema).default([{ id: "ctx_0", label: "Others", prompt: "" }]),
  // NANO-121: Model cycling
  model_rotation_enabled: z.boolean(),
  model_rotation_models: z.array(z.string()),
  model_rotation_api_key: z.string(),
  // NANO-117: Weighted arbitration
  arbitration_decay_multiplier: z.number().min(0.1).max(1.0),
  arbitration_recovery_per_cycle: z.number().min(0.05).max(0.5),
  arbitration_weight_overrides: z.record(z.string(), z.number().min(0.1).max(10.0)).default({}),
  persisted: z.boolean().optional(),
});

/** Matches backend test_twitch_credentials response */
export const TwitchCredentialsResultSchema = z.object({
  success: z.boolean(),
  error: z.string().nullable(),
});

// ============================================
// Memory Config Schemas (NANO-043, NANO-102, NANO-103, NANO-104)
// ============================================

/** Matches backend MemoryConfig (config.py) — base memory fields */
export const MemoryConfigUpdatedSchema = z.object({
  top_k: z.number().int().min(1).max(50).default(5),
  relevance_threshold: z.number().min(0).max(1).nullable().default(0.25),
  dedup_threshold: z.number().min(0).max(2).nullable().default(0.3),
  distance_metric: z.enum(["l2", "cosine"]).default("l2"),
  cross_activation: z.boolean().default(false),
  reflection_interval: z.number().int().min(1).max(1000).default(20),
  reflection_prompt: z.string().nullable().default(null),
  reflection_system_message: z.string().nullable().default(null),
  reflection_delimiter: z.string().min(1).default("{qa}"),
  enabled: z.boolean().default(false),
  persisted: z.boolean().optional(),
});

/** Matches backend CurationConfig (config.py) */
export const CurationConfigSchema = z.object({
  enabled: z.boolean().default(false),
  api_key: z.string().nullable().optional().default(null),
  model: z.string().min(1).default("anthropic/claude-haiku-4-5"),
  prompt: z.string().nullable().optional().default(null),
  timeout: z.number().min(1).max(120).default(30),
  persisted: z.boolean().optional(),
});

// ============================================
// Safe Parse Helper
// ============================================

/**
 * Validates data against a Zod schema without throwing.
 * On failure: logs the error and returns null.
 * Socket handlers should skip store updates on null.
 */
export function safeParse<T>(
  schema: z.ZodType<T>,
  data: unknown,
  eventName: string,
): T | null {
  const result = schema.safeParse(data);
  if (!result.success) {
    console.error(
      `[NANO-089] ${eventName} validation failed:`,
      result.error.issues,
    );
    return null;
  }
  return result.data;
}
