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
  patience_prompt: z.string(),
  twitch_enabled: z.boolean(),
  twitch_channel: z.string(),
  twitch_app_id: z.string(),
  twitch_app_secret: z.string(),
  twitch_buffer_size: z.number().int().min(1).max(50),
  twitch_max_message_length: z.number().int().min(50).max(1000),
  twitch_prompt_template: z.string(),
  twitch_audience_window: z.number().int().min(25).max(300).default(25),
  twitch_audience_char_cap: z.number().int().min(50).max(500).default(150),
  twitch_has_credentials: z.boolean(),
  // NANO-110: Addressing-others contexts
  addressing_others_contexts: z.array(AddressingContextSchema).default([{ id: "ctx_0", label: "Others", prompt: "" }]),
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
