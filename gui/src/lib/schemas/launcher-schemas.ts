/**
 * NANO-089 Phase 2: Zod schemas for the Launcher write-config route.
 *
 * These schemas validate the LauncherConfig POST body before YAML write
 * and fill defaults on GET hydration. Single source of truth — replaces
 * the inline TypeScript interfaces that were in write-config/route.ts.
 */
import { z } from "zod";

// ============================================
// Reusable Atoms
// ============================================

const portSchema = z.number().int().min(1).max(65535);
const timeoutSchema = z.number().min(1);
const temperatureSchema = z.number().min(0).max(2);
const contextSizeSchema = z.number().int().min(1);
const maxTokensSchema = z.number().int().min(1);
const repeatPenaltySchema = z.number().min(0).max(2);
const repeatLastNSchema = z.number().int().min(0).max(2048);
const frequencyPenaltySchema = z.number().min(-2).max(2);
const presencePenaltySchema = z.number().min(-2).max(2);

// ============================================
// Sub-Config Schemas
// ============================================

export const LLMLocalConfigSchema = z.object({
  executablePath: z.string(),
  modelPath: z.string(),
  mmprojPath: z.string().optional().default(""),
  host: z.string().default("127.0.0.1"),
  port: portSchema.default(5557),
  contextSize: contextSizeSchema.default(8192),
  gpuLayers: z.number().int().default(99),
  device: z.string().default(""),
  tensorSplit: z.string().default(""),
  extraArgs: z.string().default(""),
  timeout: timeoutSchema.default(30),
  temperature: temperatureSchema.default(0.7),
  maxTokens: maxTokensSchema.default(256),
  topP: z.number().min(0).max(1).default(0.95),
  repeatPenalty: repeatPenaltySchema.default(1.1),
  repeatLastN: repeatLastNSchema.default(64),
  frequencyPenalty: frequencyPenaltySchema.default(0.0),
  presencePenalty: presencePenaltySchema.default(0.0),
  reasoningFormat: z.string().optional().default(""),
  reasoningBudget: z.number().optional().default(-1),
});

export const LLMCloudConfigSchema = z.object({
  provider: z.string().min(1),
  apiUrl: z.string(),
  apiKey: z.string(),
  model: z.string(),
  contextSize: contextSizeSchema.default(32768),
  timeout: timeoutSchema.default(60),
  temperature: temperatureSchema.default(0.7),
  maxTokens: maxTokensSchema.default(256),
  frequencyPenalty: frequencyPenaltySchema.default(0.0),
  presencePenalty: presencePenaltySchema.default(0.0),
});

export const VLMLocalConfigSchema = z.object({
  modelType: z.string().min(1),
  executablePath: z.string(),
  modelPath: z.string(),
  mmprojPath: z.string().default(""),
  host: z.string().default("127.0.0.1"),
  port: portSchema.default(5558),
  contextSize: contextSizeSchema.default(8192),
  gpuLayers: z.number().int().default(99),
  device: z.string().default(""),
  tensorSplit: z.string().default(""),
  extraArgs: z.string().default(""),
  timeout: timeoutSchema.default(30),
  maxTokens: maxTokensSchema.default(300),
});

export const VLMCloudConfigSchema = z.object({
  apiKey: z.string(),
  baseUrl: z.string(),
  model: z.string(),
  contextSize: contextSizeSchema.default(8192),
  timeout: timeoutSchema.default(30),
  maxTokens: maxTokensSchema.default(300),
});

export const STTParakeetConfigSchema = z.object({
  host: z.string().default("127.0.0.1"),
  port: portSchema.default(5555),
  timeout: timeoutSchema.default(30),
  platform: z.enum(["native", "wsl"]).default("native"),
  wslDistro: z.string().default("Ubuntu"),
  envType: z.enum(["conda", "venv", "system", "other"]).default("conda"),
  envNameOrPath: z.string().default(""),
  customActivation: z.string().default(""),
  serverScriptPath: z.string().default(""),
});

export const STTWhisperConfigSchema = z.object({
  host: z.string().default("127.0.0.1"),
  port: portSchema.default(8080),
  timeout: timeoutSchema.default(30),
  binaryPath: z.string().default("whisper-server"),
  modelPath: z.string().default(""),
  language: z.string().default("en"),
  threads: z.number().int().min(1).default(4),
  noGpu: z.boolean().default(false),
});

export const TTSLocalConfigSchema = z.object({
  provider: z.string().min(1),
  host: z.string().default("127.0.0.1"),
  port: portSchema.default(5556),
  voice: z.string().default(""),
  language: z.string().default(""),
  modelsDirectory: z.string().default("./tts/models"),
  device: z.string().optional().default("cuda"),
  timeout: timeoutSchema.default(30),
  envType: z.enum(["conda", "venv", "system", "other"]).default("conda"),
  envNameOrPath: z.string().default(""),
  customActivation: z.string().default(""),
});

export const EmbeddingConfigSchema = z.object({
  enabled: z.boolean().default(false),
  executablePath: z.string().default(""),
  modelPath: z.string().default(""),
  host: z.string().default("127.0.0.1"),
  port: portSchema.default(5559),
  gpuLayers: z.number().int().default(99),
  contextSize: contextSizeSchema.default(2048),
  timeout: timeoutSchema.default(60),
  relevanceThreshold: z.number().min(0).max(1).default(0.25),
  topK: z.number().int().min(1).default(5),
});

// ============================================
// Top-Level LauncherConfig Schema
// ============================================

export const LauncherConfigSchema = z.object({
  llmProviderType: z.enum(["local", "cloud"]),
  llmLocal: LLMLocalConfigSchema,
  llmCloud: LLMCloudConfigSchema,
  vlmEnabled: z.boolean(),
  useLLMForVision: z.boolean(),
  vlmProviderType: z.enum(["local", "cloud"]),
  vlmLocal: VLMLocalConfigSchema,
  vlmCloud: VLMCloudConfigSchema,
  sttProvider: z.enum(["parakeet", "whisper"]),
  sttParakeet: STTParakeetConfigSchema,
  sttWhisper: STTWhisperConfigSchema,
  ttsProviderType: z.enum(["local", "cloud"]),
  ttsLocal: TTSLocalConfigSchema,
  embedding: EmbeddingConfigSchema.optional(),
});

// ============================================
// Inferred Types (replaces inline interfaces)
// ============================================

export type LauncherConfig = z.infer<typeof LauncherConfigSchema>;
export type LLMLocalConfig = z.infer<typeof LLMLocalConfigSchema>;
export type LLMCloudConfig = z.infer<typeof LLMCloudConfigSchema>;
export type VLMLocalConfig = z.infer<typeof VLMLocalConfigSchema>;
export type VLMCloudConfig = z.infer<typeof VLMCloudConfigSchema>;
export type STTParakeetConfig = z.infer<typeof STTParakeetConfigSchema>;
export type STTWhisperConfig = z.infer<typeof STTWhisperConfigSchema>;
export type TTSLocalConfig = z.infer<typeof TTSLocalConfigSchema>;
export type EmbeddingConfig = z.infer<typeof EmbeddingConfigSchema>;
