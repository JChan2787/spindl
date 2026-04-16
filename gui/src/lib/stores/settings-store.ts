import { create } from "zustand";
import type {
  ProviderInfo,
  ConfigLoadedEvent,
  LocalLLMConfig,
  LocalVLMConfig,
} from "@/types/events";

// VAD configuration state
interface VADConfig {
  threshold: number;
  min_speech_ms: number;
  min_silence_ms: number;
  speech_pad_ms: number;
}

// Pipeline configuration state (NANO-096: max_context removed — context size lives in provider config)
interface PipelineConfig {
  summarization_threshold: number;
  budget_strategy: string;
}

// Memory configuration state (NANO-043)
interface CurationConfig {
  enabled: boolean;
  api_key: string | null;
  model: string;
  prompt: string | null;
  timeout: number;
}

interface MemoryConfig {
  top_k: number;
  relevance_threshold: number | null;
  dedup_threshold: number | null;
  reflection_interval: number;
  reflection_prompt: string | null;
  reflection_system_message: string | null;
  reflection_delimiter: string;
  enabled: boolean;
  curation: CurationConfig;
}

// Generation parameters (NANO-053, NANO-108)
export interface GenerationParamsConfig {
  temperature: number;
  max_tokens: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  repeat_last_n: number;
  frequency_penalty: number;
  presence_penalty: number;
  force_role_history: "auto" | "splice" | "flatten";
}

// Prompt injection wrapper config (NANO-045d + NANO-052 follow-up)
export interface PromptConfig {
  rag_prefix: string;
  rag_suffix: string;
  codex_prefix: string;
  codex_suffix: string;
  example_dialogue_prefix: string;
  example_dialogue_suffix: string;
}

// NANO-110: Addressing-others context
export interface AddressingContextEntry {
  id: string;
  label: string;
  prompt: string;
}

// Stimuli configuration (NANO-056)
export interface StimuliConfig {
  enabled: boolean;
  patience_enabled: boolean;
  patience_seconds: number;
  patience_prompt: string;
  // Twitch integration (NANO-056b)
  twitch_enabled: boolean;
  twitch_channel: string;
  twitch_app_id: string;
  twitch_app_secret: string;
  twitch_buffer_size: number;
  twitch_max_message_length: number;
  twitch_prompt_template: string;
  twitch_audience_window: number;
  twitch_audience_char_cap: number;
  // Resolved by backend — true when credentials available (config or env vars)
  twitch_has_credentials: boolean;
  // NANO-110: Addressing-others contexts
  addressing_others_contexts: AddressingContextEntry[];
}

// Twitch module status (NANO-056b)
export interface TwitchStatus {
  connected: boolean;
  channel: string;
  buffer_count: number;
  recent_messages: string[];
}

// Stimuli progress (NANO-056)
export interface StimuliProgress {
  elapsed: number;
  total: number;
  progress: number; // 0.0–1.0
  blocked?: boolean; // True when timer is paused (playback or typing)
  blocked_reason?: "playback" | "typing" | null;
}

// Tools runtime config (NANO-065a)
export interface ToolsConfig {
  master_enabled: boolean;
  tools: Record<string, { enabled: boolean; label: string }>;
}

// LLM provider runtime config (NANO-065b)
export interface LLMRuntimeConfig {
  provider: string;
  model: string;
  context_size: number | null;
  available_providers: string[];
  // NANO-114: True when active provider's chat template benefits from
  // role-array history (Gemma-3/Gemma-4 via llama.cpp --jinja).
  supports_role_history: boolean;
}

// VLM provider runtime config (NANO-065c, extended NANO-079)
export interface VLMRuntimeConfig {
  provider: string;
  available_providers: string[];
  healthy: boolean;
  unified_vlm: boolean;
}

// Avatar config (NANO-093, NANO-094)
export interface AvatarRuntimeConfig {
  enabled: boolean;
  emotion_classifier: "classifier" | "off";
  show_emotion_in_chat: boolean;
  emotion_confidence_threshold: number;
  expression_fade_delay: number;
  subtitles_enabled: boolean; // NANO-100: subtitle overlay window
  subtitle_fade_delay: number; // NANO-100: seconds to hold subtitle after TTS
  stream_deck_enabled: boolean; // NANO-110: stream deck overlay window
  avatar_always_on_top: boolean;
  subtitle_always_on_top: boolean;
  avatar_connected: boolean; // NANO-097: avatar renderer is connected
  // NANO-110: Tauri app install status
  tauri_installed: boolean; // true if all binaries exist
  tauri_installing: boolean; // true if install is in progress
  tauri_install_message: string; // live progress message
}

// Provider states
interface ProviderStates {
  llm: ProviderInfo | null;
  tts: ProviderInfo | null;
  stt: ProviderInfo | null;
  vlm: ProviderInfo | null;
  embedding: { base_url: string; enabled: boolean } | null;
}

interface SettingsStoreState {
  // VAD configuration
  vadConfig: VADConfig;
  pendingVadChanges: Partial<VADConfig>;
  isSavingVad: boolean;

  // Pipeline configuration
  pipelineConfig: PipelineConfig;
  pendingPipelineChanges: Partial<PipelineConfig>;
  isSavingPipeline: boolean;

  // Memory configuration (NANO-043)
  memoryConfig: MemoryConfig;
  pendingMemoryChanges: Partial<MemoryConfig>;
  isSavingMemory: boolean;

  // Prompt injection wrappers (NANO-045d)
  promptConfig: PromptConfig;
  pendingPromptChanges: Partial<PromptConfig>;
  isSavingPrompt: boolean;

  // Generation parameters (NANO-053)
  generationConfig: GenerationParamsConfig;
  pendingGenerationChanges: Partial<GenerationParamsConfig>;
  isSavingGeneration: boolean;

  // Stimuli configuration (NANO-056)
  stimuliConfig: StimuliConfig;
  pendingStimuliChanges: Partial<StimuliConfig>;
  isSavingStimuli: boolean;
  stimuliProgress: StimuliProgress | null;
  twitchStatus: TwitchStatus | null;

  // Tools runtime config (NANO-065a)
  toolsConfig: ToolsConfig;
  isSavingTools: boolean;

  // LLM provider runtime config (NANO-065b)
  llmConfig: LLMRuntimeConfig;
  isSavingLLM: boolean;

  // VLM provider runtime config (NANO-065c)
  vlmConfig: VLMRuntimeConfig;
  isSavingVLM: boolean;

  // Avatar config (NANO-093)
  avatarConfig: AvatarRuntimeConfig;

  // Local LLM launch state (NANO-065b Enhancement)
  localLLMConfig: LocalLLMConfig;
  isLaunchingLLM: boolean;
  llmServerRunning: boolean;
  llmLaunchError: string | null;

  // Local VLM launch state (NANO-079)
  localVLMConfig: LocalVLMConfig;
  isLaunchingVLM: boolean;
  vlmServerRunning: boolean;
  vlmLaunchError: string | null;

  // Provider states (read-only display)
  providers: ProviderStates;

  // Raw config (for display)
  rawConfig: string | null;

  // Restart required flag
  restartRequired: boolean;

  // Error state
  error: string | null;

  // Actions - VAD
  setVadConfig: (config: VADConfig) => void;
  updatePendingVad: (changes: Partial<VADConfig>) => void;
  clearPendingVad: () => void;
  setSavingVad: (saving: boolean) => void;

  // Actions - Pipeline
  setPipelineConfig: (config: PipelineConfig) => void;
  updatePendingPipeline: (changes: Partial<PipelineConfig>) => void;
  clearPendingPipeline: () => void;
  setSavingPipeline: (saving: boolean) => void;

  // Actions - Memory
  setMemoryConfig: (config: MemoryConfig) => void;
  updatePendingMemory: (changes: Partial<MemoryConfig>) => void;
  clearPendingMemory: () => void;
  setSavingMemory: (saving: boolean) => void;

  // Actions - Prompt (NANO-045d)
  setPromptConfig: (config: PromptConfig) => void;
  updatePendingPrompt: (changes: Partial<PromptConfig>) => void;
  clearPendingPrompt: () => void;
  setSavingPrompt: (saving: boolean) => void;

  // Actions - Generation (NANO-053)
  setGenerationConfig: (config: GenerationParamsConfig) => void;
  updatePendingGeneration: (changes: Partial<GenerationParamsConfig>) => void;
  clearPendingGeneration: () => void;
  setSavingGeneration: (saving: boolean) => void;

  // Actions - Stimuli (NANO-056)
  setStimuliConfig: (config: StimuliConfig) => void;
  updatePendingStimuli: (changes: Partial<StimuliConfig>) => void;
  clearPendingStimuli: () => void;
  setSavingStimuli: (saving: boolean) => void;
  setStimuliProgress: (progress: StimuliProgress | null) => void;
  setTwitchStatus: (status: TwitchStatus | null) => void;

  // Actions - Tools (NANO-065a)
  setToolsConfig: (config: ToolsConfig) => void;
  setSavingTools: (saving: boolean) => void;

  // Actions - LLM (NANO-065b)
  setLLMConfig: (config: LLMRuntimeConfig) => void;
  setSavingLLM: (saving: boolean) => void;

  // Actions - VLM (NANO-065c)
  setVLMConfig: (config: VLMRuntimeConfig) => void;
  setSavingVLM: (saving: boolean) => void;

  // Actions - Avatar (NANO-093)
  setAvatarConfig: (config: AvatarRuntimeConfig) => void;

  // Actions - Local LLM Launch (NANO-065b Enhancement)
  setLocalLLMConfig: (config: LocalLLMConfig) => void;
  updateLocalLLMConfig: (changes: Partial<LocalLLMConfig>) => void;
  setLaunchingLLM: (launching: boolean) => void;
  setLLMServerRunning: (running: boolean) => void;
  setLLMLaunchError: (error: string | null) => void;

  // Actions - Local VLM Launch (NANO-079)
  setLocalVLMConfig: (config: LocalVLMConfig) => void;
  updateLocalVLMConfig: (changes: Partial<LocalVLMConfig>) => void;
  setLaunchingVLM: (launching: boolean) => void;
  setVLMServerRunning: (running: boolean) => void;
  setVLMLaunchError: (error: string | null) => void;
  setUnifiedVLM: (unified: boolean) => void;

  // Actions - Providers
  setProviders: (providers: ProviderStates) => void;

  // Actions - Config
  setRawConfig: (config: string) => void;
  setRestartRequired: (required: boolean) => void;

  // Actions - From config_loaded event
  loadConfig: (config: ConfigLoadedEvent) => void;

  // Actions - Error
  setError: (error: string | null) => void;
}

// Default values
const DEFAULT_VAD: VADConfig = {
  threshold: 0.5,
  min_speech_ms: 250,
  min_silence_ms: 1000,
  speech_pad_ms: 300,
};

const DEFAULT_PIPELINE: PipelineConfig = {
  summarization_threshold: 0.8,
  budget_strategy: "truncate",
};

const DEFAULT_CURATION: CurationConfig = {
  enabled: false,
  api_key: null,
  model: "anthropic/claude-haiku-4-5",
  prompt: null,
  timeout: 30,
};

const DEFAULT_MEMORY: MemoryConfig = {
  top_k: 5,
  relevance_threshold: 0.25,
  dedup_threshold: 0.30,
  reflection_interval: 20,
  reflection_prompt: null,
  reflection_system_message: null,
  reflection_delimiter: "{qa}",
  enabled: false,
  curation: DEFAULT_CURATION,
};

const DEFAULT_GENERATION: GenerationParamsConfig = {
  temperature: 0.7,
  max_tokens: 256,
  top_p: 0.95,
  top_k: 40,
  min_p: 0.05,
  repeat_penalty: 1.1,
  repeat_last_n: 64,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  force_role_history: "auto",
};

const DEFAULT_AVATAR: AvatarRuntimeConfig = {
  enabled: false,
  emotion_classifier: "off",
  show_emotion_in_chat: true,
  emotion_confidence_threshold: 0.3,
  expression_fade_delay: 1.0,
  subtitles_enabled: false,
  subtitle_fade_delay: 1.5,
  stream_deck_enabled: false,
  avatar_always_on_top: true,
  subtitle_always_on_top: true,
  avatar_connected: false,
  tauri_installed: false,
  tauri_installing: false,
  tauri_install_message: "",
};

const DEFAULT_STIMULI: StimuliConfig = {
  enabled: false,
  patience_enabled: false,
  patience_seconds: 60,
  patience_prompt: "Continue the conversation naturally. You have been idle. Think of something interesting to say or ask.",
  twitch_enabled: false,
  twitch_channel: "",
  twitch_app_id: "",
  twitch_app_secret: "",
  twitch_buffer_size: 10,
  twitch_max_message_length: 300,
  twitch_prompt_template: "Recent Twitch chat messages:\n{messages}\nPick the most interesting message and respond to it naturally.",
  twitch_audience_window: 25,
  twitch_audience_char_cap: 150,
  twitch_has_credentials: false,
  addressing_others_contexts: [{ id: "ctx_0", label: "Others", prompt: "" }],
};

const DEFAULT_PROMPT: PromptConfig = {
  rag_prefix: "The following are relevant memories about the user and past conversations. Use them to inform your response:",
  rag_suffix: "End of memories.",
  codex_prefix: "The following facts are always true in this context:",
  codex_suffix: "",
  example_dialogue_prefix: "The following are example dialogues demonstrating this character's voice, tone, and response style. Use them as style reference only — do not repeat or quote them directly:",
  example_dialogue_suffix: "End of style examples.",
};

const DEFAULT_TOOLS: ToolsConfig = {
  master_enabled: false,
  tools: {},
};

const DEFAULT_LLM: LLMRuntimeConfig = {
  provider: "llama",
  model: "",
  context_size: null,
  available_providers: [],
  supports_role_history: false,
};

const DEFAULT_VLM: VLMRuntimeConfig = {
  provider: "llama",
  available_providers: [],
  healthy: false,
  unified_vlm: false,
};

const DEFAULT_LOCAL_LLM: LocalLLMConfig = {
  host: "127.0.0.1",
  port: 5557,
  gpu_layers: 99,
  context_size: 8192,
};

const DEFAULT_LOCAL_VLM: LocalVLMConfig = {
  model_type: "gemma3",
  host: "127.0.0.1",
  port: 5558,
  gpu_layers: 99,
  context_size: 8192,
};

export const useSettingsStore = create<SettingsStoreState>((set) => ({
  // Initial state - VAD
  vadConfig: DEFAULT_VAD,
  pendingVadChanges: {},
  isSavingVad: false,

  // Initial state - Pipeline
  pipelineConfig: DEFAULT_PIPELINE,
  pendingPipelineChanges: {},
  isSavingPipeline: false,

  // Initial state - Memory
  memoryConfig: DEFAULT_MEMORY,
  pendingMemoryChanges: {},
  isSavingMemory: false,

  // Initial state - Prompt (NANO-045d)
  promptConfig: DEFAULT_PROMPT,
  pendingPromptChanges: {},
  isSavingPrompt: false,

  // Initial state - Generation (NANO-053)
  generationConfig: DEFAULT_GENERATION,
  pendingGenerationChanges: {},
  isSavingGeneration: false,

  // Initial state - Stimuli (NANO-056)
  stimuliConfig: DEFAULT_STIMULI,
  pendingStimuliChanges: {},
  isSavingStimuli: false,
  stimuliProgress: null,
  twitchStatus: null,

  // Initial state - Avatar (NANO-093)
  avatarConfig: DEFAULT_AVATAR,

  // Initial state - Tools (NANO-065a)
  toolsConfig: DEFAULT_TOOLS,
  isSavingTools: false,

  // Initial state - LLM (NANO-065b)
  llmConfig: DEFAULT_LLM,
  isSavingLLM: false,

  // Initial state - VLM (NANO-065c)
  vlmConfig: DEFAULT_VLM,
  isSavingVLM: false,

  // Initial state - Local LLM Launch (NANO-065b Enhancement)
  localLLMConfig: DEFAULT_LOCAL_LLM,
  isLaunchingLLM: false,
  llmServerRunning: false,
  llmLaunchError: null,

  // Initial state - Local VLM Launch (NANO-079)
  localVLMConfig: DEFAULT_LOCAL_VLM,
  isLaunchingVLM: false,
  vlmServerRunning: false,
  vlmLaunchError: null,

  // Initial state - Providers
  providers: {
    llm: null,
    tts: null,
    stt: null,
    vlm: null,
    embedding: null,
  },

  // Initial state - Config
  rawConfig: null,
  restartRequired: false,
  error: null,

  // VAD actions
  setVadConfig: (vadConfig) => set({ vadConfig, pendingVadChanges: {}, isSavingVad: false }),

  updatePendingVad: (changes) => set((state) => ({
    pendingVadChanges: { ...state.pendingVadChanges, ...changes },
  })),

  clearPendingVad: () => set({ pendingVadChanges: {} }),

  setSavingVad: (isSavingVad) => set({ isSavingVad }),

  // Pipeline actions
  setPipelineConfig: (pipelineConfig) => set({ pipelineConfig, pendingPipelineChanges: {}, isSavingPipeline: false }),

  updatePendingPipeline: (changes) => set((state) => ({
    pendingPipelineChanges: { ...state.pendingPipelineChanges, ...changes },
  })),

  clearPendingPipeline: () => set({ pendingPipelineChanges: {} }),

  setSavingPipeline: (isSavingPipeline) => set({ isSavingPipeline }),

  // Memory actions
  setMemoryConfig: (memoryConfig) => set({ memoryConfig, pendingMemoryChanges: {}, isSavingMemory: false }),

  updatePendingMemory: (changes) => set((state) => ({
    pendingMemoryChanges: { ...state.pendingMemoryChanges, ...changes },
  })),

  clearPendingMemory: () => set({ pendingMemoryChanges: {} }),

  setSavingMemory: (isSavingMemory) => set({ isSavingMemory }),

  // Prompt actions (NANO-045d)
  setPromptConfig: (promptConfig) => set({ promptConfig, pendingPromptChanges: {}, isSavingPrompt: false }),

  updatePendingPrompt: (changes) => set((state) => ({
    pendingPromptChanges: { ...state.pendingPromptChanges, ...changes },
  })),

  clearPendingPrompt: () => set({ pendingPromptChanges: {} }),

  setSavingPrompt: (isSavingPrompt) => set({ isSavingPrompt }),

  // Generation actions (NANO-053)
  setGenerationConfig: (generationConfig) => set({ generationConfig, pendingGenerationChanges: {}, isSavingGeneration: false }),

  updatePendingGeneration: (changes) => set((state) => ({
    pendingGenerationChanges: { ...state.pendingGenerationChanges, ...changes },
  })),

  clearPendingGeneration: () => set({ pendingGenerationChanges: {} }),

  setSavingGeneration: (isSavingGeneration) => set({ isSavingGeneration }),

  // Stimuli actions (NANO-056)
  setStimuliConfig: (stimuliConfig) => set({ stimuliConfig: { ...DEFAULT_STIMULI, ...stimuliConfig }, pendingStimuliChanges: {}, isSavingStimuli: false }),

  updatePendingStimuli: (changes) => set((state) => ({
    pendingStimuliChanges: { ...state.pendingStimuliChanges, ...changes },
  })),

  clearPendingStimuli: () => set({ pendingStimuliChanges: {} }),

  setSavingStimuli: (isSavingStimuli) => set({ isSavingStimuli }),

  setStimuliProgress: (stimuliProgress) => set({ stimuliProgress }),
  setTwitchStatus: (twitchStatus) => set({ twitchStatus }),

  // Avatar actions (NANO-093)
  setAvatarConfig: (avatarConfig) => set({ avatarConfig }),

  // Tools actions (NANO-065a)
  setToolsConfig: (toolsConfig) => set({ toolsConfig, isSavingTools: false }),
  setSavingTools: (isSavingTools) => set({ isSavingTools }),

  // LLM actions (NANO-065b)
  setLLMConfig: (llmConfig) => set({ llmConfig, isSavingLLM: false }),
  setSavingLLM: (isSavingLLM) => set({ isSavingLLM }),

  // VLM actions (NANO-065c)
  setVLMConfig: (vlmConfig) => set({ vlmConfig, isSavingVLM: false }),
  setSavingVLM: (isSavingVLM) => set({ isSavingVLM }),

  // Local LLM Launch actions (NANO-065b Enhancement)
  setLocalLLMConfig: (localLLMConfig) => set({ localLLMConfig }),
  updateLocalLLMConfig: (changes) => set((state) => ({
    localLLMConfig: { ...state.localLLMConfig, ...changes },
  })),
  setLaunchingLLM: (isLaunchingLLM) => set({ isLaunchingLLM }),
  setLLMServerRunning: (llmServerRunning) => set({ llmServerRunning }),
  setLLMLaunchError: (llmLaunchError) => set({ llmLaunchError }),

  // Local VLM Launch actions (NANO-079)
  setLocalVLMConfig: (localVLMConfig) => set({ localVLMConfig }),
  updateLocalVLMConfig: (changes) => set((state) => ({
    localVLMConfig: { ...state.localVLMConfig, ...changes },
  })),
  setLaunchingVLM: (isLaunchingVLM) => set({ isLaunchingVLM }),
  setVLMServerRunning: (vlmServerRunning) => set({ vlmServerRunning }),
  setVLMLaunchError: (vlmLaunchError) => set({ vlmLaunchError }),
  setUnifiedVLM: (unified) => set((state) => ({
    vlmConfig: { ...state.vlmConfig, unified_vlm: unified },
  })),

  // Provider actions
  setProviders: (providers) => set({ providers }),

  // Config actions
  setRawConfig: (rawConfig) => set({ rawConfig }),

  setRestartRequired: (restartRequired) => set({ restartRequired }),

  // Load from config_loaded event
  loadConfig: (config) => set({
    vadConfig: { ...DEFAULT_VAD, ...config.settings.vad },
    pipelineConfig: config.settings.pipeline,
    memoryConfig: {
      ...DEFAULT_MEMORY,
      ...(config.settings?.memory ?? {}),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      curation: { ...DEFAULT_CURATION, ...((config.settings?.memory as any)?.curation ?? {}) },
    },
    promptConfig: config.settings?.prompt ?? DEFAULT_PROMPT,
    generationConfig: { ...DEFAULT_GENERATION, ...(config.settings?.generation ?? {}) },
    stimuliConfig: { ...DEFAULT_STIMULI, ...(config.settings?.stimuli ?? {}) },
    toolsConfig: config.settings?.tools
      ? {
          master_enabled: config.settings.tools.master_enabled,
          tools: Object.fromEntries(
            Object.entries(config.settings.tools.tools).map(([name, cfg]) => [
              name,
              { enabled: cfg.enabled, label: name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) },
            ])
          ),
        }
      : DEFAULT_TOOLS,
    llmConfig: config.settings?.llm
      ? {
          provider: config.settings.llm.provider,
          model: config.settings.llm.model,
          context_size: config.settings.llm.context_size,
          available_providers: config.settings.llm.available_providers,
          supports_role_history:
            config.settings.llm.supports_role_history ?? false,
        }
      : DEFAULT_LLM,
    // VLM state hydrated via request_vlm_config socket call (NANO-065c)
    vlmConfig: DEFAULT_VLM,
    // Avatar config (NANO-093) — hydrated from config_loaded or socket event
    // Spread DEFAULT_AVATAR first to ensure avatar_connected (runtime-only) is always present
    avatarConfig: { ...DEFAULT_AVATAR, ...(config.settings?.avatar ?? {}) },
    providers: {
      llm: config.providers.llm,
      tts: config.providers.tts,
      stt: config.providers.stt,
      vlm: config.providers.vlm ?? null,
      embedding: config.providers.embedding ?? null,
    },
    pendingVadChanges: {},
    pendingPipelineChanges: {},
    pendingMemoryChanges: {},
    pendingPromptChanges: {},
    pendingGenerationChanges: {},
    pendingStimuliChanges: {},
    error: null,
  }),

  // Error actions
  setError: (error) => set({ error }),
}));

// Selector for effective VAD config (current + pending changes)
export const selectEffectiveVadConfig = (state: SettingsStoreState): VADConfig => ({
  ...state.vadConfig,
  ...state.pendingVadChanges,
});

// Selector for effective pipeline config (current + pending changes)
export const selectEffectivePipelineConfig = (state: SettingsStoreState): PipelineConfig => ({
  ...state.pipelineConfig,
  ...state.pendingPipelineChanges,
});

// Selector for effective memory config (current + pending changes)
export const selectEffectiveMemoryConfig = (state: SettingsStoreState): MemoryConfig => ({
  ...state.memoryConfig,
  ...state.pendingMemoryChanges,
});

// Selector for effective prompt config (current + pending changes) (NANO-045d)
export const selectEffectivePromptConfig = (state: SettingsStoreState): PromptConfig => ({
  ...state.promptConfig,
  ...state.pendingPromptChanges,
});

// Selector for effective generation config (current + pending changes) (NANO-053)
export const selectEffectiveGenerationConfig = (state: SettingsStoreState): GenerationParamsConfig => ({
  ...state.generationConfig,
  ...state.pendingGenerationChanges,
});

// Selector for effective stimuli config (current + pending changes) (NANO-056)
export const selectEffectiveStimuliConfig = (state: SettingsStoreState): StimuliConfig => ({
  ...state.stimuliConfig,
  ...state.pendingStimuliChanges,
});

// Selector to check if there are unsaved changes
export const selectHasUnsavedChanges = (state: SettingsStoreState): boolean =>
  Object.keys(state.pendingVadChanges).length > 0 ||
  Object.keys(state.pendingPipelineChanges).length > 0 ||
  Object.keys(state.pendingMemoryChanges).length > 0 ||
  Object.keys(state.pendingPromptChanges).length > 0 ||
  Object.keys(state.pendingGenerationChanges).length > 0 ||
  Object.keys(state.pendingStimuliChanges).length > 0;

// ============================================
// Base Animations API (NANO-099)
// ============================================

export interface BaseAnimationsConfig {
  idle: string | null;
  happy: string | null;
  sad: string | null;
  angry: string | null;
  curious: string | null;
  global_animations_dir: string;
}

export async function fetchBaseAnimations(): Promise<BaseAnimationsConfig> {
  const res = await fetch("/api/avatar/base-animations");
  if (!res.ok) throw new Error(`Failed to fetch base animations: ${res.statusText}`);
  return res.json();
}

export async function uploadBaseAnimation(
  slot: "idle" | "happy" | "sad" | "angry" | "curious",
  file: File
): Promise<{ slot: string; clip_name: string; filename: string; success: boolean }> {
  const formData = new FormData();
  formData.append("slot", slot);
  formData.append("file", file);

  const res = await fetch("/api/avatar/base-animations", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function clearBaseAnimation(
  slot: "idle" | "happy" | "sad" | "angry" | "curious"
): Promise<{ slot: string; success: boolean }> {
  const res = await fetch("/api/avatar/base-animations", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slot }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Clear failed: ${res.statusText}`);
  }
  return res.json();
}
