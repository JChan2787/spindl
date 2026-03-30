import { create } from "zustand";
import type { CloudProvider } from "@/lib/constants/cloud-providers";

// ============================================
// Type Definitions
// ============================================

export type LLMProviderType = "local" | "cloud";
export type { CloudProvider };
export type VLMModelType = "gemma3" | "qwen2_vl" | "llava" | "minicpm_v";
export type EnvironmentType = "conda" | "venv" | "system" | "other";
export type STTPlatform = "native" | "wsl";
export type TTSProvider = "kokoro" | "custom";
export type STTProviderType = "parakeet" | "whisper";

// LLM Local Configuration
interface LLMLocalConfig {
  executablePath: string;
  modelPath: string;
  mmprojPath: string;
  host: string;
  port: number;
  contextSize: number;
  gpuLayers: number;
  device: string;
  tensorSplit: string;
  extraArgs: string;
  timeout: number;
  temperature: number;
  maxTokens: number;
  topP: number;
  reasoningFormat: string;
  reasoningBudget: number;
}

// LLM Cloud Configuration
interface LLMCloudConfig {
  provider: CloudProvider;
  apiUrl: string;
  apiKey: string;
  model: string;
  contextSize: number;
  timeout: number;
  temperature: number;
  maxTokens: number;
}

// VLM Local Configuration
interface VLMLocalConfig {
  modelType: VLMModelType;
  executablePath: string;
  modelPath: string;
  mmprojPath: string;
  host: string;
  port: number;
  contextSize: number;
  gpuLayers: number;
  device: string;
  tensorSplit: string;
  extraArgs: string;
  timeout: number;
  maxTokens: number;
}

// VLM Cloud Configuration
interface VLMCloudConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
  contextSize: number;
  timeout: number;
  maxTokens: number;
}

// STT Parakeet Configuration
interface STTParakeetConfig {
  host: string;
  port: number;
  timeout: number;
  platform: STTPlatform;
  wslDistro: string;
  envType: EnvironmentType;
  envNameOrPath: string;
  customActivation: string;
  serverScriptPath: string;
}

// STT Whisper Configuration
interface STTWhisperConfig {
  host: string;
  port: number;
  timeout: number;
  binaryPath: string;
  modelPath: string;
  language: string;
  threads: number;
  noGpu: boolean;
}

// Legacy flat STT shape (used by stt-section component via activeSTT selector)
export interface STTConfig {
  provider: STTProviderType;
  host: string;
  port: number;
  timeout: number;
  platform: STTPlatform;
  wslDistro: string;
  envType: EnvironmentType;
  envNameOrPath: string;
  customActivation: string;
  serverScriptPath: string;
  binaryPath: string;
  modelPath: string;
  language: string;
  threads: number;
  noGpu: boolean;
}

// Embedding Configuration (NANO-043 Phase 5)
export interface EmbeddingConfig {
  enabled: boolean;
  executablePath: string;
  modelPath: string;
  host: string;
  port: number;
  gpuLayers: number;
  contextSize: number;
  timeout: number;
  relevanceThreshold: number;
  topK: number;
}

// TTS Local Configuration
interface TTSLocalConfig {
  provider: TTSProvider;
  host: string;
  port: number;
  voice: string;
  language: string;
  modelsDirectory: string;
  device: string;
  timeout: number;
  envType: EnvironmentType;
  envNameOrPath: string;
  customActivation: string;
}

// NANO-027 Phase 3: Launch Progress State
export type LaunchStatus = "idle" | "starting" | "loading_config" | "config_loaded" | "launching" | "complete" | "error";

// NANO-027 Phase 4: Orchestrator State
export type OrchestratorStatus = "pending" | "initializing" | "ready" | "error";

interface LaunchProgress {
  status: LaunchStatus;
  currentService: string | null;
  message: string;
  launchedServices: string[];
  error: string | null;
  // NANO-027 Phase 4: Orchestrator tracking
  orchestratorStatus: OrchestratorStatus;
  orchestratorPersona: string | null;
  orchestratorError: string | null;
}

// ============================================
// Store State
// ============================================

interface LauncherStoreState {
  // LLM Configuration
  llmProviderType: LLMProviderType;
  llmLocal: LLMLocalConfig;
  llmCloud: LLMCloudConfig;

  // Vision Configuration
  vlmEnabled: boolean;
  useLLMForVision: boolean;
  vlmProviderType: LLMProviderType;
  vlmLocal: VLMLocalConfig;
  vlmCloud: VLMCloudConfig;

  // STT Configuration
  sttProvider: STTProviderType;
  sttParakeet: STTParakeetConfig;
  sttWhisper: STTWhisperConfig;

  // TTS Configuration
  ttsProviderType: LLMProviderType;
  ttsLocal: TTSLocalConfig;

  // Embedding Configuration (NANO-043 Phase 5)
  embedding: EmbeddingConfig;

  // NANO-063: Per-provider API key recall map
  savedProviderKeys: Record<string, string>;

  // Form state
  isLoading: boolean;
  isValidating: boolean;
  validationErrors: Record<string, string>;
  isStarting: boolean;

  // NANO-027 Phase 3: Launch state
  launchProgress: LaunchProgress;

  // Actions - LLM
  setLLMProviderType: (type: LLMProviderType) => void;
  updateLLMLocal: (updates: Partial<LLMLocalConfig>) => void;
  updateLLMCloud: (updates: Partial<LLMCloudConfig>) => void;

  // Actions - Vision
  setVLMEnabled: (enabled: boolean) => void;
  setUseLLMForVision: (use: boolean) => void;
  setVLMProviderType: (type: LLMProviderType) => void;
  updateVLMLocal: (updates: Partial<VLMLocalConfig>) => void;
  updateVLMCloud: (updates: Partial<VLMCloudConfig>) => void;

  // Actions - STT
  setSTTProvider: (provider: STTProviderType) => void;
  updateSTTParakeet: (updates: Partial<STTParakeetConfig>) => void;
  updateSTTWhisper: (updates: Partial<STTWhisperConfig>) => void;
  updateSTT: (updates: Partial<STTConfig>) => void;

  // Actions - TTS
  setTTSProviderType: (type: LLMProviderType) => void;
  updateTTSLocal: (updates: Partial<TTSLocalConfig>) => void;

  // Actions - Embedding (NANO-043 Phase 5)
  updateEmbedding: (updates: Partial<EmbeddingConfig>) => void;

  // Actions - NANO-063: Provider key isolation
  setSavedProviderKeys: (keys: Record<string, string>) => void;

  // Actions - Form
  setValidationError: (field: string, error: string | null) => void;
  clearValidationErrors: () => void;
  setIsValidating: (validating: boolean) => void;
  setIsStarting: (starting: boolean) => void;

  // Actions - Hydration
  setIsLoading: (loading: boolean) => void;
  hydrate: (config: HydrateConfig) => void;

  // Actions - Launch (NANO-027 Phase 3)
  setLaunchProgress: (progress: Partial<LaunchProgress>) => void;
  setLaunchError: (error: string, service?: string | null) => void;
  setLaunchComplete: (services: string[]) => void;
  resetLaunch: () => void;

  // Actions - Orchestrator (NANO-027 Phase 4)
  setOrchestratorInitializing: () => void;
  setOrchestratorReady: (persona: string) => void;
  setOrchestratorError: (error: string) => void;

  // Actions - Reset
  reset: () => void;
}

// Hydration config type (matches API response shape)
export interface HydrateConfig {
  llmProviderType: LLMProviderType;
  llmLocal: LLMLocalConfig;
  llmCloud: Partial<LLMCloudConfig>;
  vlmEnabled: boolean;
  useLLMForVision: boolean;
  vlmProviderType: LLMProviderType;
  vlmLocal: Partial<VLMLocalConfig>;
  vlmCloud: Partial<VLMCloudConfig>;
  sttProvider: STTProviderType;
  sttParakeet: Partial<STTParakeetConfig>;
  sttWhisper: Partial<STTWhisperConfig>;
  ttsProviderType: LLMProviderType;
  ttsLocal: Partial<TTSLocalConfig>;
  embedding?: Partial<EmbeddingConfig>;
  savedProviderKeys?: Record<string, string>;
}

// ============================================
// Default Values
// ============================================

const DEFAULT_LLM_LOCAL: LLMLocalConfig = {
  executablePath: "",
  modelPath: "",
  mmprojPath: "",
  host: "127.0.0.1",
  port: 5557,
  contextSize: 8192,
  gpuLayers: 99,
  device: "",
  tensorSplit: "",
  extraArgs: "",
  timeout: 30,
  temperature: 0.7,
  maxTokens: 256,
  topP: 0.95,
  reasoningFormat: "",
  reasoningBudget: -1,
};

const DEFAULT_LLM_CLOUD: LLMCloudConfig = {
  provider: "deepseek",
  apiUrl: "",
  apiKey: "",
  model: "",
  contextSize: 32768,
  timeout: 60,
  temperature: 0.7,
  maxTokens: 256,
};

const DEFAULT_VLM_LOCAL: VLMLocalConfig = {
  modelType: "gemma3",
  executablePath: "",
  modelPath: "",
  mmprojPath: "",
  host: "127.0.0.1",
  port: 5558,
  contextSize: 8192,
  gpuLayers: 99,
  device: "",
  tensorSplit: "",
  extraArgs: "",
  timeout: 30,
  maxTokens: 300,
};

const DEFAULT_VLM_CLOUD: VLMCloudConfig = {
  apiKey: "",
  baseUrl: "https://api.openai.com",
  model: "gpt-4o",
  contextSize: 8192,
  timeout: 30,
  maxTokens: 300,
};

const DEFAULT_STT_PARAKEET: STTParakeetConfig = {
  host: "127.0.0.1",
  port: 5555,
  timeout: 30,
  platform: "native",
  wslDistro: "Ubuntu",
  envType: "conda",
  envNameOrPath: "",
  customActivation: "",
  serverScriptPath: "",
};

const DEFAULT_STT_WHISPER: STTWhisperConfig = {
  host: "127.0.0.1",
  port: 8080,
  timeout: 30,
  binaryPath: "whisper-server",
  modelPath: "",
  language: "en",
  threads: 4,
  noGpu: false,
};

const DEFAULT_TTS_LOCAL: TTSLocalConfig = {
  provider: "kokoro",
  host: "127.0.0.1",
  port: 5556,
  voice: "",
  language: "",
  modelsDirectory: "./tts/models",
  device: "cuda",
  timeout: 30,
  envType: "conda",
  envNameOrPath: "",
  customActivation: "",
};

// NANO-043 Phase 5: Default embedding config
const DEFAULT_EMBEDDING: EmbeddingConfig = {
  enabled: false,
  executablePath: "",
  modelPath: "",
  host: "127.0.0.1",
  port: 5559,
  gpuLayers: 99,
  contextSize: 2048,
  timeout: 60,
  relevanceThreshold: 0.25,
  topK: 5,
};

// NANO-027 Phase 3 & 4: Default launch progress
const DEFAULT_LAUNCH_PROGRESS: LaunchProgress = {
  status: "idle",
  currentService: null,
  message: "",
  launchedServices: [],
  error: null,
  orchestratorStatus: "pending",
  orchestratorPersona: null,
  orchestratorError: null,
};

// ============================================
// Store Implementation
// ============================================

export const useLauncherStore = create<LauncherStoreState>((set) => ({
  // Initial state - LLM
  llmProviderType: "local",
  llmLocal: DEFAULT_LLM_LOCAL,
  llmCloud: DEFAULT_LLM_CLOUD,

  // Initial state - Vision
  vlmEnabled: true,
  useLLMForVision: true,
  vlmProviderType: "local",
  vlmLocal: DEFAULT_VLM_LOCAL,
  vlmCloud: DEFAULT_VLM_CLOUD,

  // Initial state - STT
  sttProvider: "parakeet" as STTProviderType,
  sttParakeet: DEFAULT_STT_PARAKEET,
  sttWhisper: DEFAULT_STT_WHISPER,

  // Initial state - TTS
  ttsProviderType: "local",
  ttsLocal: DEFAULT_TTS_LOCAL,

  // Initial state - Embedding (NANO-043 Phase 5)
  embedding: DEFAULT_EMBEDDING,

  // NANO-063: Per-provider API key recall
  savedProviderKeys: {},

  // Initial state - Form
  isLoading: true,
  isValidating: false,
  validationErrors: {},
  isStarting: false,

  // Initial state - Launch (NANO-027 Phase 3)
  launchProgress: DEFAULT_LAUNCH_PROGRESS,

  // LLM Actions
  setLLMProviderType: (llmProviderType) => set({ llmProviderType }),
  updateLLMLocal: (updates) =>
    set((state) => ({ llmLocal: { ...state.llmLocal, ...updates } })),
  updateLLMCloud: (updates) =>
    set((state) => ({ llmCloud: { ...state.llmCloud, ...updates } })),

  // Vision Actions
  setVLMEnabled: (vlmEnabled) => set({ vlmEnabled }),
  setUseLLMForVision: (useLLMForVision) => set({ useLLMForVision }),
  setVLMProviderType: (vlmProviderType) => set({ vlmProviderType }),
  updateVLMLocal: (updates) =>
    set((state) => ({ vlmLocal: { ...state.vlmLocal, ...updates } })),
  updateVLMCloud: (updates) =>
    set((state) => ({ vlmCloud: { ...state.vlmCloud, ...updates } })),

  // STT Actions
  setSTTProvider: (sttProvider) => set({ sttProvider }),
  updateSTTParakeet: (updates) =>
    set((state) => ({ sttParakeet: { ...state.sttParakeet, ...updates } })),
  updateSTTWhisper: (updates) =>
    set((state) => ({ sttWhisper: { ...state.sttWhisper, ...updates } })),
  updateSTT: (updates) =>
    set((state) => {
      const result: Partial<LauncherStoreState> = {};
      if (updates.provider !== undefined) result.sttProvider = updates.provider;
      // Route shared + provider-specific fields to the active sub-config
      const provider = updates.provider ?? state.sttProvider;
      if (provider === "parakeet") {
        const parakeetUpdates: Partial<STTParakeetConfig> = {};
        if (updates.host !== undefined) parakeetUpdates.host = updates.host;
        if (updates.port !== undefined) parakeetUpdates.port = updates.port;
        if (updates.timeout !== undefined) parakeetUpdates.timeout = updates.timeout;
        if (updates.platform !== undefined) parakeetUpdates.platform = updates.platform;
        if (updates.wslDistro !== undefined) parakeetUpdates.wslDistro = updates.wslDistro;
        if (updates.envType !== undefined) parakeetUpdates.envType = updates.envType;
        if (updates.envNameOrPath !== undefined) parakeetUpdates.envNameOrPath = updates.envNameOrPath;
        if (updates.customActivation !== undefined) parakeetUpdates.customActivation = updates.customActivation;
        if (updates.serverScriptPath !== undefined) parakeetUpdates.serverScriptPath = updates.serverScriptPath;
        if (Object.keys(parakeetUpdates).length > 0) {
          result.sttParakeet = { ...state.sttParakeet, ...parakeetUpdates };
        }
      } else {
        const whisperUpdates: Partial<STTWhisperConfig> = {};
        if (updates.host !== undefined) whisperUpdates.host = updates.host;
        if (updates.port !== undefined) whisperUpdates.port = updates.port;
        if (updates.timeout !== undefined) whisperUpdates.timeout = updates.timeout;
        if (updates.binaryPath !== undefined) whisperUpdates.binaryPath = updates.binaryPath;
        if (updates.modelPath !== undefined) whisperUpdates.modelPath = updates.modelPath;
        if (updates.language !== undefined) whisperUpdates.language = updates.language;
        if (updates.threads !== undefined) whisperUpdates.threads = updates.threads;
        if (updates.noGpu !== undefined) whisperUpdates.noGpu = updates.noGpu;
        if (Object.keys(whisperUpdates).length > 0) {
          result.sttWhisper = { ...state.sttWhisper, ...whisperUpdates };
        }
      }
      return result;
    }),

  // TTS Actions
  setTTSProviderType: (ttsProviderType) => set({ ttsProviderType }),
  updateTTSLocal: (updates) =>
    set((state) => ({ ttsLocal: { ...state.ttsLocal, ...updates } })),

  // Embedding Actions (NANO-043 Phase 5)
  updateEmbedding: (updates) =>
    set((state) => ({ embedding: { ...state.embedding, ...updates } })),

  // NANO-063: Provider key isolation
  setSavedProviderKeys: (savedProviderKeys) => set({ savedProviderKeys }),

  // Form Actions
  setValidationError: (field, error) =>
    set((state) => {
      const newErrors = { ...state.validationErrors };
      if (error === null) {
        delete newErrors[field];
      } else {
        newErrors[field] = error;
      }
      return { validationErrors: newErrors };
    }),
  clearValidationErrors: () => set({ validationErrors: {} }),
  setIsValidating: (isValidating) => set({ isValidating }),
  setIsStarting: (isStarting) => set({ isStarting }),

  // Hydration Actions
  setIsLoading: (isLoading) => set({ isLoading }),
  hydrate: (config) =>
    set((state) => ({
      llmProviderType: config.llmProviderType,
      llmLocal: { ...state.llmLocal, ...config.llmLocal },
      llmCloud: { ...state.llmCloud, ...config.llmCloud },
      vlmEnabled: config.vlmEnabled,
      useLLMForVision: config.useLLMForVision,
      vlmProviderType: config.vlmProviderType,
      vlmLocal: { ...state.vlmLocal, ...config.vlmLocal },
      vlmCloud: { ...state.vlmCloud, ...config.vlmCloud },
      sttProvider: config.sttProvider,
      sttParakeet: { ...state.sttParakeet, ...config.sttParakeet },
      sttWhisper: { ...state.sttWhisper, ...config.sttWhisper },
      ttsProviderType: config.ttsProviderType,
      ttsLocal: { ...state.ttsLocal, ...config.ttsLocal },
      embedding: { ...state.embedding, ...(config.embedding || {}) },
      savedProviderKeys: config.savedProviderKeys || {},
      isLoading: false,
    })),

  // Launch Actions (NANO-027 Phase 3)
  setLaunchProgress: (progress) =>
    set((state) => ({
      launchProgress: { ...state.launchProgress, ...progress },
    })),
  setLaunchError: (error, service = null) =>
    set((state) => ({
      launchProgress: {
        ...state.launchProgress,
        status: "error",
        error,
        currentService: service,
      },
      isStarting: false,
    })),
  setLaunchComplete: (services) =>
    set((state) => ({
      launchProgress: {
        ...state.launchProgress,
        status: "complete",
        launchedServices: services,
        currentService: null,
        message: "All services started",
      },
      isStarting: false,
    })),
  resetLaunch: () =>
    set({
      launchProgress: DEFAULT_LAUNCH_PROGRESS,
      isStarting: false,
    }),

  // Orchestrator Actions (NANO-027 Phase 4)
  setOrchestratorInitializing: () =>
    set((state) => ({
      launchProgress: {
        ...state.launchProgress,
        orchestratorStatus: "initializing",
        message: "Initializing orchestrator...",
      },
    })),
  setOrchestratorReady: (persona) =>
    set((state) => ({
      launchProgress: {
        ...state.launchProgress,
        orchestratorStatus: "ready",
        orchestratorPersona: persona,
        message: `Orchestrator ready (${persona})`,
      },
    })),
  setOrchestratorError: (error) =>
    set((state) => ({
      launchProgress: {
        ...state.launchProgress,
        orchestratorStatus: "error",
        orchestratorError: error,
        message: `Orchestrator failed: ${error}`,
      },
    })),

  // Reset
  reset: () =>
    set({
      llmProviderType: "local",
      llmLocal: DEFAULT_LLM_LOCAL,
      llmCloud: DEFAULT_LLM_CLOUD,
      vlmEnabled: true,
      useLLMForVision: true,
      vlmProviderType: "local",
      vlmLocal: DEFAULT_VLM_LOCAL,
      vlmCloud: DEFAULT_VLM_CLOUD,
      sttProvider: "parakeet" as STTProviderType,
      sttParakeet: DEFAULT_STT_PARAKEET,
      sttWhisper: DEFAULT_STT_WHISPER,
      ttsProviderType: "local",
      ttsLocal: DEFAULT_TTS_LOCAL,
      embedding: DEFAULT_EMBEDDING,
      savedProviderKeys: {},
      isLoading: false,
      isValidating: false,
      validationErrors: {},
      isStarting: false,
      launchProgress: DEFAULT_LAUNCH_PROGRESS,
    }),
}));

// ============================================
// Selectors
// ============================================

// Compose flat STTConfig from split state (used by stt-section component)
export const selectActiveSTT = (state: LauncherStoreState): STTConfig => {
  const p = state.sttParakeet;
  const w = state.sttWhisper;
  const active = state.sttProvider === "parakeet" ? p : w;
  return {
    provider: state.sttProvider,
    host: active.host,
    port: active.port,
    timeout: active.timeout,
    // Parakeet fields
    platform: p.platform,
    wslDistro: p.wslDistro,
    envType: p.envType,
    envNameOrPath: p.envNameOrPath,
    customActivation: p.customActivation,
    serverScriptPath: p.serverScriptPath,
    // Whisper fields
    binaryPath: w.binaryPath,
    modelPath: w.modelPath,
    language: w.language,
    threads: w.threads,
    noGpu: w.noGpu,
  };
};

export const selectHasValidationErrors = (state: LauncherStoreState): boolean =>
  Object.keys(state.validationErrors).length > 0;

export const selectIsFormComplete = (state: LauncherStoreState): boolean => {
  // Check required LLM fields based on provider type
  if (state.llmProviderType === "local") {
    if (!state.llmLocal.executablePath || !state.llmLocal.modelPath) {
      return false;
    }
  } else {
    if (!state.llmCloud.apiKey || !state.llmCloud.model) {
      return false;
    }
  }

  // Check VLM if enabled and not using LLM for vision
  if (state.vlmEnabled && !state.useLLMForVision) {
    if (state.vlmProviderType === "local") {
      if (!state.vlmLocal.executablePath || !state.vlmLocal.modelPath) {
        return false;
      }
    } else {
      if (!state.vlmCloud.apiKey || !state.vlmCloud.model) {
        return false;
      }
    }
  }

  // Check STT — provider-conditional
  if (state.sttProvider === "parakeet") {
    if (!state.sttParakeet.serverScriptPath) return false;
    if ((state.sttParakeet.envType === "conda" || state.sttParakeet.envType === "venv") && !state.sttParakeet.envNameOrPath) return false;
    if (state.sttParakeet.envType === "other" && !state.sttParakeet.customActivation) return false;
  } else {
    if (!state.sttWhisper.modelPath) return false;
  }

  // Check TTS
  if (state.ttsProviderType === "local") {
    if (state.ttsLocal.envType === "conda" || state.ttsLocal.envType === "venv") {
      if (!state.ttsLocal.envNameOrPath) {
        return false;
      }
    }
    if (state.ttsLocal.envType === "other" && !state.ttsLocal.customActivation) {
      return false;
    }
  }

  // Check Embedding (optional — only validate when enabled)
  if (state.embedding.enabled) {
    if (!state.embedding.executablePath || !state.embedding.modelPath) {
      return false;
    }
  }

  return true;
};
