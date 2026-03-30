import { describe, it, expect, beforeEach } from "vitest";
import {
  useSettingsStore,
  selectEffectiveVadConfig,
  selectEffectivePipelineConfig,
  selectHasUnsavedChanges,
} from "./settings-store";
import type { ConfigLoadedEvent } from "@/types/events";

// Test fixture factories
function createMockConfigEvent(overrides?: Partial<ConfigLoadedEvent>): ConfigLoadedEvent {
  return {
    persona: {
      id: "spindle",
      name: "Spindle",
      voice: "en-US-Neural2-F",
    },
    providers: {
      llm: { name: "deepseek", config: {} },
      tts: { name: "piper", config: {} },
      stt: { name: "parakeet", config: {} },
      vlm: { name: "ollama", config: {} },
    },
    settings: {
      vad: {
        threshold: 0.6,
        min_speech_ms: 300,
        min_silence_ms: 1200,
      },
      pipeline: {
        summarization_threshold: 0.85,
        budget_strategy: "truncate",
      },
    },
    ...overrides,
  };
}

describe("useSettingsStore", () => {
  beforeEach(() => {
    // Reset store between tests
    useSettingsStore.setState({
      vadConfig: {
        threshold: 0.5,
        min_speech_ms: 250,
        min_silence_ms: 1000,
      },
      pendingVadChanges: {},
      isSavingVad: false,
      pipelineConfig: {
        summarization_threshold: 0.8,
        budget_strategy: "truncate",
      },
      pendingPipelineChanges: {},
      isSavingPipeline: false,
      providers: {
        llm: null,
        tts: null,
        stt: null,
        vlm: null,
        embedding: null,
      },
      rawConfig: null,
      restartRequired: false,
      error: null,
    });
  });

  describe("initial state", () => {
    it("should initialize with default VAD config", () => {
      const state = useSettingsStore.getState();
      expect(state.vadConfig.threshold).toBe(0.5);
      expect(state.vadConfig.min_speech_ms).toBe(250);
      expect(state.vadConfig.min_silence_ms).toBe(1000);
    });

    it("should initialize with default pipeline config", () => {
      const state = useSettingsStore.getState();
      expect(state.pipelineConfig.summarization_threshold).toBe(0.8);
    });

    it("should initialize with null providers", () => {
      const state = useSettingsStore.getState();
      expect(state.providers.llm).toBeNull();
      expect(state.providers.tts).toBeNull();
    });

    it("should initialize with restartRequired false", () => {
      const state = useSettingsStore.getState();
      expect(state.restartRequired).toBe(false);
    });
  });

  describe("VAD config actions", () => {
    it("should set VAD config", () => {
      useSettingsStore.getState().setVadConfig({
        threshold: 0.7,
        min_speech_ms: 400,
        min_silence_ms: 1500,
      });

      const state = useSettingsStore.getState();
      expect(state.vadConfig.threshold).toBe(0.7);
      expect(state.vadConfig.min_speech_ms).toBe(400);
    });

    it("should clear pending VAD changes when setting config", () => {
      useSettingsStore.setState({ pendingVadChanges: { threshold: 0.9 } });
      useSettingsStore.getState().setVadConfig({
        threshold: 0.7,
        min_speech_ms: 400,
        min_silence_ms: 1500,
      });

      expect(useSettingsStore.getState().pendingVadChanges).toEqual({});
    });

    it("should update pending VAD changes", () => {
      useSettingsStore.getState().updatePendingVad({ threshold: 0.8 });
      expect(useSettingsStore.getState().pendingVadChanges.threshold).toBe(0.8);

      useSettingsStore.getState().updatePendingVad({ min_speech_ms: 500 });
      const pending = useSettingsStore.getState().pendingVadChanges;
      expect(pending.threshold).toBe(0.8);
      expect(pending.min_speech_ms).toBe(500);
    });

    it("should clear pending VAD changes", () => {
      useSettingsStore.setState({
        pendingVadChanges: { threshold: 0.9, min_speech_ms: 600 },
      });
      useSettingsStore.getState().clearPendingVad();

      expect(useSettingsStore.getState().pendingVadChanges).toEqual({});
    });

    it("should set saving VAD state", () => {
      useSettingsStore.getState().setSavingVad(true);
      expect(useSettingsStore.getState().isSavingVad).toBe(true);

      useSettingsStore.getState().setSavingVad(false);
      expect(useSettingsStore.getState().isSavingVad).toBe(false);
    });
  });

  describe("Pipeline config actions", () => {
    it("should set pipeline config", () => {
      useSettingsStore.getState().setPipelineConfig({
        summarization_threshold: 0.9,
        budget_strategy: "drop",
      });

      const state = useSettingsStore.getState();
      expect(state.pipelineConfig.summarization_threshold).toBe(0.9);
      expect(state.pipelineConfig.budget_strategy).toBe("drop");
    });

    it("should clear pending pipeline changes when setting config", () => {
      useSettingsStore.setState({
        pendingPipelineChanges: { summarization_threshold: 0.95 },
      });
      useSettingsStore.getState().setPipelineConfig({
        summarization_threshold: 0.9,
        budget_strategy: "drop",
      });

      expect(useSettingsStore.getState().pendingPipelineChanges).toEqual({});
    });

    it("should update pending pipeline changes", () => {
      useSettingsStore.getState().updatePendingPipeline({ summarization_threshold: 0.7 });
      expect(useSettingsStore.getState().pendingPipelineChanges.summarization_threshold).toBe(0.7);
    });
  });

  describe("provider actions", () => {
    it("should set providers", () => {
      useSettingsStore.getState().setProviders({
        llm: { name: "deepseek", config: {} },
        tts: { name: "piper", config: {} },
        stt: { name: "parakeet", config: {} },
        vlm: null,
        embedding: null,
      });

      const state = useSettingsStore.getState();
      expect(state.providers.llm?.name).toBe("deepseek");
      expect(state.providers.stt?.name).toBe("parakeet");
      expect(state.providers.vlm).toBeNull();
    });
  });

  describe("loadConfig", () => {
    it("should load all config from config_loaded event", () => {
      const config = createMockConfigEvent();
      useSettingsStore.getState().loadConfig(config);

      const state = useSettingsStore.getState();
      expect(state.vadConfig.threshold).toBe(0.6);
      expect(state.pipelineConfig.summarization_threshold).toBe(0.85);
      expect(state.providers.llm?.name).toBe("deepseek");
    });

    it("should clear pending changes when loading config", () => {
      useSettingsStore.setState({
        pendingVadChanges: { threshold: 0.9 },
        pendingPipelineChanges: { summarization_threshold: 0.95 },
      });

      useSettingsStore.getState().loadConfig(createMockConfigEvent());

      const state = useSettingsStore.getState();
      expect(state.pendingVadChanges).toEqual({});
      expect(state.pendingPipelineChanges).toEqual({});
    });

    it("should clear error when loading config", () => {
      useSettingsStore.setState({ error: "Previous error" });
      useSettingsStore.getState().loadConfig(createMockConfigEvent());

      expect(useSettingsStore.getState().error).toBeNull();
    });

    it("should handle config without VLM", () => {
      const config = createMockConfigEvent();
      delete config.providers.vlm;

      useSettingsStore.getState().loadConfig(config);

      expect(useSettingsStore.getState().providers.vlm).toBeNull();
    });
  });

  describe("restart required", () => {
    it("should set restart required", () => {
      useSettingsStore.getState().setRestartRequired(true);
      expect(useSettingsStore.getState().restartRequired).toBe(true);
    });

    it("should clear restart required", () => {
      useSettingsStore.setState({ restartRequired: true });
      useSettingsStore.getState().setRestartRequired(false);
      expect(useSettingsStore.getState().restartRequired).toBe(false);
    });
  });

  describe("error handling", () => {
    it("should set error", () => {
      useSettingsStore.getState().setError("Something went wrong");
      expect(useSettingsStore.getState().error).toBe("Something went wrong");
    });

    it("should clear error", () => {
      useSettingsStore.setState({ error: "Previous error" });
      useSettingsStore.getState().setError(null);
      expect(useSettingsStore.getState().error).toBeNull();
    });
  });
});

describe("selectors", () => {
  beforeEach(() => {
    useSettingsStore.setState({
      vadConfig: {
        threshold: 0.5,
        min_speech_ms: 250,
        min_silence_ms: 1000,
      },
      pendingVadChanges: {},
      pipelineConfig: {
        summarization_threshold: 0.8,
        budget_strategy: "truncate",
      },
      pendingPipelineChanges: {},
    });
  });

  describe("selectEffectiveVadConfig", () => {
    it("should return current config when no pending changes", () => {
      const state = useSettingsStore.getState();
      const effective = selectEffectiveVadConfig(state);

      expect(effective.threshold).toBe(0.5);
      expect(effective.min_speech_ms).toBe(250);
    });

    it("should merge pending changes into effective config", () => {
      useSettingsStore.setState({ pendingVadChanges: { threshold: 0.8 } });

      const state = useSettingsStore.getState();
      const effective = selectEffectiveVadConfig(state);

      expect(effective.threshold).toBe(0.8);
      expect(effective.min_speech_ms).toBe(250); // unchanged
    });
  });

  describe("selectEffectivePipelineConfig", () => {
    it("should return current config when no pending changes", () => {
      const state = useSettingsStore.getState();
      const effective = selectEffectivePipelineConfig(state);

      expect(effective.summarization_threshold).toBe(0.8);
    });

    it("should merge pending changes into effective config", () => {
      useSettingsStore.setState({
        pendingPipelineChanges: { summarization_threshold: 0.7 },
      });

      const state = useSettingsStore.getState();
      const effective = selectEffectivePipelineConfig(state);

      expect(effective.summarization_threshold).toBe(0.7);
      expect(effective.budget_strategy).toBe("truncate"); // unchanged
    });
  });

  describe("selectHasUnsavedChanges", () => {
    it("should return false when no pending changes", () => {
      const state = useSettingsStore.getState();
      expect(selectHasUnsavedChanges(state)).toBe(false);
    });

    it("should return true when VAD changes pending", () => {
      useSettingsStore.setState({ pendingVadChanges: { threshold: 0.9 } });

      const state = useSettingsStore.getState();
      expect(selectHasUnsavedChanges(state)).toBe(true);
    });

    it("should return true when pipeline changes pending", () => {
      useSettingsStore.setState({
        pendingPipelineChanges: { summarization_threshold: 0.95 },
      });

      const state = useSettingsStore.getState();
      expect(selectHasUnsavedChanges(state)).toBe(true);
    });

    it("should return true when both VAD and pipeline changes pending", () => {
      useSettingsStore.setState({
        pendingVadChanges: { threshold: 0.9 },
        pendingPipelineChanges: { summarization_threshold: 0.95 },
      });

      const state = useSettingsStore.getState();
      expect(selectHasUnsavedChanges(state)).toBe(true);
    });
  });
});

// ============================================
// NANO-079: VLM Dashboard Launch — Store Tests
// ============================================

describe("NANO-079: VLM launch state", () => {
  beforeEach(() => {
    useSettingsStore.setState({
      localVLMConfig: { host: "127.0.0.1", port: 5558, gpu_layers: 99, context_size: 8192 },
      isLaunchingVLM: false,
      vlmServerRunning: false,
      vlmLaunchError: null,
      vlmConfig: { provider: "llama", available_providers: ["llama", "openai"], healthy: false, unified_vlm: false },
      localLLMConfig: { host: "127.0.0.1", port: 5557, gpu_layers: 99, context_size: 8192 },
    });
  });

  it("should update local VLM config fields", () => {
    useSettingsStore.getState().updateLocalVLMConfig({ executable_path: "/path/to/server" });
    expect(useSettingsStore.getState().localVLMConfig.executable_path).toBe("/path/to/server");
    // Other fields preserved
    expect(useSettingsStore.getState().localVLMConfig.port).toBe(5558);
  });

  it("should set VLM launching state", () => {
    useSettingsStore.getState().setLaunchingVLM(true);
    expect(useSettingsStore.getState().isLaunchingVLM).toBe(true);
  });

  it("should set VLM server running state", () => {
    useSettingsStore.getState().setVLMServerRunning(true);
    expect(useSettingsStore.getState().vlmServerRunning).toBe(true);
  });

  it("should set VLM launch error", () => {
    useSettingsStore.getState().setVLMLaunchError("Port already in use");
    expect(useSettingsStore.getState().vlmLaunchError).toBe("Port already in use");
  });

  it("should toggle unified VLM on vlmConfig", () => {
    useSettingsStore.getState().setUnifiedVLM(true);
    expect(useSettingsStore.getState().vlmConfig.unified_vlm).toBe(true);
    // Other vlmConfig fields preserved
    expect(useSettingsStore.getState().vlmConfig.provider).toBe("llama");
  });

  it("should toggle unified VLM off", () => {
    useSettingsStore.getState().setUnifiedVLM(true);
    useSettingsStore.getState().setUnifiedVLM(false);
    expect(useSettingsStore.getState().vlmConfig.unified_vlm).toBe(false);
  });
});

describe("NANO-079: canLaunch gate logic", () => {
  beforeEach(() => {
    useSettingsStore.setState({
      isLaunchingLLM: false,
      localLLMConfig: {
        executable_path: "/path/to/server",
        model_path: "/path/to/model.gguf",
        host: "127.0.0.1",
        port: 5557,
      },
      vlmConfig: { provider: "llama", available_providers: ["llama", "openai"], healthy: false, unified_vlm: false },
    });
  });

  it("canLaunch true when unified off and exe+model present", () => {
    const { isLaunchingLLM, localLLMConfig, vlmConfig } = useSettingsStore.getState();
    const canLaunch = !isLaunchingLLM && localLLMConfig.executable_path && localLLMConfig.model_path
      && (!vlmConfig.unified_vlm || localLLMConfig.mmproj_path);
    expect(!!canLaunch).toBe(true);
  });

  it("canLaunch false when unified on and mmproj empty", () => {
    useSettingsStore.getState().setUnifiedVLM(true);
    const { isLaunchingLLM, localLLMConfig, vlmConfig } = useSettingsStore.getState();
    const canLaunch = !isLaunchingLLM && localLLMConfig.executable_path && localLLMConfig.model_path
      && (!vlmConfig.unified_vlm || localLLMConfig.mmproj_path);
    expect(!!canLaunch).toBe(false);
  });

  it("canLaunch true when unified on and mmproj filled", () => {
    useSettingsStore.getState().setUnifiedVLM(true);
    useSettingsStore.getState().updateLocalLLMConfig({ mmproj_path: "/path/to/mmproj.gguf" });
    const { isLaunchingLLM, localLLMConfig, vlmConfig } = useSettingsStore.getState();
    const canLaunch = !isLaunchingLLM && localLLMConfig.executable_path && localLLMConfig.model_path
      && (!vlmConfig.unified_vlm || localLLMConfig.mmproj_path);
    expect(!!canLaunch).toBe(true);
  });

  it("canLaunch false when launching", () => {
    useSettingsStore.getState().setLaunchingLLM(true);
    const { isLaunchingLLM, localLLMConfig, vlmConfig } = useSettingsStore.getState();
    const canLaunch = !isLaunchingLLM && localLLMConfig.executable_path && localLLMConfig.model_path
      && (!vlmConfig.unified_vlm || localLLMConfig.mmproj_path);
    expect(!!canLaunch).toBe(false);
  });
});
