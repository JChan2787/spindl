/**
 * NANO-089 Phase 2: Zod schema validation tests.
 *
 * Tests both launcher schemas (write-config POST body) and
 * runtime schemas (socket event validation).
 */
import { describe, it, expect, vi } from "vitest";
import {
  LauncherConfigSchema,
  LLMLocalConfigSchema,
  VLMLocalConfigSchema,
  EmbeddingConfigSchema,
  STTWhisperConfigSchema,
  TTSLocalConfigSchema,
} from "./launcher-schemas";
import {
  LLMConfigUpdatedSchema,
  VLMConfigUpdatedSchema,
  LocalLLMConfigEventSchema,
  LocalVLMConfigEventSchema,
  ToolsConfigUpdatedSchema,
  AvatarExpressionsSchema,
  AvatarAnimationsSchema,
  safeParse,
} from "./runtime-schemas";

// ============================================
// Test Fixtures
// ============================================

const VALID_LLM_LOCAL = {
  executablePath: "C:\\llama\\llama-server.exe",
  modelPath: "C:\\models\\llm.gguf",
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
};

const VALID_LLM_CLOUD = {
  provider: "openai",
  apiUrl: "https://api.openai.com/v1",
  apiKey: "sk-test-key-123",
  model: "gpt-4o",
  contextSize: 32768,
  timeout: 60,
  temperature: 0.7,
  maxTokens: 256,
};

const VALID_VLM_LOCAL = {
  modelType: "gemma3",
  executablePath: "C:\\llama\\llama-server.exe",
  modelPath: "C:\\models\\vlm.gguf",
  mmprojPath: "",
  host: "127.0.0.1",
  port: 5558,
  contextSize: 8192,
  gpuLayers: 99,
  device: "",
  extraArgs: "",
  timeout: 30,
  maxTokens: 300,
};

const VALID_VLM_CLOUD = {
  apiKey: "sk-vlm-key",
  baseUrl: "https://api.xai.com/v1",
  model: "grok-2-vision",
  contextSize: 8192,
  timeout: 30,
  maxTokens: 300,
};

const VALID_STT_PARAKEET = {
  host: "127.0.0.1",
  port: 5555,
  timeout: 30,
  platform: "native" as const,
  wslDistro: "Ubuntu",
  envType: "conda" as const,
  envNameOrPath: "nemo",
  customActivation: "",
  serverScriptPath: "C:\\stt\\server.py",
};

const VALID_STT_WHISPER = {
  host: "127.0.0.1",
  port: 8080,
  timeout: 30,
  binaryPath: "whisper-server",
  modelPath: "C:\\models\\ggml-small.en.bin",
  language: "en",
  threads: 4,
  noGpu: false,
};

const VALID_TTS_LOCAL = {
  provider: "kokoro",
  host: "127.0.0.1",
  port: 5556,
  voice: "default",
  language: "en",
  modelsDirectory: "./tts/models",
  timeout: 30,
  envType: "conda" as const,
  envNameOrPath: "kokoro",
  customActivation: "",
};

function makeFullConfig(overrides: Record<string, unknown> = {}) {
  return {
    llmProviderType: "local",
    llmLocal: VALID_LLM_LOCAL,
    llmCloud: VALID_LLM_CLOUD,
    vlmEnabled: true,
    useLLMForVision: false,
    vlmProviderType: "local",
    vlmLocal: VALID_VLM_LOCAL,
    vlmCloud: VALID_VLM_CLOUD,
    sttProvider: "parakeet",
    sttParakeet: VALID_STT_PARAKEET,
    sttWhisper: VALID_STT_WHISPER,
    ttsProviderType: "local",
    ttsLocal: VALID_TTS_LOCAL,
    ...overrides,
  };
}

// ============================================
// Launcher Schemas
// ============================================

describe("LauncherConfigSchema", () => {
  it("accepts valid full config", () => {
    const result = LauncherConfigSchema.safeParse(makeFullConfig());
    expect(result.success).toBe(true);
  });

  it("fills defaults for omitted optional fields", () => {
    const result = LauncherConfigSchema.safeParse(makeFullConfig());
    expect(result.success).toBe(true);
    if (result.success) {
      // mmprojPath defaults to "" when omitted from llmLocal
      expect(result.data.llmLocal.mmprojPath).toBe("");
      // reasoningFormat defaults to ""
      expect(result.data.llmLocal.reasoningFormat).toBe("");
      // reasoningBudget defaults to -1
      expect(result.data.llmLocal.reasoningBudget).toBe(-1);
    }
  });

  it("accepts config with embedding section", () => {
    const result = LauncherConfigSchema.safeParse(
      makeFullConfig({
        embedding: {
          enabled: true,
          executablePath: "C:\\llama\\llama-server.exe",
          modelPath: "C:\\models\\embed.gguf",
          host: "127.0.0.1",
          port: 5559,
          gpuLayers: 99,
          contextSize: 2048,
          timeout: 60,
          relevanceThreshold: 0.25,
          topK: 5,
        },
      }),
    );
    expect(result.success).toBe(true);
  });

  it("accepts config without embedding section", () => {
    const config = makeFullConfig();
    delete (config as Record<string, unknown>).embedding;
    const result = LauncherConfigSchema.safeParse(config);
    expect(result.success).toBe(true);
  });

  it("rejects invalid llmProviderType", () => {
    const result = LauncherConfigSchema.safeParse(
      makeFullConfig({ llmProviderType: "invalid" }),
    );
    expect(result.success).toBe(false);
  });
});

describe("LLMLocalConfigSchema", () => {
  it("rejects port out of range", () => {
    const result = LLMLocalConfigSchema.safeParse({
      ...VALID_LLM_LOCAL,
      port: 99999,
    });
    expect(result.success).toBe(false);
  });

  it("rejects port 0", () => {
    const result = LLMLocalConfigSchema.safeParse({
      ...VALID_LLM_LOCAL,
      port: 0,
    });
    expect(result.success).toBe(false);
  });

  it("rejects temperature > 2", () => {
    const result = LLMLocalConfigSchema.safeParse({
      ...VALID_LLM_LOCAL,
      temperature: 2.5,
    });
    expect(result.success).toBe(false);
  });

  it("rejects negative maxTokens", () => {
    const result = LLMLocalConfigSchema.safeParse({
      ...VALID_LLM_LOCAL,
      maxTokens: -1,
    });
    expect(result.success).toBe(false);
  });

  it("rejects topP > 1", () => {
    const result = LLMLocalConfigSchema.safeParse({
      ...VALID_LLM_LOCAL,
      topP: 1.5,
    });
    expect(result.success).toBe(false);
  });
});

describe("VLMLocalConfigSchema", () => {
  it("fills default for omitted tensorSplit", () => {
    // This is the exact shape from existing test fixtures (BASE_VLM_LOCAL)
    // which omits tensorSplit — schema must handle it
    const configWithoutTensorSplit = { ...VALID_VLM_LOCAL };
    delete (configWithoutTensorSplit as Record<string, unknown>).tensorSplit;
    const result = VLMLocalConfigSchema.safeParse(configWithoutTensorSplit);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.tensorSplit).toBe("");
    }
  });

  it("rejects empty modelType", () => {
    const result = VLMLocalConfigSchema.safeParse({
      ...VALID_VLM_LOCAL,
      modelType: "",
    });
    expect(result.success).toBe(false);
  });
});

describe("EmbeddingConfigSchema", () => {
  it("rejects relevanceThreshold > 1", () => {
    const result = EmbeddingConfigSchema.safeParse({
      enabled: true,
      executablePath: "exe",
      modelPath: "model",
      host: "127.0.0.1",
      port: 5559,
      gpuLayers: 99,
      contextSize: 2048,
      timeout: 60,
      relevanceThreshold: 1.5,
      topK: 5,
    });
    expect(result.success).toBe(false);
  });

  it("rejects topK < 1", () => {
    const result = EmbeddingConfigSchema.safeParse({
      enabled: true,
      executablePath: "exe",
      modelPath: "model",
      host: "127.0.0.1",
      port: 5559,
      gpuLayers: 99,
      contextSize: 2048,
      timeout: 60,
      relevanceThreshold: 0.25,
      topK: 0,
    });
    expect(result.success).toBe(false);
  });
});

describe("STTWhisperConfigSchema", () => {
  it("rejects threads < 1", () => {
    const result = STTWhisperConfigSchema.safeParse({
      ...VALID_STT_WHISPER,
      threads: 0,
    });
    expect(result.success).toBe(false);
  });
});

describe("TTSLocalConfigSchema", () => {
  it("rejects empty provider", () => {
    const result = TTSLocalConfigSchema.safeParse({
      ...VALID_TTS_LOCAL,
      provider: "",
    });
    expect(result.success).toBe(false);
  });
});

// ============================================
// Runtime Schemas
// ============================================

describe("LLMConfigUpdatedSchema", () => {
  it("accepts valid event", () => {
    const result = LLMConfigUpdatedSchema.safeParse({
      provider: "llama",
      model: "qwen3-8b",
      context_size: 8192,
      available_providers: ["llama", "openrouter"],
    });
    expect(result.success).toBe(true);
  });

  it("fills defaults for missing optional fields", () => {
    const result = LLMConfigUpdatedSchema.safeParse({
      provider: "llama",
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.model).toBeNull();
      expect(result.data.context_size).toBeNull();
      expect(result.data.available_providers).toEqual([]);
    }
  });

  it("accepts missing provider (error-only payload)", () => {
    const result = LLMConfigUpdatedSchema.safeParse({
      error: "Services not launched",
    });
    expect(result.success).toBe(true);
  });

  it("rejects empty provider", () => {
    const result = LLMConfigUpdatedSchema.safeParse({
      provider: "",
      model: "qwen3-8b",
    });
    expect(result.success).toBe(false);
  });

  it("accepts event with error field", () => {
    const result = LLMConfigUpdatedSchema.safeParse({
      provider: "llama",
      error: "Provider not available",
      success: false,
    });
    expect(result.success).toBe(true);
  });
});

describe("VLMConfigUpdatedSchema", () => {
  it("accepts valid event with cloud_config", () => {
    const result = VLMConfigUpdatedSchema.safeParse({
      provider: "openai",
      available_providers: ["llama", "openai"],
      healthy: true,
      cloud_config: {
        api_key: "sk-test",
        model: "gpt-4o",
        base_url: "https://api.openai.com/v1",
      },
    });
    expect(result.success).toBe(true);
  });

  it("accepts event without cloud_config", () => {
    const result = VLMConfigUpdatedSchema.safeParse({
      provider: "llama",
      available_providers: ["llama"],
      healthy: true,
    });
    expect(result.success).toBe(true);
  });

  it("accepts missing provider (error-only payload)", () => {
    const result = VLMConfigUpdatedSchema.safeParse({
      error: "Screen vision tool not initialized",
    });
    expect(result.success).toBe(true);
  });

  it("fills healthy default to false", () => {
    const result = VLMConfigUpdatedSchema.safeParse({
      provider: "llama",
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.healthy).toBe(false);
    }
  });
});

describe("LocalLLMConfigEventSchema", () => {
  it("accepts full config event", () => {
    const result = LocalLLMConfigEventSchema.safeParse({
      config: {
        executable_path: "C:\\llama\\server.exe",
        model_path: "C:\\models\\llm.gguf",
        host: "127.0.0.1",
        port: 5557,
        gpu_layers: 99,
        context_size: 8192,
      },
      server_running: true,
    });
    expect(result.success).toBe(true);
  });

  it("accepts minimal config event (all config fields optional)", () => {
    const result = LocalLLMConfigEventSchema.safeParse({
      config: {},
      server_running: false,
    });
    expect(result.success).toBe(true);
  });

  it("rejects missing server_running", () => {
    const result = LocalLLMConfigEventSchema.safeParse({
      config: { host: "127.0.0.1" },
    });
    expect(result.success).toBe(false);
  });
});

describe("LocalVLMConfigEventSchema", () => {
  it("accepts full config event", () => {
    const result = LocalVLMConfigEventSchema.safeParse({
      config: {
        model_type: "gemma3",
        executable_path: "C:\\llama\\server.exe",
        model_path: "C:\\models\\vlm.gguf",
        host: "127.0.0.1",
        port: 5558,
      },
      server_running: true,
    });
    expect(result.success).toBe(true);
  });

  it("rejects missing server_running", () => {
    const result = LocalVLMConfigEventSchema.safeParse({
      config: { model_type: "gemma3" },
    });
    expect(result.success).toBe(false);
  });
});

// ============================================
// safeParse Helper
// ============================================

describe("safeParse", () => {
  it("returns parsed data on success", () => {
    const result = safeParse(
      LLMConfigUpdatedSchema,
      { provider: "llama", model: "test" },
      "test_event",
    );
    expect(result).not.toBeNull();
    expect(result?.provider).toBe("llama");
  });

  it("returns null on failure", () => {
    const result = safeParse(
      LLMConfigUpdatedSchema,
      { provider: "" }, // empty provider rejected by min(1)
      "test_event",
    );
    expect(result).toBeNull();
  });

  it("logs error on failure", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    safeParse(
      LLMConfigUpdatedSchema,
      { provider: "" },
      "test_event",
    );
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[NANO-089] test_event validation failed:"),
      expect.any(Array),
    );
    errorSpy.mockRestore();
  });
});

// ============================================
// NANO-089 Phase 4: ToolsConfigUpdatedSchema
// ============================================

describe("ToolsConfigUpdatedSchema", () => {
  it("accepts valid tools config event", () => {
    const result = ToolsConfigUpdatedSchema.safeParse({
      master_enabled: true,
      tools: {
        screen_vision: { enabled: true, label: "Screen Vision" },
      },
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.master_enabled).toBe(true);
      expect(result.data.tools.screen_vision.enabled).toBe(true);
    }
  });

  it("fills defaults for empty tools record", () => {
    const result = ToolsConfigUpdatedSchema.safeParse({
      master_enabled: false,
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.tools).toEqual({});
    }
  });

  it("rejects missing master_enabled", () => {
    const result = ToolsConfigUpdatedSchema.safeParse({
      tools: {},
    });
    expect(result.success).toBe(false);
  });

  it("accepts event with error field", () => {
    const result = ToolsConfigUpdatedSchema.safeParse({
      master_enabled: false,
      tools: {},
      error: "No VLM provider configured.",
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.error).toBe("No VLM provider configured.");
    }
  });

  it("rejects malformed tool entry", () => {
    const result = ToolsConfigUpdatedSchema.safeParse({
      master_enabled: true,
      tools: {
        screen_vision: { enabled: "yes" },
      },
    });
    expect(result.success).toBe(false);
  });
});

// ============================================
// Avatar Expression Schemas (NANO-098)
// ============================================

describe("AvatarExpressionsSchema", () => {
  it("accepts valid expression composites", () => {
    const result = AvatarExpressionsSchema.safeParse({
      curious: { aa: 0.4, happy: 0.3, oh: 0.2 },
    });
    expect(result.success).toBe(true);
  });

  it("accepts undefined (optional)", () => {
    const result = AvatarExpressionsSchema.safeParse(undefined);
    expect(result.success).toBe(true);
  });

  it("accepts empty record", () => {
    const result = AvatarExpressionsSchema.safeParse({});
    expect(result.success).toBe(true);
  });
});

// ============================================
// Avatar Animation Schemas (NANO-098 Session 3)
// ============================================

describe("AvatarAnimationsSchema", () => {
  it("accepts full animation config", () => {
    const result = AvatarAnimationsSchema.safeParse({
      default: "Breathing Idle",
      emotions: {
        amused: { threshold: 0.75, clip: "Happy" },
        melancholy: { threshold: 0.8, clip: "Sad Idle" },
        annoyed: { threshold: 0.75, clip: "Annoyed Idle" },
        curious: { threshold: 0.75, clip: "Surprised" },
      },
    });
    expect(result.success).toBe(true);
  });

  it("accepts partial config (default only)", () => {
    const result = AvatarAnimationsSchema.safeParse({
      default: "Breathing Idle",
    });
    expect(result.success).toBe(true);
  });

  it("accepts undefined (optional)", () => {
    const result = AvatarAnimationsSchema.safeParse(undefined);
    expect(result.success).toBe(true);
  });

  it("accepts empty object", () => {
    const result = AvatarAnimationsSchema.safeParse({});
    expect(result.success).toBe(true);
  });

  it("rejects threshold above 1.0", () => {
    const result = AvatarAnimationsSchema.safeParse({
      emotions: {
        amused: { threshold: 1.5, clip: "Happy" },
      },
    });
    expect(result.success).toBe(false);
  });

  it("rejects threshold below 0.0", () => {
    const result = AvatarAnimationsSchema.safeParse({
      emotions: {
        amused: { threshold: -0.1, clip: "Happy" },
      },
    });
    expect(result.success).toBe(false);
  });

  it("rejects empty clip name", () => {
    const result = AvatarAnimationsSchema.safeParse({
      emotions: {
        amused: { threshold: 0.75, clip: "" },
      },
    });
    expect(result.success).toBe(false);
  });

  it("rejects missing clip field", () => {
    const result = AvatarAnimationsSchema.safeParse({
      emotions: {
        amused: { threshold: 0.75 },
      },
    });
    expect(result.success).toBe(false);
  });
});
