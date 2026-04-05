/**
 * NANO-059: Write-config round-trip tests for all 8 VLM scenarios.
 *
 * Tests the POST handler (write YAML) and GET handler (hydration) for:
 *   #1 Local LLM + VLM Disabled
 *   #2 Local LLM + Local VLM
 *   #3 Local LLM + Cloud VLM
 *   #4 Local LLM + Unified VLM
 *   #5 Cloud LLM + VLM Disabled
 *   #6 Cloud LLM + Local VLM
 *   #7 Cloud LLM + Cloud VLM
 *   #8 Cloud LLM + Unified VLM
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import yaml from "yaml";

// ============================================
// Mock fs before importing route handlers
// ============================================
let writtenYaml = "";
let mockFileExists = false;
let mockFileContent = "";

vi.mock("fs", () => ({
  existsSync: vi.fn(() => mockFileExists),
  readFileSync: vi.fn(() => mockFileContent),
  writeFileSync: vi.fn((_path: string, content: string) => {
    writtenYaml = content;
    // After writing, make the file "exist" for GET handler reads
    mockFileExists = true;
    mockFileContent = content;
  }),
}));

// Import after mocking
import { POST, GET } from "./route";

// ============================================
// Test Fixtures
// ============================================

const BASE_LLM_LOCAL = {
  executablePath: "C:\\llama\\llama-server.exe",
  modelPath: "C:\\models\\llm.gguf",
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
};

const BASE_LLM_CLOUD = {
  provider: "openai",
  apiUrl: "https://api.openai.com/v1",
  apiKey: "sk-test-key-123",
  model: "gpt-4o",
  contextSize: 32768,
  timeout: 60,
  temperature: 0.7,
  maxTokens: 256,
};

const BASE_VLM_LOCAL = {
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

const BASE_VLM_CLOUD = {
  apiKey: "sk-vlm-cloud-key",
  baseUrl: "https://api.xai.com/v1",
  model: "grok-2-vision",
  contextSize: 8192,
  timeout: 30,
  maxTokens: 300,
};

const BASE_STT_PARAKEET = {
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

const BASE_STT_WHISPER = {
  host: "127.0.0.1",
  port: 8080,
  timeout: 30,
  binaryPath: "whisper-server",
  modelPath: "C:\\models\\ggml-small.en.bin",
  language: "en",
  threads: 4,
  noGpu: false,
};

const BASE_TTS_LOCAL = {
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

function makeBody(overrides: Record<string, unknown> = {}) {
  return {
    llmProviderType: "local",
    llmLocal: BASE_LLM_LOCAL,
    llmCloud: BASE_LLM_CLOUD,
    vlmEnabled: true,
    useLLMForVision: false,
    vlmProviderType: "local",
    vlmLocal: BASE_VLM_LOCAL,
    vlmCloud: BASE_VLM_CLOUD,
    sttEnabled: true,
    sttProvider: "parakeet",
    sttParakeet: BASE_STT_PARAKEET,
    sttWhisper: BASE_STT_WHISPER,
    ttsEnabled: true,
    ttsProviderType: "local",
    ttsLocal: BASE_TTS_LOCAL,
    ...overrides,
  };
}

function makeRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost:3000/api/launcher/write-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function parseWrittenYaml(): Record<string, unknown> {
  return yaml.parse(writtenYaml);
}

// ============================================
// Helper: validate service entries have required fields
// ============================================
function assertServiceValid(parsed: Record<string, unknown>, serviceName: string) {
  const services = (parsed.launcher as Record<string, unknown>)?.services as Record<string, unknown> | undefined;
  const service = services?.[serviceName] as Record<string, unknown> | undefined;
  if (service) {
    expect(service).toHaveProperty("platform");
    expect(service).toHaveProperty("health_check");
  }
}

function assertNoService(parsed: Record<string, unknown>, serviceName: string) {
  const services = (parsed.launcher as Record<string, unknown>)?.services as Record<string, unknown> | undefined;
  expect(services?.[serviceName]).toBeUndefined();
}

// ============================================
// Tests
// ============================================

describe("write-config POST handler — 8 VLM scenarios", () => {
  beforeEach(() => {
    writtenYaml = "";
    mockFileExists = false;
    mockFileContent = "";
  });

  // ----------------------------------------
  // Scenario #1: Local LLM + VLM Disabled
  // ----------------------------------------
  it("Scenario #1: Local LLM + VLM Disabled — vlm section preserved with provider: none, no vlm service, no screen_vision", async () => {
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: false,
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section preserved but provider set to "none"
    expect(parsed.vlm).toBeDefined();
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("none");

    // No launcher.services.vlm entry
    assertNoService(parsed, "vlm");

    // LLM service should exist and be valid
    assertServiceValid(parsed, "llm");
  });

  // ----------------------------------------
  // Scenario #2: Local LLM + Local VLM (WORKS — verify it stays working)
  // ----------------------------------------
  it("Scenario #2: Local LLM + Local VLM — dedicated local server", async () => {
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "local",
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section exists with provider: llama
    expect(parsed.vlm).toBeDefined();
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("llama");
    expect((vlm.providers as Record<string, unknown>)?.llama).toBeDefined();

    // launcher.services.vlm present with platform + health_check
    assertServiceValid(parsed, "vlm");
    const services = (parsed.launcher as Record<string, unknown>).services as Record<string, unknown>;
    const vlmService = services.vlm as Record<string, unknown>;
    expect(vlmService.enabled).toBe(true);
    expect(vlmService.platform).toBe("native");

    // tools.tools.screen_vision.vlm_provider = "llama"
    const tools = parsed.tools as Record<string, unknown> | undefined;
    const toolsTools = tools?.tools as Record<string, unknown> | undefined;
    const screenVision = toolsTools?.screen_vision as Record<string, unknown> | undefined;
    expect(screenVision?.vlm_provider).toBe("llama");
  });

  // ----------------------------------------
  // Scenario #3: Local LLM + Cloud VLM
  // ----------------------------------------
  it("Scenario #3: Local LLM + Cloud VLM — no vlm service, cloud credentials", async () => {
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "cloud",
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section exists with provider: openai
    expect(parsed.vlm).toBeDefined();
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("openai");
    const providers = vlm.providers as Record<string, unknown>;
    const openai = providers.openai as Record<string, unknown>;
    expect(openai.api_key).toBe("sk-vlm-cloud-key");
    expect(openai.model).toBe("grok-2-vision");

    // NO launcher.services.vlm entry
    assertNoService(parsed, "vlm");

    // tools.tools.screen_vision.vlm_provider = "openai"
    const tools = parsed.tools as Record<string, unknown>;
    const toolsTools = tools.tools as Record<string, unknown>;
    const screenVision = toolsTools.screen_vision as Record<string, unknown>;
    expect(screenVision.vlm_provider).toBe("openai");
  });

  // ----------------------------------------
  // Scenario #4: Local LLM + Unified VLM
  // ----------------------------------------
  it("Scenario #4: Local LLM + Unified VLM — provider: llm, screen_vision: llm", async () => {
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: true,
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section exists with provider: "llm" (NANO-030 routing value)
    expect(parsed.vlm).toBeDefined();
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("llm");

    // NO launcher.services.vlm entry
    assertNoService(parsed, "vlm");

    // tools.tools.screen_vision.vlm_provider = "llm"
    const tools = parsed.tools as Record<string, unknown>;
    const toolsTools = tools.tools as Record<string, unknown>;
    const screenVision = toolsTools.screen_vision as Record<string, unknown>;
    expect(screenVision.vlm_provider).toBe("llm");

    // NANO-084: mmproj absent when not provided (BASE_LLM_LOCAL.mmprojPath is "")
    const llm = parsed.llm as Record<string, unknown>;
    const llamaProvider = (llm.providers as Record<string, unknown>).llama as Record<string, unknown>;
    expect(llamaProvider.mmproj_path).toBeUndefined();
  });

  // ----------------------------------------
  // Scenario #5: Cloud LLM + VLM Disabled
  // ----------------------------------------
  it("Scenario #5: Cloud LLM + VLM Disabled — vlm section preserved with provider: none, no vlm service", async () => {
    const body = makeBody({
      llmProviderType: "cloud",
      vlmEnabled: false,
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section preserved but provider set to "none"
    expect(parsed.vlm).toBeDefined();
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("none");

    // No launcher.services.vlm entry
    assertNoService(parsed, "vlm");

    // LLM should be cloud provider
    const llm = parsed.llm as Record<string, unknown>;
    expect(llm.provider).toBe("openai");
  });

  // ----------------------------------------
  // Scenario #6: Cloud LLM + Local VLM (WORKS — verify it stays working)
  // ----------------------------------------
  it("Scenario #6: Cloud LLM + Local VLM — dedicated local vision server", async () => {
    const body = makeBody({
      llmProviderType: "cloud",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "local",
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section with llama provider
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("llama");

    // launcher.services.vlm present and valid
    assertServiceValid(parsed, "vlm");

    // LLM is cloud
    const llm = parsed.llm as Record<string, unknown>;
    expect(llm.provider).toBe("openai");
  });

  // ----------------------------------------
  // Scenario #7: Cloud LLM + Cloud VLM
  // ----------------------------------------
  it("Scenario #7: Cloud LLM + Cloud VLM — no local servers for either", async () => {
    const body = makeBody({
      llmProviderType: "cloud",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "cloud",
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section with openai provider
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("openai");

    // NO launcher.services.vlm entry
    assertNoService(parsed, "vlm");

    // Both LLM and VLM are cloud
    const llm = parsed.llm as Record<string, unknown>;
    expect(llm.provider).toBe("openai");
  });

  // ----------------------------------------
  // Scenario #8: Cloud LLM + Unified VLM
  // ----------------------------------------
  it("Scenario #8: Cloud LLM + Unified VLM — cloud LLM handles vision", async () => {
    const body = makeBody({
      llmProviderType: "cloud",
      vlmEnabled: true,
      useLLMForVision: true,
    });
    const res = await POST(makeRequest(body));
    const json = await res.json();
    expect(json.success).toBe(true);

    const parsed = parseWrittenYaml();

    // vlm section with provider: "llm"
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("llm");

    // NO launcher.services.vlm entry
    assertNoService(parsed, "vlm");

    // tools.tools.screen_vision.vlm_provider = "llm"
    const tools = parsed.tools as Record<string, unknown>;
    const toolsTools = tools.tools as Record<string, unknown>;
    const screenVision = toolsTools.screen_vision as Record<string, unknown>;
    expect(screenVision.vlm_provider).toBe("llm");

    // LLM is cloud
    const llm = parsed.llm as Record<string, unknown>;
    expect(llm.provider).toBe("openai");
  });
});

describe("write-config POST handler — service entry validation", () => {
  beforeEach(() => {
    writtenYaml = "";
    mockFileExists = false;
    mockFileContent = "";
  });

  it("all service entries in launcher.services have platform and health_check", async () => {
    // Test with the most complex scenario (local everything)
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "local",
      ttsProviderType: "local",
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const services = (parsed.launcher as Record<string, unknown>).services as Record<string, unknown>;

    for (const [name, config] of Object.entries(services)) {
      const svc = config as Record<string, unknown>;
      expect(svc.platform, `service '${name}' missing platform`).toBeDefined();
      expect(svc.health_check, `service '${name}' missing health_check`).toBeDefined();
    }
  });

  it("cloud TTS preserves enabled flag in launcher.services.tts (NANO-112)", async () => {
    const body = makeBody({
      ttsProviderType: "cloud",
      ttsEnabled: true,
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const services = (parsed.launcher as Record<string, unknown>)?.services as Record<string, unknown>;
    const tts = services?.tts as Record<string, unknown>;
    expect(tts).toBeDefined();
    expect(tts.enabled).toBe(true);
  });

  it("cloud TTS disabled writes enabled: false (NANO-112)", async () => {
    const body = makeBody({
      ttsProviderType: "cloud",
      ttsEnabled: false,
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const services = (parsed.launcher as Record<string, unknown>)?.services as Record<string, unknown>;
    const tts = services?.tts as Record<string, unknown>;
    expect(tts).toBeDefined();
    expect(tts.enabled).toBe(false);
  });
});

describe("write-config GET handler — hydration round-trip", () => {
  beforeEach(() => {
    writtenYaml = "";
    mockFileExists = false;
    mockFileContent = "";
  });

  it("Scenario #1 round-trip: VLM disabled hydrates vlmEnabled=false", async () => {
    // Write scenario #1
    const body = makeBody({ llmProviderType: "local", vlmEnabled: false });
    await POST(makeRequest(body));

    // Read it back
    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.exists).toBe(true);
    expect(getJson.config.vlmEnabled).toBe(false);
  });

  it("Scenario #4 round-trip: Unified VLM hydrates useLLMForVision=true", async () => {
    // Write scenario #4
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: true,
    });
    await POST(makeRequest(body));

    // Read it back
    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.vlmEnabled).toBe(true);
    expect(getJson.config.useLLMForVision).toBe(true);
  });

  it("Scenario #2 round-trip: Local VLM hydrates correctly", async () => {
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "local",
    });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.vlmEnabled).toBe(true);
    expect(getJson.config.useLLMForVision).toBe(false);
    expect(getJson.config.vlmProviderType).toBe("local");
    expect(getJson.config.vlmLocal.executablePath).toBe(BASE_VLM_LOCAL.executablePath);
  });

  it("Scenario #3 round-trip: Cloud VLM hydrates correctly", async () => {
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "cloud",
    });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.vlmEnabled).toBe(true);
    expect(getJson.config.useLLMForVision).toBe(false);
    expect(getJson.config.vlmCloud.apiKey).toBe(BASE_VLM_CLOUD.apiKey);
  });

  it("Scenario #8 round-trip: Cloud LLM + Unified VLM hydrates correctly", async () => {
    const body = makeBody({
      llmProviderType: "cloud",
      vlmEnabled: true,
      useLLMForVision: true,
    });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.vlmEnabled).toBe(true);
    expect(getJson.config.useLLMForVision).toBe(true);
    expect(getJson.config.llmProviderType).toBe("cloud");
  });

  it("no config file returns exists=false", async () => {
    mockFileExists = false;
    const getRes = await GET();
    const getJson = await getRes.json();
    expect(getJson.exists).toBe(false);
  });
});

describe("write-config GET handler — NANO-063: savedProviderKeys", () => {
  beforeEach(() => {
    writtenYaml = "";
    mockFileExists = false;
    mockFileContent = "";
  });

  it("emits saved API keys for all cloud providers after sequential writes", async () => {
    // Write openai config first
    const body1 = makeBody({
      llmProviderType: "cloud",
      llmCloud: { ...BASE_LLM_CLOUD, provider: "openai", apiKey: "sk-openai-key" },
    });
    await POST(makeRequest(body1));

    // Write deepseek config on top (openai preserved via provider spread)
    const body2 = makeBody({
      llmProviderType: "cloud",
      llmCloud: { ...BASE_LLM_CLOUD, provider: "deepseek", apiKey: "sk-deepseek-key", apiUrl: "https://api.deepseek.com/v1", model: "deepseek-chat" },
    });
    await POST(makeRequest(body2));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.savedProviderKeys).toBeDefined();
    expect(getJson.config.savedProviderKeys.deepseek).toBe("sk-deepseek-key");
    expect(getJson.config.savedProviderKeys.openai).toBe("sk-openai-key");
  });

  it("excludes llama (local provider) from savedProviderKeys", async () => {
    const body = makeBody({ llmProviderType: "local" });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.savedProviderKeys).toBeDefined();
    expect(getJson.config.savedProviderKeys.llama).toBeUndefined();
  });

  it("returns empty savedProviderKeys when only local provider is configured", async () => {
    const body = makeBody({ llmProviderType: "local" });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(Object.keys(getJson.config.savedProviderKeys)).toHaveLength(0);
  });

  it("preserves env var placeholder strings in savedProviderKeys", async () => {
    const body = makeBody({
      llmProviderType: "cloud",
      llmCloud: { ...BASE_LLM_CLOUD, provider: "deepseek", apiKey: "${DEEPSEEK_API_KEY}" },
    });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.savedProviderKeys.deepseek).toBe("${DEEPSEEK_API_KEY}");
  });
});

describe("write-config — NANO-084: mmproj on LLM card for unified vision mode", () => {
  beforeEach(() => {
    writtenYaml = "";
    mockFileExists = false;
    mockFileContent = "";
  });

  it("Local LLM + Unified VLM + mmprojPath set — writes mmproj_path under llm.providers.llama", async () => {
    const body = makeBody({
      llmProviderType: "local",
      llmLocal: { ...BASE_LLM_LOCAL, mmprojPath: "C:\\models\\mmproj.gguf" },
      vlmEnabled: true,
      useLLMForVision: true,
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const llm = parsed.llm as Record<string, unknown>;
    const llamaProvider = (llm.providers as Record<string, unknown>).llama as Record<string, unknown>;
    expect(llamaProvider.mmproj_path).toBe("C:\\models\\mmproj.gguf");
  });

  it("Local LLM + Unified VLM + mmprojPath empty — mmproj_path key absent from YAML", async () => {
    const body = makeBody({
      llmProviderType: "local",
      llmLocal: { ...BASE_LLM_LOCAL, mmprojPath: "" },
      vlmEnabled: true,
      useLLMForVision: true,
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const llm = parsed.llm as Record<string, unknown>;
    const llamaProvider = (llm.providers as Record<string, unknown>).llama as Record<string, unknown>;
    expect(llamaProvider.mmproj_path).toBeUndefined();
  });

  it("Local LLM + Dedicated VLM — llmLocal.mmprojPath empty (GUI hides field), mmproj_path absent from llm.providers.llama", async () => {
    // In dedicated VLM mode, the mmproj field on the LLM card is hidden by the GUI.
    // llmLocal.mmprojPath arrives empty; verify it is not written to the LLM provider block.
    const body = makeBody({
      llmProviderType: "local",
      llmLocal: { ...BASE_LLM_LOCAL, mmprojPath: "" },
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "local",
      vlmLocal: { ...BASE_VLM_LOCAL, mmprojPath: "C:\\models\\vlm-mmproj.gguf" },
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const llm = parsed.llm as Record<string, unknown>;
    const llamaProvider = (llm.providers as Record<string, unknown>).llama as Record<string, unknown>;
    expect(llamaProvider.mmproj_path).toBeUndefined();
  });

  it("Hydration round-trip: write mmproj on LLM unified config, GET returns mmprojPath populated", async () => {
    const body = makeBody({
      llmProviderType: "local",
      llmLocal: { ...BASE_LLM_LOCAL, mmprojPath: "C:\\models\\mmproj.gguf" },
      vlmEnabled: true,
      useLLMForVision: true,
    });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.llmLocal.mmprojPath).toBe("C:\\models\\mmproj.gguf");
  });
});

describe("write-config — NANO-085: TTS device selector", () => {
  beforeEach(() => {
    writtenYaml = "";
    mockFileExists = false;
    mockFileContent = "";
  });

  it("TTS device set to cpu — writes device: cpu under tts.providers.kokoro", async () => {
    const body = makeBody({
      ttsLocal: { ...BASE_TTS_LOCAL, device: "cpu" },
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const tts = parsed.tts as Record<string, unknown>;
    const kokoroProvider = (tts.providers as Record<string, unknown>).kokoro as Record<string, unknown>;
    expect(kokoroProvider.device).toBe("cpu");
  });

  it("TTS device set to cuda:1 — writes device: cuda:1 under tts.providers.kokoro", async () => {
    const body = makeBody({
      ttsLocal: { ...BASE_TTS_LOCAL, device: "cuda:1" },
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const tts = parsed.tts as Record<string, unknown>;
    const kokoroProvider = (tts.providers as Record<string, unknown>).kokoro as Record<string, unknown>;
    expect(kokoroProvider.device).toBe("cuda:1");
  });

  it("TTS device empty — device key absent from YAML", async () => {
    const body = makeBody({
      ttsLocal: { ...BASE_TTS_LOCAL, device: "" },
    });
    const res = await POST(makeRequest(body));
    expect((await res.json()).success).toBe(true);

    const parsed = parseWrittenYaml();
    const tts = parsed.tts as Record<string, unknown>;
    const kokoroProvider = (tts.providers as Record<string, unknown>).kokoro as Record<string, unknown>;
    expect(kokoroProvider.device).toBeUndefined();
  });

  it("Hydration round-trip: write device on TTS config, GET returns device populated", async () => {
    const body = makeBody({
      ttsLocal: { ...BASE_TTS_LOCAL, device: "cpu" },
    });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    expect(getJson.config.ttsLocal.device).toBe("cpu");
  });

  it("Hydration default: no device in YAML hydrates to cuda", async () => {
    const body = makeBody({
      ttsLocal: { ...BASE_TTS_LOCAL, device: "" },
    });
    await POST(makeRequest(body));

    const getRes = await GET();
    const getJson = await getRes.json();

    // Empty device → conditional spread omits key → GET hydrates with fallback "cuda"
    expect(getJson.config.ttsLocal.device).toBe("cuda");
  });
});

// ============================================
// NANO-089 Phase 3: Write-protection invariants
// ============================================
describe("write-config — NANO-089 Phase 3: write-protection invariants", () => {
  beforeEach(() => {
    writtenYaml = "";
    mockFileExists = false;
    mockFileContent = "";
  });

  it("VLM section present in written YAML when VLM disabled", async () => {
    const body = makeBody({ vlmEnabled: false });
    await POST(makeRequest(body));

    const parsed = parseWrittenYaml();
    expect(parsed.vlm).toBeDefined();
    expect(typeof parsed.vlm).toBe("object");
    expect((parsed.vlm as Record<string, unknown>).provider).toBe("none");
  });

  it("Existing VLM providers preserved when VLM disabled", async () => {
    // Pre-populate with existing VLM config containing multiple providers
    mockFileExists = true;
    mockFileContent = yaml.stringify({
      vlm: {
        provider: "llama",
        providers: {
          llama: { port: 5558, model_path: "/path/to/vlm.gguf" },
          openai: { api_key: "sk-existing", base_url: "https://api.xai.com/v1" },
        },
      },
    });

    const body = makeBody({ vlmEnabled: false });
    await POST(makeRequest(body));

    const parsed = parseWrittenYaml();
    const vlm = parsed.vlm as Record<string, unknown>;
    expect(vlm.provider).toBe("none");
    // Providers block must survive — "I don't want VLM right now" != "delete my configs"
    const providers = vlm.providers as Record<string, unknown>;
    expect(providers.llama).toBeDefined();
    expect(providers.openai).toBeDefined();
  });

  it("${ENV_VAR} patterns survive POST write in raw YAML", async () => {
    const body = makeBody({
      llmProviderType: "cloud",
      llmCloud: { ...BASE_LLM_CLOUD, provider: "deepseek", apiKey: "${DEEPSEEK_API_KEY}" },
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "cloud",
      vlmCloud: { ...BASE_VLM_CLOUD, apiKey: "${XAI_API_KEY}" },
    });
    await POST(makeRequest(body));

    // Check raw YAML string — env var patterns must survive yaml.stringify
    expect(writtenYaml).toContain("${DEEPSEEK_API_KEY}");
    expect(writtenYaml).toContain("${XAI_API_KEY}");
  });

  it("Full POST → GET roundtrip preserves all values", async () => {
    const body = makeBody({
      llmProviderType: "local",
      vlmEnabled: true,
      useLLMForVision: false,
      vlmProviderType: "local",
      sttProvider: "parakeet",
      ttsProviderType: "local",
    });
    const postRes = await POST(makeRequest(body));
    const postJson = await postRes.json();
    expect(postJson.success).toBe(true);

    const getRes = await GET();
    const getJson = await getRes.json();
    expect(getJson.exists).toBe(true);

    // LLM values roundtrip
    expect(getJson.config.llmProviderType).toBe("local");
    expect(getJson.config.llmLocal.port).toBe(BASE_LLM_LOCAL.port);
    expect(getJson.config.llmLocal.temperature).toBe(BASE_LLM_LOCAL.temperature);

    // VLM values roundtrip
    expect(getJson.config.vlmEnabled).toBe(true);
    expect(getJson.config.vlmProviderType).toBe("local");
    expect(getJson.config.vlmLocal.port).toBe(BASE_VLM_LOCAL.port);
    expect(getJson.config.vlmLocal.modelType).toBe(BASE_VLM_LOCAL.modelType);

    // STT values roundtrip
    expect(getJson.config.sttProvider).toBe("parakeet");
    expect(getJson.config.sttParakeet.port).toBe(BASE_STT_PARAKEET.port);

    // TTS values roundtrip
    expect(getJson.config.ttsLocal.port).toBe(BASE_TTS_LOCAL.port);
  });
});
