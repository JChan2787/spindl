import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";
import yaml, { Scalar } from "yaml";
import { CLOUD_PROVIDER_IDS } from "@/lib/constants/cloud-providers";
import {
  LauncherConfigSchema,
  type LauncherConfig,
  type STTWhisperConfig,
  type STTParakeetConfig,
} from "@/lib/schemas/launcher-schemas";

// Configuration file path relative to the gui directory (NANO-081: env var override for E2E)
const CONFIG_PATH = process.env.SPINDL_CONFIG || path.resolve(process.cwd(), "..", "config", "spindl.yaml");

// ============================================
// Utility Functions
// ============================================

// YAML reserved words that get coerced to non-string types
const YAML_RESERVED = new Set([
  "true", "false", "yes", "no", "on", "off", "null", "~",
  "True", "False", "Yes", "No", "On", "Off", "Null",
  "TRUE", "FALSE", "YES", "NO", "ON", "OFF", "NULL",
]);

function parseExtraArgs(argsString: string): string[] {
  if (!argsString.trim()) return [];
  const args: string[] = [];
  let current = "";
  let inQuote = false;
  let quoteChar = "";

  for (const char of argsString) {
    if ((char === '"' || char === "'") && !inQuote) {
      inQuote = true;
      quoteChar = char;
    } else if (char === quoteChar && inQuote) {
      inQuote = false;
      quoteChar = "";
    } else if (char === " " && !inQuote) {
      if (current) {
        args.push(current);
        current = "";
      }
    } else {
      current += char;
    }
  }
  if (current) args.push(current);
  return args;
}

/**
 * Wrap extra_args for safe YAML serialization.
 * Uses yaml Scalar with QUOTE_SINGLE for values that would otherwise
 * be coerced to booleans/null by YAML parsers.
 */
function safeExtraArgs(args: string[]): (string | Scalar)[] {
  return args.map((arg) => {
    if (YAML_RESERVED.has(arg)) {
      const scalar = new Scalar(arg);
      scalar.type = Scalar.QUOTE_SINGLE;
      return scalar;
    }
    return arg;
  });
}

function buildWhisperCommand(w: STTWhisperConfig): string {
  const parts = [w.binaryPath, "-m", w.modelPath, "--host", w.host, "--port", String(w.port), "-l", w.language, "-t", String(w.threads)];
  if (w.noGpu) parts.push("--no-gpu");
  return parts.join(" ");
}

function buildParakeetCommand(stt: STTParakeetConfig): string {
  const parts: string[] = [];

  // Native Windows + conda: use conda run (eval/source are bash-only)
  if (stt.envType === "conda" && stt.platform !== "wsl") {
    const script = stt.serverScriptPath || "stt/server/nemo_server.py";
    const cmdParts = [`conda run --live-stream -n ${stt.envNameOrPath} python -u ${script}`];
    if (stt.host) cmdParts[0] += ` --host ${stt.host}`;
    if (stt.port) cmdParts[0] += ` --port ${stt.port}`;
    return cmdParts[0];
  }

  // WSL or non-conda: build activation string (bash is available)
  let activation = "";
  switch (stt.envType) {
    case "conda":
      activation = `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate ${stt.envNameOrPath} &&`;
      break;
    case "venv":
      activation = `source ${stt.envNameOrPath}/bin/activate &&`;
      break;
    case "other":
      if (stt.customActivation) {
        activation = `${stt.customActivation} &&`;
      }
      break;
    case "system":
      break;
  }

  if (stt.platform === "wsl") {
    // Resolve relative script paths against the project root, then convert to WSL /mnt/ format.
    // Project root is derived from CONFIG_PATH (which is <project_root>/config/spindl.yaml).
    let scriptPath = stt.serverScriptPath.replace(/\\/g, "/");
    const isAbsolute = scriptPath.match(/^[A-Za-z]:\//) || scriptPath.startsWith("/");
    if (!isAbsolute) {
      const projectRoot = path.resolve(CONFIG_PATH, "..", "..").replace(/\\/g, "/");
      scriptPath = `${projectRoot}/${scriptPath}`;
    }
    // Convert Windows absolute path (C:/...) to WSL /mnt/ format
    const winMatch = scriptPath.match(/^([A-Za-z]):\/(.*)$/);
    if (winMatch) {
      scriptPath = `/mnt/${winMatch[1].toLowerCase()}/${winMatch[2]}`;
    }
    const lastSlash = scriptPath.lastIndexOf("/");
    const scriptDir = lastSlash > 0 ? scriptPath.substring(0, lastSlash) : ".";
    const scriptName = scriptPath.substring(lastSlash + 1);
    parts.push(
      `cd "${scriptDir}" && ${activation} python3 -u ${scriptName}`
    );
  } else {
    parts.push(`${activation} python3 -u ${stt.serverScriptPath}`.trim());
  }

  return parts.join(" ");
}

// ============================================
// POST Handler
// ============================================

export async function POST(request: Request) {
  try {
    // NANO-089 Phase 2: Validate request body before any YAML operations
    const rawBody = await request.json();
    const parseResult = LauncherConfigSchema.safeParse(rawBody);
    if (!parseResult.success) {
      return NextResponse.json(
        {
          success: false,
          error: `Validation failed: ${parseResult.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ")}`,
        },
        { status: 400 },
      );
    }
    const body: LauncherConfig = parseResult.data;

    // Read existing config
    let existingConfig: Record<string, unknown> = {};
    if (fs.existsSync(CONFIG_PATH)) {
      const existingContent = fs.readFileSync(CONFIG_PATH, "utf-8");
      existingConfig = yaml.parse(existingContent) || {};
    }

    // ========================================
    // LLM Configuration
    // ========================================
    if (body.llmProviderType === "local") {
      // Preserve existing llama keys not managed by the launcher (e.g. repeat_penalty from NANO-108)
      const existingLlmProviders = (typeof existingConfig.llm === "object" && existingConfig.llm !== null
        ? (existingConfig.llm as { providers?: Record<string, unknown> }).providers || {}
        : {}) as Record<string, unknown>;
      const existingLlama = (existingLlmProviders.llama || {}) as Record<string, unknown>;

      // Build llama config — spread existingLlama first, then explicitly
      // set every managed key. Keys cleared in the GUI (empty string) MUST
      // delete the zombie from existingLlama, not silently preserve it.
      const llmLlama: Record<string, unknown> = {
        ...existingLlama,
        executable_path: body.llmLocal.executablePath,
        model_path: body.llmLocal.modelPath,
        host: body.llmLocal.host,
        port: body.llmLocal.port,
        context_size: body.llmLocal.contextSize,
        gpu_layers: body.llmLocal.gpuLayers,
        timeout: body.llmLocal.timeout,
        temperature: body.llmLocal.temperature,
        max_tokens: body.llmLocal.maxTokens,
        top_p: body.llmLocal.topP,
      };

      // GPU selection — tensor_split takes precedence. Delete the other to prevent zombies.
      if (body.llmLocal.tensorSplit) {
        llmLlama.tensor_split = body.llmLocal.tensorSplit.split(",").map((v: string) => parseFloat(v.trim())).filter((v: number) => !isNaN(v));
        delete llmLlama.device;
      } else if (body.llmLocal.device) {
        llmLlama.device = body.llmLocal.device;
        delete llmLlama.tensor_split;
      } else {
        delete llmLlama.tensor_split;
        delete llmLlama.device;
      }

      // Extra args — empty string = user cleared it, delete the zombie
      if (body.llmLocal.extraArgs) {
        llmLlama.extra_args = safeExtraArgs(parseExtraArgs(body.llmLocal.extraArgs));
      } else {
        delete llmLlama.extra_args;
      }

      // NANO-042: Reasoning model config — empty = cleared
      if (body.llmLocal.reasoningFormat) {
        llmLlama.reasoning_format = body.llmLocal.reasoningFormat;
      } else {
        delete llmLlama.reasoning_format;
      }
      if (body.llmLocal.reasoningBudget !== undefined && body.llmLocal.reasoningBudget !== -1) {
        llmLlama.reasoning_budget = body.llmLocal.reasoningBudget;
      } else {
        delete llmLlama.reasoning_budget;
      }

      // NANO-084: mmproj for unified vision mode — empty = cleared, MUST delete zombie
      if (body.llmLocal.mmprojPath) {
        llmLlama.mmproj_path = body.llmLocal.mmprojPath;
      } else {
        delete llmLlama.mmproj_path;
      }

      existingConfig.llm = {
        provider: "llama",
        plugin_paths: ["./plugins/llm"],
        providers: {
          ...existingLlmProviders,
          llama: llmLlama,
        },
      };
    } else {
      const providerName = body.llmCloud.provider;
      // Preserve existing cloud provider keys not managed by the launcher
      const existingCloudProviders = (typeof existingConfig.llm === "object" && existingConfig.llm !== null
        ? (existingConfig.llm as { providers?: Record<string, unknown> }).providers || {}
        : {}) as Record<string, unknown>;
      const existingCloudProvider = (existingCloudProviders[providerName] || {}) as Record<string, unknown>;

      existingConfig.llm = {
        provider: providerName,
        plugin_paths: ["./plugins/llm"],
        providers: {
          ...existingCloudProviders,
          [providerName]: {
            ...existingCloudProvider,
            url: body.llmCloud.apiUrl,
            api_key: body.llmCloud.apiKey,
            model: body.llmCloud.model,
            context_size: body.llmCloud.contextSize,
            timeout: body.llmCloud.timeout,
            temperature: body.llmCloud.temperature,
            max_tokens: body.llmCloud.maxTokens,
          },
        },
      };
    }

    // ========================================
    // VLM Configuration (NANO-059)
    // ========================================
    if (!body.vlmEnabled) {
      // VLM disabled — preserve existing providers but mark as disabled.
      // "I don't want VLM right now" != "delete my VLM configs."
      if (typeof existingConfig.vlm === "object" && existingConfig.vlm !== null) {
        (existingConfig.vlm as Record<string, unknown>).provider = "none";
      } else {
        existingConfig.vlm = { provider: "none" };
      }

      // When VLM is disabled, strip mmproj_path from LLM llama config.
      // Unified VLM is off — a stale mmproj_path here causes llama-server
      // to load an mmproj for the wrong architecture (Session 606 bug).
      const llmSection = existingConfig.llm as Record<string, unknown> | undefined;
      if (llmSection && typeof llmSection === "object") {
        const llmProviders = llmSection.providers as Record<string, Record<string, unknown>> | undefined;
        if (llmProviders?.llama) {
          delete llmProviders.llama.mmproj_path;
        }
      }
    } else if (body.useLLMForVision) {
      // Unified: VLM uses LLM endpoint — provider must be "llm" (NANO-030 routing value)
      existingConfig.vlm = {
        provider: "llm",
        plugin_paths: ["./plugins/vlm"],
        capture: {
          monitor: 1,
          width: 1920,
          height: 1080,
          jpeg_quality: 95,
        },
        providers: {
          ...(typeof existingConfig.vlm === "object" && existingConfig.vlm !== null
            ? (existingConfig.vlm as { providers?: Record<string, unknown> }).providers || {}
            : {}),
        },
      };
    } else {
      // Separate VLM configuration
      if (body.vlmProviderType === "local") {
        // Preserve existing VLM llama keys not managed by the launcher
        const existingVlmProviders = (typeof existingConfig.vlm === "object" && existingConfig.vlm !== null
          ? (existingConfig.vlm as { providers?: Record<string, unknown> }).providers || {}
          : {}) as Record<string, unknown>;
        const existingVlmLlama = (existingVlmProviders.llama || {}) as Record<string, unknown>;

        // Build VLM llama config — same zombie-killing pattern as LLM.
        const vlmLlama: Record<string, unknown> = {
          ...existingVlmLlama,
          model_type: body.vlmLocal.modelType,
          executable_path: body.vlmLocal.executablePath,
          model_path: body.vlmLocal.modelPath,
          host: body.vlmLocal.host,
          port: body.vlmLocal.port,
          context_size: body.vlmLocal.contextSize,
          gpu_layers: body.vlmLocal.gpuLayers,
          timeout: body.vlmLocal.timeout,
          max_tokens: body.vlmLocal.maxTokens,
          prompt: "Describe what you see in this image concisely.",
        };

        // mmproj — empty = cleared, MUST delete zombie
        if (body.vlmLocal.mmprojPath) {
          vlmLlama.mmproj_path = body.vlmLocal.mmprojPath;
        } else {
          delete vlmLlama.mmproj_path;
        }

        // GPU selection — tensor_split takes precedence
        if (body.vlmLocal.tensorSplit) {
          vlmLlama.tensor_split = body.vlmLocal.tensorSplit.split(",").map((v: string) => parseFloat(v.trim())).filter((v: number) => !isNaN(v));
          delete vlmLlama.device;
        } else if (body.vlmLocal.device) {
          vlmLlama.device = body.vlmLocal.device;
          delete vlmLlama.tensor_split;
        } else {
          delete vlmLlama.tensor_split;
          delete vlmLlama.device;
        }

        // Extra args — empty = cleared
        if (body.vlmLocal.extraArgs) {
          vlmLlama.extra_args = safeExtraArgs(parseExtraArgs(body.vlmLocal.extraArgs));
        } else {
          delete vlmLlama.extra_args;
        }

        existingConfig.vlm = {
          provider: "llama",
          plugin_paths: ["./plugins/vlm"],
          capture: {
            monitor: 1,
            width: 1920,
            height: 1080,
            jpeg_quality: 95,
          },
          providers: {
            ...existingVlmProviders,
            llama: vlmLlama,
          },
        };
      } else {
        // Preserve existing VLM cloud keys not managed by the launcher
        const existingVlmProviders = (typeof existingConfig.vlm === "object" && existingConfig.vlm !== null
          ? (existingConfig.vlm as { providers?: Record<string, unknown> }).providers || {}
          : {}) as Record<string, unknown>;
        const existingVlmCloud = (existingVlmProviders.openai || {}) as Record<string, unknown>;

        existingConfig.vlm = {
          provider: "openai",
          plugin_paths: ["./plugins/vlm"],
          capture: {
            monitor: 1,
            width: 1920,
            height: 1080,
            jpeg_quality: 95,
          },
          providers: {
            ...existingVlmProviders,
            openai: {
              ...existingVlmCloud,
              api_key: body.vlmCloud.apiKey,
              base_url: body.vlmCloud.baseUrl,
              model: body.vlmCloud.model,
              context_size: body.vlmCloud.contextSize,
              timeout: body.vlmCloud.timeout,
              max_tokens: body.vlmCloud.maxTokens,
              prompt: "Describe what you see in this image concisely.",
            },
          },
        };
      }
    }

    // NANO-089 Phase 3: VLM section must always exist after config mutation.
    // Safety net — the branches above handle this, but future changes must not break it.
    if (!existingConfig.vlm || typeof existingConfig.vlm !== "object") {
      existingConfig.vlm = { provider: "none" };
    }

    // ========================================
    // Tools Configuration (NANO-059: screen_vision routing)
    // ========================================
    if (body.vlmEnabled) {
      if (!existingConfig.tools) existingConfig.tools = {};
      const tools = existingConfig.tools as Record<string, unknown>;
      // VLM requires the tool system — ensure tools are enabled at launch
      tools.enabled = true;
      if (!tools.tools) tools.tools = {};
      const toolsTools = tools.tools as Record<string, unknown>;
      const screenVision = (toolsTools.screen_vision || {}) as Record<string, unknown>;

      if (body.useLLMForVision) {
        screenVision.vlm_provider = "llm";
      } else {
        screenVision.vlm_provider = body.vlmProviderType === "local" ? "llama" : "openai";
      }
      toolsTools.screen_vision = screenVision;
    }

    // ========================================
    // STT Configuration — always write BOTH providers to preserve configs
    // ========================================
    existingConfig.stt = {
      provider: body.sttProvider,
      plugin_paths: [],
      providers: {
        parakeet: {
          host: body.sttParakeet.host,
          port: body.sttParakeet.port,
          timeout: body.sttParakeet.timeout,
          server_script: body.sttParakeet.serverScriptPath || undefined,
          conda_env: (body.sttParakeet.envType === "conda" && body.sttParakeet.envNameOrPath)
            ? body.sttParakeet.envNameOrPath : undefined,
        },
        whisper: {
          host: body.sttWhisper.host,
          port: body.sttWhisper.port,
          timeout: body.sttWhisper.timeout,
          binary_path: body.sttWhisper.binaryPath,
          model_path: body.sttWhisper.modelPath,
          language: body.sttWhisper.language,
          threads: body.sttWhisper.threads,
          no_gpu: body.sttWhisper.noGpu,
        },
      },
    };

    // ========================================
    // TTS Configuration
    // ========================================
    if (body.ttsProviderType === "local") {
      existingConfig.tts = {
        provider: body.ttsLocal.provider,
        plugin_paths: ["./plugins/tts"],
        providers: {
          ...(typeof existingConfig.tts === "object" && existingConfig.tts !== null
            ? (existingConfig.tts as { providers?: Record<string, unknown> }).providers || {}
            : {}),
          [body.ttsLocal.provider]: {
            host: body.ttsLocal.host,
            port: body.ttsLocal.port,
            voice: body.ttsLocal.voice,
            language: body.ttsLocal.language,
            models_dir: body.ttsLocal.modelsDirectory,
            timeout: body.ttsLocal.timeout,
            ...(body.ttsLocal.device ? { device: body.ttsLocal.device } : {}),
            ...(body.ttsLocal.envType === "conda" && body.ttsLocal.envNameOrPath
              ? { conda_env: body.ttsLocal.envNameOrPath }
              : {}),
          },
        },
      };
    }
    // Cloud TTS is a placeholder for now - no write logic

    // ========================================
    // Launcher Service Configuration
    // ========================================
    if (!existingConfig.launcher) {
      existingConfig.launcher = {
        log_file: "./logs/launcher.log",
        log_level: "info",
        health_check_timeout: 60,
        services: {},
      };
    }

    const launcher = existingConfig.launcher as {
      services?: Record<string, unknown>;
      logging?: unknown;
    };
    if (!launcher.services) {
      launcher.services = {};
    }

    // LLM service config
    launcher.services.llm = {
      enabled: true,
      platform: "native",
      health_check: {
        type: "provider",
        timeout: 90,
      },
    };

    // VLM service config (NANO-059: only write entry for dedicated local VLM)
    if (body.vlmEnabled && !body.useLLMForVision && body.vlmProviderType === "local") {
      launcher.services.vlm = {
        enabled: true,
        platform: "native",
        health_check: {
          type: "provider",
          timeout: 60,
        },
      };
    } else {
      // Disabled, unified, or cloud — no local VLM server to launch. Omit entry entirely.
      delete launcher.services.vlm;
    }

    // STT service config — always regenerate command from GUI fields
    // NANO-112: Respect user's enabled/disabled choice from GUI toggle
    const activeSTT = body.sttProvider === "parakeet" ? body.sttParakeet : body.sttWhisper;
    const sttCommand = body.sttProvider === "whisper"
      ? buildWhisperCommand(body.sttWhisper)
      : buildParakeetCommand(body.sttParakeet);

    launcher.services.stt = {
      enabled: body.sttEnabled,
      platform: body.sttProvider === "parakeet" ? body.sttParakeet.platform : "native",
      ...(body.sttProvider === "parakeet" && body.sttParakeet.platform === "wsl" ? { wsl_distro: body.sttParakeet.wslDistro } : {}),
      command: sttCommand,
      health_check: {
        type: "tcp",
        host: activeSTT.host,
        port: activeSTT.port,
        timeout: 60,
      },
    };

    // TTS service config
    // NANO-112: Respect user's enabled/disabled choice from GUI toggle
    if (body.ttsProviderType === "local") {
      launcher.services.tts = {
        enabled: body.ttsEnabled,
        platform: "native",
        health_check: {
          type: "provider",
          timeout: 60,
        },
      };
    } else if (body.ttsEnabled) {
      // Cloud TTS (enabled) — no local server, but preserve the enabled flag
      launcher.services.tts = {
        enabled: true,
        platform: "native",
        health_check: {
          type: "provider",
          timeout: 60,
        },
      };
    } else {
      // Cloud TTS disabled — preserve disabled state
      launcher.services.tts = {
        enabled: false,
        platform: "native",
        health_check: {
          type: "none",
          timeout: 60,
        },
      };
    }

    // ========================================
    // Embedding Server Configuration (NANO-043 Phase 5)
    // ========================================
    if (body.embedding?.enabled) {
      const emb = body.embedding;
      const embeddingCommand = `${emb.executablePath} --embedding -m ${emb.modelPath} --host ${emb.host} --port ${emb.port} -ngl ${emb.gpuLayers} -c ${emb.contextSize} -b ${emb.contextSize} -ub ${emb.contextSize} --split-mode none`;

      launcher.services.embedding = {
        enabled: true,
        platform: "native",
        command: embeddingCommand,
        health_check: {
          type: "http",
          url: `http://${emb.host}:${emb.port}/health`,
          timeout: emb.timeout,
        },
      };

      // Dual-write: update memory.embedding so the orchestrator picks up the correct URL
      if (!existingConfig.memory) {
        existingConfig.memory = { enabled: true };
      }
      const memory = existingConfig.memory as Record<string, unknown>;
      memory.enabled = true;
      if (!memory.embedding || typeof memory.embedding !== "object") {
        memory.embedding = {};
      }
      (memory.embedding as Record<string, unknown>).base_url = `http://${emb.host}:${emb.port}`;
      (memory.embedding as Record<string, unknown>).timeout = emb.timeout;
      memory.relevance_threshold = emb.relevanceThreshold;
      memory.top_k = emb.topK;
    } else {
      // Embedding disabled — write disabled entry
      launcher.services.embedding = {
        enabled: false,
        platform: "native",
        command: "",
        health_check: { type: "none" },
      };
    }

    // ========================================
    // Write Config
    // ========================================
    const yamlContent = yaml.stringify(existingConfig, {
      indent: 2,
      lineWidth: 0,
    });

    fs.writeFileSync(CONFIG_PATH, yamlContent, "utf-8");

    return NextResponse.json({
      success: true,
      message: "Configuration saved successfully",
      configPath: CONFIG_PATH,
    });
  } catch (error) {
    console.error("Error writing config:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

// ============================================
// GET Handler - Returns full config for hydration
// ============================================

function arrayToString(arr: unknown): string {
  if (!arr) return "";
  if (Array.isArray(arr)) return arr.join(" ");
  return String(arr);
}

interface ParsedSTTCommand {
  serverScriptPath: string;
  envType: "conda" | "venv" | "system" | "other";
  envNameOrPath: string;
  customActivation: string;
}

/**
 * Parse STT command string to extract script path and environment info.
 *
 * Expected patterns:
 * - cd "..." && eval "$(conda shell.bash hook ...)" && conda activate {env} && python3 -u {script}
 * - source {venv}/bin/activate && python3 -u {script}
 * - python3 -u {script}
 */
function parseSTTCommand(command: string): ParsedSTTCommand {
  const result: ParsedSTTCommand = {
    serverScriptPath: "",
    envType: "system",
    envNameOrPath: "",
    customActivation: "",
  };

  if (!command) return result;

  // Extract directory from cd command
  let directory = "";
  const cdMatch = command.match(/cd\s+"([^"]+)"/);
  if (cdMatch) {
    directory = cdMatch[1];
  }

  // Check for conda activation
  const condaMatch = command.match(/conda\s+activate\s+(\S+)/);
  if (condaMatch) {
    result.envType = "conda";
    result.envNameOrPath = condaMatch[1];
  }

  // Check for venv activation
  const venvMatch = command.match(/source\s+([^\s]+)\/bin\/activate/);
  if (venvMatch && !command.includes("conda shell.bash hook") && !command.includes("conda activate")) {
    result.envType = "venv";
    result.envNameOrPath = venvMatch[1];
  }

  // Extract script name from python command
  const pythonMatch = command.match(/python3?\s+-u\s+(\S+)/);
  if (pythonMatch) {
    const scriptName = pythonMatch[1];
    if (directory) {
      // Combine directory with script name
      // Handle WSL paths - convert /mnt/c/... to C:/...
      let fullPath = `${directory}/${scriptName}`;
      if (fullPath.startsWith("/mnt/")) {
        // Convert /mnt/c/path to C:/path
        const wslMatch = fullPath.match(/^\/mnt\/([a-z])\/(.*)$/);
        if (wslMatch) {
          fullPath = `${wslMatch[1].toUpperCase()}:/${wslMatch[2]}`;
        }
      }
      result.serverScriptPath = fullPath;
    } else {
      result.serverScriptPath = scriptName;
    }
  }

  return result;
}

export async function GET() {
  try {
    const exists = fs.existsSync(CONFIG_PATH);

    if (!exists) {
      return NextResponse.json({
        exists: false,
        configPath: CONFIG_PATH,
        config: null,
      });
    }

    const content = fs.readFileSync(CONFIG_PATH, "utf-8");
    const raw = yaml.parse(content);

    // Transform yaml config to launcher store format
    const llmProvider = raw?.llm?.provider;
    const llmIsLocal = llmProvider === "llama";

    // Always read from llama provider config for local fields, regardless of active provider
    // This mirrors the cloud hydration pattern below — both sides preserve their fields
    const llmLocalConfig = raw?.llm?.providers?.llama || {};

    // For cloud config, read from the first available cloud provider
    // NANO-063: Registry-driven fallback (no hardcoded provider names)
    const llmCloudProvider = llmIsLocal
      ? (CLOUD_PROVIDER_IDS.find(id => raw?.llm?.providers?.[id]) ?? "deepseek")
      : llmProvider;
    const llmCloudConfig = raw?.llm?.providers?.[llmCloudProvider] || {};

    // NANO-063: Collect saved API keys for all cloud providers
    const llmProviders = raw?.llm?.providers || {};
    const savedProviderKeys: Record<string, string> = {};
    for (const [name, cfg] of Object.entries(llmProviders)) {
      if (name === "llama") continue;
      const providerCfg = cfg as Record<string, unknown>;
      if (providerCfg?.api_key) {
        savedProviderKeys[name] = String(providerCfg.api_key);
      }
    }

    // NANO-059: VLM enabled = vlm section exists with a real provider (not "none")
    const vlmSection = raw?.vlm;
    const vlmProvider = vlmSection?.provider;
    const vlmEnabled = vlmSection != null && vlmProvider !== "none";
    const vlmIsLocal = vlmProvider === "llama";

    // Always read from llama provider config for local VLM fields, regardless of active provider
    const vlmLocalConfig = vlmSection?.providers?.llama || {};

    // For VLM cloud config, read from openai provider regardless of current provider
    const vlmCloudConfig = vlmSection?.providers?.openai || {};

    const ttsProvider = raw?.tts?.provider;
    const ttsConfig = raw?.tts?.providers?.[ttsProvider] || {};

    // Parse launcher.services for environment info
    const sttLauncher = raw?.launcher?.services?.stt || {};
    const ttsLauncher = raw?.launcher?.services?.tts || {};

    // Parse embedding server config (NANO-043 Phase 5)
    const embeddingLauncher = raw?.launcher?.services?.embedding || {};
    const memoryEmbedding = raw?.memory?.embedding || {};
    const embeddingCommand: string = embeddingLauncher.command || "";
    const embExeMatch = embeddingCommand.match(/^"?([^"]+?)"?\s+--embedding/);
    const embModelMatch = embeddingCommand.match(/-m\s+"?([^"]+?)"?\s+--host/);
    const embHostMatch = embeddingCommand.match(/--host\s+(\S+)/);
    const embPortMatch = embeddingCommand.match(/--port\s+(\d+)/);
    const embNglMatch = embeddingCommand.match(/-ngl\s+(\d+)/);
    const embCtxMatch = embeddingCommand.match(/-c\s+(\d+)/);
    // Fallback to memory.embedding.base_url for host/port
    const embBaseUrl: string = memoryEmbedding.base_url || "http://127.0.0.1:5559";
    const embUrlParts = embBaseUrl.match(/\/\/([^:]+):(\d+)/);

    const config = {
      // LLM
      llmProviderType: llmIsLocal ? "local" : "cloud",
      llmLocal: {
        executablePath: llmLocalConfig.executable_path || "",
        modelPath: llmLocalConfig.model_path || "",
        host: llmLocalConfig.host || "127.0.0.1",
        port: llmLocalConfig.port || 5557,
        contextSize: llmLocalConfig.context_size || 8192,
        gpuLayers: llmLocalConfig.gpu_layers ?? 99,
        device: llmLocalConfig.device || "",
        tensorSplit: Array.isArray(llmLocalConfig.tensor_split) ? llmLocalConfig.tensor_split.join(",") : "",
        extraArgs: arrayToString(llmLocalConfig.extra_args),
        timeout: llmLocalConfig.timeout || 30,
        temperature: llmLocalConfig.temperature ?? 0.7,
        maxTokens: llmLocalConfig.max_tokens || 256,
        topP: llmLocalConfig.top_p ?? 0.95,
        // NANO-042: Reasoning model config
        reasoningFormat: llmLocalConfig.reasoning_format || "",
        reasoningBudget: llmLocalConfig.reasoning_budget ?? -1,
        // NANO-084: mmproj for unified vision mode
        mmprojPath: llmLocalConfig.mmproj_path || "",
      },
      llmCloud: {
        provider: llmCloudProvider || "deepseek",
        apiUrl: llmCloudConfig.url || "",
        apiKey: llmCloudConfig.api_key || "",
        model: llmCloudConfig.model || "",
        contextSize: llmCloudConfig.context_size || 8192,
        timeout: llmCloudConfig.timeout || 60,
        temperature: llmCloudConfig.temperature ?? 0.7,
        maxTokens: llmCloudConfig.max_tokens || 256,
      },

      // VLM (NANO-059: vlmEnabled + improved useLLMForVision detection)
      // Source of truth is vlm.provider only — tools.screen_vision.vlm_provider
      // can be stale after runtime VLM swaps (Session 560 bug).
      vlmEnabled,
      useLLMForVision: vlmProvider === "llm",
      vlmProviderType: vlmIsLocal ? "local" : "cloud",
      vlmLocal: {
        modelType: vlmLocalConfig.model_type || "gemma3",
        executablePath: vlmLocalConfig.executable_path || "",
        modelPath: vlmLocalConfig.model_path || "",
        mmprojPath: vlmLocalConfig.mmproj_path || "",
        host: vlmLocalConfig.host || "127.0.0.1",
        port: vlmLocalConfig.port || 5558,
        contextSize: vlmLocalConfig.context_size || 8192,
        gpuLayers: vlmLocalConfig.gpu_layers ?? 99,
        device: vlmLocalConfig.device || "",
        tensorSplit: Array.isArray(vlmLocalConfig.tensor_split) ? vlmLocalConfig.tensor_split.join(",") : "",
        extraArgs: arrayToString(vlmLocalConfig.extra_args),
        timeout: vlmLocalConfig.timeout || 30,
        maxTokens: vlmLocalConfig.max_tokens || 300,
      },
      vlmCloud: {
        apiKey: vlmCloudConfig.api_key || "",
        baseUrl: vlmCloudConfig.base_url || "https://api.openai.com",
        model: vlmCloudConfig.model || "gpt-4o",
        contextSize: vlmCloudConfig.context_size || 8192,
        timeout: vlmCloudConfig.timeout || 30,
        maxTokens: vlmCloudConfig.max_tokens || 300,
      },

      // STT - dual-provider hydration (always return BOTH configs)
      // NANO-112: Read enabled flag from launcher services
      sttEnabled: sttLauncher.enabled !== false,
      sttProvider: (raw?.stt?.provider || "parakeet") as "parakeet" | "whisper",
      sttParakeet: (() => {
        const sttSection = raw?.stt || {};
        const sttProviders = sttSection.providers || {};
        const parakeetConfig = sttProviders.parakeet || {};
        // Parakeet env/script: prefer provider YAML fields, fall back to launcher command parsing
        const parsed = parseSTTCommand(sttLauncher.command || "");
        // Legacy: if no providers block, read flat stt.host/port/timeout
        const host = parakeetConfig.host || sttSection.host || "127.0.0.1";
        const port = parakeetConfig.port || sttSection.port || 5555;
        const timeout = parakeetConfig.timeout || sttSection.timeout || 30;
        return {
          host,
          port,
          timeout,
          platform: sttLauncher.platform || "native",
          wslDistro: sttLauncher.wsl_distro || "Ubuntu",
          envType: parakeetConfig.conda_env ? "conda" as const : parsed.envType,
          envNameOrPath: parakeetConfig.conda_env || parsed.envNameOrPath,
          customActivation: parsed.customActivation,
          serverScriptPath: parakeetConfig.server_script || parsed.serverScriptPath,
        };
      })(),
      sttWhisper: (() => {
        const sttProviders = raw?.stt?.providers || {};
        const whisperConfig = sttProviders.whisper || {};
        return {
          host: whisperConfig.host || "127.0.0.1",
          port: whisperConfig.port || 8080,
          timeout: whisperConfig.timeout || 30,
          binaryPath: whisperConfig.binary_path || "whisper-server",
          modelPath: whisperConfig.model_path || "",
          language: whisperConfig.language || "en",
          threads: whisperConfig.threads || 4,
          noGpu: whisperConfig.no_gpu ?? false,
        };
      })(),

      // TTS
      // NANO-112: Read enabled flag from launcher services
      ttsEnabled: ttsLauncher.enabled !== false,
      ttsProviderType: "local" as const, // Cloud not implemented yet
      ttsLocal: {
        provider: ttsProvider || "kokoro",
        host: ttsConfig.host || "127.0.0.1",
        port: ttsConfig.port || 5556,
        voice: ttsConfig.voice || "",
        language: ttsConfig.language || "",
        modelsDirectory: ttsConfig.models_dir || "./tts/models",
        device: ttsConfig.device || "cuda",
        timeout: ttsConfig.timeout || 30,
        envType: ttsConfig.conda_env ? "conda" : "system",
        envNameOrPath: ttsConfig.conda_env || "",
        customActivation: "",
      },

      // Embedding Server (NANO-043 Phase 5)
      embedding: {
        enabled: embeddingLauncher.enabled ?? false,
        executablePath: embExeMatch?.[1] || "",
        modelPath: embModelMatch?.[1] || "",
        host: embHostMatch?.[1] || embUrlParts?.[1] || "127.0.0.1",
        port: parseInt(embPortMatch?.[1] || embUrlParts?.[2] || "5559"),
        gpuLayers: parseInt(embNglMatch?.[1] || "99"),
        contextSize: parseInt(embCtxMatch?.[1] || "2048"),
        timeout: embeddingLauncher.health_check?.timeout || memoryEmbedding.timeout || 60,
        relevanceThreshold: raw?.memory?.relevance_threshold ?? 0.25,
        topK: raw?.memory?.top_k ?? 5,
      },

      // NANO-063: Per-provider API key map for key isolation on provider switch
      savedProviderKeys,
    };

    // NANO-089 Phase 2: Soft validation — fill defaults for missing fields,
    // but don't block hydration if validation fails (prevents locked-out YAML states).
    // Use passthrough() to preserve extra fields like savedProviderKeys.
    const validated = LauncherConfigSchema.passthrough().safeParse(config);
    if (validated.success) {
      return NextResponse.json({
        exists: true,
        configPath: CONFIG_PATH,
        config: validated.data,
      });
    } else {
      console.warn("[write-config GET] Hydration validation issues:", validated.error.issues);
      return NextResponse.json({
        exists: true,
        configPath: CONFIG_PATH,
        config,
      });
    }
  } catch (error) {
    console.error("Error reading config:", error);
    return NextResponse.json(
      {
        exists: false,
        error: error instanceof Error ? error.message : "Unknown error",
        config: null,
      },
      { status: 500 }
    );
  }
}
