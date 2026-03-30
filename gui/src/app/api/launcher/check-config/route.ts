import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";
import * as yaml from "yaml";

/**
 * NANO-027 Phase 4: Config State Check Endpoint
 *
 * Determines if the configuration is "empty" (requiring Launcher redirect).
 * Config is considered empty if:
 * - config file does not exist, OR
 * - llm.provider is missing/empty, OR
 * - llm.providers.{active_provider} section is missing required fields
 */

const CONFIG_PATH = process.env.SPINDL_CONFIG || path.resolve(process.cwd(), "..", "config", "spindl.yaml");

interface ConfigCheckResult {
  needsSetup: boolean;
  reason: string | null;
  configPath: string;
  details?: {
    fileExists: boolean;
    hasLLMProvider: boolean;
    hasLLMConfig: boolean;
    hasSTTConfig: boolean;
    hasTTSConfig: boolean;
  };
}

export async function GET(): Promise<NextResponse<ConfigCheckResult>> {
  try {
    // Check 1: File exists?
    if (!fs.existsSync(CONFIG_PATH)) {
      return NextResponse.json({
        needsSetup: true,
        reason: "Configuration file does not exist",
        configPath: CONFIG_PATH,
        details: {
          fileExists: false,
          hasLLMProvider: false,
          hasLLMConfig: false,
          hasSTTConfig: false,
          hasTTSConfig: false,
        },
      });
    }

    // Read and parse config
    const content = fs.readFileSync(CONFIG_PATH, "utf-8");
    const config = yaml.parse(content);

    if (!config) {
      return NextResponse.json({
        needsSetup: true,
        reason: "Configuration file is empty",
        configPath: CONFIG_PATH,
        details: {
          fileExists: true,
          hasLLMProvider: false,
          hasLLMConfig: false,
          hasSTTConfig: false,
          hasTTSConfig: false,
        },
      });
    }

    // Check 2: LLM provider defined?
    const llmProvider = config?.llm?.provider;
    if (!llmProvider) {
      return NextResponse.json({
        needsSetup: true,
        reason: "LLM provider not configured",
        configPath: CONFIG_PATH,
        details: {
          fileExists: true,
          hasLLMProvider: false,
          hasLLMConfig: false,
          hasSTTConfig: !!config?.stt?.host,
          hasTTSConfig: !!config?.tts?.provider,
        },
      });
    }

    // Check 3: LLM provider config has required fields?
    const llmConfig = config?.llm?.providers?.[llmProvider];
    const hasRequiredLLMFields = llmProvider === "llama"
      ? !!(llmConfig?.executable_path && llmConfig?.model_path)
      : !!(llmConfig?.api_key && llmConfig?.model);

    if (!hasRequiredLLMFields) {
      return NextResponse.json({
        needsSetup: true,
        reason: `LLM provider '${llmProvider}' is missing required configuration`,
        configPath: CONFIG_PATH,
        details: {
          fileExists: true,
          hasLLMProvider: true,
          hasLLMConfig: false,
          hasSTTConfig: !!config?.stt?.host,
          hasTTSConfig: !!config?.tts?.provider,
        },
      });
    }

    // Config is complete
    return NextResponse.json({
      needsSetup: false,
      reason: null,
      configPath: CONFIG_PATH,
      details: {
        fileExists: true,
        hasLLMProvider: true,
        hasLLMConfig: true,
        hasSTTConfig: !!config?.stt?.host,
        hasTTSConfig: !!config?.tts?.provider,
      },
    });
  } catch (error) {
    console.error("Error checking config:", error);
    return NextResponse.json(
      {
        needsSetup: true,
        reason: `Error reading configuration: ${error instanceof Error ? error.message : "Unknown error"}`,
        configPath: CONFIG_PATH,
      },
      { status: 500 }
    );
  }
}
