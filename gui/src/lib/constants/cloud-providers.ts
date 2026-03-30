/**
 * NANO-063: Cloud Provider Registry.
 * Single source of truth for all cloud LLM provider metadata.
 * Adding a provider = one entry here. UI, defaults, env var pattern all derive automatically.
 */

export const CLOUD_PROVIDERS = {
  deepseek: {
    label: "DeepSeek",
    defaultUrl: "https://api.deepseek.com/v1",
    defaultModel: "deepseek-chat",
    envVarKey: "DEEPSEEK_API_KEY",
  },
  openai: {
    label: "OpenAI",
    defaultUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-4o",
    envVarKey: "OPENAI_API_KEY",
  },
  openrouter: {
    label: "OpenRouter",
    defaultUrl: "https://openrouter.ai/api/v1",
    defaultModel: "",
    envVarKey: "OPENROUTER_API_KEY",
  },
} as const;

export type CloudProvider = keyof typeof CLOUD_PROVIDERS;

/** Ordered list for UI rendering (stable insertion order). */
export const CLOUD_PROVIDER_IDS = Object.keys(CLOUD_PROVIDERS) as CloudProvider[];
