import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { LLMConfig } from "./llm-config";
import { useSettingsStore } from "@/lib/stores";

// Mock getSocket to return a fake socket
vi.mock("@/lib/socket", () => ({
  getSocket: vi.fn(() => ({
    emit: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    connected: false,
  })),
}));

describe("LLMConfig — context_size input (NANO-096)", () => {
  beforeEach(() => {
    useSettingsStore.setState({
      llmConfig: {
        provider: "openrouter",
        model: "google/gemini-2.5-pro",
        context_size: 200000,
        available_providers: ["llama", "openrouter", "deepseek"],
      },
      isSavingLLM: false,
      localLLMConfig: {
        executable_path: "",
        model_path: "",
        host: "127.0.0.1",
        port: 5557,
        gpu_layers: 99,
        context_size: 8192,
      },
      isLaunchingLLM: false,
      llmServerRunning: false,
      llmLaunchError: null,
      vlmConfig: {
        provider: "none",
        available_providers: [],
        healthy: false,
        unified_vlm: false,
      },
    });
  });

  it("should show editable context_size input for cloud provider", () => {
    render(<LLMConfig />);
    const label = screen.getByText("Context Size");
    expect(label).toBeInTheDocument();

    // Should be an editable input with the provider's context_size value
    const input = screen.getByDisplayValue("200000");
    expect(input).toBeInTheDocument();
    expect(input.tagName).toBe("INPUT");
    expect(input).toHaveAttribute("type", "number");
  });

  it("should show read-only context display for local llama provider", () => {
    useSettingsStore.setState({
      llmConfig: {
        provider: "llama",
        model: "gemma-3-27b",
        context_size: 32768,
        available_providers: ["llama", "openrouter", "deepseek"],
      },
      llmServerRunning: true,
    });

    render(<LLMConfig />);

    // Should show read-only "Context" label with formatted token count
    expect(screen.getByText("Context")).toBeInTheDocument();
    expect(screen.getByText("32,768 tokens")).toBeInTheDocument();

    // Should NOT have an editable context_size input with the "Context Size" label
    expect(screen.queryByText("Context Size")).not.toBeInTheDocument();
  });

  it("should not show runtime context display for local llama when context_size is null", () => {
    useSettingsStore.setState({
      llmConfig: {
        provider: "llama",
        model: "",
        context_size: null,
        available_providers: ["llama", "openrouter", "deepseek"],
      },
      llmServerRunning: true,
    });

    render(<LLMConfig />);

    // The read-only "Context" + "N tokens" display should not appear when null
    expect(screen.queryByText(/tokens$/)).not.toBeInTheDocument();
  });

  it("should show context_size input for deepseek cloud provider", () => {
    useSettingsStore.setState({
      llmConfig: {
        provider: "deepseek",
        model: "deepseek-chat",
        context_size: 128000,
        available_providers: ["llama", "openrouter", "deepseek"],
      },
    });

    render(<LLMConfig />);
    expect(screen.getByText("Context Size")).toBeInTheDocument();
    expect(screen.getByDisplayValue("128000")).toBeInTheDocument();
  });
});
