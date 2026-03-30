import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock global fetch for the route handler
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

import { POST } from "./route";

function makeRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/api/launcher/fetch-models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("POST /api/launcher/fetch-models", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("returns 400 if apiKey is missing", async () => {
    const res = await POST(makeRequest({ provider: "openrouter", apiKey: "" }));
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.success).toBe(false);
    expect(data.error).toContain("API key");
  });

  it("returns 400 for unsupported provider", async () => {
    const res = await POST(
      makeRequest({ provider: "deepseek", apiKey: "sk-test" })
    );
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toContain("Unsupported provider");
  });

  it("returns sorted model list on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: [
          {
            id: "openai/gpt-4o",
            name: "OpenAI: GPT-4o",
            context_length: 128000,
            pricing: { prompt: "0.0000025", completion: "0.00001" },
          },
          {
            id: "google/gemini-2.5-pro",
            name: "Google: Gemini 2.5 Pro",
            context_length: 1048576,
            pricing: { prompt: "0.00000125", completion: "0.0000050" },
          },
          {
            id: "anthropic/claude-3.5-sonnet",
            name: "Anthropic: Claude 3.5 Sonnet",
            context_length: 200000,
            pricing: { prompt: "0.000003", completion: "0.000015" },
          },
        ],
      }),
    });

    const res = await POST(
      makeRequest({ provider: "openrouter", apiKey: "sk-test" })
    );
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.success).toBe(true);
    expect(data.models).toHaveLength(3);
    // Sorted alphabetically by name
    expect(data.models[0].name).toBe("Anthropic: Claude 3.5 Sonnet");
    expect(data.models[1].name).toBe("Google: Gemini 2.5 Pro");
    expect(data.models[2].name).toBe("OpenAI: GPT-4o");
    // Fields present
    expect(data.models[0]).toHaveProperty("id");
    expect(data.models[0]).toHaveProperty("context_length");
    expect(data.models[0]).toHaveProperty("pricing");
  });

  it("returns 401 for invalid API key", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: async () => "Unauthorized",
    });

    const res = await POST(
      makeRequest({ provider: "openrouter", apiKey: "bad-key" })
    );
    expect(res.status).toBe(401);
    const data = await res.json();
    expect(data.error).toContain("Invalid API key");
  });

  it("returns 502 for upstream server error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Internal Server Error",
    });

    const res = await POST(
      makeRequest({ provider: "openrouter", apiKey: "sk-test" })
    );
    expect(res.status).toBe(502);
  });

  it("handles empty model list", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ data: [] }),
    });

    const res = await POST(
      makeRequest({ provider: "openrouter", apiKey: "sk-test" })
    );
    const data = await res.json();
    expect(data.success).toBe(true);
    expect(data.models).toHaveLength(0);
  });

  it("uses custom apiUrl when provided", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ data: [] }),
    });

    await POST(
      makeRequest({
        provider: "openrouter",
        apiKey: "sk-test",
        apiUrl: "https://custom.openrouter.ai/api/v1",
      })
    );

    expect(mockFetch).toHaveBeenCalledWith(
      "https://custom.openrouter.ai/api/v1/models",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer sk-test",
        }),
      })
    );
  });

  it("handles models without pricing gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: [
          { id: "test/model", name: "Test Model", context_length: 4096 },
        ],
      }),
    });

    const res = await POST(
      makeRequest({ provider: "openrouter", apiKey: "sk-test" })
    );
    const data = await res.json();
    expect(data.models[0].pricing).toBeNull();
  });
});
