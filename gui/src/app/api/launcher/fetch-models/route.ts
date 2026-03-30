import { NextResponse } from "next/server";

interface FetchModelsRequest {
  provider: string;
  apiKey: string;
  apiUrl?: string;
}

interface OpenRouterModel {
  id: string;
  name: string;
  context_length: number | null;
  pricing?: {
    prompt: string;
    completion: string;
  };
}

export interface ModelInfo {
  id: string;
  name: string;
  context_length: number | null;
  pricing: { prompt: string; completion: string } | null;
}

export async function POST(request: Request) {
  try {
    const body: FetchModelsRequest = await request.json();

    if (!body.apiKey) {
      return NextResponse.json(
        { success: false, models: [], error: "API key is required" },
        { status: 400 }
      );
    }

    if (body.provider !== "openrouter") {
      return NextResponse.json(
        {
          success: false,
          models: [],
          error: `Unsupported provider: ${body.provider}`,
        },
        { status: 400 }
      );
    }

    const baseUrl = (body.apiUrl || "https://openrouter.ai/api/v1").replace(
      /\/$/,
      ""
    );

    const response = await fetch(`${baseUrl}/models`, {
      headers: {
        Authorization: `Bearer ${body.apiKey}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        {
          success: false,
          models: [],
          error:
            response.status === 401
              ? "Invalid API key"
              : `OpenRouter API error (${response.status}): ${errorText.slice(0, 200)}`,
        },
        { status: response.status === 401 ? 401 : 502 }
      );
    }

    const data = await response.json();
    const rawModels: OpenRouterModel[] = data.data || [];

    const models: ModelInfo[] = rawModels
      .map((m) => ({
        id: m.id,
        name: m.name,
        context_length: m.context_length,
        pricing: m.pricing
          ? { prompt: m.pricing.prompt, completion: m.pricing.completion }
          : null,
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json({ success: true, models });
  } catch (error) {
    console.error("Error fetching models:", error);
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { success: false, models: [], error: message },
      { status: 500 }
    );
  }
}
