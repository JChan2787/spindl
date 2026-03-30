import { NextResponse } from "next/server";

import type { CharacterCardData, CharacterBook } from "@/types/events";

// ============================================
// Type Definitions
// ============================================

interface ValidateImportRequest {
  json_data: string | CharacterCardData;
}

interface ImportPreview {
  name: string;
  description: string;
  has_personality: boolean;
  has_system_prompt: boolean;
  has_codex: boolean;
  codex_count: number;
  has_spindl: boolean;
  tags: string[];
}

interface ValidationSuccessResponse {
  valid: true;
  preview: ImportPreview;
  warnings: string[];
}

interface ValidationErrorResponse {
  valid: false;
  errors: string[];
}

// ============================================
// Validation Helpers
// ============================================

function validateCharacterCard(card: unknown): { valid: boolean; errors: string[]; warnings: string[] } {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!card || typeof card !== "object") {
    errors.push("Invalid JSON: must be an object");
    return { valid: false, errors, warnings };
  }

  const c = card as Record<string, unknown>;

  // Check spec field
  if (c.spec !== "chara_card_v2") {
    if (c.spec === "chara_card_v1" || !c.spec) {
      warnings.push("Card appears to be V1 format; will be upgraded to V2");
    } else {
      errors.push(`Invalid spec: expected 'chara_card_v2', got '${c.spec}'`);
    }
  }

  // Check data field exists
  if (!c.data || typeof c.data !== "object") {
    errors.push("Missing or invalid 'data' field");
    return { valid: false, errors, warnings };
  }

  const data = c.data as Record<string, unknown>;

  // Check required fields
  if (!data.name || typeof data.name !== "string") {
    errors.push("Missing or invalid 'data.name' field");
  } else if (data.name.trim().length === 0) {
    errors.push("Character name cannot be empty");
  }

  // Validate character_book if present
  if (data.character_book !== undefined && data.character_book !== null) {
    if (typeof data.character_book !== "object") {
      errors.push("Invalid 'data.character_book': must be an object or null");
    } else {
      const book = data.character_book as Record<string, unknown>;
      if (book.entries !== undefined && !Array.isArray(book.entries)) {
        errors.push("Invalid 'data.character_book.entries': must be an array");
      }
    }
  }

  // Warnings for missing but recommended fields
  if (!data.description && !data.personality && !data.system_prompt) {
    warnings.push("Card has no description, personality, or system prompt");
  }

  return { valid: errors.length === 0, errors, warnings };
}

function buildPreview(card: CharacterCardData): ImportPreview {
  const data = card.data;
  const characterBook = data.character_book as CharacterBook | undefined;
  const spindlExt = data.extensions?.spindl as Record<string, unknown> | undefined;

  return {
    name: data.name,
    description: (data.description || "").substring(0, 200),
    has_personality: Boolean(data.personality && data.personality.trim().length > 0),
    has_system_prompt: Boolean(data.system_prompt && data.system_prompt.trim().length > 0),
    has_codex: Boolean(characterBook?.entries && characterBook.entries.length > 0),
    codex_count: characterBook?.entries?.length || 0,
    has_spindl: Boolean(spindlExt && Object.keys(spindlExt).length > 0),
    tags: data.tags || [],
  };
}

// ============================================
// POST Handler - Validate import (dry-run)
// ============================================

export async function POST(request: Request) {
  try {
    const body: ValidateImportRequest = await request.json();
    const { json_data } = body;

    // Parse JSON if string
    let card: CharacterCardData;
    if (typeof json_data === "string") {
      try {
        card = JSON.parse(json_data);
      } catch {
        return NextResponse.json<ValidationErrorResponse>({
          valid: false,
          errors: ["Invalid JSON string: failed to parse"],
        });
      }
    } else {
      card = json_data;
    }

    // Validate card structure
    const validation = validateCharacterCard(card);

    if (!validation.valid) {
      return NextResponse.json<ValidationErrorResponse>({
        valid: false,
        errors: validation.errors,
      });
    }

    // Build preview
    const preview = buildPreview(card);

    return NextResponse.json<ValidationSuccessResponse>({
      valid: true,
      preview,
      warnings: validation.warnings,
    });
  } catch (error) {
    console.error("Error validating import:", error);
    return NextResponse.json<ValidationErrorResponse>({
      valid: false,
      errors: [error instanceof Error ? error.message : "Unknown error"],
    });
  }
}
