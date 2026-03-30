import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type { CharacterCardData, CharacterBook } from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CARD_FILENAME = "card.json";
const AVATAR_FILENAME = "avatar.png";

// PNG signature bytes
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// ============================================
// Type Definitions
// ============================================

interface ImportCharacterRequest {
  json_data: string | CharacterCardData;
  character_id?: string;
  overwrite?: boolean;
}

interface ImportCompleteResponse {
  character_id: string;
  name: string;
  was_overwrite: boolean;
  has_avatar: boolean;
  success: boolean;
}

interface ImportErrorResponse {
  error: string;
  exists?: boolean;
}

// ============================================
// PNG tEXt Chunk Extraction
// ============================================

/**
 * Extract character card JSON from a PNG's tEXt chunks.
 *
 * SillyTavern embeds character data as:
 *   tEXt chunk → keyword: "chara" → text: base64(JSON)
 *
 * Also checks for 'ccv3' keyword (Character Card V3 draft spec).
 */
function extractCharaFromPng(pngBuffer: Buffer): string | null {
  if (!pngBuffer.subarray(0, 8).equals(PNG_SIGNATURE)) {
    return null;
  }

  let offset = 8; // Skip PNG signature

  while (offset < pngBuffer.length) {
    if (offset + 8 > pngBuffer.length) break;

    const length = pngBuffer.readUInt32BE(offset);
    const chunkType = pngBuffer.subarray(offset + 4, offset + 8).toString("ascii");

    if (offset + 12 + length > pngBuffer.length) break;

    if (chunkType === "tEXt") {
      const chunkData = pngBuffer.subarray(offset + 8, offset + 8 + length);
      const nullIdx = chunkData.indexOf(0x00);

      if (nullIdx !== -1) {
        const keyword = chunkData.subarray(0, nullIdx).toString("latin1");
        const textData = chunkData.subarray(nullIdx + 1);

        if (keyword === "chara" || keyword === "ccv3") {
          try {
            return Buffer.from(textData.toString("latin1"), "base64").toString("utf-8");
          } catch {
            // Try treating as raw UTF-8
            try {
              return textData.toString("utf-8");
            } catch {
              // Skip this chunk
            }
          }
        }
      }
    } else if (chunkType === "IEND") {
      break;
    }

    // Advance: 4 (length) + 4 (type) + length (data) + 4 (CRC)
    offset += 12 + length;
  }

  return null;
}

// ============================================
// Validation Helpers
// ============================================

function validateCharacterCard(card: unknown): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!card || typeof card !== "object") {
    errors.push("Invalid JSON: must be an object");
    return { valid: false, errors };
  }

  const c = card as Record<string, unknown>;

  // Check spec field
  if (c.spec !== "chara_card_v2") {
    errors.push(`Invalid spec: expected 'chara_card_v2', got '${c.spec}'`);
  }

  // Check data field exists
  if (!c.data || typeof c.data !== "object") {
    errors.push("Missing or invalid 'data' field");
    return { valid: false, errors };
  }

  const data = c.data as Record<string, unknown>;

  // Check required fields
  if (!data.name || typeof data.name !== "string") {
    errors.push("Missing or invalid 'data.name' field");
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

  return { valid: errors.length === 0, errors };
}

function normalizeCard(card: CharacterCardData): CharacterCardData {
  // Ensure all optional fields have sensible defaults
  const data = card.data;

  return {
    spec: "chara_card_v2",
    spec_version: card.spec_version || "2.0",
    data: {
      name: data.name,
      description: data.description || "",
      personality: data.personality || "",
      scenario: data.scenario || "",
      first_mes: data.first_mes || "",
      mes_example: data.mes_example || "",
      creator_notes: data.creator_notes || "",
      system_prompt: data.system_prompt || "",
      post_history_instructions: data.post_history_instructions || "",
      alternate_greetings: data.alternate_greetings || [],
      tags: data.tags || [],
      creator: data.creator || "",
      character_version: data.character_version || "1.0",
      extensions: data.extensions || {},
      character_book: data.character_book,
    },
  };
}

// ============================================
// Shared import logic
// ============================================

function importCard(
  card: CharacterCardData,
  characterIdOverride: string | undefined,
  overwrite: boolean,
  charactersDir: string,
  pngBytes?: Buffer,
): NextResponse<ImportCompleteResponse | ImportErrorResponse> {
  // Validate card structure
  const validation = validateCharacterCard(card);
  if (!validation.valid) {
    return NextResponse.json<ImportErrorResponse>(
      { error: `Validation failed: ${validation.errors.join(", ")}` },
      { status: 400 }
    );
  }

  // Normalize the card
  card = normalizeCard(card);

  // Determine character ID
  const spindlExt = card.data.extensions?.spindl as { id?: string } | undefined;
  const resolvedId =
    characterIdOverride ||
    spindlExt?.id ||
    card.data.name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

  if (!resolvedId) {
    return NextResponse.json<ImportErrorResponse>(
      { error: "Could not determine character ID" },
      { status: 400 }
    );
  }

  // Ensure characters directory exists
  if (!fs.existsSync(charactersDir)) {
    fs.mkdirSync(charactersDir, { recursive: true });
  }

  const characterDir = path.join(charactersDir, resolvedId);
  const cardPath = path.join(characterDir, CARD_FILENAME);
  const exists = fs.existsSync(characterDir);

  // Handle existing character
  if (exists && !overwrite) {
    return NextResponse.json<ImportErrorResponse>(
      { error: `Character '${resolvedId}' already exists`, exists: true },
      { status: 409 }
    );
  }

  // Create directory if needed
  if (!exists) {
    fs.mkdirSync(characterDir, { recursive: true });
  }

  // Ensure spindl.id is set
  if (!card.data.extensions) {
    card.data.extensions = {};
  }
  if (!card.data.extensions.spindl) {
    card.data.extensions.spindl = {};
  }
  (card.data.extensions.spindl as { id: string }).id = resolvedId;

  // Write card.json
  fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");

  // Write avatar.png if PNG import
  let hasAvatar = false;
  if (pngBytes) {
    const avatarPath = path.join(characterDir, AVATAR_FILENAME);
    fs.writeFileSync(avatarPath, pngBytes);
    hasAvatar = true;
  }

  return NextResponse.json<ImportCompleteResponse>({
    character_id: resolvedId,
    name: card.data.name,
    was_overwrite: exists,
    has_avatar: hasAvatar,
    success: true,
  }, { status: exists ? 200 : 201 });
}

// ============================================
// POST Handler - Import character (JSON or PNG)
// ============================================

export async function POST(request: Request) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const contentType = request.headers.get("content-type") || "";

    // PNG upload via multipart/form-data
    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const file = formData.get("file") as File | null;
      const characterId = formData.get("character_id") as string | null;
      const overwrite = formData.get("overwrite") === "true";

      if (!file) {
        return NextResponse.json<ImportErrorResponse>(
          { error: "No file provided" },
          { status: 400 }
        );
      }

      const buffer = Buffer.from(await file.arrayBuffer());
      const filename = file.name.toLowerCase();

      if (filename.endsWith(".png")) {
        // Extract character JSON from PNG tEXt chunk
        const jsonStr = extractCharaFromPng(buffer);
        if (!jsonStr) {
          return NextResponse.json<ImportErrorResponse>(
            { error: "PNG file does not contain character data. Expected a tEXt chunk with key 'chara' (SillyTavern format)." },
            { status: 400 }
          );
        }

        let card: CharacterCardData;
        try {
          card = JSON.parse(jsonStr);
        } catch {
          return NextResponse.json<ImportErrorResponse>(
            { error: "Invalid JSON found in PNG tEXt chunk" },
            { status: 400 }
          );
        }

        return importCard(card, characterId || undefined, overwrite, CHARACTERS_DIR, buffer);
      } else if (filename.endsWith(".json")) {
        // JSON file uploaded via form
        let card: CharacterCardData;
        try {
          card = JSON.parse(buffer.toString("utf-8"));
        } catch {
          return NextResponse.json<ImportErrorResponse>(
            { error: "Invalid JSON file" },
            { status: 400 }
          );
        }

        return importCard(card, characterId || undefined, overwrite, CHARACTERS_DIR);
      } else {
        return NextResponse.json<ImportErrorResponse>(
          { error: `Unsupported file type. Expected .json or .png, got: ${file.name}` },
          { status: 400 }
        );
      }
    }

    // Existing JSON body path
    const body: ImportCharacterRequest = await request.json();
    const { json_data, character_id, overwrite = false } = body;

    // Parse JSON if string
    let card: CharacterCardData;
    if (typeof json_data === "string") {
      try {
        card = JSON.parse(json_data);
      } catch {
        return NextResponse.json<ImportErrorResponse>(
          { error: "Invalid JSON string" },
          { status: 400 }
        );
      }
    } else {
      card = json_data;
    }

    return importCard(card, character_id, overwrite, CHARACTERS_DIR);
  } catch (error) {
    console.error("Error importing character:", error);
    return NextResponse.json<ImportErrorResponse>(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
