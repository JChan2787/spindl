import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type {
  CharacterInfo,
  CharacterCardData,
  CharacterListEvent,
  CharacterCreatedEvent,
} from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CARD_FILENAME = "card.json";
const AVATAR_FILENAME = "avatar.png";

// ============================================
// GET Handler - List all characters
// ============================================

export async function GET() {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    // Ensure characters directory exists
    if (!fs.existsSync(CHARACTERS_DIR)) {
      return NextResponse.json<CharacterListEvent>({
        characters: [],
        active: null,
      });
    }

    const entries = fs.readdirSync(CHARACTERS_DIR, { withFileTypes: true });
    const characters: CharacterInfo[] = [];

    for (const entry of entries) {
      // Skip non-directories and special directories
      if (!entry.isDirectory() || entry.name.startsWith("_")) {
        continue;
      }

      const cardPath = path.join(CHARACTERS_DIR, entry.name, CARD_FILENAME);

      // Skip if no card.json exists
      if (!fs.existsSync(cardPath)) {
        continue;
      }

      try {
        const cardContent = fs.readFileSync(cardPath, "utf-8");
        const card: CharacterCardData = JSON.parse(cardContent);

        // Check if avatar exists
        const avatarPath = path.join(CHARACTERS_DIR, entry.name, AVATAR_FILENAME);
        const hasAvatar = fs.existsSync(avatarPath);

        // Extract spindl extensions for id
        const spindlExt = card.data.extensions?.spindl as {
          id?: string;
          voice?: string;
        } | undefined;

        characters.push({
          id: spindlExt?.id || entry.name,
          name: card.data.name,
          description: card.data.description?.substring(0, 100) || "",
          voice: spindlExt?.voice || null,
          has_avatar: hasAvatar,
          tags: card.data.tags || [],
        });
      } catch (parseError) {
        console.error(`Error parsing card.json for ${entry.name}:`, parseError);
        // Skip invalid characters
        continue;
      }
    }

    // Sort by name
    characters.sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json<CharacterListEvent>({
      characters,
      active: null, // Active character is managed by orchestrator, not filesystem
    });
  } catch (error) {
    console.error("Error listing characters:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// POST Handler - Create new character
// ============================================

interface CreateCharacterRequest {
  card: CharacterCardData;
  character_id?: string;
  avatar_data?: string;
  original_avatar_data?: string;
  crop_settings?: { scale: number; offsetX: number; offsetY: number };
}

export async function POST(request: Request) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const body: CreateCharacterRequest = await request.json();
    const { card, character_id, avatar_data, original_avatar_data, crop_settings } = body;

    // Validate card structure
    if (!card || card.spec !== "chara_card_v2" || !card.data?.name) {
      return NextResponse.json(
        { error: "Invalid character card: must be Character Card V2 format with a name" },
        { status: 400 }
      );
    }

    // Determine character ID
    // Priority: explicit character_id > spindl.id > sanitized name
    const spindlExt = card.data.extensions?.spindl as { id?: string } | undefined;
    const resolvedId =
      character_id ||
      spindlExt?.id ||
      card.data.name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

    if (!resolvedId) {
      return NextResponse.json(
        { error: "Could not determine character ID" },
        { status: 400 }
      );
    }

    // Ensure characters directory exists
    if (!fs.existsSync(CHARACTERS_DIR)) {
      fs.mkdirSync(CHARACTERS_DIR, { recursive: true });
    }

    // Check if character already exists
    const characterDir = path.join(CHARACTERS_DIR, resolvedId);
    if (fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${resolvedId}' already exists`, exists: true },
        { status: 409 }
      );
    }

    // Create character directory
    fs.mkdirSync(characterDir, { recursive: true });

    // Ensure spindl.id is set in the card
    if (!card.data.extensions) {
      card.data.extensions = {};
    }
    if (!card.data.extensions.spindl) {
      card.data.extensions.spindl = {};
    }
    (card.data.extensions.spindl as { id: string }).id = resolvedId;

    // Write card.json
    const cardPath = path.join(characterDir, CARD_FILENAME);
    fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");

    // Write avatar files if provided (atomic creation)
    if (avatar_data) {
      try {
        // Write cropped avatar
        let base64Data = avatar_data;
        if (avatar_data.includes(",")) {
          base64Data = avatar_data.split(",")[1];
        }
        const imageBuffer = Buffer.from(base64Data, "base64");
        fs.writeFileSync(path.join(characterDir, AVATAR_FILENAME), imageBuffer);

        // Write original avatar for re-editing
        if (original_avatar_data) {
          let originalBase64 = original_avatar_data;
          if (original_avatar_data.includes(",")) {
            originalBase64 = original_avatar_data.split(",")[1];
          }
          const originalBuffer = Buffer.from(originalBase64, "base64");
          const mimeMatch = original_avatar_data.match(/data:image\/([a-z]+);base64/);
          const ext = mimeMatch ? (mimeMatch[1] === "jpg" ? "jpeg" : mimeMatch[1]) : "png";
          fs.writeFileSync(path.join(characterDir, `avatar_original.${ext}`), originalBuffer);
        }

        // Write crop settings
        if (crop_settings) {
          fs.writeFileSync(
            path.join(characterDir, "crop_settings.json"),
            JSON.stringify(crop_settings, null, 2),
            "utf-8"
          );
        }
      } catch (avatarError) {
        // Rollback: delete the entire character directory
        console.error("Error writing avatar files, rolling back:", avatarError);
        fs.rmSync(characterDir, { recursive: true, force: true });
        return NextResponse.json(
          { error: "Failed to write avatar files" },
          { status: 500 }
        );
      }
    }

    return NextResponse.json<CharacterCreatedEvent>({
      character_id: resolvedId,
      success: true,
    }, { status: 201 });
  } catch (error) {
    console.error("Error creating character:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
