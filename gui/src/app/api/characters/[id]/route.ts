import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type {
  CharacterCardData,
  CharacterDetailEvent,
  CharacterUpdatedEvent,
  CharacterDeletedEvent,
} from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CARD_FILENAME = "card.json";
const AVATAR_FILENAME = "avatar.png";

// ============================================
// GET Handler - Get character detail
// ============================================

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const characterDir = path.join(CHARACTERS_DIR, id);
    const cardPath = path.join(characterDir, CARD_FILENAME);

    // Check if character exists
    if (!fs.existsSync(cardPath)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Read and parse card.json
    const cardContent = fs.readFileSync(cardPath, "utf-8");
    const card: CharacterCardData = JSON.parse(cardContent);

    // Check if avatar exists
    const avatarPath = path.join(characterDir, AVATAR_FILENAME);
    const hasAvatar = fs.existsSync(avatarPath);

    return NextResponse.json<CharacterDetailEvent>({
      character_id: id,
      card,
      has_avatar: hasAvatar,
    });
  } catch (error) {
    console.error("Error getting character:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// PUT Handler - Update character
// ============================================

interface UpdateCharacterRequest {
  card: CharacterCardData;
}

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const body: UpdateCharacterRequest = await request.json();
    const { card } = body;

    // Validate card structure
    if (!card || card.spec !== "chara_card_v2" || !card.data?.name) {
      return NextResponse.json(
        { error: "Invalid character card: must be Character Card V2 format with a name" },
        { status: 400 }
      );
    }

    const characterDir = path.join(CHARACTERS_DIR, id);
    const cardPath = path.join(characterDir, CARD_FILENAME);

    // Check if character exists
    if (!fs.existsSync(cardPath)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Ensure spindl.id matches the directory name
    if (!card.data.extensions) {
      card.data.extensions = {};
    }
    if (!card.data.extensions.spindl) {
      card.data.extensions.spindl = {};
    }
    (card.data.extensions.spindl as { id: string }).id = id;

    // Write updated card.json
    fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");

    return NextResponse.json<CharacterUpdatedEvent>({
      character_id: id,
      success: true,
    });
  } catch (error) {
    console.error("Error updating character:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// DELETE Handler - Delete character
// ============================================

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const characterDir = path.join(CHARACTERS_DIR, id);

    // Check if character exists
    if (!fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Delete the entire character directory
    fs.rmSync(characterDir, { recursive: true, force: true });

    return NextResponse.json<CharacterDeletedEvent>({
      character_id: id,
      success: true,
    });
  } catch (error) {
    console.error("Error deleting character:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
