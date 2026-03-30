import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type { CharacterCardData, CharacterBookEntry } from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CARD_FILENAME = "card.json";

// ============================================
// PUT Handler - Update single entry in character codex
// ============================================

interface UpdateEntryRequest {
  entry: CharacterBookEntry;
}

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string; entryId: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id, entryId } = await params;
    const entryIdNum = parseInt(entryId, 10);

    if (isNaN(entryIdNum)) {
      return NextResponse.json(
        { error: "Invalid entry ID" },
        { status: 400 }
      );
    }

    const body: UpdateEntryRequest = await request.json();
    const { entry } = body;

    if (!entry) {
      return NextResponse.json(
        { error: "entry is required" },
        { status: 400 }
      );
    }

    const cardPath = path.join(CHARACTERS_DIR, id, CARD_FILENAME);

    if (!fs.existsSync(cardPath)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Load existing card
    const content = fs.readFileSync(cardPath, "utf-8");
    const card: CharacterCardData = JSON.parse(content);

    // Check character_book exists
    if (!card.data.character_book || !card.data.character_book.entries) {
      return NextResponse.json(
        { error: "Character has no codex entries" },
        { status: 404 }
      );
    }

    // Find and update entry
    const entryIndex = card.data.character_book.entries.findIndex(
      (e) => e.id === entryIdNum
    );

    if (entryIndex === -1) {
      return NextResponse.json(
        { error: `Entry with ID ${entryIdNum} not found` },
        { status: 404 }
      );
    }

    // Update entry, preserving ID
    card.data.character_book.entries[entryIndex] = {
      ...entry,
      id: entryIdNum,
    };

    // Write back
    fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");

    return NextResponse.json({
      character_id: id,
      entry_id: entryIdNum,
      success: true,
    });
  } catch (error) {
    console.error("Error updating character codex entry:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// DELETE Handler - Remove single entry from character codex
// ============================================

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string; entryId: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id, entryId } = await params;
    const entryIdNum = parseInt(entryId, 10);

    if (isNaN(entryIdNum)) {
      return NextResponse.json(
        { error: "Invalid entry ID" },
        { status: 400 }
      );
    }

    const cardPath = path.join(CHARACTERS_DIR, id, CARD_FILENAME);

    if (!fs.existsSync(cardPath)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Load existing card
    const content = fs.readFileSync(cardPath, "utf-8");
    const card: CharacterCardData = JSON.parse(content);

    // Check character_book exists
    if (!card.data.character_book || !card.data.character_book.entries) {
      return NextResponse.json(
        { error: "Character has no codex entries" },
        { status: 404 }
      );
    }

    // Find entry
    const entryIndex = card.data.character_book.entries.findIndex(
      (e) => e.id === entryIdNum
    );

    if (entryIndex === -1) {
      return NextResponse.json(
        { error: `Entry with ID ${entryIdNum} not found` },
        { status: 404 }
      );
    }

    // Remove entry
    card.data.character_book.entries.splice(entryIndex, 1);

    // Write back
    fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");

    return NextResponse.json({
      character_id: id,
      entry_id: entryIdNum,
      success: true,
    });
  } catch (error) {
    console.error("Error deleting character codex entry:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
