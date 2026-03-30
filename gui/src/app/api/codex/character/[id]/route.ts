import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type {
  CharacterCardData,
  CharacterBook,
  CharacterBookEntry,
  CharacterCodexEvent,
} from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CARD_FILENAME = "card.json";

// ============================================
// GET Handler - Get character codex entries
// ============================================

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const cardPath = path.join(CHARACTERS_DIR, id, CARD_FILENAME);

    if (!fs.existsSync(cardPath)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    const content = fs.readFileSync(cardPath, "utf-8");
    const card: CharacterCardData = JSON.parse(content);

    // Extract character_book entries
    const book = card.data.character_book;
    const entries = book?.entries || [];

    // Ensure entries have IDs (auto-assign if missing)
    const entriesWithIds = entries.map((entry, index) => ({
      ...entry,
      id: entry.id ?? index,
    }));

    return NextResponse.json<CharacterCodexEvent>({
      character_id: id,
      entries: entriesWithIds,
    });
  } catch (error) {
    console.error("Error reading character codex:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// PUT Handler - Save entire character codex
// ============================================

interface SaveCharacterCodexRequest {
  entries: CharacterBookEntry[];
}

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const body: SaveCharacterCodexRequest = await request.json();
    const { entries } = body;

    if (!Array.isArray(entries)) {
      return NextResponse.json(
        { error: "entries must be an array" },
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

    // Auto-assign IDs to entries that don't have them
    let nextId = 0;
    const existingIds = new Set(entries.filter(e => e.id !== undefined).map(e => e.id));
    for (const entryId of existingIds) {
      if (typeof entryId === "number" && entryId >= nextId) {
        nextId = entryId + 1;
      }
    }

    const entriesWithIds = entries.map((entry) => ({
      ...entry,
      id: entry.id ?? nextId++,
    }));

    // Update or create character_book
    if (!card.data.character_book) {
      card.data.character_book = {
        entries: [],
        extensions: {},
      };
    }
    card.data.character_book.entries = entriesWithIds;

    // Write back
    fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");

    return NextResponse.json<CharacterCodexEvent>({
      character_id: id,
      entries: entriesWithIds,
    });
  } catch (error) {
    console.error("Error saving character codex:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// POST Handler - Add single entry to character codex
// ============================================

interface CreateEntryRequest {
  entry: CharacterBookEntry;
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const body: CreateEntryRequest = await request.json();
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

    // Ensure character_book exists
    if (!card.data.character_book) {
      card.data.character_book = {
        entries: [],
        extensions: {},
      };
    }

    // Find next available ID
    let nextId = 0;
    for (const e of card.data.character_book.entries) {
      if (e.id !== undefined && e.id >= nextId) {
        nextId = e.id + 1;
      }
    }

    // Add entry with assigned ID
    const newEntry: CharacterBookEntry = {
      ...entry,
      id: nextId,
    };
    card.data.character_book.entries.push(newEntry);

    // Write back
    fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");

    return NextResponse.json({
      character_id: id,
      entry_id: nextId,
      success: true,
    }, { status: 201 });
  } catch (error) {
    console.error("Error creating character codex entry:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
