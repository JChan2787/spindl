import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type { CharacterBook, CharacterBookEntry } from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CODEX_FILENAME = "codex.json";

// ============================================
// PUT Handler - Update single entry in global codex
// ============================================

interface UpdateEntryRequest {
  entry: CharacterBookEntry;
}

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ entryId: string }> }
) {
  try {
    const GLOBAL_DIR = path.join(getCharactersDir(), "_global");
    const { entryId } = await params;
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

    // Load existing codex
    const codexPath = path.join(GLOBAL_DIR, CODEX_FILENAME);

    if (!fs.existsSync(codexPath)) {
      return NextResponse.json(
        { error: "Global codex does not exist" },
        { status: 404 }
      );
    }

    const content = fs.readFileSync(codexPath, "utf-8");
    const book: CharacterBook = JSON.parse(content);

    // Find and update entry
    const entryIndex = book.entries.findIndex((e) => e.id === entryIdNum);

    if (entryIndex === -1) {
      return NextResponse.json(
        { error: `Entry with ID ${entryIdNum} not found` },
        { status: 404 }
      );
    }

    // Update entry, preserving ID
    book.entries[entryIndex] = {
      ...entry,
      id: entryIdNum,
    };

    // Write back
    fs.writeFileSync(codexPath, JSON.stringify(book, null, 2), "utf-8");

    return NextResponse.json({
      entry_id: entryIdNum,
      success: true,
    });
  } catch (error) {
    console.error("Error updating global codex entry:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// DELETE Handler - Remove single entry from global codex
// ============================================

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ entryId: string }> }
) {
  try {
    const GLOBAL_DIR = path.join(getCharactersDir(), "_global");
    const { entryId } = await params;
    const entryIdNum = parseInt(entryId, 10);

    if (isNaN(entryIdNum)) {
      return NextResponse.json(
        { error: "Invalid entry ID" },
        { status: 400 }
      );
    }

    // Load existing codex
    const codexPath = path.join(GLOBAL_DIR, CODEX_FILENAME);

    if (!fs.existsSync(codexPath)) {
      return NextResponse.json(
        { error: "Global codex does not exist" },
        { status: 404 }
      );
    }

    const content = fs.readFileSync(codexPath, "utf-8");
    const book: CharacterBook = JSON.parse(content);

    // Find entry
    const entryIndex = book.entries.findIndex((e) => e.id === entryIdNum);

    if (entryIndex === -1) {
      return NextResponse.json(
        { error: `Entry with ID ${entryIdNum} not found` },
        { status: 404 }
      );
    }

    // Remove entry
    book.entries.splice(entryIndex, 1);

    // Write back
    fs.writeFileSync(codexPath, JSON.stringify(book, null, 2), "utf-8");

    return NextResponse.json({
      entry_id: entryIdNum,
      success: true,
    });
  } catch (error) {
    console.error("Error deleting global codex entry:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
