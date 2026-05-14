import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type {
  CharacterBook,
  CharacterBookEntry,
  CodexVolume,
  GlobalCodexEvent,
} from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CODEX_FILENAME = "codex.json";

// ============================================
// GET Handler - Get global codex entries
// ============================================

export async function GET() {
  try {
    const GLOBAL_DIR = path.join(getCharactersDir(), "_global");
    const codexPath = path.join(GLOBAL_DIR, CODEX_FILENAME);

    // If no global codex exists, return empty with default volume
    if (!fs.existsSync(codexPath)) {
      return NextResponse.json<GlobalCodexEvent>({
        entries: [],
        volumes: [{ id: "vol_default", name: "Default", enabled: true, insertion_order: 0 }],
        name: "Global Codex",
      });
    }

    const content = fs.readFileSync(codexPath, "utf-8");
    const book: CharacterBook = JSON.parse(content);

    // Ensure entries have IDs (auto-assign if missing)
    const entries = book.entries.map((entry, index) => ({
      ...entry,
      id: entry.id ?? index,
    }));

    // Ensure volumes exist with at least the default
    let volumes: CodexVolume[] = book.volumes || [];
    if (!volumes.some((v) => v.id === "vol_default")) {
      volumes = [{ id: "vol_default", name: "Default", enabled: true, insertion_order: 0 }, ...volumes];
    }

    return NextResponse.json<GlobalCodexEvent>({
      entries,
      volumes,
      name: book.name || "Global Codex",
    });
  } catch (error) {
    console.error("Error reading global codex:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// PUT Handler - Save entire global codex
// ============================================

interface SaveGlobalCodexRequest {
  entries: CharacterBookEntry[];
  volumes?: CodexVolume[];
  name?: string;
}

export async function PUT(request: Request) {
  try {
    const GLOBAL_DIR = path.join(getCharactersDir(), "_global");
    const body: SaveGlobalCodexRequest = await request.json();
    const { entries, volumes, name } = body;

    if (!Array.isArray(entries)) {
      return NextResponse.json(
        { error: "entries must be an array" },
        { status: 400 }
      );
    }

    // Ensure _global directory exists
    if (!fs.existsSync(GLOBAL_DIR)) {
      fs.mkdirSync(GLOBAL_DIR, { recursive: true });
    }

    // Auto-assign IDs to entries that don't have them
    let nextId = 0;
    const existingIds = new Set(entries.filter(e => e.id !== undefined).map(e => e.id));
    for (const id of existingIds) {
      if (typeof id === "number" && id >= nextId) {
        nextId = id + 1;
      }
    }

    const entriesWithIds = entries.map((entry) => ({
      ...entry,
      id: entry.id ?? nextId++,
    }));

    // Ensure default volume exists in the volumes array
    const savedVolumes = volumes || [];
    if (!savedVolumes.some((v) => v.id === "vol_default")) {
      savedVolumes.unshift({ id: "vol_default", name: "Default", enabled: true, insertion_order: 0 });
    }

    // Build the CharacterBook structure
    const book: CharacterBook = {
      name: name || "Global Codex",
      description: "Entries active across all characters",
      entries: entriesWithIds,
      volumes: savedVolumes,
      extensions: {},
    };

    // Write to file
    const codexPath = path.join(GLOBAL_DIR, CODEX_FILENAME);
    fs.writeFileSync(codexPath, JSON.stringify(book, null, 2), "utf-8");

    return NextResponse.json<GlobalCodexEvent>({
      entries: entriesWithIds,
      volumes: savedVolumes,
      name: book.name || "Global Codex",
    });
  } catch (error) {
    console.error("Error saving global codex:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// POST Handler - Add single entry to global codex
// ============================================

interface CreateEntryRequest {
  entry: CharacterBookEntry;
}

export async function POST(request: Request) {
  try {
    const GLOBAL_DIR = path.join(getCharactersDir(), "_global");
    const body: CreateEntryRequest = await request.json();
    const { entry } = body;

    if (!entry) {
      return NextResponse.json(
        { error: "entry is required" },
        { status: 400 }
      );
    }

    // Load existing codex
    const codexPath = path.join(GLOBAL_DIR, CODEX_FILENAME);
    let book: CharacterBook;

    if (fs.existsSync(codexPath)) {
      const content = fs.readFileSync(codexPath, "utf-8");
      book = JSON.parse(content);
    } else {
      // Ensure _global directory exists
      if (!fs.existsSync(GLOBAL_DIR)) {
        fs.mkdirSync(GLOBAL_DIR, { recursive: true });
      }
      book = {
        name: "Global Codex",
        description: "Entries active across all characters",
        entries: [],
        volumes: [{ id: "vol_default", name: "Default", enabled: true, insertion_order: 0 }],
        extensions: {},
      };
    }

    // Find next available ID
    let nextId = 0;
    for (const e of book.entries) {
      if (e.id !== undefined && e.id >= nextId) {
        nextId = e.id + 1;
      }
    }

    // Add entry with assigned ID
    const newEntry: CharacterBookEntry = {
      ...entry,
      id: nextId,
    };
    book.entries.push(newEntry);

    // Write back
    fs.writeFileSync(codexPath, JSON.stringify(book, null, 2), "utf-8");

    return NextResponse.json({
      entry_id: nextId,
      success: true,
    }, { status: 201 });
  } catch (error) {
    console.error("Error creating global codex entry:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
