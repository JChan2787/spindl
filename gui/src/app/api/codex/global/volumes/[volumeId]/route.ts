import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type { CharacterBook } from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CODEX_FILENAME = "codex.json";
const DEFAULT_VOLUME_ID = "vol_default";

function loadBook(): { book: CharacterBook; codexPath: string; globalDir: string } {
  const globalDir = path.join(getCharactersDir(), "_global");
  const codexPath = path.join(globalDir, CODEX_FILENAME);

  if (!fs.existsSync(codexPath)) {
    return {
      book: {
        name: "Global Codex",
        description: "Entries active across all characters",
        entries: [],
        volumes: [{ id: DEFAULT_VOLUME_ID, name: "Default", enabled: true, insertion_order: 0 }],
        extensions: {},
      },
      codexPath,
      globalDir,
    };
  }

  const content = fs.readFileSync(codexPath, "utf-8");
  const book: CharacterBook = JSON.parse(content);
  if (!book.volumes) book.volumes = [];
  if (!book.volumes.some((v) => v.id === DEFAULT_VOLUME_ID)) {
    book.volumes.unshift({ id: DEFAULT_VOLUME_ID, name: "Default", enabled: true, insertion_order: 0 });
  }
  return { book, codexPath, globalDir };
}

function saveBook(book: CharacterBook, codexPath: string, globalDir: string) {
  if (!fs.existsSync(globalDir)) {
    fs.mkdirSync(globalDir, { recursive: true });
  }
  fs.writeFileSync(codexPath, JSON.stringify(book, null, 2), "utf-8");
}

// PUT — update a volume (name, description, enabled, insertion_order)
export async function PUT(
  request: Request,
  { params }: { params: Promise<{ volumeId: string }> }
) {
  try {
    const { volumeId } = await params;
    const { book, codexPath, globalDir } = loadBook();
    const body = await request.json();

    const vol = (book.volumes || []).find((v) => v.id === volumeId);
    if (!vol) {
      return NextResponse.json({ error: "Volume not found" }, { status: 404 });
    }

    if (body.name !== undefined) vol.name = body.name;
    if (body.description !== undefined) vol.description = body.description;
    if (body.enabled !== undefined) vol.enabled = body.enabled;
    if (body.insertion_order !== undefined) vol.insertion_order = body.insertion_order;

    saveBook(book, codexPath, globalDir);

    return NextResponse.json({ volume: vol, success: true });
  } catch (error) {
    console.error("Error updating volume:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// DELETE — delete a volume, reassign entries to Default
export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ volumeId: string }> }
) {
  try {
    const { volumeId } = await params;

    if (volumeId === DEFAULT_VOLUME_ID) {
      return NextResponse.json(
        { error: "Cannot delete the Default volume" },
        { status: 400 }
      );
    }

    const { book, codexPath, globalDir } = loadBook();
    const volIndex = (book.volumes || []).findIndex((v) => v.id === volumeId);

    if (volIndex === -1) {
      return NextResponse.json({ error: "Volume not found" }, { status: 404 });
    }

    book.volumes!.splice(volIndex, 1);

    for (const entry of book.entries) {
      if (entry.volume_id === volumeId) {
        entry.volume_id = DEFAULT_VOLUME_ID;
      }
    }

    saveBook(book, codexPath, globalDir);

    return NextResponse.json({ success: true, volume_id: volumeId });
  } catch (error) {
    console.error("Error deleting volume:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
