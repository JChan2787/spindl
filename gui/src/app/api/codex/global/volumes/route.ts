import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type { CharacterBook, CodexVolume } from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CODEX_FILENAME = "codex.json";

function loadBook(): { book: CharacterBook; codexPath: string; globalDir: string } {
  const globalDir = path.join(getCharactersDir(), "_global");
  const codexPath = path.join(globalDir, CODEX_FILENAME);

  if (!fs.existsSync(codexPath)) {
    return {
      book: {
        name: "Global Codex",
        description: "Entries active across all characters",
        entries: [],
        volumes: [{ id: "vol_default", name: "Default", enabled: true, insertion_order: 0 }],
        extensions: {},
      },
      codexPath,
      globalDir,
    };
  }

  const content = fs.readFileSync(codexPath, "utf-8");
  const book: CharacterBook = JSON.parse(content);
  if (!book.volumes) book.volumes = [];
  if (!book.volumes.some((v) => v.id === "vol_default")) {
    book.volumes.unshift({ id: "vol_default", name: "Default", enabled: true, insertion_order: 0 });
  }
  return { book, codexPath, globalDir };
}

function saveBook(book: CharacterBook, codexPath: string, globalDir: string) {
  if (!fs.existsSync(globalDir)) {
    fs.mkdirSync(globalDir, { recursive: true });
  }
  fs.writeFileSync(codexPath, JSON.stringify(book, null, 2), "utf-8");
}

function slugify(name: string): string {
  const slug = name.toLowerCase().trim().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
  return slug || "volume";
}

// GET — list all volumes
export async function GET() {
  try {
    const { book } = loadBook();
    return NextResponse.json({ volumes: book.volumes });
  } catch (error) {
    console.error("Error reading volumes:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// POST — create a new volume
export async function POST(request: Request) {
  try {
    const { book, codexPath, globalDir } = loadBook();
    const body = await request.json();
    const { name, description } = body as { name: string; description?: string };

    if (!name || typeof name !== "string" || !name.trim()) {
      return NextResponse.json({ error: "name is required" }, { status: 400 });
    }

    let slug = `vol_${slugify(name)}`;
    const existingIds = new Set((book.volumes || []).map((v) => v.id));
    if (existingIds.has(slug)) {
      let counter = 2;
      while (existingIds.has(`${slug}_${counter}`)) counter++;
      slug = `${slug}_${counter}`;
    }

    const maxOrder = Math.max(...(book.volumes || []).map((v) => v.insertion_order ?? 0), -1);

    const volume: CodexVolume = {
      id: slug,
      name: name.trim(),
      enabled: true,
      insertion_order: maxOrder + 1,
      ...(description ? { description } : {}),
    };

    book.volumes = book.volumes || [];
    book.volumes.push(volume);
    saveBook(book, codexPath, globalDir);

    return NextResponse.json({ volume, success: true }, { status: 201 });
  } catch (error) {
    console.error("Error creating volume:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
