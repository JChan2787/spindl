import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import { getCharactersDir } from "@/lib/characters-dir";

const CARD_FILENAME = "card.json";

// ============================================
// GET Handler - Check VRM status
// ============================================

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const characterDir = path.join(CHARACTERS_DIR, id);

    if (!fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Find any .vrm file in the character directory
    const files = fs.readdirSync(characterDir);
    const vrmFile = files.find((f) => f.toLowerCase().endsWith(".vrm"));

    return NextResponse.json({
      character_id: id,
      has_vrm: !!vrmFile,
      vrm_filename: vrmFile || null,
    });
  } catch (error) {
    console.error("Error checking VRM:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// POST Handler - Upload VRM file (FormData)
// ============================================

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const characterDir = path.join(CHARACTERS_DIR, id);

    if (!fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    const formData = await request.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json(
        { error: "file field is required (FormData with .vrm file)" },
        { status: 400 }
      );
    }

    // Use original filename if available, default to avatar.vrm
    const originalName =
      file instanceof File && file.name ? file.name : "avatar.vrm";
    const filename = originalName.toLowerCase().endsWith(".vrm")
      ? originalName
      : "avatar.vrm";

    // Write VRM file to character directory
    const buffer = Buffer.from(await file.arrayBuffer());
    const vrmPath = path.join(characterDir, filename);
    fs.writeFileSync(vrmPath, buffer);

    // Patch card.json to set avatar_vrm
    const cardPath = path.join(characterDir, CARD_FILENAME);
    if (fs.existsSync(cardPath)) {
      const cardContent = fs.readFileSync(cardPath, "utf-8");
      const card = JSON.parse(cardContent);

      if (!card.data) card.data = {};
      if (!card.data.extensions) card.data.extensions = {};
      if (!card.data.extensions.spindl) card.data.extensions.spindl = {};

      card.data.extensions.spindl.avatar_vrm = filename;

      fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");
    }

    return NextResponse.json({
      character_id: id,
      vrm_filename: filename,
      success: true,
    });
  } catch (error) {
    console.error("Error uploading VRM:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// DELETE Handler - Remove VRM file
// ============================================

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const characterDir = path.join(CHARACTERS_DIR, id);

    if (!fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Remove all .vrm files from character directory
    const files = fs.readdirSync(characterDir);
    for (const f of files) {
      if (f.toLowerCase().endsWith(".vrm")) {
        fs.unlinkSync(path.join(characterDir, f));
      }
    }

    // Clear avatar_vrm from card.json
    const cardPath = path.join(characterDir, CARD_FILENAME);
    if (fs.existsSync(cardPath)) {
      const cardContent = fs.readFileSync(cardPath, "utf-8");
      const card = JSON.parse(cardContent);

      if (card.data?.extensions?.spindl) {
        delete card.data.extensions.spindl.avatar_vrm;
        fs.writeFileSync(cardPath, JSON.stringify(card, null, 2), "utf-8");
      }
    }

    return NextResponse.json({
      character_id: id,
      success: true,
    });
  } catch (error) {
    console.error("Error deleting VRM:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
