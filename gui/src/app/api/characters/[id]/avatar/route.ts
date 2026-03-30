import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

import type { AvatarUploadedEvent, AvatarDataEvent, AvatarDataEventExtended } from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const AVATAR_FILENAME = "avatar.png";
const ORIGINAL_EXTENSIONS = ["png", "jpeg", "webp"];

// ============================================
// GET Handler - Serve avatar image
// ============================================

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const characterDir = path.join(CHARACTERS_DIR, id);
    const avatarPath = path.join(characterDir, AVATAR_FILENAME);

    // Check if character directory exists
    if (!fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Check if avatar exists
    let imageData: string | null = null;
    if (fs.existsSync(avatarPath)) {
      const avatarBuffer = fs.readFileSync(avatarPath);
      const base64 = avatarBuffer.toString("base64");
      imageData = `data:image/png;base64,${base64}`;
    }

    // Check if extended data requested
    const { searchParams } = new URL(request.url);
    const includeOriginal = searchParams.get("include_original") === "true";

    if (includeOriginal) {
      let originalData: string | null = null;
      let cropSettings: { scale: number; offsetX: number; offsetY: number } | null = null;

      // Find original avatar file (any supported extension)
      for (const ext of ORIGINAL_EXTENSIONS) {
        const originalPath = path.join(characterDir, `avatar_original.${ext}`);
        if (fs.existsSync(originalPath)) {
          const buffer = fs.readFileSync(originalPath);
          const base64 = buffer.toString("base64");
          originalData = `data:image/${ext};base64,${base64}`;
          break;
        }
      }

      // Load crop settings
      const settingsPath = path.join(characterDir, "crop_settings.json");
      if (fs.existsSync(settingsPath)) {
        const settingsContent = fs.readFileSync(settingsPath, "utf-8");
        cropSettings = JSON.parse(settingsContent);
      }

      return NextResponse.json<AvatarDataEventExtended>({
        character_id: id,
        image_data: imageData,
        original_data: originalData,
        crop_settings: cropSettings,
      });
    }

    return NextResponse.json<AvatarDataEvent>({
      character_id: id,
      image_data: imageData,
    });
  } catch (error) {
    console.error("Error getting avatar:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// POST Handler - Upload avatar image
// ============================================

interface UploadAvatarRequest {
  image_data: string;
  original_avatar_data?: string;
  crop_settings?: { scale: number; offsetX: number; offsetY: number };
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const body: UploadAvatarRequest = await request.json();
    const { image_data, original_avatar_data, crop_settings } = body;

    if (!image_data) {
      return NextResponse.json(
        { error: "image_data is required" },
        { status: 400 }
      );
    }

    const characterDir = path.join(CHARACTERS_DIR, id);
    const avatarPath = path.join(characterDir, AVATAR_FILENAME);

    // Check if character directory exists
    if (!fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 }
      );
    }

    // Write cropped avatar
    let base64Data = image_data;
    if (image_data.includes(",")) {
      base64Data = image_data.split(",")[1];
    }
    const imageBuffer = Buffer.from(base64Data, "base64");
    fs.writeFileSync(avatarPath, imageBuffer);

    // Write original avatar for re-editing
    if (original_avatar_data) {
      // Remove any previous original files (extension may differ)
      for (const ext of ORIGINAL_EXTENSIONS) {
        const oldPath = path.join(characterDir, `avatar_original.${ext}`);
        if (fs.existsSync(oldPath)) {
          fs.unlinkSync(oldPath);
        }
      }

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

    return NextResponse.json<AvatarUploadedEvent>({
      character_id: id,
      success: true,
    });
  } catch (error) {
    console.error("Error uploading avatar:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
