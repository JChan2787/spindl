import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";
import * as zlib from "zlib";

import type { CharacterCardData } from "@/types/events";
import { getCharactersDir } from "@/lib/characters-dir";

const CARD_FILENAME = "card.json";
const AVATAR_FILENAME = "avatar.png";

// PNG signature bytes
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// ============================================
// Type Definitions
// ============================================

interface ExportCharacterRequest {
  character_id: string;
  format?: "json" | "png";
  include_spindl?: boolean;
  include_codex?: boolean;
}

interface ExportCompleteResponse {
  character_id: string;
  json_data: string;
  filename: string;
  success: boolean;
}

interface ExportErrorResponse {
  error: string;
}

// ============================================
// PNG tEXt Chunk Embedding
// ============================================

/**
 * Embed character card JSON into a PNG as a tEXt chunk.
 *
 * Inserts a tEXt chunk with keyword 'chara' and base64-encoded JSON
 * before the IEND chunk. Removes any existing 'chara' tEXt chunk first.
 */
function embedCharaInPng(pngBuffer: Buffer, jsonStr: string): Buffer {
  if (!pngBuffer.subarray(0, 8).equals(PNG_SIGNATURE)) {
    throw new Error("Input is not a valid PNG file");
  }

  // Build tEXt chunk payload: keyword\0base64_data
  const b64Data = Buffer.from(jsonStr, "utf-8").toString("base64");
  const textPayload = Buffer.concat([
    Buffer.from("chara\0", "latin1"),
    Buffer.from(b64Data, "latin1"),
  ]);

  // Calculate CRC over chunk type + data
  const crcInput = Buffer.concat([Buffer.from("tEXt", "ascii"), textPayload]);
  const crc = zlib.crc32(crcInput);

  // Build the full tEXt chunk: length + type + data + crc
  const lengthBuf = Buffer.alloc(4);
  lengthBuf.writeUInt32BE(textPayload.length, 0);
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc >>> 0, 0);

  const textChunk = Buffer.concat([lengthBuf, Buffer.from("tEXt", "ascii"), textPayload, crcBuf]);

  // Walk chunks, skip existing 'chara' tEXt, insert ours before IEND
  const parts: Buffer[] = [PNG_SIGNATURE];
  let offset = 8;

  while (offset < pngBuffer.length) {
    if (offset + 8 > pngBuffer.length) break;

    const chunkLength = pngBuffer.readUInt32BE(offset);
    const chunkType = pngBuffer.subarray(offset + 4, offset + 8).toString("ascii");
    const chunkEnd = offset + 12 + chunkLength;

    if (chunkEnd > pngBuffer.length) break;

    const rawChunk = pngBuffer.subarray(offset, chunkEnd);

    if (chunkType === "IEND") {
      parts.push(textChunk);
      parts.push(rawChunk);
      break;
    } else if (chunkType === "tEXt") {
      // Skip existing 'chara' chunks
      const chunkData = pngBuffer.subarray(offset + 8, offset + 8 + chunkLength);
      const nullIdx = chunkData.indexOf(0x00);
      if (nullIdx !== -1) {
        const keyword = chunkData.subarray(0, nullIdx).toString("latin1");
        if (keyword === "chara") {
          offset = chunkEnd;
          continue;
        }
      }
      parts.push(rawChunk);
    } else {
      parts.push(rawChunk);
    }

    offset = chunkEnd;
  }

  return Buffer.concat(parts);
}

// ============================================
// POST Handler - Export character (JSON or PNG)
// ============================================

export async function POST(request: Request) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const body: ExportCharacterRequest = await request.json();
    const {
      character_id,
      format = "json",
      include_spindl = true,
      include_codex = true,
    } = body;

    if (!character_id) {
      return NextResponse.json<ExportErrorResponse>(
        { error: "character_id is required" },
        { status: 400 }
      );
    }

    const characterDir = path.join(CHARACTERS_DIR, character_id);
    const cardPath = path.join(characterDir, CARD_FILENAME);

    // Check if character exists
    if (!fs.existsSync(cardPath)) {
      return NextResponse.json<ExportErrorResponse>(
        { error: `Character '${character_id}' not found` },
        { status: 404 }
      );
    }

    // Read card.json
    const cardContent = fs.readFileSync(cardPath, "utf-8");
    const card: CharacterCardData = JSON.parse(cardContent);

    // Create export copy (don't modify original)
    const exportCard: CharacterCardData = JSON.parse(JSON.stringify(card));

    // Remove spindl extensions if not requested
    if (!include_spindl && exportCard.data.extensions?.spindl) {
      delete exportCard.data.extensions.spindl;
      // Clean up empty extensions object
      if (Object.keys(exportCard.data.extensions).length === 0) {
        exportCard.data.extensions = {};
      }
    }

    // Remove character_book if not requested
    if (!include_codex) {
      delete exportCard.data.character_book;
    }

    // PNG export path
    if (format === "png") {
      const avatarPath = path.join(characterDir, AVATAR_FILENAME);

      if (!fs.existsSync(avatarPath)) {
        return NextResponse.json<ExportErrorResponse>(
          { error: `Character '${character_id}' has no avatar.png. PNG export requires an avatar image.` },
          { status: 400 }
        );
      }

      const pngBuffer = fs.readFileSync(avatarPath);
      const jsonStr = JSON.stringify(exportCard);
      const resultPng = embedCharaInPng(pngBuffer, jsonStr);

      const safeName = card.data.name.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_|_$/g, "");
      const filename = `${safeName}_card.png`;

      return new NextResponse(new Uint8Array(resultPng), {
        status: 200,
        headers: {
          "Content-Type": "image/png",
          "Content-Disposition": `attachment; filename="${filename}"`,
          "Content-Length": String(resultPng.length),
        },
      });
    }

    // JSON export path (existing behavior)
    const safeName = card.data.name.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_|_$/g, "");
    const filename = `${safeName}_card.json`;
    const jsonData = JSON.stringify(exportCard, null, 2);

    return NextResponse.json<ExportCompleteResponse>({
      character_id,
      json_data: jsonData,
      filename,
      success: true,
    });
  } catch (error) {
    console.error("Error exporting character:", error);
    return NextResponse.json<ExportErrorResponse>(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
