import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";
import yaml from "yaml";

import { getCharactersDir } from "@/lib/characters-dir";

const CONFIG_PATH =
  process.env.SPINDL_CONFIG ||
  path.resolve(process.cwd(), "..", "config", "spindl.yaml");

const DEFAULT_GLOBAL_ANIMATIONS_DIR = "spindl-avatar/public/animations";

/**
 * Read global_animations_dir from spindl.yaml avatar config.
 * Falls back to the default global pool path if not configured.
 */
function getGlobalAnimationsDir(): string {
  const projectRoot = path.resolve(process.cwd(), "..");
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const content = fs.readFileSync(CONFIG_PATH, "utf-8");
      const config = yaml.parse(content);
      const dir = config?.avatar?.global_animations_dir;
      if (dir && typeof dir === "string") {
        return path.resolve(projectRoot, dir);
      }
    }
  } catch {
    // Fall through to default
  }
  return path.resolve(projectRoot, DEFAULT_GLOBAL_ANIMATIONS_DIR);
}

/**
 * Scan a directory for .fbx files and return animation names.
 */
function scanFbxDir(
  dir: string,
  source: "character" | "global",
): { name: string; source: string }[] {
  if (!fs.existsSync(dir)) return [];
  try {
    const entries = fs.readdirSync(dir);
    return entries
      .filter((f) => f.toLowerCase().endsWith(".fbx"))
      .map((f) => ({
        name: f.replace(/\.fbx$/i, ""),
        source,
      }));
  } catch {
    return [];
  }
}

// ============================================
// GET Handler - List available animations
// ============================================

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const CHARACTERS_DIR = getCharactersDir();
    const { id } = await params;
    const characterDir = path.join(CHARACTERS_DIR, id);

    if (!fs.existsSync(characterDir)) {
      return NextResponse.json(
        { error: `Character '${id}' not found` },
        { status: 404 },
      );
    }

    // Scan character-local animations
    const charAnimDir = path.join(characterDir, "animations");
    const charAnims = scanFbxDir(charAnimDir, "character");

    // Scan global pool
    const globalDir = getGlobalAnimationsDir();
    const globalAnims = scanFbxDir(globalDir, "global");

    // Merge: character-local wins on name collision
    const seen = new Set<string>();
    const merged: { name: string; source: string }[] = [];
    for (const anim of charAnims) {
      seen.add(anim.name);
      merged.push(anim);
    }
    for (const anim of globalAnims) {
      if (!seen.has(anim.name)) {
        merged.push(anim);
      }
    }

    // Sort alphabetically
    merged.sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json({ animations: merged });
  } catch (error) {
    console.error("Error scanning animations:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 },
    );
  }
}
