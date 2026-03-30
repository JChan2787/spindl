import { NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";
import yaml from "yaml";

const VALID_SLOTS = ["idle", "happy", "sad", "angry", "curious"] as const;
type Slot = (typeof VALID_SLOTS)[number];

/**
 * Resolve config path and parse avatar section.
 */
function getConfig(): {
  configPath: string;
  raw: Record<string, unknown>;
  avatarSection: Record<string, unknown>;
  globalAnimationsDir: string;
} {
  const configPath =
    process.env.SPINDL_CONFIG ||
    path.resolve(process.cwd(), "..", "config", "spindl.yaml");

  if (!fs.existsSync(configPath)) {
    throw new Error(`Config file not found: ${configPath}`);
  }

  const content = fs.readFileSync(configPath, "utf-8");
  const raw = yaml.parse(content) as Record<string, unknown>;
  const avatarSection = (raw.avatar || {}) as Record<string, unknown>;

  // Resolve global animations dir relative to project root
  const projectRoot = path.resolve(process.cwd(), "..");
  const animDir = (avatarSection.global_animations_dir as string) || "spindl-avatar/public/animations";
  const globalAnimationsDir = path.isAbsolute(animDir)
    ? animDir
    : path.resolve(projectRoot, animDir);

  return { configPath, raw, avatarSection, globalAnimationsDir };
}

/**
 * Write base_animations back to the YAML config using line-level surgery.
 * Preserves comments and formatting in the rest of the file.
 *
 * Strategy: find the avatar section, find any existing base_animations block
 * within it (replace), or append at the end of the avatar section.
 */
function persistBaseAnimations(
  configPath: string,
  baseAnimations: Record<string, string | null>
): void {
  const content = fs.readFileSync(configPath, "utf-8");
  const lines = content.split("\n");

  // Pass 1: find avatar section boundaries
  let avatarStart = -1;
  let avatarEnd = lines.length; // default: avatar runs to EOF

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const stripped = line.trim();
    if (!stripped || stripped.startsWith("#")) continue;

    // Top-level key: no leading whitespace, has a colon
    const isTopLevel = !line.startsWith(" ") && !line.startsWith("\t") && stripped.includes(":");

    if (isTopLevel) {
      if (stripped === "avatar:") {
        avatarStart = i;
      } else if (avatarStart >= 0 && i > avatarStart) {
        // First top-level key after avatar: — that's where avatar ends
        avatarEnd = i;
        break;
      }
    }
  }

  if (avatarStart < 0) {
    // No avatar section — nothing to do
    return;
  }

  // Pass 2: within avatar section, find existing base_animations block
  let baStart = -1;
  let baEnd = -1;

  for (let i = avatarStart + 1; i < avatarEnd; i++) {
    const line = lines[i];
    const stripped = line.trim();
    if (!stripped || stripped.startsWith("#")) continue;

    if (stripped === "base_animations:" && baStart < 0) {
      baStart = i;
      // Find the end: scan forward for lines with indent > this line's indent
      const myIndent = line.length - line.trimStart().length;
      baEnd = i + 1; // at minimum, replace just the header
      for (let k = i + 1; k < avatarEnd; k++) {
        const kLine = lines[k];
        const kStripped = kLine.trim();
        if (!kStripped || kStripped.startsWith("#")) {
          baEnd = k + 1;
          continue;
        }
        const kIndent = kLine.length - kLine.trimStart().length;
        if (kIndent > myIndent) {
          baEnd = k + 1; // child line, include it
        } else {
          break; // same or less indent — end of block
        }
      }
      break;
    }
  }

  const baLines = [
    "  base_animations:",
    `    idle: ${baseAnimations.idle ?? "null"}`,
    `    happy: ${baseAnimations.happy ?? "null"}`,
    `    sad: ${baseAnimations.sad ?? "null"}`,
    `    angry: ${baseAnimations.angry ?? "null"}`,
    `    curious: ${baseAnimations.curious ?? "null"}`,
  ];

  const result = [...lines];
  if (baStart >= 0) {
    // Replace existing block
    result.splice(baStart, baEnd - baStart, ...baLines);
  } else {
    // Append before the end of avatar section
    result.splice(avatarEnd, 0, ...baLines);
  }

  fs.writeFileSync(configPath, result.join("\n"), "utf-8");
}

// ============================================
// GET - Return current base animation config
// ============================================

export async function GET() {
  try {
    const { avatarSection, globalAnimationsDir } = getConfig();
    const baseAnimations = (avatarSection.base_animations || {}) as Record<
      string,
      string | null
    >;

    return NextResponse.json({
      idle: baseAnimations.idle ?? null,
      happy: baseAnimations.happy ?? null,
      sad: baseAnimations.sad ?? null,
      angry: baseAnimations.angry ?? null,
      curious: baseAnimations.curious ?? null,
      global_animations_dir: globalAnimationsDir,
    });
  } catch (error) {
    console.error("Error reading base animations config:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// POST - Upload FBX file to a slot
// ============================================

export async function POST(request: Request) {
  try {
    const { configPath, avatarSection, globalAnimationsDir } = getConfig();

    const formData = await request.formData();
    const slot = formData.get("slot") as string;
    const file = formData.get("file");

    if (!slot || !VALID_SLOTS.includes(slot as Slot)) {
      return NextResponse.json(
        { error: `Invalid slot: ${slot}. Must be one of: ${VALID_SLOTS.join(", ")}` },
        { status: 400 }
      );
    }

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json(
        { error: "file field is required (FormData with .fbx file)" },
        { status: 400 }
      );
    }

    const originalName =
      file instanceof File && file.name ? file.name : `${slot}.fbx`;
    if (!originalName.toLowerCase().endsWith(".fbx")) {
      return NextResponse.json(
        { error: "Only .fbx files are accepted" },
        { status: 400 }
      );
    }

    // Ensure animations directory exists
    if (!fs.existsSync(globalAnimationsDir)) {
      fs.mkdirSync(globalAnimationsDir, { recursive: true });
    }

    // Write FBX file to global animations directory
    const buffer = Buffer.from(await file.arrayBuffer());
    const destPath = path.join(globalAnimationsDir, originalName);
    fs.writeFileSync(destPath, buffer);

    // Derive clip name (filename without extension)
    const clipName = path.basename(originalName, path.extname(originalName));

    // Update config
    const baseAnimations = (avatarSection.base_animations || {}) as Record<
      string,
      string | null
    >;
    baseAnimations[slot] = clipName;

    persistBaseAnimations(configPath, {
      idle: baseAnimations.idle ?? null,
      happy: baseAnimations.happy ?? null,
      sad: baseAnimations.sad ?? null,
      angry: baseAnimations.angry ?? null,
      curious: baseAnimations.curious ?? null,
    });

    console.log(
      `[API] Base animation uploaded: ${slot} = ${clipName} (${destPath})`,
    );

    return NextResponse.json({
      slot,
      clip_name: clipName,
      filename: originalName,
      success: true,
    });
  } catch (error) {
    console.error("Error uploading base animation:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// ============================================
// DELETE - Clear a slot and remove the FBX file
// ============================================

export async function DELETE(request: Request) {
  try {
    const { configPath, avatarSection, globalAnimationsDir } = getConfig();

    const { slot } = (await request.json()) as { slot: string };

    if (!slot || !VALID_SLOTS.includes(slot as Slot)) {
      return NextResponse.json(
        { error: `Invalid slot: ${slot}. Must be one of: ${VALID_SLOTS.join(", ")}` },
        { status: 400 }
      );
    }

    const baseAnimations = (avatarSection.base_animations || {}) as Record<
      string,
      string | null
    >;
    const clipName = baseAnimations[slot];

    // Remove FBX file if it exists
    if (clipName) {
      const fbxPath = path.join(globalAnimationsDir, `${clipName}.fbx`);
      if (fs.existsSync(fbxPath)) {
        fs.unlinkSync(fbxPath);
        console.log(`[API] Removed base animation file: ${fbxPath}`);
      }
    }

    // Clear config
    baseAnimations[slot] = null;
    persistBaseAnimations(configPath, {
      idle: baseAnimations.idle ?? null,
      happy: baseAnimations.happy ?? null,
      sad: baseAnimations.sad ?? null,
      angry: baseAnimations.angry ?? null,
      curious: baseAnimations.curious ?? null,
    });

    console.log(`[API] Base animation cleared: ${slot}`);

    return NextResponse.json({
      slot,
      success: true,
    });
  } catch (error) {
    console.error("Error clearing base animation:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
