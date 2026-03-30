import * as fs from "fs";
import * as path from "path";
import yaml from "yaml";

const CONFIG_PATH = process.env.SPINDL_CONFIG || path.resolve(process.cwd(), "..", "config", "spindl.yaml");
const DEFAULT_CHARACTERS_DIR = path.resolve(process.cwd(), "..", "characters");

/**
 * Get the characters directory from the SpindL config file.
 * Falls back to ../characters if config is missing or doesn't specify a directory.
 *
 * Reads config on every call (not cached) because the E2E harness swaps
 * the config file at runtime before starting the frontend.
 */
export function getCharactersDir(): string {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const content = fs.readFileSync(CONFIG_PATH, "utf-8");
      const config = yaml.parse(content);
      const dir = config?.character?.directory;
      if (dir && typeof dir === "string") {
        const projectRoot = path.resolve(process.cwd(), "..");
        return path.resolve(projectRoot, dir);
      }
    }
  } catch {
    // Fall through to default on any parse error
  }
  return DEFAULT_CHARACTERS_DIR;
}
