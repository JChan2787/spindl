import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import type { VRM } from '@pixiv/three-vrm';
import { invoke } from '@tauri-apps/api/core';
import { remapMixamoAnimationToVrm } from './retarget';

// ── Types ────────────────────────────────────────────────────────────

export interface EmotionAnimationEntry {
  threshold: number;
  clip: string;
}

export interface AnimationConfig {
  default?: string;
  emotions?: Record<string, EmotionAnimationEntry>;
}

// ── Module state ─────────────────────────────────────────────────────

let mixer: THREE.AnimationMixer | null = null;
let currentAction: THREE.AnimationAction | null = null;
let currentVRM: VRM | null = null;
const loadedClips: Map<string, THREE.AnimationClip> = new Map();
let clipPlaying = false;

// Per-character animation config from character card (NANO-098 Session 3)
let animationConfig: AnimationConfig | null = null;
// Track which clip name is currently playing for dedup
let currentClipName: string | null = null;

// NANO-099: Base animations fallback (global defaults from Settings)
export interface BaseAnimationsConfig {
  idle: string | null;
  happy: string | null;
  sad: string | null;
  angry: string | null;
  curious: string | null;
}
let baseAnimations: BaseAnimationsConfig = { idle: null, happy: null, sad: null, angry: null, curious: null };

// Map classifier moods → base animation slots
const MOOD_TO_BASE_SLOT: Record<string, keyof BaseAnimationsConfig> = {
  amused: 'happy',
  melancholy: 'sad',
  annoyed: 'angry',
  curious: 'curious',
  surprised: 'curious',
};

// ── Public API ───────────────────────────────────────────────────────

/** Create a mixer targeting the VRM's scene graph. Call after every loadAvatar(). */
export function initMixer(vrm: VRM): void {
  disposeMixer();
  currentVRM = vrm;
  mixer = new THREE.AnimationMixer(vrm.scene);
}

/** Tear down mixer + actions. Call before VRM disposal. */
export function disposeMixer(): void {
  if (mixer) {
    mixer.stopAllAction();
    mixer.uncacheRoot(mixer.getRoot());
  }
  mixer = null;
  currentAction = null;
  currentVRM = null;
  loadedClips.clear();
  clipPlaying = false;
  currentClipName = null;
  // animationConfig is NOT cleared here — it's character-level state
  // set externally via setAnimationConfig(), not owned by the mixer.
}

/** Advance the mixer clock. Call once per frame in the render loop. */
export function updateMixer(delta: number): void {
  mixer?.update(delta);
}

/** True when a keyframed animation clip is driving the skeleton. */
export function isClipPlaying(): boolean {
  return clipPlaying;
}

/**
 * Load a Mixamo FBX file from disk, retarget it to the current VRM,
 * and store the clip by name for later playback.
 */
export async function loadFBXClip(name: string, filePath: string): Promise<void> {
  if (!currentVRM || !mixer) return;

  const bytes = await invoke<number[]>('read_file_bytes', { path: filePath });
  const blob = new Blob([new Uint8Array(bytes)], { type: 'application/octet-stream' });
  const blobUrl = URL.createObjectURL(blob);

  try {
    const fbxLoader = new FBXLoader();
    const asset = await new Promise<THREE.Group>((resolve, reject) => {
      fbxLoader.load(blobUrl, resolve, undefined, reject);
    });

    const clip = remapMixamoAnimationToVrm(currentVRM, asset);
    clip.name = name;
    loadedClips.set(name, clip);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

/**
 * Crossfade to a named clip. If already playing a different clip,
 * crossfades from the current action.
 */
export function playClip(name: string, crossfadeDuration = 0.3): void {
  if (!mixer) return;
  const clip = loadedClips.get(name);
  if (!clip) return;

  const newAction = mixer.clipAction(clip);
  newAction.setLoop(THREE.LoopRepeat, Infinity);

  if (currentAction && currentAction !== newAction) {
    newAction.reset();
    newAction.play();
    currentAction.crossFadeTo(newAction, crossfadeDuration, true);
  } else {
    newAction.reset();
    newAction.play();
  }

  currentAction = newAction;
  currentClipName = name;
  clipPlaying = true;
}

/** Fade out the current action and return to procedural idle. */
export function stopClip(crossfadeDuration = 0.3): void {
  if (!currentAction || !mixer) {
    clipPlaying = false;
    currentClipName = null;
    return;
  }
  currentAction.fadeOut(crossfadeDuration);
  // After fade completes, mark as not playing
  setTimeout(() => {
    clipPlaying = false;
  }, crossfadeDuration * 1000);
  currentAction = null;
  currentClipName = null;
}

/** Names of all loaded clips. */
export function getLoadedClipNames(): string[] {
  return Array.from(loadedClips.keys());
}

/**
 * Scan a directory for .fbx files via Tauri's list_directory command.
 * Returns filenames (without extension) mapped to full paths.
 */
export async function scanAnimations(dir: string): Promise<Map<string, string>> {
  const result = new Map<string, string>();
  try {
    const entries = await invoke<string[]>('list_directory', { path: dir });
    for (const entry of entries) {
      if (entry.toLowerCase().endsWith('.fbx')) {
        const name = entry.replace(/\.fbx$/i, '');
        // Build full path — dir may or may not end with separator
        const sep = dir.includes('/') ? '/' : '\\';
        const fullPath = dir.endsWith(sep) ? dir + entry : dir + sep + entry;
        result.set(name, fullPath);
      }
    }
  } catch {
    // Directory doesn't exist or is empty — graceful no-op
  }
  return result;
}

/**
 * Scan, load, and optionally auto-play animations.
 * Called after mixer init to populate clips from disk.
 */
export async function loadAnimationsFromDir(dir: string): Promise<void> {
  const animations = await scanAnimations(dir);
  for (const [name, path] of animations) {
    try {
      await loadFBXClip(name, path);
    } catch (err) {
      console.warn(`[SpindL] Failed to load animation "${name}":`, err);
    }
  }
}

// ── Animation Config (NANO-098 Session 3) ───────────────────────────

/** Set per-character animation config from character card. */
export function setAnimationConfig(config: AnimationConfig | null): void {
  animationConfig = config;
}

/** Get the current animation config. */
export function getAnimationConfig(): AnimationConfig | null {
  return animationConfig;
}

/** Set base animations config from Settings (NANO-099). */
export function setBaseAnimations(config: BaseAnimationsConfig): void {
  baseAnimations = config;
}

/** Get the current base animations config. */
export function getBaseAnimations(): BaseAnimationsConfig {
  return baseAnimations;
}

/**
 * Scan character-local + global animation directories, deduplicate
 * (character-local wins on name collision), and load all clips.
 */
export async function loadAnimationsWithFallback(
  characterDir: string,
  globalDir: string,
): Promise<void> {
  const charAnims = await scanAnimations(characterDir);
  const globalAnims = await scanAnimations(globalDir);

  // Merge: character-local takes priority
  const merged = new Map(globalAnims);
  for (const [name, path] of charAnims) {
    merged.set(name, path);
  }

  for (const [name, path] of merged) {
    try {
      await loadFBXClip(name, path);
    } catch (err) {
      console.warn(`[SpindL] Failed to load animation "${name}":`, err);
    }
  }
}

/**
 * Emotion-driven animation crossfade (NANO-098 Session 3).
 *
 * If the mood's confidence exceeds its threshold in the character's
 * animation config, crossfade to the emotion clip. Otherwise, crossfade
 * back to the default idle clip. Skips if the target clip is already playing.
 *
 * No-op when animationConfig is null (context menu behavior preserved).
 */
export function updateEmotionAnimation(mood: string | null, confidence: number): void {
  if (!mixer) return;
  // Allow base animations fallback even without per-character animationConfig
  const emotions = animationConfig?.emotions;

  // Check if current mood has a per-character animation above threshold
  if (mood && emotions && mood in emotions) {
    const { threshold, clip } = emotions[mood];
    if (confidence >= threshold && loadedClips.has(clip)) {
      if (currentClipName === clip) return; // already playing
      playClip(clip);
      return;
    }
  }

  // NANO-099: Fallback to base animations (global defaults from Settings)
  if (mood && mood in MOOD_TO_BASE_SLOT) {
    const slot = MOOD_TO_BASE_SLOT[mood];
    const baseClip = baseAnimations[slot];
    if (baseClip && loadedClips.has(baseClip)) {
      if (currentClipName === baseClip) return;
      playClip(baseClip);
      return;
    }
  }

  // Below threshold / no config / no base animation — return to default idle
  const defaultClip = animationConfig?.default ?? baseAnimations.idle;
  if (defaultClip && loadedClips.has(defaultClip)) {
    if (currentClipName === defaultClip) return; // already playing
    playClip(defaultClip);
  }
}
