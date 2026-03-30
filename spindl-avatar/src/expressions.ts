import { setExpression } from './avatar';

export type BlendShapeTarget = Record<string, number>;

const MOOD_SHAPES: Record<string, BlendShapeTarget> = {
  default:      { relaxed: 1.0 },
  amused:       { happy: 1.0 },
  melancholy:   { sad: 1.0 },
  annoyed:      { angry: 1.0 },
  // curious has no built-in preset — falls through to per-character
  // expressionComposites (avatar_expressions.curious on the character card).
  // If no composite is set, falls back to neutral.
};

const ONSET_MS = 300;
const OFFSET_MS = 600;

const BROW_KEYS = new Set([
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
]);
const MOUTH_KEYS = new Set([
  'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthOpen', 'mouthPucker', 'jawOpen', 'happy', 'sad', 'angry',
]);
const BROW_OFFSET_MS = -50;
const MOUTH_OFFSET_MS = 100;

const OVERSHOOT_FACTOR = 0.05;

let currentShapes: BlendShapeTarget = { neutral: 1.0 };
let targetShapes: BlendShapeTarget = { neutral: 1.0 };
let transitionProgress = 1.0;
let transitionDurationMs = ONSET_MS;
let transitionSpeed = 1.0 / ONSET_MS;
let isOnset = false;

// Per-character expression composites (overrides for moods that the VRM
// doesn't support natively, e.g. "curious" built from multiple blendshapes).
// Set via setExpressionComposites() when a character loads.
let expressionComposites: Record<string, BlendShapeTarget> = {};

/** Set per-character expression composites. Overrides MOOD_SHAPES for matching moods. */
export function setExpressionComposites(composites: Record<string, BlendShapeTarget>): void {
  expressionComposites = composites;
}

export function getBlendShapesForMood(mood: string | null): BlendShapeTarget {
  if (!mood) return { neutral: 1.0 };
  // Composites take priority over built-in presets
  if (mood in expressionComposites) return { ...expressionComposites[mood] };
  if (mood in MOOD_SHAPES) return { ...MOOD_SHAPES[mood] };
  return { neutral: 1.0 };
}

export function lerpBlendShapes(
  from: BlendShapeTarget,
  to: BlendShapeTarget,
  t: number,
): BlendShapeTarget {
  const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
  const result: BlendShapeTarget = {};
  for (const key of allKeys) {
    const a = from[key] ?? 0;
    const b = to[key] ?? 0;
    const v = a + (b - a) * t;
    if (v > 0.001) result[key] = v;
  }
  return result;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function ease(t: number): number {
  return easeOutCubic(t);
}

export function setMood(mood: string | null, confidence: number = 1.0): void {
  currentShapes = getCurrentLerpedShapes();
  const raw = getBlendShapesForMood(mood);
  // Scale all blend shape values by confidence (subtle text → subtle expression)
  if (confidence < 1.0 && mood !== null) {
    for (const key of Object.keys(raw)) {
      raw[key] *= confidence;
    }
  }
  targetShapes = raw;
  transitionProgress = 0;
  isOnset = mood !== null;
  transitionDurationMs = isOnset ? ONSET_MS : OFFSET_MS;
  transitionSpeed = 1.0 / transitionDurationMs;
}

function getCurrentLerpedShapes(): BlendShapeTarget {
  if (transitionProgress >= 1.0) return { ...targetShapes };
  return lerpBlendShapes(currentShapes, targetShapes, ease(transitionProgress));
}

function staggerProgress(key: string, rawProgress: number): number {
  let offsetMs = 0;
  if (BROW_KEYS.has(key)) offsetMs = BROW_OFFSET_MS;
  else if (MOUTH_KEYS.has(key)) offsetMs = MOUTH_OFFSET_MS;
  if (offsetMs === 0) return rawProgress;

  const elapsedMs = rawProgress * transitionDurationMs;
  const shifted = Math.max(0, Math.min(transitionDurationMs, elapsedMs - offsetMs));
  return shifted / transitionDurationMs;
}

function applyOvershoot(easedT: number): number {
  return easedT + Math.sin(easedT * Math.PI) * OVERSHOOT_FACTOR;
}

export function updateExpressions(deltaMs: number): void {
  if (transitionProgress >= 1.0) {
    // Transition complete — keep applying target shapes every frame
    // (VRM expressionManager needs values pushed each frame)
    for (const key of Object.keys(targetShapes)) {
      setExpression(key, targetShapes[key]);
    }
    return;
  }

  transitionProgress = Math.min(1.0, transitionProgress + deltaMs * transitionSpeed);

  for (const key of Object.keys(currentShapes)) {
    setExpression(key, 0);
  }

  const allKeys = new Set([...Object.keys(currentShapes), ...Object.keys(targetShapes)]);
  for (const key of allKeys) {
    const keyProgress = isOnset
      ? staggerProgress(key, transitionProgress)
      : transitionProgress;

    let easedT = ease(keyProgress);

    if (isOnset && transitionProgress < 1.0) {
      easedT = applyOvershoot(easedT);
    }

    const from = currentShapes[key] ?? 0;
    const to = targetShapes[key] ?? 0;
    const v = from + (to - from) * easedT;
    if (v > 0.001) setExpression(key, v);
  }
}

export function getEmotionIntensity(): number {
  const EMOTION_KEYS = ['happy', 'angry', 'sad', 'surprised', 'relaxed'];
  const shapes = getCurrentLerpedShapes();
  let max = 0;
  for (const key of EMOTION_KEYS) {
    const v = shapes[key] ?? 0;
    if (v > max) max = v;
  }
  return max;
}

export function isTransitioning(): boolean {
  return transitionProgress < 1.0;
}

export function getAvailableMoods(): string[] {
  return Object.keys(MOOD_SHAPES);
}
