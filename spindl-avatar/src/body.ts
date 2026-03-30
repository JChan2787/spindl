import { getVRM } from './avatar';
import { isClipPlaying } from './mixer';
import { createSpring, springDamped, springUnderdamped, SpringState } from './spring';
import { fbm } from './noise';
import type { Mode, Mood } from './state';

const POSTURE_HL = 0.25;
const POSTURE_FAST_HL = 0.12;

interface PostureTarget {
  spineX: number;
  spineY: number;
  chestX: number;
  headX: number;
  headY: number;
  headZ: number;
  shoulderL: number;
  shoulderR: number;
  upperArmLX: number;
  upperArmRX: number;
  lowerArmLZ: number;
  lowerArmRZ: number;
}

const POSTURE_KEYS: (keyof PostureTarget)[] = [
  'spineX', 'spineY', 'chestX',
  'headX', 'headY', 'headZ',
  'shoulderL', 'shoulderR',
  'upperArmLX', 'upperArmRX',
  'lowerArmLZ', 'lowerArmRZ',
];

const ZERO_POSTURE: PostureTarget = {
  spineX: 0, spineY: 0, chestX: 0,
  headX: 0, headY: 0, headZ: 0,
  shoulderL: 0, shoulderR: 0,
  upperArmLX: 0, upperArmRX: 0,
  lowerArmLZ: 0, lowerArmRZ: 0,
};

const MODE_POSTURES: Record<Mode, PostureTarget> = {
  idle: { ...ZERO_POSTURE },
  thinking: {
    ...ZERO_POSTURE,
    headX: 0.1,
    headZ: 0.12,
    headY: 0.08,
    chestX: -0.04,
    upperArmRX: -0.3,
    lowerArmRZ: 0.4,
    shoulderR: 0.05,
  },
  speaking: {
    ...ZERO_POSTURE,
    spineX: 0.12,
    chestX: 0.08,
    headX: -0.06,
    shoulderL: -0.03,
    shoulderR: -0.03,
  },
};

const MOOD_POSTURES: Record<string, Partial<PostureTarget>> = {
  error: {
    shoulderL: 0.1,
    shoulderR: 0.1,
    headX: -0.06,
    spineX: 0.04,
  },
  success: {
    spineX: -0.06,
    chestX: -0.04,
    headX: -0.05,
  },
  warn: {
    shoulderL: 0.06,
    shoulderR: 0.06,
    headY: 0.1,
  },
  melancholy: {
    headX: 0.15,
    spineX: 0.08,
    chestX: 0.04,
    shoulderL: 0.08,
    shoulderR: 0.08,
  },
};

const DAMPING_RATIOS: Record<keyof PostureTarget, number> = {
  spineX: 0.82,
  spineY: 0.85,
  chestX: 0.78,
  headX: 0.72,
  headY: 0.72,
  headZ: 0.75,
  shoulderL: 0.80,
  shoulderR: 0.80,
  upperArmLX: 0.65,
  upperArmRX: 0.65,
  lowerArmLZ: 0.60,
  lowerArmRZ: 0.60,
};

const springs: Record<keyof PostureTarget, SpringState> = {} as any;
for (const key of POSTURE_KEYS) {
  springs[key] = createSpring();
}

let target: PostureTarget = { ...ZERO_POSTURE };

let speakingPhase = 0;
let speakingElapsed = 0;
const SPEAKING_SWAY_CYCLE = 3200;

const SPEAK_GESTURE_SHOULDER_AMOUNT = 0.06;
const SPEAK_GESTURE_ARM_AMOUNT = 0.10;

const AMP_NOD_THRESHOLD = 0.28;
const AMP_NOD_STRENGTH = 0.08;
const AMP_NOD_COOLDOWN_MS = 120;
let ampNodSpring: SpringState = createSpring();
let ampNodTarget = 0;
let ampNodCooldown = 0;
let prevAmplitude = 0;

const BEAT_BOB_PRIMARY_HZ = 3.2;
const BEAT_BOB_SECONDARY_HZ = 1.8;
const BEAT_BOB_TERTIARY_HZ = 0.7;
const BEAT_BOB_PRIMARY_AMOUNT = 0.025;
const BEAT_BOB_SECONDARY_AMOUNT = 0.015;
const BEAT_BOB_TERTIARY_AMOUNT = 0.035;

const GESTURE_TURN_RANGE = 0.10;
const GESTURE_TURN_MIN_INTERVAL = 2500;
const GESTURE_TURN_MAX_INTERVAL = 5000;
let gestureTurnSpring: SpringState = createSpring();
let gestureTurnTarget = 0;
let gestureTurnTimer = 0;
let gestureTurnNextInterval = 3000;

const SECONDARY_CHEST_DAMPING = 0.3;
const SECONDARY_SPINE_DAMPING = 0.12;
let secondaryChestSpringY: SpringState = createSpring();
let secondaryChestSpringX: SpringState = createSpring();
let secondarySpineSpringY: SpringState = createSpring();
let secondarySpineSpringX: SpringState = createSpring();
const SECONDARY_HL = 0.15;

function randomGestureInterval(): number {
  return GESTURE_TURN_MIN_INTERVAL + Math.random() * (GESTURE_TURN_MAX_INTERVAL - GESTURE_TURN_MIN_INTERVAL);
}

function randomGestureTurnTarget(): number {
  const sign = Math.random() > 0.5 ? 1 : -1;
  return sign * (0.3 + Math.random() * 0.7) * GESTURE_TURN_RANGE;
}

function addPosture(base: PostureTarget, overlay: Partial<PostureTarget>): PostureTarget {
  return {
    spineX: base.spineX + (overlay.spineX ?? 0),
    spineY: base.spineY + (overlay.spineY ?? 0),
    chestX: base.chestX + (overlay.chestX ?? 0),
    headX: base.headX + (overlay.headX ?? 0),
    headY: base.headY + (overlay.headY ?? 0),
    headZ: base.headZ + (overlay.headZ ?? 0),
    shoulderL: base.shoulderL + (overlay.shoulderL ?? 0),
    shoulderR: base.shoulderR + (overlay.shoulderR ?? 0),
    upperArmLX: base.upperArmLX + (overlay.upperArmLX ?? 0),
    upperArmRX: base.upperArmRX + (overlay.upperArmRX ?? 0),
    lowerArmLZ: base.lowerArmLZ + (overlay.lowerArmLZ ?? 0),
    lowerArmRZ: base.lowerArmRZ + (overlay.lowerArmRZ ?? 0),
  };
}

// Anticipation
const ANTICIPATION_DURATION = 120;
const ANTICIPATION_STRENGTH = 0.4;
let anticipationTimer = 0;
let anticipationTarget: PostureTarget = { ...ZERO_POSTURE };
let prevTarget: PostureTarget = { ...ZERO_POSTURE };
let lastModeForAnticipation: Mode | null = null;

// Keystroke reaction
const KEYSTROKE_HEAD_IMPULSE = 0.45;
const KEYSTROKE_HEAD_TILT_IMPULSE = 0.27;
const KEYSTROKE_SHOULDER_IMPULSE = 0.36;
const KEYSTROKE_COOLDOWN_MS = 80;
let keystrokeCooldown = 0;

export function triggerKeystrokeReaction(): void {
  if (keystrokeCooldown > 0) return;
  keystrokeCooldown = KEYSTROKE_COOLDOWN_MS;
  const jitter = 0.7 + Math.random() * 0.6;
  springs.headX.vel += KEYSTROKE_HEAD_IMPULSE * jitter;
  const tiltDir = Math.random() > 0.5 ? 1 : -1;
  springs.headZ.vel += KEYSTROKE_HEAD_TILT_IMPULSE * jitter * tiltDir;
  if (Math.random() < 0.2) {
    springs.headX.vel += KEYSTROKE_HEAD_IMPULSE * 1.8;
  }
  const side = Math.random() > 0.5 ? 1 : -1;
  const sJitter = 0.7 + Math.random() * 0.6;
  springs.shoulderL.vel += KEYSTROKE_SHOULDER_IMPULSE * sJitter * (side > 0 ? 1 : 0.3);
  springs.shoulderR.vel += KEYSTROKE_SHOULDER_IMPULSE * sJitter * (side > 0 ? 0.3 : 1);
}

export function updateBody(deltaMs: number, mode: Mode, mood: Mood, amplitude: number): void {
  const vrm = getVRM();
  if (!vrm?.humanoid) return;
  const dt = deltaMs / 1000;

  keystrokeCooldown = Math.max(0, keystrokeCooldown - deltaMs);

  const modePosture = MODE_POSTURES[mode] ?? ZERO_POSTURE;
  const moodOverlay = mood && MOOD_POSTURES[mood] ? MOOD_POSTURES[mood] : {};
  const realTarget = addPosture(modePosture, moodOverlay);

  if (mode !== lastModeForAnticipation && lastModeForAnticipation !== null) {
    anticipationTimer = ANTICIPATION_DURATION;
    for (const key of POSTURE_KEYS) {
      const delta = realTarget[key] - prevTarget[key];
      anticipationTarget[key] = prevTarget[key] - delta * ANTICIPATION_STRENGTH;
    }
  }
  lastModeForAnticipation = mode;

  if (anticipationTimer > 0) {
    anticipationTimer -= deltaMs;
    target = anticipationTimer > 0 ? anticipationTarget : realTarget;
  } else {
    target = realTarget;
  }
  prevTarget = realTarget;

  const hl = (mood === 'annoyed') ? POSTURE_FAST_HL : POSTURE_HL;

  for (const key of POSTURE_KEYS) {
    springs[key] = springUnderdamped(springs[key], target[key], hl, DAMPING_RATIOS[key], dt);
  }

  let speakingSway = 0;
  let speakingNod = 0;
  let beatBob = 0;
  let gestureTurnOffset = 0;
  if (mode === 'speaking') {
    speakingPhase += (deltaMs / SPEAKING_SWAY_CYCLE) * Math.PI * 2;
    speakingElapsed += dt;
    const energy = Math.min(amplitude * 2.0, 1.0);
    speakingSway = Math.sin(speakingPhase) * 0.05 * energy;
    speakingNod = Math.sin(speakingPhase * 1.7) * 0.03 * energy;

    const t = speakingElapsed;
    beatBob = (
      Math.sin(t * BEAT_BOB_PRIMARY_HZ * Math.PI * 2) * BEAT_BOB_PRIMARY_AMOUNT +
      Math.sin(t * BEAT_BOB_SECONDARY_HZ * Math.PI * 2 + 0.7) * BEAT_BOB_SECONDARY_AMOUNT +
      Math.sin(t * BEAT_BOB_TERTIARY_HZ * Math.PI * 2 + 1.4) * BEAT_BOB_TERTIARY_AMOUNT
    ) * energy;

    ampNodCooldown = Math.max(0, ampNodCooldown - deltaMs);
    const ampDelta = amplitude - prevAmplitude;
    if (ampDelta > 0.05 && amplitude > AMP_NOD_THRESHOLD && ampNodCooldown <= 0) {
      ampNodTarget = AMP_NOD_STRENGTH * Math.min(amplitude * 2, 1.0);
      ampNodCooldown = AMP_NOD_COOLDOWN_MS;
    } else if (ampNodCooldown <= 0) {
      ampNodTarget = 0;
    }
    const nodHL = ampNodTarget > ampNodSpring.pos ? 0.04 : 0.15;
    ampNodSpring = springDamped(ampNodSpring, ampNodTarget, nodHL, dt);

    gestureTurnTimer += deltaMs;
    if (gestureTurnTimer >= gestureTurnNextInterval) {
      gestureTurnTarget = randomGestureTurnTarget();
      gestureTurnNextInterval = randomGestureInterval();
      gestureTurnTimer = 0;
    }
    gestureTurnSpring = springDamped(gestureTurnSpring, gestureTurnTarget, 0.3, dt);
    gestureTurnOffset = gestureTurnSpring.pos;
  } else {
    ampNodSpring = springDamped(ampNodSpring, 0, 0.2, dt);
    gestureTurnSpring = springDamped(gestureTurnSpring, 0, 0.4, dt);
    gestureTurnTimer = 0;
    ampNodCooldown = 0;
  }
  prevAmplitude = amplitude;

  const head = vrm.humanoid.getNormalizedBoneNode('head');
  if (head) {
    head.rotation.x += -springs.headX.pos - ampNodSpring.pos - beatBob;
    head.rotation.y += springs.headY.pos + gestureTurnOffset;
    head.rotation.z += springs.headZ.pos;

    const headY = head.rotation.y;
    const headX = head.rotation.x;

    secondaryChestSpringY = springDamped(secondaryChestSpringY, headY * SECONDARY_CHEST_DAMPING, SECONDARY_HL, dt);
    secondaryChestSpringX = springDamped(secondaryChestSpringX, headX * SECONDARY_CHEST_DAMPING, SECONDARY_HL, dt);
    secondarySpineSpringY = springDamped(secondarySpineSpringY, headY * SECONDARY_SPINE_DAMPING, SECONDARY_HL * 1.5, dt);
    secondarySpineSpringX = springDamped(secondarySpineSpringX, headX * SECONDARY_SPINE_DAMPING, SECONDARY_HL * 1.5, dt);
  }

  const clipActive = isClipPlaying();

  const chest = vrm.humanoid.getNormalizedBoneNode('chest');
  if (chest) {
    chest.rotation.x += -springs.chestX.pos - speakingNod + secondaryChestSpringX.pos;
    if (clipActive) {
      chest.rotation.y += secondaryChestSpringY.pos;
    } else {
      chest.rotation.y = secondaryChestSpringY.pos;
    }
  }

  const spine = vrm.humanoid.getNormalizedBoneNode('spine');
  if (spine) {
    if (clipActive) {
      spine.rotation.y += springs.spineY.pos + speakingSway + secondarySpineSpringY.pos;
      // Don't zero out spine.rotation.z — clip is driving it
    } else {
      spine.rotation.y = springs.spineY.pos + speakingSway + secondarySpineSpringY.pos;
      spine.rotation.z = 0;
    }
    spine.rotation.x += -springs.spineX.pos + secondarySpineSpringX.pos;
  }

  let speakShoulderL = 0, speakShoulderR = 0;
  let speakArmLX = 0, speakArmRX = 0;
  if (mode === 'speaking') {
    const t = speakingElapsed;
    const energy = Math.min(amplitude * 2.0, 1.0);
    speakShoulderL = fbm(t * 0.4 + 1700, 2) * SPEAK_GESTURE_SHOULDER_AMOUNT * energy;
    speakShoulderR = fbm(t * 0.35 + 1800, 2) * SPEAK_GESTURE_SHOULDER_AMOUNT * energy;
    speakArmLX = fbm(t * 0.25 + 1900, 2) * SPEAK_GESTURE_ARM_AMOUNT * energy;
    speakArmRX = fbm(t * 0.3 + 2000, 2) * SPEAK_GESTURE_ARM_AMOUNT * energy;
  }

  const leftShoulder = vrm.humanoid.getNormalizedBoneNode('leftShoulder');
  if (leftShoulder) {
    leftShoulder.rotation.z += springs.shoulderL.pos + speakShoulderL;
  }

  const rightShoulder = vrm.humanoid.getNormalizedBoneNode('rightShoulder');
  if (rightShoulder) {
    rightShoulder.rotation.z += springs.shoulderR.pos + speakShoulderR;
  }

  const leftUpperArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
  if (leftUpperArm) {
    leftUpperArm.rotation.x += springs.upperArmLX.pos + speakArmLX;
  }
  const rightUpperArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
  if (rightUpperArm) {
    rightUpperArm.rotation.x += springs.upperArmRX.pos + speakArmRX;
  }
  const leftLowerArm = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
  if (leftLowerArm && springs.lowerArmLZ.pos !== 0) {
    leftLowerArm.rotation.z += springs.lowerArmLZ.pos;
  }
  const rightLowerArm = vrm.humanoid.getNormalizedBoneNode('rightLowerArm');
  if (rightLowerArm && springs.lowerArmRZ.pos !== 0) {
    rightLowerArm.rotation.z += springs.lowerArmRZ.pos;
  }
}

export function resetBody(): void {
  for (const key of POSTURE_KEYS) {
    springs[key] = createSpring();
  }
  target = { ...ZERO_POSTURE };
  anticipationTimer = 0;
  anticipationTarget = { ...ZERO_POSTURE };
  prevTarget = { ...ZERO_POSTURE };
  lastModeForAnticipation = null;
  speakingPhase = 0;
  speakingElapsed = 0;
  ampNodSpring = createSpring();
  ampNodTarget = 0;
  ampNodCooldown = 0;
  prevAmplitude = 0;
  gestureTurnSpring = createSpring();
  gestureTurnTarget = 0;
  gestureTurnTimer = 0;
  gestureTurnNextInterval = 3000;
  secondaryChestSpringY = createSpring();
  secondaryChestSpringX = createSpring();
  secondarySpineSpringY = createSpring();
  secondarySpineSpringX = createSpring();
}
