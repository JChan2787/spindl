import { setExpression, getVRM, REST_ARM_Z, REST_FOREARM_Z } from './avatar';
import { getEmotionIntensity, isTransitioning } from './expressions';
import { isClipPlaying } from './mixer';
import { fbm } from './noise';
import { createSpring, springDamped, SpringState } from './spring';

// Blink config
const BLINK_MIN_INTERVAL = 2500;
const BLINK_MAX_INTERVAL = 7000;
const BLINK_CLOSE_MS = 75;
const BLINK_HOLD_MS = 40;
const BLINK_OPEN_MS = 200;
const BLINK_TOTAL_MS = BLINK_CLOSE_MS + BLINK_HOLD_MS + BLINK_OPEN_MS;
const DOUBLE_BLINK_CHANCE = 0.2;
const DOUBLE_BLINK_GAP = 120;

// Breathing config
const BREATHE_PRIMARY_CYCLE = 4200;
const BREATHE_CHEST_X = 0.012;
const BREATHE_CHEST_SCALE = 0.003;
const BREATHE_SPINE_AMOUNT = 0.006;
const BREATHE_SHOULDER_AMOUNT = 0.008;
const BREATHE_SHOULDER_PHASE = 0.3;
const BREATHE_HEAD_AMOUNT = 0.004;
const BREATHE_NECK_AMOUNT = 0.003;

// Asymmetric idle pose
const ASYM_SHOULDER_OFFSET = 0.008;
const ASYM_HEAD_TILT = 0.015;
const ASYM_HIP_LEAN = 0.004;

// Head tilt holds
const TILT_HOLD_MIN_INTERVAL = 8000;
const TILT_HOLD_MAX_INTERVAL = 20000;
const TILT_HOLD_DURATION = 2000;
const TILT_HOLD_AMOUNT = 0.06;

// Saccade config
const SACCADE_MIN_INTERVAL = 2000;
const SACCADE_MAX_INTERVAL = 6000;
const SACCADE_RANGE_X = 0.35;
const SACCADE_RANGE_Y = 0.2;

// Mouse tracking config
const MOUSE_EYE_STRENGTH_X = 0.7;
const MOUSE_EYE_STRENGTH_Y = 0.45;
const MOUSE_HEAD_STRENGTH_X = 0.30;
const MOUSE_HEAD_STRENGTH_Y = 0.18;
const MOUSE_SACCADE_BLEND = 0.12;

// Spring half-lives
const EYE_SPRING_HL = 0.06;
const SACCADE_SNAP_HL = 0.03;
const SACCADE_SETTLE_HL = 0.3;
const HEAD_SPRING_HL = 0.18;
const TILT_SPRING_HL = 0.4;

// Gaze-breaking
const GAZE_BREAK_MIN_INTERVAL = 5000;
const GAZE_BREAK_MAX_INTERVAL = 15000;
const GAZE_BREAK_DURATION = 400;
const GAZE_BREAK_AMOUNT = 0.4;

// Contrapposto weight shift
const CONTRA_SHIFT_MIN_INTERVAL = 5000;
const CONTRA_SHIFT_MAX_INTERVAL = 15000;
const CONTRA_HL = 0.6;
const CONTRA_HIP_Z = 0.035;
const CONTRA_SPINE_Z = 0.021;
const CONTRA_CHEST_Z = 0.014;
const CONTRA_NECK_Z = 0.01;

// Micro-fidgets
const FIDGET_MIN_INTERVAL = 4000;
const FIDGET_MAX_INTERVAL = 10000;

type FidgetType = 'weightShift' | 'deepBreath' | 'postureAdjust' | 'shoulderSettle';

interface FidgetDef {
  duration: number;
  hipZ?: number;
  spineX?: number;
  shoulderL?: number;
  shoulderR?: number;
  chestScale?: number;
}

const FIDGET_DEFS: Record<FidgetType, FidgetDef> = {
  weightShift: { duration: 1800, hipZ: 0.025 },
  deepBreath: { duration: 2400, chestScale: 2.5, shoulderL: 0.01, shoulderR: 0.01 },
  postureAdjust: { duration: 1200, spineX: -0.03 },
  shoulderSettle: { duration: 1000, shoulderL: 0.015, shoulderR: -0.008 },
};

const FIDGET_TYPES: FidgetType[] = ['weightShift', 'deepBreath', 'postureAdjust', 'shoulderSettle'];

// State
let blinkTimer = randomRange(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL);
let blinkProgress = -1;
let doubleBlink = false;
let doubleBlinkGap = -1;

let breathePrimaryPhase = 0;
let saccadeTimer = randomRange(SACCADE_MIN_INTERVAL, SACCADE_MAX_INTERVAL);
let saccadeTargetX = 0;
let saccadeTargetY = 0;
let saccadeJustMoved = false;
let saccadeSettleTimer = 0;

let eyeSpringX: SpringState = createSpring();
let eyeSpringY: SpringState = createSpring();
let headSpringX: SpringState = createSpring();
let headSpringY: SpringState = createSpring();
let saccadeSpringX: SpringState = createSpring();
let saccadeSpringY: SpringState = createSpring();
let tiltSpring: SpringState = createSpring();

let elapsed = 0;

let tiltHoldTimer = randomRange(TILT_HOLD_MIN_INTERVAL, TILT_HOLD_MAX_INTERVAL);
let tiltHoldActive = false;
let tiltHoldProgress = 0;
let tiltHoldTarget = 0;

let mouseNormX = 0;
let mouseNormY = 0;

let contraSpring: SpringState = createSpring();
let contraTimer = randomRange(CONTRA_SHIFT_MIN_INTERVAL, CONTRA_SHIFT_MAX_INTERVAL);
let contraTarget = 0;

let fidgetTimer = randomRange(FIDGET_MIN_INTERVAL, FIDGET_MAX_INTERVAL);
let fidgetActive = false;
let fidgetProgress = 0;
let fidgetDuration = 0;
let fidgetSign = 1;
let fidgetHipZ: SpringState = createSpring();
let fidgetSpineX: SpringState = createSpring();
let fidgetShoulderL: SpringState = createSpring();
let fidgetShoulderR: SpringState = createSpring();
let fidgetBreathScale = 1.0;
let currentFidget: FidgetDef | null = null;

let gazeBreakTimer = randomRange(GAZE_BREAK_MIN_INTERVAL, GAZE_BREAK_MAX_INTERVAL);
let gazeBreakActive = false;
let gazeBreakProgress = 0;
let gazeBreakOffsetX = 0;
let gazeBreakOffsetY = 0;

// Track when cursor last moved — stale cursor decays gaze back to camera
let lastMouseMoveTime = 0;
const MOUSE_STALE_MS = 1500; // after 1.5s of no movement, gaze returns to center

export function setMousePosition(normX: number, normY: number): void {
  if (Math.abs(normX - mouseNormX) > 0.01 || Math.abs(normY - mouseNormY) > 0.01) {
    lastMouseMoveTime = performance.now();
  }
  mouseNormX = normX;
  mouseNormY = normY;
}

function randomRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function blinkValue(progress: number): number {
  if (progress < BLINK_CLOSE_MS) {
    const t = progress / BLINK_CLOSE_MS;
    return t * t;
  } else if (progress < BLINK_CLOSE_MS + BLINK_HOLD_MS) {
    return 1.0;
  } else {
    const t = (progress - BLINK_CLOSE_MS - BLINK_HOLD_MS) / BLINK_OPEN_MS;
    return 1.0 - (1.0 - Math.pow(1.0 - t, 3));
  }
}

let prevMode = 'idle';

export function triggerBlink(): void {
  if (blinkProgress < 0 && doubleBlinkGap < 0) {
    startBlink();
  }
}

function startBlink(): void {
  blinkProgress = 0;
  const wobble = (Math.random() - 0.5) * 0.06;
  eyeSpringX.vel += wobble * 8;
  saccadeSpringX.vel += wobble * 4;
}

export function updateIdle(deltaMs: number, isSpeaking: boolean, mode: string = 'idle', mood: string | null = null): void {
  elapsed += deltaMs;
  const vrm = getVRM();
  const dt = deltaMs / 1000;

  if (mode !== prevMode) {
    if (blinkProgress < 0 && doubleBlinkGap < 0) {
      startBlink();
    }
    prevMode = mode;
  }

  const moodTension = (mood === 'annoyed') ? 1.25 :
                       (mood === 'melancholy') ? 0.7 :
                       (mood === 'amused') ? 0.85 : 1.0;
  const moodBreathSpeed = (mood === 'annoyed') ? 1.3 :
                           (mood === 'melancholy') ? 0.8 : 1.0;

  const breatheSpeed = (mode === 'speaking' ? 1.4 : mode === 'thinking' ? 0.7 : 1.0) * moodBreathSpeed;
  const headAmplitude = (mode === 'speaking' ? 2.0 : mode === 'thinking' ? 0.6 : 1.2) * (moodTension > 1.0 ? 1.2 : 1.0);
  const bodyAmplitude = (mode === 'speaking' ? 1.0 : mode === 'thinking' ? 0.4 : 1.0) * moodTension;
  const noiseTime = elapsed / 1000;

  // Blinking
  const emotionIntensity = getEmotionIntensity();
  const blinkScale = 1 - emotionIntensity * 0.5;

  if (blinkProgress >= 0) {
    blinkProgress += deltaMs;
    if (blinkProgress < BLINK_TOTAL_MS) {
      const bv = blinkValue(blinkProgress) * blinkScale;
      setExpression('blink', bv);
    } else {
      setExpression('blink', 0);
      blinkProgress = -1;
      if (doubleBlink) {
        doubleBlink = false;
        doubleBlinkGap = DOUBLE_BLINK_GAP;
      } else {
        const blinkRateMultiplier = mode === 'speaking' ? 0.8 :
          emotionIntensity > 0.3 ? 0.7 : 1.0;
        blinkTimer = randomRange(
          BLINK_MIN_INTERVAL * blinkRateMultiplier,
          BLINK_MAX_INTERVAL * blinkRateMultiplier
        );
      }
    }
  } else if (doubleBlinkGap >= 0) {
    doubleBlinkGap -= deltaMs;
    if (doubleBlinkGap <= 0) {
      doubleBlinkGap = -1;
      startBlink();
      blinkTimer = randomRange(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL);
    }
  } else {
    blinkTimer -= deltaMs;
    if (blinkTimer <= 0) {
      startBlink();
      doubleBlink = Math.random() < DOUBLE_BLINK_CHANCE;
    }
  }

  // Breathing
  const breatheRateVar = 1.0 + fbm(noiseTime * 0.05 + 3000, 2) * 0.15;
  breathePrimaryPhase += (deltaMs / BREATHE_PRIMARY_CYCLE) * Math.PI * 2 * breatheSpeed * breatheRateVar;
  const breatheDepth = 0.003 + fbm(noiseTime * 0.08 + 300, 2) * 0.002;
  const breatheAmount = Math.sin(breathePrimaryPhase) * breatheDepth;
  const breatheCycle = Math.sin(breathePrimaryPhase);
  const breatheShoulder = Math.sin(breathePrimaryPhase - BREATHE_SHOULDER_PHASE);
  const shoulderInhaleRise = Math.max(0, breatheShoulder) * 0.002;

  // Head tilt holds
  if (!tiltHoldActive) {
    tiltHoldTimer -= deltaMs;
    if (tiltHoldTimer <= 0) {
      tiltHoldActive = true;
      tiltHoldProgress = 0;
      tiltHoldTarget = (Math.random() > 0.5 ? 1 : -1) * (0.5 + Math.random() * 0.5) * TILT_HOLD_AMOUNT;
    }
  } else {
    tiltHoldProgress += deltaMs;
    if (tiltHoldProgress >= TILT_HOLD_DURATION) {
      tiltHoldActive = false;
      tiltHoldTarget = 0;
      tiltHoldTimer = randomRange(TILT_HOLD_MIN_INTERVAL, TILT_HOLD_MAX_INTERVAL);
    }
  }
  tiltSpring = springDamped(tiltSpring, tiltHoldTarget, TILT_SPRING_HL, dt);

  // Micro-fidgets
  const fidgetRateScale = moodTension > 1.2 ? 0.5 : moodTension < 0.8 ? 1.5 : 1.0;
  if (mode === 'idle') {
    if (!fidgetActive) {
      fidgetTimer -= deltaMs;
      if (fidgetTimer <= 0) {
        fidgetActive = true;
        fidgetProgress = 0;
        const type = FIDGET_TYPES[Math.floor(Math.random() * FIDGET_TYPES.length)];
        currentFidget = FIDGET_DEFS[type];
        fidgetDuration = currentFidget.duration;
        fidgetSign = Math.random() > 0.5 ? 1 : -1;
      }
    } else {
      fidgetProgress += deltaMs;
      if (fidgetProgress >= fidgetDuration) {
        fidgetActive = false;
        currentFidget = null;
        fidgetTimer = randomRange(FIDGET_MIN_INTERVAL * fidgetRateScale, FIDGET_MAX_INTERVAL * fidgetRateScale);
      }
    }
  }

  const fidgetPhase = currentFidget && fidgetActive
    ? (fidgetProgress < fidgetDuration * 0.4 ? fidgetProgress / (fidgetDuration * 0.4) :
       fidgetProgress < fidgetDuration * 0.7 ? 1.0 :
       1.0 - (fidgetProgress - fidgetDuration * 0.7) / (fidgetDuration * 0.3))
    : 0;
  const fHL = 0.12;
  fidgetHipZ = springDamped(fidgetHipZ, (currentFidget?.hipZ ?? 0) * fidgetPhase * fidgetSign, fHL, dt);
  fidgetSpineX = springDamped(fidgetSpineX, (currentFidget?.spineX ?? 0) * fidgetPhase, fHL, dt);
  fidgetShoulderL = springDamped(fidgetShoulderL, (currentFidget?.shoulderL ?? 0) * fidgetPhase, fHL, dt);
  fidgetShoulderR = springDamped(fidgetShoulderR, (currentFidget?.shoulderR ?? 0) * fidgetPhase, fHL, dt);
  fidgetBreathScale = 1.0 + ((currentFidget?.chestScale ?? 1.0) - 1.0) * fidgetPhase;

  // Contrapposto
  if (mode === 'idle') {
    contraTimer -= deltaMs;
    if (contraTimer <= 0) {
      contraTarget = [-1, 0, 1][Math.floor(Math.random() * 3)];
      contraTimer = randomRange(CONTRA_SHIFT_MIN_INTERVAL, CONTRA_SHIFT_MAX_INTERVAL);
    }
  }
  contraSpring = springDamped(contraSpring, mode === 'idle' ? contraTarget : 0, CONTRA_HL, dt);
  const cw = contraSpring.pos;

  const clipActive = isClipPlaying();

  if (vrm?.humanoid) {
    // When a keyframed clip is playing, the mixer drives the skeleton.
    // Only keep additive breathing chest scale — gate everything else.
    if (!clipActive) {
      const chest = vrm.humanoid.getNormalizedBoneNode('chest');
      if (chest) {
        chest.scale.y = 1.0 + breatheCycle * BREATHE_CHEST_SCALE * fidgetBreathScale;
        chest.rotation.x = -breatheCycle * BREATHE_CHEST_X * fidgetBreathScale;
        chest.rotation.z = cw * -CONTRA_CHEST_Z;
      }
      const upperChest = vrm.humanoid.getNormalizedBoneNode('upperChest');
      if (upperChest) {
        upperChest.rotation.x = -breatheCycle * BREATHE_CHEST_X * 0.5 * fidgetBreathScale;
        upperChest.scale.y = 1.0 + breatheAmount * 1.5 * fidgetBreathScale;
      }

      const spine = vrm.humanoid.getNormalizedBoneNode('spine');
      if (spine) {
        spine.rotation.x = breatheCycle * BREATHE_SPINE_AMOUNT + fidgetSpineX.pos;
        spine.rotation.z = cw * -CONTRA_SPINE_Z;
      }

      const hips = vrm.humanoid.getNormalizedBoneNode('hips');
      if (hips) {
        hips.rotation.z = fbm(noiseTime * 0.04 + 400, 2) * 0.015 * bodyAmplitude + ASYM_HIP_LEAN + fidgetHipZ.pos + cw * CONTRA_HIP_Z;
        hips.rotation.x = fbm(noiseTime * 0.03 + 500, 2) * 0.008 * bodyAmplitude;
      }

      const leftShoulder = vrm.humanoid.getNormalizedBoneNode('leftShoulder');
      const rightShoulder = vrm.humanoid.getNormalizedBoneNode('rightShoulder');
      if (leftShoulder) {
        leftShoulder.rotation.z =
          fbm(noiseTime * 0.05 + 600, 2) * 0.01 * bodyAmplitude
          + breatheShoulder * BREATHE_SHOULDER_AMOUNT * fidgetBreathScale
          + fidgetShoulderL.pos;
        leftShoulder.position.y = shoulderInhaleRise * fidgetBreathScale;
      }
      if (rightShoulder) {
        rightShoulder.rotation.z =
          fbm(noiseTime * 0.05 + 700, 2) * 0.01 * bodyAmplitude
          + breatheShoulder * BREATHE_SHOULDER_AMOUNT * fidgetBreathScale
          + ASYM_SHOULDER_OFFSET
          + fidgetShoulderR.pos;
        rightShoulder.position.y = shoulderInhaleRise * fidgetBreathScale;
      }

      const leftUpperArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
      const rightUpperArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
      if (leftUpperArm) {
        leftUpperArm.rotation.x = fbm(noiseTime * 0.035 + 800, 2) * 0.02 * bodyAmplitude;
        leftUpperArm.rotation.z = REST_ARM_Z + fbm(noiseTime * 0.03 + 900, 2) * 0.015 * bodyAmplitude;
      }
      if (rightUpperArm) {
        rightUpperArm.rotation.x = fbm(noiseTime * 0.035 + 1000, 2) * 0.02 * bodyAmplitude;
        rightUpperArm.rotation.z = -REST_ARM_Z + fbm(noiseTime * 0.03 + 1100, 2) * -0.015 * bodyAmplitude;
      }

      const leftLowerArm = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
      const rightLowerArm = vrm.humanoid.getNormalizedBoneNode('rightLowerArm');
      if (leftLowerArm) {
        leftLowerArm.rotation.z = REST_FOREARM_Z + fbm(noiseTime * 0.025 + 2300, 2) * 0.03 * bodyAmplitude;
        leftLowerArm.rotation.y = fbm(noiseTime * 0.02 + 2400, 2) * 0.015 * bodyAmplitude;
      }
      if (rightLowerArm) {
        rightLowerArm.rotation.z = -REST_FOREARM_Z + fbm(noiseTime * 0.025 + 2500, 2) * -0.03 * bodyAmplitude;
        rightLowerArm.rotation.y = fbm(noiseTime * 0.02 + 2600, 2) * -0.015 * bodyAmplitude;
      }
    } else {
      // Clip active: only additive breathing scale on chest (subtle life on top of clip)
      const chest = vrm.humanoid.getNormalizedBoneNode('chest');
      if (chest) {
        chest.scale.y = 1.0 + breatheCycle * BREATHE_CHEST_SCALE * 0.5;
      }
      const upperChest = vrm.humanoid.getNormalizedBoneNode('upperChest');
      if (upperChest) {
        upperChest.scale.y = 1.0 + breatheAmount * 0.8;
      }
    }
  }

  // Eye saccades
  saccadeTimer -= deltaMs;
  if (saccadeTimer <= 0) {
    saccadeTargetX = (Math.random() - 0.5) * SACCADE_RANGE_X * 2;
    saccadeTargetY = (Math.random() - 0.5) * SACCADE_RANGE_Y * 2;
    saccadeTimer = randomRange(SACCADE_MIN_INTERVAL, SACCADE_MAX_INTERVAL);
    saccadeJustMoved = true;
    saccadeSettleTimer = 200;
  }

  const saccadeHL = saccadeJustMoved ? SACCADE_SNAP_HL : SACCADE_SETTLE_HL;
  saccadeSpringX = springDamped(saccadeSpringX, saccadeTargetX, saccadeHL, dt);
  saccadeSpringY = springDamped(saccadeSpringY, saccadeTargetY, saccadeHL, dt);

  if (saccadeJustMoved) {
    saccadeSettleTimer -= deltaMs;
    if (saccadeSettleTimer <= 0) saccadeJustMoved = false;
  }

  // Gaze-breaking
  if (!gazeBreakActive) {
    gazeBreakTimer -= deltaMs;
    if (gazeBreakTimer <= 0 && !isSpeaking) {
      gazeBreakActive = true;
      gazeBreakProgress = 0;
      const angle = Math.random() * Math.PI * 2;
      gazeBreakOffsetX = Math.cos(angle) * GAZE_BREAK_AMOUNT;
      gazeBreakOffsetY = Math.sin(angle) * GAZE_BREAK_AMOUNT * 0.5;
    }
  } else {
    gazeBreakProgress += deltaMs;
    if (gazeBreakProgress >= GAZE_BREAK_DURATION) {
      gazeBreakActive = false;
      gazeBreakOffsetX = 0;
      gazeBreakOffsetY = 0;
      gazeBreakTimer = randomRange(GAZE_BREAK_MIN_INTERVAL, GAZE_BREAK_MAX_INTERVAL);
    }
  }

  // Mode-specific gaze bias
  let gazeBiasX = 0;
  let gazeBiasY = 0;
  if (mode === 'thinking') {
    gazeBiasX += -0.25 + fbm(noiseTime * 0.15 + 2100, 2) * 0.15;
    gazeBiasY += 0.2 + fbm(noiseTime * 0.12 + 2200, 2) * 0.1;
  } else if (mode === 'speaking') {
    gazeBiasX = 0;
    gazeBiasY = 0;
  }

  // Mouse tracking — decay to center (look at camera) when cursor is stale
  const breakX = (gazeBreakActive && !isSpeaking) ? gazeBreakOffsetX : 0;
  const breakY = (gazeBreakActive && !isSpeaking) ? gazeBreakOffsetY : 0;

  const timeSinceMouseMove = performance.now() - lastMouseMoveTime;
  const mouseStale = lastMouseMoveTime > 0 && timeSinceMouseMove > MOUSE_STALE_MS;
  // Smooth fade: 0 = fully following cursor, 1 = fully centered on camera
  const staleFade = mouseStale ? Math.min(1.0, (timeSinceMouseMove - MOUSE_STALE_MS) / 1000) : 0;

  const mouseWeight = mode === 'thinking' ? 0.3 : mode === 'speaking' ? 0.6 : 1.0;
  const effectiveMouseX = mouseNormX * (1 - staleFade);
  const effectiveMouseY = mouseNormY * (1 - staleFade);
  const mouseTargetX = effectiveMouseX * MOUSE_EYE_STRENGTH_X * mouseWeight + breakX + gazeBiasX;
  const mouseTargetY = -effectiveMouseY * MOUSE_EYE_STRENGTH_Y * mouseWeight + breakY + gazeBiasY;
  eyeSpringX = springDamped(eyeSpringX, mouseTargetX, EYE_SPRING_HL, dt);
  eyeSpringY = springDamped(eyeSpringY, mouseTargetY, EYE_SPRING_HL, dt);

  const headTargetX = effectiveMouseX * MOUSE_HEAD_STRENGTH_X * mouseWeight + breakX * 0.3 + gazeBiasX * 0.4;
  const headTargetY = -effectiveMouseY * MOUSE_HEAD_STRENGTH_Y * mouseWeight + breakY * 0.3 + gazeBiasY * 0.3;
  headSpringX = springDamped(headSpringX, headTargetX, HEAD_SPRING_HL, dt);
  headSpringY = springDamped(headSpringY, headTargetY, HEAD_SPRING_HL, dt);

  const saccadeWeight = isSpeaking ? 0.05 : MOUSE_SACCADE_BLEND;
  const eyeX = eyeSpringX.pos + saccadeSpringX.pos * saccadeWeight;
  const eyeY = eyeSpringY.pos + saccadeSpringY.pos * saccadeWeight;

  setExpression('lookLeft', Math.max(0, -eyeX));
  setExpression('lookRight', Math.max(0, eyeX));
  setExpression('lookUp', Math.max(0, eyeY));
  setExpression('lookDown', Math.max(0, -eyeY));

  // Eye bones for convergence and finer control
  if (vrm?.humanoid) {
    const leftEye = vrm.humanoid.getNormalizedBoneNode('leftEye');
    const rightEye = vrm.humanoid.getNormalizedBoneNode('rightEye');
    const eyeRotX = -eyeY * 0.08;
    const eyeRotY = eyeX * 0.08;
    const convergence = mode === 'speaking' ? 0.035 : mode === 'thinking' ? 0.005 : 0.02;
    const tremorX = Math.sin(noiseTime * 47.3) * 0.001 + Math.sin(noiseTime * 31.7) * 0.0007;
    const tremorY = Math.sin(noiseTime * 53.1) * 0.001 + Math.sin(noiseTime * 37.9) * 0.0007;
    if (leftEye) {
      leftEye.rotation.x = eyeRotX + tremorX;
      leftEye.rotation.y = eyeRotY + convergence + tremorY;
    }
    if (rightEye) {
      rightEye.rotation.x = eyeRotX + tremorX * 0.95;
      rightEye.rotation.y = eyeRotY - convergence + tremorY * 1.05;
    }
  }

  // Facial micro-expressions — skip when mood is active or transitioning (mood system owns these keys)
  if (!mood && !isTransitioning()) {
    const browNoise = fbm(noiseTime * 0.1 + 1200, 2);
    if (browNoise > 0.3) {
      setExpression('surprised', (browNoise - 0.3) * 0.15);
    } else {
      setExpression('surprised', 0);
    }

    const mouthNoise = fbm(noiseTime * 0.06 + 1300, 2);
    if (!isSpeaking && mouthNoise > 0.2) {
      setExpression('happy', (mouthNoise - 0.2) * 0.1);
    } else if (!isSpeaking) {
      setExpression('happy', 0);
    }

    // Speaking micro-expressions
    if (isSpeaking) {
      const speakBrowNoise = fbm(noiseTime * 0.5 + 1400, 2);
      if (speakBrowNoise > 0.25) {
        setExpression('surprised', (speakBrowNoise - 0.25) * 0.2);
      }
      const speakSquintNoise = fbm(noiseTime * 0.3 + 1500, 2);
      if (speakSquintNoise > 0.4) {
        setExpression('squint', (speakSquintNoise - 0.4) * 0.15);
      }
      setExpression('happy', 0.05 + fbm(noiseTime * 0.2 + 1600, 2) * 0.05);
    }
  }

  // Swallow animation
  if (!isSpeaking && vrm?.humanoid) {
    const swallowPeriod = 35 + fbm(noiseTime * 0.01 + 4000, 1) * 25;
    const swallowPhase = (noiseTime % swallowPeriod) / swallowPeriod;
    if (swallowPhase < 0.012) {
      const st = swallowPhase / 0.012;
      const swallowCurve = Math.sin(st * Math.PI);
      const jaw = vrm.humanoid.getNormalizedBoneNode('jaw');
      if (jaw) jaw.rotation.x += swallowCurve * 0.02;
      const head = vrm.humanoid.getNormalizedBoneNode('head');
      if (head) head.rotation.x += swallowCurve * -0.008;
    }
  }

  // Head + neck movement
  if (vrm?.humanoid) {
    const totalY = fbm(noiseTime * 0.08, 3) * 0.06 * headAmplitude + headSpringX.pos;
    const totalX = fbm(noiseTime * 0.06 + 100, 3) * 0.04 * headAmplitude
      + headSpringY.pos
      + breatheCycle * BREATHE_HEAD_AMOUNT * fidgetBreathScale;
    const totalZ = fbm(noiseTime * 0.05 + 200, 2) * 0.025 * headAmplitude
      + tiltSpring.pos
      + ASYM_HEAD_TILT;

    const rotMag = Math.sqrt(totalY * totalY + totalX * totalX);
    const neckShare = 0.30 + Math.min(rotMag / 0.3, 1.0) * 0.20;
    const headShare = 1.0 - neckShare;

    const neck = vrm.humanoid.getNormalizedBoneNode('neck');
    if (neck) {
      neck.rotation.y = totalY * neckShare;
      neck.rotation.x = totalX * neckShare + breatheCycle * -BREATHE_NECK_AMOUNT * fidgetBreathScale;
      neck.rotation.z = totalZ * neckShare + cw * CONTRA_NECK_Z;
    }

    const head = vrm.humanoid.getNormalizedBoneNode('head');
    if (head) {
      head.rotation.y = totalY * headShare;
      head.rotation.x = totalX * headShare;
      head.rotation.z = totalZ * headShare;
    }
  }
}

export function resetIdle(): void {
  blinkProgress = -1;
  doubleBlink = false;
  doubleBlinkGap = -1;
  blinkTimer = randomRange(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL);
  saccadeSpringX = createSpring();
  saccadeSpringY = createSpring();
  eyeSpringX = createSpring();
  eyeSpringY = createSpring();
  headSpringX = createSpring();
  headSpringY = createSpring();
  tiltSpring = createSpring();
  tiltHoldActive = false;
  tiltHoldTarget = 0;
  tiltHoldTimer = randomRange(TILT_HOLD_MIN_INTERVAL, TILT_HOLD_MAX_INTERVAL);
  gazeBreakActive = false;
  gazeBreakTimer = randomRange(GAZE_BREAK_MIN_INTERVAL, GAZE_BREAK_MAX_INTERVAL);
  contraSpring = createSpring();
  contraTimer = randomRange(CONTRA_SHIFT_MIN_INTERVAL, CONTRA_SHIFT_MAX_INTERVAL);
  contraTarget = 0;
  fidgetTimer = randomRange(FIDGET_MIN_INTERVAL, FIDGET_MAX_INTERVAL);
  fidgetActive = false;
  fidgetHipZ = createSpring();
  fidgetSpineX = createSpring();
  fidgetShoulderL = createSpring();
  fidgetShoulderR = createSpring();
  fidgetBreathScale = 1.0;
  currentFidget = null;
}
