import { createScene } from './scene';
import { loadAvatar, updateAvatar, setForceOpaqueMaterials } from './avatar';
import { createState } from './state';
import { connectStatus, StatusMessage } from './ws-client';
import { setMood, updateExpressions, getAvailableMoods, setExpressionComposites } from './expressions';
import { updateIdle, setMousePosition, triggerBlink, resetIdle } from './idle';
import { updateLipSyncAmplitude, resetLipSync } from './lipsync';
import { updateBody, triggerKeystrokeReaction } from './body';
import { initMixer, disposeMixer, updateMixer, loadAnimationsFromDir, loadAnimationsWithFallback, playClip, stopClip, getLoadedClipNames, setAnimationConfig, getAnimationConfig, updateEmotionAnimation, setBaseAnimations } from './mixer';
import { initWind, updateWind } from './wind';
import { createSpring, springDamped, SpringState } from './spring';
import { applyBackground, loadBackgroundConfig, saveBackgroundConfig, BACKGROUND_PRESETS, BackgroundConfig } from './background';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { invoke } from '@tauri-apps/api/core';
import { open as dialogOpen } from '@tauri-apps/plugin-dialog';
import * as THREE from 'three';

// Animation directory — resolved by Rust to handle CWD differences (dev vs prod)
let ANIMATIONS_DIR = 'public/animations'; // fallback, overwritten at startup

// Default WebSocket URL — configurable via localStorage
// Socket.IO URL — connects to SpindL's GUI server (same server the dashboard uses)
const WS_URL = localStorage.getItem('spindl-avatar-ws-url') ?? 'http://127.0.0.1:8765';

const canvas = document.getElementById('avatar') as HTMLCanvasElement;
const ctx = createScene(canvas);
const state = createState();

// Draggable window
const appWindow = getCurrentWindow();
canvas.addEventListener('mousedown', (e) => {
  if (e.button === 0) appWindow.startDragging();
});

document.addEventListener('contextmenu', (e) => {
  e.preventDefault();
});

// Border (decorations) toggle
let borderEnabled = localStorage.getItem('spindl-avatar-border') === '1';
// Apply initial state — config defaults to decorations:false, so only enable if persisted
if (borderEnabled) appWindow.setDecorations(true);

(window as any).__SPINDL_TOGGLE_BORDER = async () => {
  borderEnabled = !borderEnabled;
  localStorage.setItem('spindl-avatar-border', borderEnabled ? '1' : '0');
  await appWindow.setDecorations(borderEnabled);
};

// Global cursor tracking at ~20Hz
setInterval(async () => {
  try {
    const pos = await invoke<[number, number] | null>('get_cursor_position');
    if (pos) setMousePosition(pos[0], pos[1]);
  } catch { /* ignore if command not available yet */ }
}, 50);

// Global keystroke detection
let lastKeystrokeCount = 0;
setInterval(async () => {
  try {
    const count = await invoke<number>('get_keystroke_count');
    if (lastKeystrokeCount > 0 && count !== lastKeystrokeCount) {
      triggerKeystrokeReaction();
    }
    lastKeystrokeCount = count;
  } catch { /* ignore — keystroke command may not exist on this platform */ }
}, 50);

let toolMoodTimer: ReturnType<typeof setTimeout> | null = null;
let speakingStopTimer: ReturnType<typeof setTimeout> | null = null;
let expressionFadeTimer: ReturnType<typeof setTimeout> | null = null;
let expressionFadeDelay = 1.0; // seconds — updated from backend config

async function onStatusMessage(msg: StatusMessage): Promise<void> {
  switch (msg.type) {
    case 'state':
      if (msg.value === 'idle') {
        state.mode = 'idle';
        // Face + body emotion hold until next mood event — don't reset on idle.
        // The classifier will push a new mood (or neutral) on the next response,
        // which resets both expression blend shapes and animation clip.
      } else if (msg.value === 'thinking') {
        state.mode = 'thinking';
      }
      break;

    case 'speaking':
      if (msg.value) {
        if (speakingStopTimer) { clearTimeout(speakingStopTimer); speakingStopTimer = null; }
        // TTS started — cancel any pending expression fade
        if (expressionFadeTimer) { clearTimeout(expressionFadeTimer); expressionFadeTimer = null; }
        state.mode = 'speaking';
        state.speaking = true;
      } else {
        speakingStopTimer = setTimeout(() => {
          state.speaking = false;
          state.amplitude = 0;
          resetLipSync();
          // Only force idle if state hasn't already moved (e.g. interrupt → thinking)
          if (state.mode === 'speaking') {
            state.mode = 'idle';
          }
          speakingStopTimer = null;

          // Start expression fade timer — clear face after delay, body holds until next mood
          if (expressionFadeTimer) clearTimeout(expressionFadeTimer);
          const delayMs = expressionFadeDelay * 1000;
          expressionFadeTimer = setTimeout(() => {
            state.mood = null;
            setMood(null);
            updateEmotionAnimation(null, 0);
            expressionFadeTimer = null;
          }, delayMs);
        }, 300);
      }
      break;

    case 'mood':
      // Cancel pending expression fade — new mood takes over
      if (expressionFadeTimer) { clearTimeout(expressionFadeTimer); expressionFadeTimer = null; }
      triggerBlink();
      setMood(msg.value, msg.confidence);
      // NANO-098 Session 3: Emotion-driven animation selection
      updateEmotionAnimation(msg.value, msg.confidence);
      break;

    case 'toolMood':
      if (toolMoodTimer) clearTimeout(toolMoodTimer);
      state.toolMood = msg.value as any;
      setMood('curious');
      toolMoodTimer = setTimeout(() => {
        state.toolMood = null;
        setMood(null);
        toolMoodTimer = null;
      }, 2000);
      break;

    // NANO-097: Character switch → load new VRM model
    case 'loadModel':
      try {
        if (msg.value) {
          // Load character-specific VRM from filesystem
          const bytes = await invoke<number[]>('read_file_bytes', { path: msg.value });
          const blob = new Blob([new Uint8Array(bytes)], { type: 'application/octet-stream' });
          await saveAvatarBlob(blob);
          const blobUrl = URL.createObjectURL(blob);
          await loadAvatar(ctx.scene, blobUrl);
          URL.revokeObjectURL(blobUrl);
        } else {
          // No VRM for this character — load bundled default
          await clearAvatarBlob();
          await loadAvatar(ctx.scene, '/models/avatar.vrm');
        }
        frameCameraToModel();
        resetIdle();
        initWind();
        // NANO-098 Session 3: Set animation config BEFORE mixer setup (setup reads config for default clip)
        setAnimationConfig(msg.animations ?? null);
        // NANO-099: Set base animations before mixer setup (fallback defaults)
        if (msg.baseAnimations) setBaseAnimations(msg.baseAnimations);
        await setupMixerForAvatar(msg.characterAnimationsDir);
        // NANO-098: Apply per-character expression composites
        setExpressionComposites(msg.expressions ?? {});
        if (ctx.transparent) setForceOpaqueMaterials(true);
        if (import.meta.env.DEV) console.log('[SpindL] Model loaded via avatar_load_model:', msg.value || '(default)');
      } catch (err) {
        console.error('[SpindL] avatar_load_model failed:', err);
      }
      break;

    // NANO-098: Live preview of expression composites from character editor
    case 'previewExpressions':
      setExpressionComposites(msg.expressions);
      // Show the mood being edited, or clear back to neutral on revert
      setMood(msg.previewMood);
      break;

    // NANO-098 Session 3: Live preview of animation clip from character editor
    case 'previewAnimation':
      if (msg.clip && getLoadedClipNames().includes(msg.clip)) {
        playClip(msg.clip);
      } else {
        stopClip();
      }
      break;

    // NANO-098 Session 3: Update animation config from editor save
    case 'updateAnimationConfig':
      setAnimationConfig(msg.animations);
      break;

    // NANO-099: Rescan animation files after base animation upload/clear
    case 'rescanAnimations':
      try {
        if (msg.baseAnimations) {
          setBaseAnimations(msg.baseAnimations);
        }
        await setupMixerForAvatar();
        if (import.meta.env.DEV) console.log('[SpindL] Animations rescanned');
      } catch (err) {
        console.error('[SpindL] Animation rescan failed:', err);
      }
      break;
  }
}

// Apply background
applyBackground(ctx.scene, loadBackgroundConfig());

// Restore transparent mode if previously enabled
if (localStorage.getItem('spindl-avatar-transparent') === '1') {
  ctx.setTransparent(true);
}

// Background change (context menu)
function pickImageFile(): Promise<string | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.style.display = 'none';
    let resolved = false;
    const cleanup = () => {
      window.removeEventListener('focus', onFocus);
      input.remove();
    };
    const resolveOnce = (value: string | null) => {
      if (resolved) return;
      resolved = true;
      cleanup();
      resolve(value);
    };
    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (!file) { resolveOnce(null); return; }
      const reader = new FileReader();
      reader.onload = () => resolveOnce(reader.result as string);
      reader.onerror = () => resolveOnce(null);
      reader.readAsDataURL(file);
    });
    function onFocus() {
      setTimeout(() => {
        if (!input.files?.length) resolveOnce(null);
      }, 500);
    }
    window.addEventListener('focus', onFocus);
    document.body.appendChild(input);
    input.click();
  });
}

(window as any).__SPINDL_SET_TRANSPARENT = (on: boolean) => {
  ctx.setTransparent(on);
  localStorage.setItem('spindl-avatar-transparent', on ? '1' : '0');
};

(window as any).__SPINDL_CHANGE_BG = async (presetOrCustom: string, blur?: number) => {
  let config: BackgroundConfig;
  if (presetOrCustom === 'custom') {
    const dataUrl = await pickImageFile();
    if (!dataUrl) return;
    config = { type: 'image', imagePath: dataUrl, blur: blur ?? 0, vignette: true };
  } else {
    config = BACKGROUND_PRESETS[presetOrCustom] ?? BACKGROUND_PRESETS.darkPurple;
  }
  await applyBackground(ctx.scene, config);
  saveBackgroundConfig(config);
};

(window as any).__SPINDL_SET_BLUR = async (blur: number) => {
  const config = loadBackgroundConfig();
  if (config.type === 'image') {
    config.blur = blur;
    await applyBackground(ctx.scene, config);
    saveBackgroundConfig(config);
  }
};

// Avatar persistence (IndexedDB)
function openAvatarDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('spindl-avatar', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('avatar');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function saveAvatarBlob(blob: Blob): Promise<void> {
  const db = await openAvatarDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('avatar', 'readwrite');
    tx.objectStore('avatar').put(blob, 'model');
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

async function loadAvatarBlob(): Promise<Blob | null> {
  const db = await openAvatarDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('avatar', 'readonly');
    const req = tx.objectStore('avatar').get('model');
    req.onsuccess = () => { db.close(); resolve(req.result ?? null); };
    req.onerror = () => { db.close(); reject(req.error); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

async function clearAvatarBlob(): Promise<void> {
  const db = await openAvatarDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('avatar', 'readwrite');
    tx.objectStore('avatar').delete('model');
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

(window as any).__SPINDL_LOAD_AVATAR = async () => {
  if (import.meta.env.DEV) console.log('[SpindL] Load Avatar triggered');
  try {
    const selected = await dialogOpen({
      title: 'Load VRM Avatar',
      filters: [{ name: 'VRM Models', extensions: ['vrm'] }],
      multiple: false,
    });
    if (!selected) return;

    const filePath = typeof selected === 'string' ? selected : (selected as any).path ?? String(selected);
    if (!filePath) return;

    const bytes = await invoke<number[]>('read_file_bytes', { path: filePath });
    const blob = new Blob([new Uint8Array(bytes)], { type: 'application/octet-stream' });
    await saveAvatarBlob(blob);

    // Load into Three.js
    const blobUrl = URL.createObjectURL(blob);
    await loadAvatar(ctx.scene, blobUrl);
    URL.revokeObjectURL(blobUrl);

    frameCameraToModel();

    resetIdle();
    initWind();
    await setupMixerForAvatar();
    if (ctx.transparent) setForceOpaqueMaterials(true);
    if (import.meta.env.DEV) console.log('[SpindL] Avatar loaded OK');
  } catch (err) {
    console.error('[SpindL] Avatar load failed:', err);
    const errDiv = document.createElement('div');
    errDiv.style.cssText = 'position:fixed;top:10px;left:50%;transform:translateX(-50%);color:#ff4444;font:12px sans-serif;background:rgba(0,0,0,0.8);padding:8px 12px;border-radius:4px;z-index:9999;';
    errDiv.textContent = `Load failed: ${err}`;
    document.body.appendChild(errDiv);
    setTimeout(() => errDiv.remove(), 5000);
  }
};

(window as any).__SPINDL_RESET_AVATAR = async () => {
  await clearAvatarBlob().catch(() => {});
  try {
    await loadAvatar(ctx.scene, '/models/avatar.vrm');
    resetIdle();
    initWind();
    await setupMixerForAvatar();
    if (import.meta.env.DEV) console.log('[SpindL] Avatar reset to default');
  } catch (err) {
    if (import.meta.env.DEV) console.error('[SpindL] Failed to load default avatar:', err);
  }
};

// Camera controls
const CAMERA_BASE_Z = 1.0;
const CAMERA_MIN_ZOOM = 0.1;
const CAMERA_MAX_ZOOM = 10.0;
const CAMERA_ZOOM_HL = 0.08;

// Portrait camera defaults (standard humanoid VRM framing)
const PORTRAIT_CAMERA_Y = 1.42;
const PORTRAIT_CAMERA_Z = 1.0;
const PORTRAIT_LOOK_Y = 1.44;

/** Frame camera to loaded model. Uses portrait defaults for standard VRMs,
 *  falls back to bounding-box auto-fit for non-standard scale models. */
async function frameCameraToModel(): Promise<void> {
  const vrm = (await import('./avatar')).getVRM();
  if (!vrm) return;

  const box = new THREE.Box3().setFromObject(vrm.scene);
  const center = box.getCenter(new THREE.Vector3());

  // Standard humanoid VRMs have center Y roughly in 0.5–3.0 range
  if (center.y >= 0.5 && center.y <= 3.0) {
    // Portrait defaults
    ctx.camera.position.set(0, PORTRAIT_CAMERA_Y, PORTRAIT_CAMERA_Z);
    ctx.camera.lookAt(0, PORTRAIT_LOOK_Y, 0);
    cameraBaseY = PORTRAIT_CAMERA_Y;
  } else {
    // Non-standard model — auto-fit from bounding box
    const size = box.getSize(new THREE.Vector3());
    ctx.camera.position.set(center.x, center.y, center.z + Math.max(size.y, size.x) * 0.8);
    ctx.camera.lookAt(center.x, center.y, center.z);
    cameraBaseY = center.y;
  }
  cameraZoomTarget = 1.0;
  cameraZoomSpring.pos = 1.0;
  cameraOrbitTarget = 0;
  cameraOrbitSpring.pos = 0;
  saveCameraState();
}

function loadCameraState(): { zoom: number; panY: number; orbitAngle: number } {
  try {
    const saved = localStorage.getItem('spindl-avatar-camera');
    if (saved) return { orbitAngle: 0, ...JSON.parse(saved) };
  } catch { /* ignore */ }
  return { zoom: 1.0, panY: ctx.camera.position.y, orbitAngle: 0 };
}

function saveCameraState(): void {
  localStorage.setItem('spindl-avatar-camera', JSON.stringify({ zoom: cameraZoomTarget, panY: cameraBaseY, orbitAngle: cameraOrbitTarget }));
}

const savedCamera = loadCameraState();
let cameraZoomTarget = savedCamera.zoom;
let cameraZoomSpring: SpringState = createSpring();
cameraZoomSpring.pos = cameraZoomTarget;

// Orbit — horizontal rotation around model center (radians)
let cameraOrbitTarget = savedCamera.orbitAngle;
let cameraOrbitSpring: SpringState = createSpring(cameraOrbitTarget);
const CAMERA_ORBIT_HL = 0.1; // responsive orbit spring

(window as any).__SPINDL_ZOOM_IN = () => {
  cameraZoomTarget = Math.min(CAMERA_MAX_ZOOM, cameraZoomTarget + 0.15);
  saveCameraState();
};
(window as any).__SPINDL_ZOOM_OUT = () => {
  cameraZoomTarget = Math.max(CAMERA_MIN_ZOOM, cameraZoomTarget - 0.15);
  saveCameraState();
};

// Expression preview
let previewMoodTimer: ReturnType<typeof setTimeout> | null = null;

(window as any).__SPINDL_PREVIEW_MOOD = (mood: string) => {
  if (previewMoodTimer) clearTimeout(previewMoodTimer);
  state.mood = mood as any;
  setMood(mood);
  previewMoodTimer = setTimeout(() => {
    state.mood = null;
    setMood(null);
    previewMoodTimer = null;
  }, 3000);
};

(window as any).__SPINDL_GET_MOODS = () => getAvailableMoods();

// Animation playback from context menu
(window as any).__SPINDL_PLAY_ANIMATION = (name: string) => {
  if (!name) {
    stopClip();
    localStorage.removeItem('spindl-avatar-animation');
    return;
  }
  // The menu sends lowercase + underscores — find matching clip by case-insensitive comparison
  const clipNames = getLoadedClipNames();
  const match = clipNames.find(c => c.toLowerCase().replace(/\s+/g, '_') === name);
  if (match) {
    playClip(match);
    localStorage.setItem('spindl-avatar-animation', match);
  }
};

/** Init mixer for current VRM + load animations + auto-play saved/default animation. */
async function setupMixerForAvatar(characterAnimDir?: string): Promise<void> {
  const vrm = (await import('./avatar')).getVRM();
  if (!vrm) return;

  disposeMixer();
  initMixer(vrm);

  // Resolve global animations dir via Rust (handles CWD differences)
  try {
    ANIMATIONS_DIR = await invoke<string>('get_animations_dir');
  } catch { /* keep fallback */ }

  // NANO-098 Session 3: Scan both character-local + global pool, or global only
  if (characterAnimDir) {
    await loadAnimationsWithFallback(characterAnimDir, ANIMATIONS_DIR);
  } else {
    await loadAnimationsFromDir(ANIMATIONS_DIR);
  }

  // Auto-play: animation config takes full authority when present.
  // Fallback to localStorage (context menu selection) when no config.
  const config = getAnimationConfig();
  const clips = getLoadedClipNames();
  if (config) {
    if (config.default && clips.includes(config.default)) {
      playClip(config.default, 0);
    }
  } else if (clips.length > 0) {
    // No character config — check localStorage for previously selected animation
    const saved = localStorage.getItem('spindl-avatar-animation');
    if (saved) {
      const match = clips.find(c => c.toLowerCase() === saved.toLowerCase());
      if (match) playClip(match, 0);
    }
  }
}

// Camera pan
let cameraBaseY = savedCamera.panY;
let cameraTime = 0;

(window as any).__SPINDL_CAMERA_UP = () => {
  cameraBaseY += 0.03;
  saveCameraState();
};
(window as any).__SPINDL_CAMERA_DOWN = () => {
  cameraBaseY -= 0.03;
  saveCameraState();
};

// Scroll: zoom (default) or orbit (Shift held)
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (e.shiftKey) {
    // Shift+scroll = orbit horizontally
    const orbitDelta = e.deltaY > 0 ? -0.05 : 0.05;
    cameraOrbitTarget += orbitDelta;
    saveCameraState();
  } else {
    // Regular scroll = zoom
    const delta = e.deltaY > 0 ? -0.01 : 0.01;
    cameraZoomTarget = Math.max(CAMERA_MIN_ZOOM, Math.min(CAMERA_MAX_ZOOM, cameraZoomTarget + delta));
    saveCameraState();
  }
}, { passive: false });

// Right-click drag to pan camera vertically
let rightDragStartY = 0;
let rightDragBaseY = 0;
let rightDragging = false;
const RIGHT_DRAG_THRESHOLD = 3;
const CAMERA_PAN_SENSITIVITY = 0.003;

canvas.addEventListener('mousedown', (e) => {
  if (e.button === 2) {
    rightDragStartY = e.clientY;
    rightDragBaseY = cameraBaseY;
    rightDragging = false;
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (e.buttons & 2) {
    const dy = e.clientY - rightDragStartY;
    if (!rightDragging && Math.abs(dy) > RIGHT_DRAG_THRESHOLD) {
      rightDragging = true;
    }
    if (rightDragging) {
      cameraBaseY = rightDragBaseY + dy * CAMERA_PAN_SENSITIVITY;
    }
  }
});

canvas.addEventListener('mouseup', (e) => {
  if (e.button === 2) {
    if (!rightDragging) {
      invoke('show_context_menu').catch(() => {});
    } else {
      saveCameraState();
    }
    rightDragging = false;
  }
});

// Smoothed amplitude spring
let ampSpring: SpringState = createSpring();
const AMP_HL = 0.03;

// Mood-reactive camera
const MOOD_ZOOM: Record<string, number> = {
  error:      0.08,
  melancholy: 0.06,
  warn:       0.04,
  success:   -0.05,
};

const MODE_ZOOM: Record<string, number> = {
  speaking:   0.03,
  thinking:  -0.03,
  idle:       0,
};

const MOOD_DUTCH: Record<string, number> = {
  error:      0.015,
  melancholy: -0.01,
  warn:       0.008,
};

let moodZoomSpring: SpringState = createSpring();
let dutchTiltSpring: SpringState = createSpring();
const MOOD_ZOOM_HL = 0.5;
const DUTCH_TILT_HL = 0.6;
let dutchTiltEnabled = localStorage.getItem('spindl-avatar-dutch') !== '0';

(window as any).__SPINDL_TOGGLE_DUTCH = () => {
  dutchTiltEnabled = !dutchTiltEnabled;
  localStorage.setItem('spindl-avatar-dutch', dutchTiltEnabled ? '1' : '0');
  if (!dutchTiltEnabled) {
    dutchTiltSpring.pos = 0;
    dutchTiltSpring.vel = 0;
  }
};

// Load avatar: check IndexedDB for saved model, fall back to default
(async () => {
  let avatarLoaded = false;
  const savedBlob = await loadAvatarBlob().catch(() => null);
  if (savedBlob) {
    const url = URL.createObjectURL(savedBlob);
    try {
      await loadAvatar(ctx.scene, url);
      avatarLoaded = true;
      if (import.meta.env.DEV) console.log('[SpindL] Loaded saved avatar from IndexedDB');
    } catch (err) {
      if (import.meta.env.DEV) console.warn('[SpindL] Saved avatar failed, clearing and falling back:', err);
      await clearAvatarBlob().catch(() => {});
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  if (!avatarLoaded) {
    try {
      await loadAvatar(ctx.scene, '/models/avatar.vrm');
      if (import.meta.env.DEV) console.log('[SpindL] Loaded default avatar');
    } catch (err) {
      if (import.meta.env.DEV) console.error('[SpindL] Avatar load FAILED:', err);
      const msg = document.createElement('div');
      msg.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);color:#fff;font:14px sans-serif;text-align:center;opacity:0.7;pointer-events:none;';
      msg.textContent = 'No avatar model found — place avatar.vrm in public/models/ or right-click → Load Avatar';
      document.body.appendChild(msg);
    }
  }
  initWind();
  await setupMixerForAvatar();
  // Re-apply force-opaque if transparent mode was restored before avatar loaded
  if (ctx.transparent) setForceOpaqueMaterials(true);
  connectStatus(WS_URL, state, onStatusMessage, (delay) => { expressionFadeDelay = delay; });
})();

function animate(): void {
  requestAnimationFrame(animate);
  const delta = ctx.clock.getDelta();
  const deltaMs = delta * 1000;
  const dt = delta;
  cameraTime += dt;

  // Amplitude smoothing
  ampSpring = springDamped(ampSpring, state.amplitude, AMP_HL, dt);
  const smoothAmp = Math.max(0, ampSpring.pos);

  updateMixer(delta);
  updateExpressions(deltaMs);
  updateIdle(deltaMs, state.speaking, state.mode, state.mood);
  updateBody(deltaMs, state.mode, state.mood, smoothAmp);

  if (state.speaking) {
    updateLipSyncAmplitude(smoothAmp, deltaMs);
  }

  updateWind(deltaMs);

  // Smooth zoom
  cameraZoomSpring = springDamped(cameraZoomSpring, cameraZoomTarget, CAMERA_ZOOM_HL, dt);

  // Mood-reactive camera
  const moodZoomTarget = (MOOD_ZOOM[state.mood ?? ''] ?? 0) + (MODE_ZOOM[state.mode] ?? 0);
  moodZoomSpring = springDamped(moodZoomSpring, moodZoomTarget, MOOD_ZOOM_HL, dt);
  const dutchTarget = dutchTiltEnabled ? (MOOD_DUTCH[state.mood ?? ''] ?? 0) : 0;
  dutchTiltSpring = springDamped(dutchTiltSpring, dutchTarget, DUTCH_TILT_HL, dt);

  const effectiveZoom = Math.max(CAMERA_MIN_ZOOM, Math.min(CAMERA_MAX_ZOOM, cameraZoomSpring.pos + moodZoomSpring.pos));
  const camDist = CAMERA_BASE_Z / effectiveZoom;

  // Orbit — spring-smoothed horizontal rotation around model center
  cameraOrbitSpring = springDamped(cameraOrbitSpring, cameraOrbitTarget, CAMERA_ORBIT_HL, dt);
  const orbitAngle = cameraOrbitSpring.pos;

  // Camera breathing
  const camBreathY = Math.sin(cameraTime * 0.4) * 0.002;
  const camBreathX = Math.sin(cameraTime * 0.25) * 0.001;

  // Position camera on orbit circle at current distance + breathing offset
  ctx.camera.position.x = Math.sin(orbitAngle) * camDist + camBreathX;
  ctx.camera.position.z = Math.cos(orbitAngle) * camDist;
  ctx.camera.position.y = cameraBaseY + camBreathY;

  // Always look at model center (orbit pivot)
  ctx.camera.lookAt(0, cameraBaseY, 0);

  // Dutch tilt — applied after lookAt so it layers on top
  const dutch = Math.abs(dutchTiltSpring.pos) > 0.0005 ? dutchTiltSpring.pos : 0;
  ctx.camera.rotation.z += dutch;

  updateAvatar(delta);

  ctx.composer.render();
}

animate();
