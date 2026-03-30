/**
 * SpindL Subtitle Overlay (NANO-100)
 *
 * Renders LLM response text as typewriter-animated subtitles in a standalone
 * Tauri window. Designed for OBS Window Capture as a separate source.
 *
 * Data flow:
 *   response event      -> buffer full text
 *   tts_status {started} -> begin typewriter reveal synced to TTS duration
 *   tts_status {completed|interrupted} -> hold briefly, then fade out
 *
 * Reference: V1R4 overlay-effects.ts (typewriter, sentence crop, timing)
 */

import { io } from 'socket.io-client';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { invoke } from '@tauri-apps/api/core';

// --- Config ---

const WS_URL = localStorage.getItem('spindl-subtitles-ws-url') ?? 'http://127.0.0.1:8765';
const MAX_SENTENCES = 2;
const FALLBACK_CHAR_RATE = 0.06; // seconds per char when no TTS duration
const FALLBACK_TIMEOUT_MS = 500; // ms to wait for TTS before using fallback
let HOLD_AFTER_TTS_MS = 1500;  // ms to hold text after TTS completes — updated from backend config

// --- State machine ---

type SubtitleState = 'idle' | 'buffered' | 'revealing' | 'holding' | 'fading';

let state: SubtitleState = 'idle';
let fullText = '';
let startTime = 0;
let duration = 0;
let charsRevealed = 0;
let animFrameId: number | null = null;
let fallbackTimer: ReturnType<typeof setTimeout> | null = null;
let holdTimer: ReturnType<typeof setTimeout> | null = null;

// --- DOM refs ---

const pill = document.getElementById('subtitle-pill')!;
const textEl = document.getElementById('subtitle-text')!;

// --- Draggable window + context menu ---

const appWindow = getCurrentWindow();
document.addEventListener('mousedown', (e) => {
  if (e.button === 0) appWindow.startDragging();
});

document.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  invoke('show_context_menu').catch(() => {});
});

// --- Background swap (called from Rust menu) ---

(window as Record<string, unknown>).__SPINDL_SET_BG = (color: string) => {
  document.body.style.background = color;
  localStorage.setItem('spindl-subtitles-bg', color);
};

// Restore saved background
const savedBg = localStorage.getItem('spindl-subtitles-bg');
if (savedBg) {
  document.body.style.background = savedBg;
}

// --- Text processing ---

/** Crop to last N complete sentences + trailing partial (V1R4 pattern). */
function cropToRecentSentences(text: string, maxSentences: number): string {
  const sentences = text.match(/[^.!?]*[.!?]+\s*/g);
  if (!sentences || sentences.length <= maxSentences) return text;

  const lastN = sentences.slice(-maxSentences).join('');
  const lastSentence = sentences[sentences.length - 1];
  const lastIdx = text.lastIndexOf(lastSentence);
  const trailing = text.slice(lastIdx + lastSentence.length);
  return (lastN + trailing).trim();
}

/** Word wrap text to fit within max chars per line. */
function wordWrap(text: string, maxChars: number): string {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = '';

  for (const word of words) {
    if (current.length === 0) {
      current = word;
    } else if (current.length + 1 + word.length <= maxChars) {
      current += ' ' + word;
    } else {
      lines.push(current);
      current = word;
    }
  }
  if (current.length > 0) lines.push(current);

  return lines.join('\n');
}

// --- Rendering ---

function showPill(): void {
  pill.classList.remove('fading');
  pill.classList.add('visible');
}

function hidePill(): void {
  pill.classList.add('fading');
  pill.classList.remove('visible');
}

function updateDisplay(text: string): void {
  const cropped = cropToRecentSentences(text, MAX_SENTENCES);
  const wrapped = wordWrap(cropped, 50);
  textEl.textContent = wrapped;
}

// --- State transitions ---

function startReveal(ttsDuration: number): void {
  if (fallbackTimer) { clearTimeout(fallbackTimer); fallbackTimer = null; }
  if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }

  duration = ttsDuration > 0 ? ttsDuration : fullText.length * FALLBACK_CHAR_RATE;
  startTime = performance.now() / 1000;
  charsRevealed = 0;
  state = 'revealing';
  showPill();
  scheduleFrame();
}

function onTTSComplete(): void {
  if (state === 'revealing' || state === 'holding') {
    // Show remaining text immediately
    charsRevealed = fullText.length;
    updateDisplay(fullText);
    state = 'holding';

    console.log(`[Subtitles] TTS complete — holding for ${HOLD_AFTER_TTS_MS}ms`);

    // Hold then fade
    holdTimer = setTimeout(() => {
      state = 'fading';
      hidePill();
      console.log('[Subtitles] Fading out');
      // After CSS transition completes, go idle
      setTimeout(() => {
        state = 'idle';
        fullText = '';
        textEl.textContent = '';
      }, 200);
      holdTimer = null;
    }, HOLD_AFTER_TTS_MS);
  }
}

function onNewResponse(text: string): void {
  // Cancel any in-progress animation
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  if (fallbackTimer) { clearTimeout(fallbackTimer); fallbackTimer = null; }
  if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }

  if (!text || text.trim().length === 0) {
    state = 'idle';
    hidePill();
    return;
  }

  fullText = text;
  charsRevealed = 0;
  state = 'buffered';

  // Start fallback timer — if TTS doesn't start within timeout, use estimated duration
  fallbackTimer = setTimeout(() => {
    if (state === 'buffered') {
      startReveal(0); // 0 triggers fallback rate
    }
    fallbackTimer = null;
  }, FALLBACK_TIMEOUT_MS);
}

// --- Animation loop ---

function scheduleFrame(): void {
  animFrameId = requestAnimationFrame(animate);
}

function animate(): void {
  if (state !== 'revealing') {
    animFrameId = null;
    return;
  }

  const elapsed = performance.now() / 1000 - startTime;
  const progress = Math.min(elapsed / duration, 1.0);
  charsRevealed = Math.floor(progress * fullText.length);

  const revealed = fullText.slice(0, charsRevealed);
  updateDisplay(revealed);

  if (progress >= 1.0) {
    // All chars revealed — enter holding state
    state = 'holding';
    animFrameId = null;
    return;
  }

  scheduleFrame();
}

// --- Socket.IO connection ---

const socket = io(WS_URL, {
  reconnection: true,
  reconnectionDelay: 2000,
  reconnectionDelayMax: 30000,
  transports: ['websocket', 'polling'],
});

socket.on('connect', () => {
  if (import.meta.env.DEV) console.log('[Subtitles] Connected:', WS_URL);
  // Request initial config for fade delay hydration
  socket.emit('request_config', {});
});

socket.on('disconnect', (reason) => {
  if (import.meta.env.DEV) console.log('[Subtitles] Disconnected:', reason);
});

// LLM response — full text arrives in one event
socket.on('response', (data: { text: string; is_final: boolean }) => {
  if (data.is_final && data.text) {
    onNewResponse(data.text);
  }
});

// TTS lifecycle — timing for typewriter sync
socket.on('tts_status', (data: { status: string; duration?: number }) => {
  if (data.status === 'started') {
    if (state === 'buffered') {
      startReveal(data.duration ?? 0);
    }
  } else if (data.status === 'completed' || data.status === 'interrupted') {
    onTTSComplete();
  }
});

// Config hydration — avatar_config_updated is authoritative, config_loaded is fallback
let configReceivedFromUpdate = false;

socket.on('config_loaded', (data: { settings?: { avatar?: { subtitle_fade_delay?: number; subtitle_always_on_top?: boolean } } }) => {
  if (configReceivedFromUpdate) return; // avatar_config_updated already set the value
  const delay = data.settings?.avatar?.subtitle_fade_delay;
  if (delay !== undefined) {
    HOLD_AFTER_TTS_MS = delay * 1000;
    if (import.meta.env.DEV) console.log('[Subtitles] Hydrated fade delay:', delay, 's');
  }
  const onTop = data.settings?.avatar?.subtitle_always_on_top;
  if (onTop !== undefined) {
    getCurrentWindow().setAlwaysOnTop(onTop);
  }
});

socket.on('avatar_config_updated', (data: { subtitle_fade_delay?: number; subtitle_always_on_top?: boolean }) => {
  if (data.subtitle_fade_delay !== undefined) {
    HOLD_AFTER_TTS_MS = data.subtitle_fade_delay * 1000;
    configReceivedFromUpdate = true;
    if (import.meta.env.DEV) console.log('[Subtitles] Fade delay updated:', data.subtitle_fade_delay, 's');
  }
  if (data.subtitle_always_on_top !== undefined) {
    getCurrentWindow().setAlwaysOnTop(data.subtitle_always_on_top);
  }
});
