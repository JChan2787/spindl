import { io, Socket } from 'socket.io-client';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { AvatarState } from './state';
import type { AnimationConfig, BaseAnimationsConfig } from './mixer';

export type StatusMessage =
  | { type: 'state'; value: 'thinking' | 'idle' }
  | { type: 'speaking'; value: boolean }
  | { type: 'amplitude'; value: number }
  | { type: 'mood'; value: string; confidence: number }
  | { type: 'toolMood'; value: string }
  | { type: 'loadModel'; value: string; expressions?: Record<string, Record<string, number>>; animations?: AnimationConfig; characterAnimationsDir?: string; baseAnimations?: BaseAnimationsConfig }
  | { type: 'previewExpressions'; expressions: Record<string, Record<string, number>>; previewMood: string | null }
  | { type: 'previewAnimation'; clip: string | null }
  | { type: 'updateAnimationConfig'; animations: AnimationConfig | null }
  | { type: 'rescanAnimations'; baseAnimations?: BaseAnimationsConfig };

/**
 * Connect to SpindL's GUI server via Socket.IO and map events to
 * StatusMessage callbacks. Reuses the same event infrastructure the
 * dashboard uses — no bespoke WebSocket endpoint needed.
 *
 * Socket.IO events consumed:
 *   state_changed  → { from, to, trigger }
 *   audio_level    → { level: 0.0-1.0 }
 *   tts_status     → { status: "started"|"completed"|"interrupted" }
 *   avatar_mood    → { mood: string }
 *   avatar_tool_mood → { tool_mood: string }
 */
export function connectStatus(
  url: string,
  state: AvatarState,
  onMessage?: (msg: StatusMessage) => void,
  onFadeDelayUpdate?: (delay: number) => void,
): void {
  const socket: Socket = io(url, {
    reconnection: true,
    reconnectionDelay: 2000,
    reconnectionDelayMax: 30000,
    transports: ['websocket', 'polling'],
  });

  socket.on('connect', () => {
    if (import.meta.env.DEV) console.log('[SpindL] Socket.IO connected:', url);
    // NANO-097: Identify as avatar renderer for client tracking
    socket.emit('register_avatar_client', {});
    // Request config for initial hydration (always_on_top, expression_fade_delay)
    socket.emit('request_config', {});
  });

  // Initial config hydration — mirrors subtitle renderer pattern
  socket.on('config_loaded', (data: { settings?: { avatar?: { avatar_always_on_top?: boolean; expression_fade_delay?: number } } }) => {
    const avatarCfg = data.settings?.avatar;
    if (!avatarCfg) return;
    if (avatarCfg.avatar_always_on_top !== undefined) {
      getCurrentWindow().setAlwaysOnTop(avatarCfg.avatar_always_on_top);
    }
    if (avatarCfg.expression_fade_delay !== undefined && onFadeDelayUpdate) {
      onFadeDelayUpdate(avatarCfg.expression_fade_delay);
    }
  });

  socket.on('disconnect', (reason) => {
    if (import.meta.env.DEV) console.log('[SpindL] Socket.IO disconnected:', reason);
  });

  // State machine transitions — map AgentState values to avatar modes
  socket.on('state_changed', (data: { from: string; to: string; trigger: string }) => {
    let avatarState: 'thinking' | 'idle' | null = null;

    switch (data.to) {
      case 'listening':
      case 'idle':
        avatarState = 'idle';
        break;
      case 'processing':
      case 'user_speaking':
        avatarState = 'thinking';
        break;
      // system_speaking handled by tts_status
    }

    if (avatarState) {
      state.mode = avatarState;
      const msg: StatusMessage = { type: 'state', value: avatarState };
      onMessage?.(msg);
    }
  });

  // TTS amplitude for lipsync — already emitted at ~50ms intervals
  socket.on('audio_level', (data: { level: number }) => {
    state.amplitude = data.level;
    const msg: StatusMessage = { type: 'amplitude', value: data.level };
    onMessage?.(msg);
  });

  // TTS lifecycle — speaking start/stop
  socket.on('tts_status', (data: { status: string; duration?: number }) => {
    if (data.status === 'started') {
      state.speaking = true;
      state.mode = 'speaking';
      const msg: StatusMessage = { type: 'speaking', value: true };
      onMessage?.(msg);
    } else if (data.status === 'completed' || data.status === 'interrupted') {
      const msg: StatusMessage = { type: 'speaking', value: false };
      onMessage?.(msg);
    }
  });

  // Avatar mood from emotion classifier (NANO-094, NANO-098: +confidence)
  socket.on('avatar_mood', (data: { mood: string; confidence?: number }) => {
    state.mood = data.mood as AvatarState['mood'];
    const msg: StatusMessage = { type: 'mood', value: data.mood, confidence: data.confidence ?? 1.0 };
    onMessage?.(msg);
  });

  // Avatar tool mood from tool invocation mapping
  socket.on('avatar_tool_mood', (data: { tool_mood: string }) => {
    state.toolMood = data.tool_mood as AvatarState['toolMood'];
    const msg: StatusMessage = { type: 'toolMood', value: data.tool_mood };
    onMessage?.(msg);
  });

  // Avatar config updates (expression_fade_delay, always_on_top, etc.)
  socket.on('avatar_config_updated', (data: { expression_fade_delay?: number; avatar_always_on_top?: boolean }) => {
    if (data.expression_fade_delay !== undefined && onFadeDelayUpdate) {
      onFadeDelayUpdate(data.expression_fade_delay);
    }
    if (data.avatar_always_on_top !== undefined) {
      getCurrentWindow().setAlwaysOnTop(data.avatar_always_on_top);
    }
  });

  // NANO-097: Character switch → load new VRM model (NANO-098: +expressions, +animations)
  socket.on('avatar_load_model', (data: {
    path: string;
    expressions?: Record<string, Record<string, number>>;
    animations?: AnimationConfig;
    character_animations_dir?: string;
    base_animations?: BaseAnimationsConfig;
  }) => {
    if (import.meta.env.DEV) console.log('[SpindL] avatar_load_model:', data.path, data.expressions ? '(+expressions)' : '', data.animations ? '(+animations)' : '');
    const msg: StatusMessage = {
      type: 'loadModel',
      value: data.path,
      expressions: data.expressions,
      animations: data.animations,
      characterAnimationsDir: data.character_animations_dir,
      baseAnimations: data.base_animations,
    };
    onMessage?.(msg);
  });

  // NANO-098: Live preview of expression composites from character editor
  socket.on('avatar_preview_expressions', (data: { expressions: Record<string, Record<string, number>>; previewMood?: string }) => {
    const msg: StatusMessage = { type: 'previewExpressions', expressions: data.expressions, previewMood: data.previewMood ?? null };
    onMessage?.(msg);
  });

  // NANO-098 Session 3: Live preview of animation clip from character editor
  socket.on('avatar_preview_animation', (data: { clip: string | null }) => {
    const msg: StatusMessage = { type: 'previewAnimation', clip: data.clip };
    onMessage?.(msg);
  });

  // NANO-098 Session 3: Update animation config from editor save
  socket.on('avatar_update_animation_config', (data: { animations: AnimationConfig | null }) => {
    const msg: StatusMessage = { type: 'updateAnimationConfig', animations: data.animations ?? null };
    onMessage?.(msg);
  });

  // NANO-099: Rescan animation files (base animation uploaded/cleared)
  socket.on('avatar_rescan_animations', (data?: { base_animations?: BaseAnimationsConfig }) => {
    if (import.meta.env.DEV) console.log('[SpindL] avatar_rescan_animations');
    const msg: StatusMessage = { type: 'rescanAnimations', baseAnimations: data?.base_animations };
    onMessage?.(msg);
  });
}
