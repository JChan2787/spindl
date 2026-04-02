/**
 * Socket.IO client for SpindL Stream Deck (NANO-110).
 *
 * Connects to the same GUI server as the avatar and dashboard.
 * Receives addressing-others context config, emits start/stop events.
 */

import { io, Socket } from "socket.io-client";

// ── Types ────────────────────────────────────────────────────────────

export interface AddressingContext {
  id: string;
  label: string;
  prompt: string;
}

export interface StimuliConfigEvent {
  addressing_others_contexts: AddressingContext[];
}

export interface AddressingOthersState {
  active: boolean;
  context_id: string | null;
}

export type OnContextsUpdated = (contexts: AddressingContext[]) => void;
export type OnStateChanged = (state: AddressingOthersState) => void;
export type OnConnectionChanged = (connected: boolean) => void;

// ── Client ───────────────────────────────────────────────────────────

const DEFAULT_PORT = 8765;
const RECONNECT_DELAY = 2000;

let socket: Socket | null = null;
let onContextsUpdated: OnContextsUpdated | null = null;
let onStateChanged: OnStateChanged | null = null;
let onConnectionChanged: OnConnectionChanged | null = null;

export function connect(port: number = DEFAULT_PORT): void {
  if (socket?.connected) return;

  const url = `http://127.0.0.1:${port}`;
  console.log(`[StreamDeck] Connecting to ${url}`);

  socket = io(url, {
    transports: ["websocket"],
    reconnection: true,
    reconnectionDelay: RECONNECT_DELAY,
    timeout: 5000,
  });

  socket.on("connect", () => {
    console.log("[StreamDeck] Connected");
    socket!.emit("register_stream_deck_client", {});
    onConnectionChanged?.(true);
  });

  socket.on("disconnect", (reason: string) => {
    console.log(`[StreamDeck] Disconnected: ${reason}`);
    onConnectionChanged?.(false);
  });

  // Receive full stimuli config — extract addressing contexts
  socket.on("stimuli_config_updated", (event: StimuliConfigEvent) => {
    if (event.addressing_others_contexts) {
      console.log(
        `[StreamDeck] Contexts updated: ${event.addressing_others_contexts.length}`
      );
      onContextsUpdated?.(event.addressing_others_contexts);
    }
  });

  // Receive addressing-others state broadcast
  socket.on("addressing_others_state", (state: AddressingOthersState) => {
    console.log(
      `[StreamDeck] State: active=${state.active}, context=${state.context_id}`
    );
    onStateChanged?.(state);
  });

  // Hydrate contexts from config_loaded (sent on connect when orchestrator is running)
  socket.on("config_loaded", (event: { settings?: { stimuli?: StimuliConfigEvent } }) => {
    const contexts = event.settings?.stimuli?.addressing_others_contexts;
    if (contexts) {
      console.log(`[StreamDeck] Initial contexts from config_loaded: ${contexts.length}`);
      onContextsUpdated?.(contexts);
    }
  });
}

export function disconnect(): void {
  socket?.disconnect();
  socket = null;
}

// ── Event emitters ───────────────────────────────────────────────────

export function emitAddressingStart(contextId: string): void {
  socket?.emit("addressing_others_start", { context_id: contextId });
}

export function emitAddressingStop(): void {
  socket?.emit("addressing_others_stop", {});
}

// ── Callback registration ────────────────────────────────────────────

export function setOnContextsUpdated(cb: OnContextsUpdated): void {
  onContextsUpdated = cb;
}

export function setOnStateChanged(cb: OnStateChanged): void {
  onStateChanged = cb;
}

export function setOnConnectionChanged(cb: OnConnectionChanged): void {
  onConnectionChanged = cb;
}
