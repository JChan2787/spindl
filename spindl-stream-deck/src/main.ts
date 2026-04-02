/**
 * SpindL Stream Deck — Main entry point (NANO-110).
 *
 * Renders a dynamic button grid for addressing-others contexts.
 * Each button: hold to activate, release to deactivate.
 * One active context at a time.
 */

import {
  connect,
  emitAddressingStart,
  emitAddressingStop,
  setOnContextsUpdated,
  setOnStateChanged,
  setOnConnectionChanged,
  type AddressingContext,
} from "./ws-client";

import "./style.css";

// ── State ────────────────────────────────────────────────────────────

let contexts: AddressingContext[] = [
  { id: "ctx_0", label: "Others", prompt: "" },
];
let activeContextId: string | null = null;
let connected = false;

// ── DOM ──────────────────────────────────────────────────────────────

const app = document.getElementById("app")!;

function render(): void {
  app.innerHTML = "";

  // Status bar
  const status = document.createElement("div");
  status.className = `status ${connected ? "connected" : "disconnected"}`;
  status.textContent = connected ? "LIVE" : "DISCONNECTED";
  app.appendChild(status);

  // Button container
  const grid = document.createElement("div");
  grid.className = "button-grid";

  for (const ctx of contexts) {
    const btn = document.createElement("button");
    btn.className = `deck-btn ${activeContextId === ctx.id ? "active" : ""}`;
    btn.textContent = ctx.label || "Others";
    btn.dataset.contextId = ctx.id;

    // Hold to activate
    btn.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      activateContext(ctx.id);
    });

    // Release to deactivate
    btn.addEventListener("pointerup", (e) => {
      e.preventDefault();
      deactivateContext();
    });

    // Handle pointer leaving button while held
    btn.addEventListener("pointerleave", (e) => {
      e.preventDefault();
      if (activeContextId === ctx.id) {
        deactivateContext();
      }
    });

    // Prevent context menu on long press
    btn.addEventListener("contextmenu", (e) => e.preventDefault());

    grid.appendChild(btn);
  }

  app.appendChild(grid);
}

// ── Actions ──────────────────────────────────────────────────────────

function activateContext(contextId: string): void {
  // If another context is active, release it first
  if (activeContextId && activeContextId !== contextId) {
    emitAddressingStop();
  }
  activeContextId = contextId;
  emitAddressingStart(contextId);
  render();
}

function deactivateContext(): void {
  if (!activeContextId) return;
  activeContextId = null;
  emitAddressingStop();
  render();
}

// ── Window resize ────────────────────────────────────────────────────

function resizeWindow(): void {
  // Each button is ~48px tall + 8px gap. Status bar is ~24px. Padding is 12px.
  const buttonHeight = 48;
  const gap = 8;
  const statusHeight = 24;
  const padding = 24; // 12px top + 12px bottom
  const targetHeight = statusHeight + padding + contexts.length * buttonHeight + (contexts.length - 1) * gap;

  // Dynamically resize the Tauri window
  import("@tauri-apps/api/window").then(({ getCurrentWindow, LogicalSize }) => {
    const win = getCurrentWindow();
    win.setSize(new LogicalSize(220, Math.max(80, targetHeight)));
  }).catch(() => {
    // Not in Tauri context (dev browser) — skip resize
  });
}

// ── Socket callbacks ─────────────────────────────────────────────────

setOnContextsUpdated((newContexts) => {
  contexts = newContexts;
  render();
  resizeWindow();
});

setOnStateChanged((state) => {
  activeContextId = state.active ? state.context_id : null;
  render();
});

setOnConnectionChanged((isConnected) => {
  connected = isConnected;
  render();
});

// ── Init ─────────────────────────────────────────────────────────────

render();
connect();
