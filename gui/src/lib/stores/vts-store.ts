/**
 * VTubeStudio Store - State management for VTS dashboard (NANO-060b)
 *
 * Dedicated store following block-editor-store pattern.
 * VTS controls are instant-fire (buttons), not debounced.
 */

import { create } from "zustand";
import type { VTSExpression } from "@/types/events";

interface VTSState {
  // Connection / status
  connected: boolean;
  authenticated: boolean;
  enabled: boolean;
  modelName: string | null;

  // Cached lists from VTS
  hotkeys: string[];
  expressions: VTSExpression[];

  // Actions
  setStatus: (status: {
    connected: boolean;
    authenticated: boolean;
    enabled: boolean;
    model_name: string | null;
    hotkeys: string[];
    expressions: VTSExpression[];
  }) => void;
  setEnabled: (enabled: boolean) => void;
  setHotkeys: (hotkeys: string[]) => void;
  setExpressions: (expressions: VTSExpression[]) => void;
}

export const useVTSStore = create<VTSState>((set) => ({
  connected: false,
  authenticated: false,
  enabled: false,
  modelName: null,
  hotkeys: [],
  expressions: [],

  setStatus: (status) =>
    set({
      connected: status.connected,
      authenticated: status.authenticated,
      enabled: status.enabled,
      modelName: status.model_name,
      hotkeys: status.hotkeys,
      expressions: status.expressions,
    }),

  setEnabled: (enabled) => set({ enabled }),

  setHotkeys: (hotkeys) => set({ hotkeys }),

  setExpressions: (expressions) => set({ expressions }),
}));
