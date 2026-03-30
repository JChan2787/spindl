/**
 * Prompt Store - State management for prompt inspection
 * NANO-025 Phase 3
 */

import { create } from "zustand";
import type { PromptSnapshotEvent } from "@/types/events";

interface PromptStoreState {
  // Current prompt snapshot (most recent LLM call)
  currentSnapshot: PromptSnapshotEvent | null;

  // History of snapshots (last N for comparison)
  snapshotHistory: PromptSnapshotEvent[];

  // UI state
  selectedMessageIndex: number | null;

  // Actions
  setSnapshot: (snapshot: PromptSnapshotEvent) => void;
  clearSnapshot: () => void;
  selectMessage: (index: number | null) => void;
}

const MAX_HISTORY = 10;

export const usePromptStore = create<PromptStoreState>((set) => ({
  currentSnapshot: null,
  snapshotHistory: [],
  selectedMessageIndex: null,

  setSnapshot: (snapshot) =>
    set((state) => ({
      currentSnapshot: snapshot,
      snapshotHistory: [snapshot, ...state.snapshotHistory.slice(0, MAX_HISTORY - 1)],
    })),

  clearSnapshot: () =>
    set({
      currentSnapshot: null,
      selectedMessageIndex: null,
    }),

  selectMessage: (index) => set({ selectedMessageIndex: index }),
}));
