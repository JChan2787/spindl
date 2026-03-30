/**
 * Block Editor Store - State management for block config API + editor UI
 * NANO-045c-1 (socket plumbing) + NANO-045c-2 (UI state)
 */

import { create } from "zustand";
import type { BlockConfigLoadedEvent, BlockConfigUpdatedEvent } from "@/types/events";

interface BlockEditorState {
  // --- Server state (045c-1) ---

  /** Current block config from server (null until first load) */
  blockConfig: BlockConfigLoadedEvent | null;

  /** Whether a save operation is in progress */
  isSaving: boolean;

  // --- UI state (045c-2) ---

  /** Currently selected block ID in the editor */
  selectedBlockId: string | null;

  /** Whether the editor is in edit mode (vs view-only) */
  editMode: boolean;

  /** Pending override edits (local, not yet saved to server) */
  pendingOverrides: Record<string, string | null>;

  /** Pending disabled list (null = no changes from server state) */
  pendingDisabled: string[] | null;

  /** Pending block order (null = no changes from server state) */
  pendingOrder: string[] | null;

  // --- Server actions (045c-1) ---

  /** Set full block config (from block_config_loaded event) */
  setBlockConfig: (config: BlockConfigLoadedEvent) => void;

  /** Handle block_config_updated response (clears saving + pending) */
  setBlockConfigUpdated: (event: BlockConfigUpdatedEvent) => void;

  /** Toggle saving flag (set before emitting set_block_config) */
  setSaving: (saving: boolean) => void;

  // --- UI actions (045c-2) ---

  /** Select a block in the editor */
  selectBlock: (blockId: string | null) => void;

  /** Toggle between view and edit mode */
  toggleEditMode: () => void;

  /** Toggle a block's enabled/disabled state (locally) */
  toggleBlockEnabled: (blockId: string) => void;

  /** Set an override for a block (locally) */
  setOverride: (blockId: string, content: string | null) => void;

  /** Set pending block order (from drag-and-drop reorder) */
  setPendingOrder: (order: string[]) => void;

  /** Discard all pending changes without saving */
  clearPending: () => void;

  /** Check if there are unsaved changes */
  hasPendingChanges: () => boolean;
}

export const useBlockEditorStore = create<BlockEditorState>((set, get) => ({
  // Server state
  blockConfig: null,
  isSaving: false,

  // UI state
  selectedBlockId: null,
  editMode: false,
  pendingOverrides: {},
  pendingDisabled: null,
  pendingOrder: null,

  // --- Server actions ---

  setBlockConfig: (config) => set({ blockConfig: config }),

  setBlockConfigUpdated: (event) =>
    set((state) => {
      if (!event.success) {
        return { isSaving: false };
      }
      // Merge updated fields into existing config (preserves blocks metadata)
      const updated: BlockConfigLoadedEvent = state.blockConfig
        ? {
            ...state.blockConfig,
            order: event.order,
            disabled: event.disabled,
            overrides: event.overrides,
          }
        : { order: event.order, disabled: event.disabled, overrides: event.overrides, blocks: [] };
      // Clear pending state on successful save
      return {
        blockConfig: updated,
        isSaving: false,
        pendingOverrides: {},
        pendingDisabled: null,
        pendingOrder: null,
      };
    }),

  setSaving: (saving) => set({ isSaving: saving }),

  // --- UI actions ---

  selectBlock: (blockId) => set({ selectedBlockId: blockId }),

  toggleEditMode: () =>
    set((state) => ({
      editMode: !state.editMode,
      // Clear selection and pending changes when leaving edit mode
      ...(!state.editMode
        ? {}
        : { selectedBlockId: null, pendingOverrides: {}, pendingDisabled: null, pendingOrder: null }),
    })),

  toggleBlockEnabled: (blockId) =>
    set((state) => {
      const config = state.blockConfig;
      if (!config) return {};

      // Initialize from server state if first edit
      const current = state.pendingDisabled ?? [...config.disabled];
      const idx = current.indexOf(blockId);
      const next = idx >= 0 ? current.filter((id) => id !== blockId) : [...current, blockId];

      return { pendingDisabled: next };
    }),

  setOverride: (blockId, content) =>
    set((state) => {
      const next = { ...state.pendingOverrides };
      if (content === null) {
        // Explicit null means "remove override"
        next[blockId] = null;
      } else {
        next[blockId] = content;
      }
      return { pendingOverrides: next };
    }),

  setPendingOrder: (order) => set({ pendingOrder: order }),

  clearPending: () =>
    set({ pendingOverrides: {}, pendingDisabled: null, pendingOrder: null, selectedBlockId: null }),

  hasPendingChanges: () => {
    const state = get();
    const hasOverrideChanges = Object.keys(state.pendingOverrides).length > 0;
    const hasDisabledChanges = state.pendingDisabled !== null;
    const hasOrderChanges = state.pendingOrder !== null;
    return hasOverrideChanges || hasDisabledChanges || hasOrderChanges;
  },
}));
