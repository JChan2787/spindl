import { create } from "zustand";
import type { CharacterBookEntry, GlobalCodexEvent, CharacterCodexEvent } from "@/types/events";

interface CodexStoreState {
  // Global codex
  globalEntries: CharacterBookEntry[];
  globalCodexName: string;
  isLoadingGlobal: boolean;

  // Character codex (for editing within character editor)
  characterEntries: CharacterBookEntry[];
  characterCodexId: string | null;
  isLoadingCharacter: boolean;

  // Entry being edited
  editingEntry: CharacterBookEntry | null;
  editingEntryCharacterId: string | null; // null = global
  isNewEntry: boolean;

  // Action feedback
  lastAction: {
    type: "create" | "update" | "delete" | null;
    entryId: number | null;
    characterId: string | null;
    success: boolean | null;
    error: string | null;
  };

  // Actions
  setGlobalCodex: (entries: CharacterBookEntry[], name: string) => void;
  setLoadingGlobal: (loading: boolean) => void;
  setCharacterCodex: (characterId: string, entries: CharacterBookEntry[]) => void;
  setLoadingCharacter: (loading: boolean) => void;
  clearCharacterCodex: () => void;

  // Entry editing
  startEditEntry: (entry: CharacterBookEntry, characterId: string | null) => void;
  startNewEntry: (characterId: string | null) => void;
  updateEditingEntry: (entry: CharacterBookEntry) => void;
  cancelEditEntry: () => void;

  // Action results
  setActionResult: (
    type: "create" | "update" | "delete",
    entryId: number,
    characterId: string | null,
    success: boolean,
    error?: string
  ) => void;
  setActionError: (error: string) => void;
  clearActionResult: () => void;

  // Entry mutations (local state updates after server confirms)
  addEntry: (entry: CharacterBookEntry, characterId: string | null) => void;
  updateEntry: (entry: CharacterBookEntry, characterId: string | null) => void;
  removeEntry: (entryId: number, characterId: string | null) => void;
}

// Helper to create an empty codex entry
export function createEmptyCodexEntry(): CharacterBookEntry {
  return {
    keys: [],
    content: "",
    extensions: {},
    enabled: true,
    insertion_order: 0,
    case_sensitive: false,
    name: "",
    priority: 10,
    id: undefined, // Will be auto-assigned by backend
    comment: "",
    selective: false,
    secondary_keys: [],
    constant: false,
    position: "after_char",
    sticky: undefined,
    cooldown: undefined,
    delay: undefined,
  };
}

export const useCodexStore = create<CodexStoreState>((set) => ({
  globalEntries: [],
  globalCodexName: "Global Codex",
  isLoadingGlobal: false,

  characterEntries: [],
  characterCodexId: null,
  isLoadingCharacter: false,

  editingEntry: null,
  editingEntryCharacterId: null,
  isNewEntry: false,

  lastAction: {
    type: null,
    entryId: null,
    characterId: null,
    success: null,
    error: null,
  },

  setGlobalCodex: (entries, name) =>
    set({ globalEntries: entries, globalCodexName: name, isLoadingGlobal: false }),

  setLoadingGlobal: (isLoadingGlobal) => set({ isLoadingGlobal }),

  setCharacterCodex: (characterId, entries) =>
    set({
      characterEntries: entries,
      characterCodexId: characterId,
      isLoadingCharacter: false,
    }),

  setLoadingCharacter: (isLoadingCharacter) => set({ isLoadingCharacter }),

  clearCharacterCodex: () =>
    set({
      characterEntries: [],
      characterCodexId: null,
      isLoadingCharacter: false,
    }),

  startEditEntry: (entry, characterId) =>
    set({
      editingEntry: { ...entry },
      editingEntryCharacterId: characterId,
      isNewEntry: false,
    }),

  startNewEntry: (characterId) =>
    set({
      editingEntry: createEmptyCodexEntry(),
      editingEntryCharacterId: characterId,
      isNewEntry: true,
    }),

  updateEditingEntry: (entry) => set({ editingEntry: entry }),

  cancelEditEntry: () =>
    set({
      editingEntry: null,
      editingEntryCharacterId: null,
      isNewEntry: false,
    }),

  setActionResult: (type, entryId, characterId, success, error) =>
    set({
      lastAction: {
        type,
        entryId,
        characterId,
        success,
        error: error ?? null,
      },
      // Clear editing state on successful create/update
      ...(success && (type === "create" || type === "update")
        ? { editingEntry: null, editingEntryCharacterId: null, isNewEntry: false }
        : {}),
    }),

  setActionError: (error) =>
    set({
      lastAction: {
        type: null,
        entryId: null,
        characterId: null,
        success: false,
        error,
      },
    }),

  clearActionResult: () =>
    set({
      lastAction: {
        type: null,
        entryId: null,
        characterId: null,
        success: null,
        error: null,
      },
    }),

  addEntry: (entry, characterId) =>
    set((state) => {
      if (characterId === null) {
        return { globalEntries: [...state.globalEntries, entry] };
      } else if (state.characterCodexId === characterId) {
        return { characterEntries: [...state.characterEntries, entry] };
      }
      return {};
    }),

  updateEntry: (entry, characterId) =>
    set((state) => {
      if (characterId === null) {
        return {
          globalEntries: state.globalEntries.map((e) =>
            e.id === entry.id ? entry : e
          ),
        };
      } else if (state.characterCodexId === characterId) {
        return {
          characterEntries: state.characterEntries.map((e) =>
            e.id === entry.id ? entry : e
          ),
        };
      }
      return {};
    }),

  removeEntry: (entryId, characterId) =>
    set((state) => {
      if (characterId === null) {
        return {
          globalEntries: state.globalEntries.filter((e) => e.id !== entryId),
        };
      } else if (state.characterCodexId === characterId) {
        return {
          characterEntries: state.characterEntries.filter((e) => e.id !== entryId),
        };
      }
      return {};
    }),
}));

// ============================================
// REST API Functions (NANO-035 Phase 2)
// ============================================

/**
 * Fetch global codex entries via REST API
 */
export async function fetchGlobalCodex(): Promise<GlobalCodexEvent> {
  const response = await fetch("/api/codex/global");
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to fetch global codex");
  }
  return response.json();
}

/**
 * Fetch character codex entries via REST API
 */
export async function fetchCharacterCodex(characterId: string): Promise<CharacterCodexEvent> {
  const response = await fetch(`/api/codex/character/${characterId}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to fetch character codex");
  }
  return response.json();
}

/**
 * Create a new codex entry via REST API
 * @param entry The entry to create
 * @param characterId If provided, add to character; otherwise add to global
 */
export async function createCodexEntryApi(
  entry: CharacterBookEntry,
  characterId: string | null = null
): Promise<{ entry_id: number; success: boolean }> {
  const url = characterId
    ? `/api/codex/character/${characterId}`
    : "/api/codex/global";

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entry }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to create codex entry");
  }
  return response.json();
}

/**
 * Update a codex entry via REST API
 * @param entry The updated entry
 * @param entryId The entry ID to update
 * @param characterId If provided, update in character; otherwise update in global
 */
export async function updateCodexEntryApi(
  entry: CharacterBookEntry,
  entryId: number,
  characterId: string | null = null
): Promise<{ entry_id: number; success: boolean }> {
  const url = characterId
    ? `/api/codex/character/${characterId}/${entryId}`
    : `/api/codex/global/${entryId}`;

  const response = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entry }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to update codex entry");
  }
  return response.json();
}

/**
 * Delete a codex entry via REST API
 * @param entryId The entry ID to delete
 * @param characterId If provided, delete from character; otherwise delete from global
 */
export async function deleteCodexEntryApi(
  entryId: number,
  characterId: string | null = null
): Promise<{ entry_id: number; success: boolean }> {
  const url = characterId
    ? `/api/codex/character/${characterId}/${entryId}`
    : `/api/codex/global/${entryId}`;

  const response = await fetch(url, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to delete codex entry");
  }
  return response.json();
}
