import { create } from "zustand";
import type {
  CharacterInfo,
  CharacterCardData,
  CharacterDetailEvent,
  CharacterListEvent,
  AvatarDataEvent,
  AvatarDataEventExtended,
  CropSettings,
  ReloadCharacterResponse,
} from "@/types/events";
import { getSocket } from "@/lib/socket";

// ============================================
// REST API Functions
// ============================================

export async function fetchCharacters(): Promise<CharacterListEvent> {
  const response = await fetch("/api/characters");
  if (!response.ok) {
    throw new Error(`Failed to fetch characters: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchCharacterDetail(characterId: string): Promise<CharacterDetailEvent> {
  const response = await fetch(`/api/characters/${encodeURIComponent(characterId)}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch character: ${response.statusText}`);
  }
  return response.json();
}

export async function createCharacterApi(
  card: CharacterCardData,
  characterId?: string,
  avatarData?: { cropped: string; original: string; settings: CropSettings },
): Promise<{ character_id: string; success: boolean }> {
  const body: Record<string, unknown> = { card, character_id: characterId };

  if (avatarData) {
    body.avatar_data = avatarData.cropped;
    body.original_avatar_data = avatarData.original;
    body.crop_settings = avatarData.settings;
  }

  const response = await fetch("/api/characters", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || `Failed to create character: ${response.statusText}`);
  }
  return response.json();
}

export async function updateCharacterApi(characterId: string, card: CharacterCardData): Promise<{ character_id: string; success: boolean }> {
  const response = await fetch(`/api/characters/${encodeURIComponent(characterId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ card }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || `Failed to update character: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteCharacterApi(characterId: string): Promise<{ character_id: string; success: boolean }> {
  const response = await fetch(`/api/characters/${encodeURIComponent(characterId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || `Failed to delete character: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchAvatar(characterId: string): Promise<AvatarDataEvent> {
  const response = await fetch(`/api/characters/${encodeURIComponent(characterId)}/avatar`);
  if (!response.ok) {
    throw new Error(`Failed to fetch avatar: ${response.statusText}`);
  }
  return response.json();
}

export async function uploadAvatarApi(
  characterId: string,
  imageData: string,
  originalData?: string,
  cropSettings?: CropSettings,
): Promise<{ character_id: string; success: boolean }> {
  const body: Record<string, unknown> = { image_data: imageData };

  if (originalData) {
    body.original_avatar_data = originalData;
  }
  if (cropSettings) {
    body.crop_settings = cropSettings;
  }

  const response = await fetch(`/api/characters/${encodeURIComponent(characterId)}/avatar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || `Failed to upload avatar: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchAvatarExtended(characterId: string): Promise<AvatarDataEventExtended> {
  const response = await fetch(
    `/api/characters/${encodeURIComponent(characterId)}/avatar?include_original=true`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch extended avatar data: ${response.statusText}`);
  }
  return response.json();
}

export async function validateImportApi(jsonData: string): Promise<{
  valid: boolean;
  preview?: {
    name: string;
    description: string;
    has_personality: boolean;
    has_system_prompt: boolean;
    has_codex: boolean;
    codex_count: number;
    has_spindl: boolean;
    tags: string[];
  };
  errors?: string[];
  warnings?: string[];
}> {
  const response = await fetch("/api/characters/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ json_data: jsonData }),
  });
  return response.json();
}

export async function importCharacterApi(
  jsonData: string,
  characterId?: string,
  overwrite?: boolean
): Promise<{
  character_id?: string;
  name?: string;
  was_overwrite?: boolean;
  has_avatar?: boolean;
  success?: boolean;
  error?: string;
  exists?: boolean;
}> {
  const response = await fetch("/api/characters/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ json_data: jsonData, character_id: characterId, overwrite }),
  });
  return response.json();
}

export async function importPngCharacterApi(
  file: File,
  characterId?: string,
  overwrite?: boolean
): Promise<{
  character_id?: string;
  name?: string;
  was_overwrite?: boolean;
  has_avatar?: boolean;
  success?: boolean;
  error?: string;
  exists?: boolean;
}> {
  const formData = new FormData();
  formData.append("file", file);
  if (characterId) formData.append("character_id", characterId);
  if (overwrite) formData.append("overwrite", "true");

  const response = await fetch("/api/characters/import", {
    method: "POST",
    body: formData,
  });
  return response.json();
}

export async function exportCharacterApi(
  characterId: string,
  includeSpindl?: boolean,
  includeCodex?: boolean
): Promise<{
  character_id?: string;
  json_data?: string;
  filename?: string;
  success?: boolean;
  error?: string;
}> {
  const response = await fetch("/api/characters/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      character_id: characterId,
      include_spindl: includeSpindl,
      include_codex: includeCodex,
    }),
  });
  return response.json();
}

export async function exportPngCharacterApi(
  characterId: string,
  includeSpindl?: boolean,
  includeCodex?: boolean
): Promise<{ blob?: Blob; filename?: string; error?: string }> {
  const response = await fetch("/api/characters/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      character_id: characterId,
      format: "png",
      include_spindl: includeSpindl,
      include_codex: includeCodex,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    return { error: errorData.error || "PNG export failed" };
  }

  const blob = await response.blob();
  const disposition = response.headers.get("content-disposition") || "";
  const filenameMatch = disposition.match(/filename="?([^"]+)"?/);
  const filename = filenameMatch?.[1] || `${characterId}_card.png`;

  return { blob, filename };
}

// ============================================
// NANO-036: Character Hot-Reload
// ============================================

/**
 * Emit reload_character event to trigger backend hot-reload.
 * Returns a promise that resolves with the reload result.
 */
export function reloadCharacter(): Promise<ReloadCharacterResponse> {
  return new Promise((resolve) => {
    const socket = getSocket();
    if (!socket.connected) {
      resolve({
        success: false,
        error: "Socket not connected",
      });
      return;
    }
    socket.emit("reload_character", {}, (response: ReloadCharacterResponse) => {
      resolve(response);
    });
  });
}

// ============================================
// Store Types
// ============================================

interface CharacterStoreState {
  // Character list
  characters: CharacterInfo[];
  activeCharacterId: string | null;
  isLoading: boolean;

  // NANO-077: Character switching state
  isSwitchingCharacter: boolean;

  // Selected character for editing
  selectedCharacterId: string | null;
  selectedCharacterCard: CharacterCardData | null;
  selectedCharacterHasAvatar: boolean;
  isLoadingDetail: boolean;

  // Avatar cache
  avatarCache: Record<string, string | null>;

  // Action feedback
  lastAction: {
    type: "create" | "update" | "delete" | "avatar" | null;
    characterId: string | null;
    success: boolean | null;
    error: string | null;
  };

  // Unsaved changes tracking
  hasUnsavedChanges: boolean;
  editedCard: CharacterCardData | null;

  // Actions
  setCharacters: (characters: CharacterInfo[], active: string | null) => void;
  setLoading: (loading: boolean) => void;
  selectCharacter: (characterId: string | null) => void;
  setCharacterDetail: (detail: CharacterDetailEvent) => void;
  setLoadingDetail: (loading: boolean) => void;
  setActionResult: (
    type: "create" | "update" | "delete" | "avatar",
    characterId: string,
    success: boolean,
    error?: string
  ) => void;
  setActionError: (error: string) => void;
  clearActionResult: () => void;
  removeCharacter: (characterId: string) => void;
  setAvatar: (characterId: string, imageData: string | null) => void;
  setEditedCard: (card: CharacterCardData | null) => void;
  setHasUnsavedChanges: (hasChanges: boolean) => void;
  resetEditor: () => void;

  // NANO-077: Character switching
  setSwitchingCharacter: (switching: boolean) => void;
}

export const useCharacterStore = create<CharacterStoreState>((set) => ({
  characters: [],
  activeCharacterId: null,
  isLoading: false,

  isSwitchingCharacter: false,

  selectedCharacterId: null,
  selectedCharacterCard: null,
  selectedCharacterHasAvatar: false,
  isLoadingDetail: false,

  avatarCache: {},

  lastAction: {
    type: null,
    characterId: null,
    success: null,
    error: null,
  },

  hasUnsavedChanges: false,
  editedCard: null,

  setCharacters: (characters, active) =>
    set((state) => ({
      characters,
      activeCharacterId: active ?? state.activeCharacterId,
      isLoading: false,
    })),

  setLoading: (isLoading) => set({ isLoading }),

  selectCharacter: (characterId) =>
    set((state) => {
      if (characterId === state.selectedCharacterId) {
        return state;
      }
      return {
        selectedCharacterId: characterId,
        selectedCharacterCard: null,
        selectedCharacterHasAvatar: false,
        isLoadingDetail: characterId !== null,
        hasUnsavedChanges: false,
        editedCard: null,
      };
    }),

  setCharacterDetail: (detail) =>
    set((state) => ({
      selectedCharacterCard: detail.card,
      selectedCharacterHasAvatar: detail.has_avatar,
      isLoadingDetail: false,
      // Initialize edited card with loaded data
      editedCard: detail.card,
      hasUnsavedChanges: false,
      // Update the selected ID if it matches
      selectedCharacterId:
        state.selectedCharacterId === detail.character_id
          ? detail.character_id
          : state.selectedCharacterId,
    })),

  setLoadingDetail: (isLoadingDetail) => set({ isLoadingDetail }),

  setActionResult: (type, characterId, success, error) =>
    set({
      lastAction: {
        type,
        characterId,
        success,
        error: error ?? null,
      },
    }),

  setActionError: (error) =>
    set({
      lastAction: {
        type: null,
        characterId: null,
        success: false,
        error,
      },
    }),

  clearActionResult: () =>
    set({
      lastAction: {
        type: null,
        characterId: null,
        success: null,
        error: null,
      },
    }),

  removeCharacter: (characterId) =>
    set((state) => ({
      characters: state.characters.filter((c) => c.id !== characterId),
      selectedCharacterId:
        state.selectedCharacterId === characterId
          ? null
          : state.selectedCharacterId,
      selectedCharacterCard:
        state.selectedCharacterId === characterId
          ? null
          : state.selectedCharacterCard,
      editedCard:
        state.selectedCharacterId === characterId ? null : state.editedCard,
      hasUnsavedChanges:
        state.selectedCharacterId === characterId
          ? false
          : state.hasUnsavedChanges,
    })),

  setAvatar: (characterId, imageData) =>
    set((state) => ({
      avatarCache: {
        ...state.avatarCache,
        [characterId]: imageData,
      },
      // Update has_avatar flag if this is the selected character
      selectedCharacterHasAvatar:
        state.selectedCharacterId === characterId && imageData !== null
          ? true
          : state.selectedCharacterHasAvatar,
    })),

  setEditedCard: (card) =>
    set((state) => ({
      editedCard: card,
      hasUnsavedChanges:
        card !== null &&
        JSON.stringify(card) !== JSON.stringify(state.selectedCharacterCard),
    })),

  setHasUnsavedChanges: (hasChanges) => set({ hasUnsavedChanges: hasChanges }),

  resetEditor: () =>
    set({
      selectedCharacterId: null,
      selectedCharacterCard: null,
      selectedCharacterHasAvatar: false,
      editedCard: null,
      hasUnsavedChanges: false,
      isLoadingDetail: false,
    }),

  // NANO-077: Character switching
  setSwitchingCharacter: (isSwitchingCharacter) => set({ isSwitchingCharacter }),
}));

// Helper function to create an empty character card
export function createEmptyCharacterCard(name: string = "New Character"): CharacterCardData {
  return {
    spec: "chara_card_v2",
    spec_version: "2.0",
    data: {
      name,
      description: "",
      personality: "",
      scenario: "",
      first_mes: "",
      mes_example: "",
      creator_notes: "",
      system_prompt: "",
      post_history_instructions: "",
      alternate_greetings: [],
      tags: [],
      creator: "",
      character_version: "1.0",
      extensions: {
        spindl: {
          id: "",
          voice: "",
          language: "a",
          appearance: "",
          rules: [],
        },
      },
    },
  };
}
