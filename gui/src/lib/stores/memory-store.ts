import { create } from "zustand";
import type {
  MemoryCollectionType,
  MemoryDocument,
  MemorySearchResult,
} from "@/types/events";

interface MemoryStoreState {
  // Collection counts
  counts: { global: number; general: number; flashcards: number; summaries: number };
  memoryEnabled: boolean;

  // Current active collection tab
  activeCollection: MemoryCollectionType;

  // Memory lists per collection (cached)
  globalMemories: MemoryDocument[];
  generalMemories: MemoryDocument[];
  flashcardMemories: MemoryDocument[];
  summaryMemories: MemoryDocument[];

  // Loading states
  isLoadingCounts: boolean;
  isLoadingList: boolean;

  // Selected memory (for detail/edit panel)
  selectedMemory: MemoryDocument | null;
  selectedCollection: MemoryCollectionType | null;

  // Editing state (general memories only)
  isEditing: boolean;
  isNewMemory: boolean;
  editContent: string;

  // Search
  searchQuery: string;
  searchResults: MemorySearchResult[];
  isSearching: boolean;
  isSearchMode: boolean;

  // Action feedback (auto-clear toast pattern)
  lastAction: {
    type: "add" | "edit" | "delete" | "promote" | "clear" | null;
    success: boolean | null;
    error: string | null;
  };

  // Actions - Data loading
  setCounts: (counts: { global: number; general: number; flashcards: number; summaries: number }, enabled: boolean) => void;
  setMemories: (collection: MemoryCollectionType, memories: MemoryDocument[]) => void;
  setLoadingCounts: (loading: boolean) => void;
  setLoadingList: (loading: boolean) => void;

  // Actions - Navigation
  setActiveCollection: (collection: MemoryCollectionType) => void;
  selectMemory: (memory: MemoryDocument | null, collection: MemoryCollectionType | null) => void;

  // Actions - Editing (general only)
  startNewMemory: () => void;
  startEditMemory: (memory: MemoryDocument) => void;
  setEditContent: (content: string) => void;
  cancelEdit: () => void;

  // Actions - Search
  setSearchQuery: (query: string) => void;
  setSearchResults: (results: MemorySearchResult[], query: string) => void;
  setSearching: (searching: boolean) => void;
  setSearchMode: (active: boolean) => void;

  // Actions - Mutations (optimistic local state updates)
  addMemory: (memory: MemoryDocument, collection?: MemoryCollectionType) => void;
  editMemory: (oldId: string, memory: MemoryDocument) => void;
  removeMemory: (collection: MemoryCollectionType, id: string) => void;
  clearAllFlashcards: () => void;

  // Actions - Action feedback
  setActionResult: (type: "add" | "edit" | "delete" | "promote" | "clear", success: boolean, error?: string) => void;
  clearActionResult: () => void;
}

export const useMemoryStore = create<MemoryStoreState>((set) => ({
  counts: { global: 0, general: 0, flashcards: 0, summaries: 0 },
  memoryEnabled: false,

  activeCollection: "global",

  globalMemories: [],
  generalMemories: [],
  flashcardMemories: [],
  summaryMemories: [],

  isLoadingCounts: false,
  isLoadingList: false,

  selectedMemory: null,
  selectedCollection: null,

  isEditing: false,
  isNewMemory: false,
  editContent: "",

  searchQuery: "",
  searchResults: [],
  isSearching: false,
  isSearchMode: false,

  lastAction: { type: null, success: null, error: null },

  // Data loading
  setCounts: (counts, enabled) => set({
    counts,
    memoryEnabled: enabled,
    isLoadingCounts: false,
  }),

  setMemories: (collection, memories) => set((state) => {
    const update: Partial<MemoryStoreState> = { isLoadingList: false };
    if (collection === "global") update.globalMemories = memories;
    else if (collection === "general") update.generalMemories = memories;
    else if (collection === "flashcards") update.flashcardMemories = memories;
    else if (collection === "summaries") update.summaryMemories = memories;
    return update;
  }),

  setLoadingCounts: (isLoadingCounts) => set({ isLoadingCounts }),
  setLoadingList: (isLoadingList) => set({ isLoadingList }),

  // Navigation
  setActiveCollection: (activeCollection) => set({
    activeCollection,
    selectedMemory: null,
    selectedCollection: null,
    isEditing: false,
    isNewMemory: false,
    editContent: "",
  }),

  selectMemory: (memory, collection) => set({
    selectedMemory: memory,
    selectedCollection: collection,
    isEditing: false,
    isNewMemory: false,
    editContent: "",
  }),

  // Editing (general only)
  startNewMemory: () => set((state) => ({
    isNewMemory: true,
    isEditing: true,
    editContent: "",
    selectedMemory: null,
    selectedCollection: state.activeCollection === "global" ? "global" : "general",
  })),

  startEditMemory: (memory) => set((state) => ({
    isEditing: true,
    isNewMemory: false,
    editContent: memory.content,
    selectedMemory: memory,
    selectedCollection: state.activeCollection === "global" ? "global" : "general",
  })),

  setEditContent: (editContent) => set({ editContent }),

  cancelEdit: () => set({
    isEditing: false,
    isNewMemory: false,
    editContent: "",
  }),

  // Search
  setSearchQuery: (searchQuery) => set({ searchQuery }),

  setSearchResults: (results, query) => set({
    searchResults: results,
    searchQuery: query,
    isSearching: false,
  }),

  setSearching: (isSearching) => set({ isSearching }),

  setSearchMode: (isSearchMode) => set((state) => ({
    isSearchMode,
    // Clear search results when exiting search mode
    ...(!isSearchMode ? { searchResults: [], searchQuery: "", isSearching: false } : {}),
  })),

  // Mutations
  addMemory: (memory, collection?: MemoryCollectionType) => set((state) => {
    const target = collection || state.activeCollection;
    if (target === "global") {
      return {
        globalMemories: [...state.globalMemories, memory],
        counts: { ...state.counts, global: state.counts.global + 1 },
        isEditing: false, isNewMemory: false, editContent: "",
        selectedMemory: memory, selectedCollection: "global",
      };
    }
    return {
      generalMemories: [...state.generalMemories, memory],
      counts: { ...state.counts, general: state.counts.general + 1 },
      isEditing: false, isNewMemory: false, editContent: "",
      selectedMemory: memory, selectedCollection: "general",
    };
  }),

  editMemory: (oldId, memory) => set((state) => {
    const isGlobal = state.activeCollection === "global";
    if (isGlobal) {
      return {
        globalMemories: state.globalMemories.map((m) => m.id === oldId ? memory : m),
        isEditing: false, isNewMemory: false, editContent: "",
        selectedMemory: memory, selectedCollection: "global",
      };
    }
    return {
      generalMemories: state.generalMemories.map((m) => m.id === oldId ? memory : m),
      isEditing: false, isNewMemory: false, editContent: "",
      selectedMemory: memory, selectedCollection: "general",
    };
  }),

  removeMemory: (collection, id) => set((state) => {
    const countKey = collection === "global" ? "global" : collection;
    const update: Partial<MemoryStoreState> = {
      counts: { ...state.counts, [countKey]: Math.max(0, state.counts[countKey] - 1) },
    };

    if (collection === "global") {
      update.globalMemories = state.globalMemories.filter((m) => m.id !== id);
    } else if (collection === "general") {
      update.generalMemories = state.generalMemories.filter((m) => m.id !== id);
    } else if (collection === "flashcards") {
      update.flashcardMemories = state.flashcardMemories.filter((m) => m.id !== id);
    } else if (collection === "summaries") {
      update.summaryMemories = state.summaryMemories.filter((m) => m.id !== id);
    }

    // Deselect if the deleted memory was selected
    if (state.selectedMemory?.id === id) {
      update.selectedMemory = null;
      update.selectedCollection = null;
    }

    return update;
  }),

  clearAllFlashcards: () => set((state) => ({
    flashcardMemories: [],
    counts: { ...state.counts, flashcards: 0 },
    // Deselect if viewing a flashcard
    ...(state.selectedCollection === "flashcards"
      ? { selectedMemory: null, selectedCollection: null }
      : {}),
  })),

  // Action feedback
  setActionResult: (type, success, error) => set({
    lastAction: { type, success, error: error ?? null },
  }),

  clearActionResult: () => set({
    lastAction: { type: null, success: null, error: null },
  }),
}));
