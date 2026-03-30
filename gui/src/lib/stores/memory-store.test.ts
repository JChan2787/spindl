import { describe, it, expect, beforeEach } from "vitest";
import { useMemoryStore } from "./memory-store";
import type { MemoryDocument, MemorySearchResult } from "@/types/events";

// Test fixture factories
function createMockMemory(overrides?: Partial<MemoryDocument>): MemoryDocument {
  return {
    id: "mem-1",
    content: "User likes coffee",
    metadata: { type: "general", timestamp: "2026-02-07T12:00:00Z" },
    ...overrides,
  };
}

function createMockSearchResult(
  overrides?: Partial<MemorySearchResult>,
): MemorySearchResult {
  return {
    id: "sr-1",
    content: "User likes coffee",
    metadata: {},
    collection: "general",
    distance: 0.3,
    ...overrides,
  };
}

describe("useMemoryStore", () => {
  beforeEach(() => {
    // Reset store between tests
    useMemoryStore.setState({
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
    });
  });

  // ===========================================================================
  // Initial state
  // ===========================================================================

  describe("initial state", () => {
    it("should initialize with zero counts", () => {
      const state = useMemoryStore.getState();
      expect(state.counts).toEqual({ global: 0, general: 0, flashcards: 0, summaries: 0 });
    });

    it("should initialize as disabled", () => {
      const state = useMemoryStore.getState();
      expect(state.memoryEnabled).toBe(false);
    });

    it("should default to global collection", () => {
      const state = useMemoryStore.getState();
      expect(state.activeCollection).toBe("global");
    });

    it("should initialize with empty memory arrays", () => {
      const state = useMemoryStore.getState();
      expect(state.globalMemories).toEqual([]);
      expect(state.generalMemories).toEqual([]);
      expect(state.flashcardMemories).toEqual([]);
      expect(state.summaryMemories).toEqual([]);
    });

    it("should initialize with no selection", () => {
      const state = useMemoryStore.getState();
      expect(state.selectedMemory).toBeNull();
      expect(state.selectedCollection).toBeNull();
    });

    it("should initialize with editing off", () => {
      const state = useMemoryStore.getState();
      expect(state.isEditing).toBe(false);
      expect(state.isNewMemory).toBe(false);
      expect(state.editContent).toBe("");
    });

    it("should initialize with search off", () => {
      const state = useMemoryStore.getState();
      expect(state.searchQuery).toBe("");
      expect(state.searchResults).toEqual([]);
      expect(state.isSearching).toBe(false);
      expect(state.isSearchMode).toBe(false);
    });
  });

  // ===========================================================================
  // Data loading actions
  // ===========================================================================

  describe("setCounts", () => {
    it("should update counts and enabled state", () => {
      const { setCounts } = useMemoryStore.getState();
      setCounts({ global: 2, general: 5, flashcards: 3, summaries: 1 }, true);

      const state = useMemoryStore.getState();
      expect(state.counts).toEqual({ global: 2, general: 5, flashcards: 3, summaries: 1 });
      expect(state.memoryEnabled).toBe(true);
      expect(state.isLoadingCounts).toBe(false);
    });

    it("should clear loading state", () => {
      useMemoryStore.setState({ isLoadingCounts: true });
      const { setCounts } = useMemoryStore.getState();
      setCounts({ global: 0, general: 0, flashcards: 0, summaries: 0 }, false);

      expect(useMemoryStore.getState().isLoadingCounts).toBe(false);
    });
  });

  describe("setMemories", () => {
    it("should set general memories", () => {
      const memories = [createMockMemory(), createMockMemory({ id: "mem-2" })];
      const { setMemories } = useMemoryStore.getState();
      setMemories("general", memories);

      expect(useMemoryStore.getState().generalMemories).toEqual(memories);
    });

    it("should set flashcard memories", () => {
      const memories = [createMockMemory({ id: "fc-1" })];
      const { setMemories } = useMemoryStore.getState();
      setMemories("flashcards", memories);

      expect(useMemoryStore.getState().flashcardMemories).toEqual(memories);
    });

    it("should set summary memories", () => {
      const memories = [createMockMemory({ id: "sum-1" })];
      const { setMemories } = useMemoryStore.getState();
      setMemories("summaries", memories);

      expect(useMemoryStore.getState().summaryMemories).toEqual(memories);
    });

    it("should clear loading state", () => {
      useMemoryStore.setState({ isLoadingList: true });
      const { setMemories } = useMemoryStore.getState();
      setMemories("general", []);

      expect(useMemoryStore.getState().isLoadingList).toBe(false);
    });
  });

  // ===========================================================================
  // Navigation actions
  // ===========================================================================

  describe("setActiveCollection", () => {
    it("should switch active collection", () => {
      const { setActiveCollection } = useMemoryStore.getState();
      setActiveCollection("flashcards");

      expect(useMemoryStore.getState().activeCollection).toBe("flashcards");
    });

    it("should clear selection and editing state", () => {
      useMemoryStore.setState({
        selectedMemory: createMockMemory(),
        selectedCollection: "general",
        isEditing: true,
        isNewMemory: true,
        editContent: "some text",
      });

      const { setActiveCollection } = useMemoryStore.getState();
      setActiveCollection("summaries");

      const state = useMemoryStore.getState();
      expect(state.selectedMemory).toBeNull();
      expect(state.selectedCollection).toBeNull();
      expect(state.isEditing).toBe(false);
      expect(state.isNewMemory).toBe(false);
      expect(state.editContent).toBe("");
    });
  });

  describe("selectMemory", () => {
    it("should set selected memory and collection", () => {
      const memory = createMockMemory();
      const { selectMemory } = useMemoryStore.getState();
      selectMemory(memory, "general");

      const state = useMemoryStore.getState();
      expect(state.selectedMemory).toEqual(memory);
      expect(state.selectedCollection).toBe("general");
    });

    it("should clear editing state when selecting", () => {
      useMemoryStore.setState({
        isEditing: true,
        isNewMemory: true,
        editContent: "draft",
      });

      const { selectMemory } = useMemoryStore.getState();
      selectMemory(createMockMemory(), "general");

      const state = useMemoryStore.getState();
      expect(state.isEditing).toBe(false);
      expect(state.isNewMemory).toBe(false);
      expect(state.editContent).toBe("");
    });

    it("should allow deselecting with null", () => {
      useMemoryStore.setState({
        selectedMemory: createMockMemory(),
        selectedCollection: "general",
      });

      const { selectMemory } = useMemoryStore.getState();
      selectMemory(null, null);

      const state = useMemoryStore.getState();
      expect(state.selectedMemory).toBeNull();
      expect(state.selectedCollection).toBeNull();
    });
  });

  // ===========================================================================
  // Editing actions
  // ===========================================================================

  describe("startNewMemory", () => {
    it("should enter new memory creation mode (global tab)", () => {
      const { startNewMemory } = useMemoryStore.getState();
      startNewMemory();

      const state = useMemoryStore.getState();
      expect(state.isNewMemory).toBe(true);
      expect(state.isEditing).toBe(true);
      expect(state.editContent).toBe("");
      expect(state.selectedMemory).toBeNull();
      expect(state.selectedCollection).toBe("global");
    });

    it("should enter new memory creation mode (general tab)", () => {
      useMemoryStore.setState({ activeCollection: "general" });
      const { startNewMemory } = useMemoryStore.getState();
      startNewMemory();

      const state = useMemoryStore.getState();
      expect(state.selectedCollection).toBe("general");
    });
  });

  describe("startEditMemory", () => {
    it("should enter edit mode with existing content (global tab)", () => {
      const memory = createMockMemory({ content: "edit this" });
      const { startEditMemory } = useMemoryStore.getState();
      startEditMemory(memory);

      const state = useMemoryStore.getState();
      expect(state.isEditing).toBe(true);
      expect(state.isNewMemory).toBe(false);
      expect(state.editContent).toBe("edit this");
      expect(state.selectedMemory).toEqual(memory);
      expect(state.selectedCollection).toBe("global");
    });

    it("should enter edit mode on general tab", () => {
      useMemoryStore.setState({ activeCollection: "general" });
      const memory = createMockMemory({ content: "edit this" });
      const { startEditMemory } = useMemoryStore.getState();
      startEditMemory(memory);

      expect(useMemoryStore.getState().selectedCollection).toBe("general");
    });
  });

  describe("setEditContent", () => {
    it("should update edit content", () => {
      const { setEditContent } = useMemoryStore.getState();
      setEditContent("new text");

      expect(useMemoryStore.getState().editContent).toBe("new text");
    });
  });

  describe("cancelEdit", () => {
    it("should clear editing state", () => {
      useMemoryStore.setState({
        isEditing: true,
        isNewMemory: true,
        editContent: "draft text",
      });

      const { cancelEdit } = useMemoryStore.getState();
      cancelEdit();

      const state = useMemoryStore.getState();
      expect(state.isEditing).toBe(false);
      expect(state.isNewMemory).toBe(false);
      expect(state.editContent).toBe("");
    });
  });

  // ===========================================================================
  // Search actions
  // ===========================================================================

  describe("search actions", () => {
    it("setSearchQuery should update query", () => {
      const { setSearchQuery } = useMemoryStore.getState();
      setSearchQuery("coffee");

      expect(useMemoryStore.getState().searchQuery).toBe("coffee");
    });

    it("setSearchResults should update results and clear searching state", () => {
      useMemoryStore.setState({ isSearching: true });
      const results = [createMockSearchResult()];
      const { setSearchResults } = useMemoryStore.getState();
      setSearchResults(results, "coffee");

      const state = useMemoryStore.getState();
      expect(state.searchResults).toEqual(results);
      expect(state.searchQuery).toBe("coffee");
      expect(state.isSearching).toBe(false);
    });

    it("setSearching should update searching state", () => {
      const { setSearching } = useMemoryStore.getState();
      setSearching(true);

      expect(useMemoryStore.getState().isSearching).toBe(true);
    });

    it("setSearchMode(true) should enable search mode", () => {
      const { setSearchMode } = useMemoryStore.getState();
      setSearchMode(true);

      expect(useMemoryStore.getState().isSearchMode).toBe(true);
    });

    it("setSearchMode(false) should clear search state", () => {
      useMemoryStore.setState({
        isSearchMode: true,
        searchResults: [createMockSearchResult()],
        searchQuery: "test",
        isSearching: true,
      });

      const { setSearchMode } = useMemoryStore.getState();
      setSearchMode(false);

      const state = useMemoryStore.getState();
      expect(state.isSearchMode).toBe(false);
      expect(state.searchResults).toEqual([]);
      expect(state.searchQuery).toBe("");
      expect(state.isSearching).toBe(false);
    });
  });

  // ===========================================================================
  // Mutation actions
  // ===========================================================================

  describe("addMemory", () => {
    it("should add to general memories when collection is general", () => {
      useMemoryStore.setState({
        activeCollection: "general",
        generalMemories: [createMockMemory()],
        counts: { global: 0, general: 1, flashcards: 0, summaries: 0 },
      });

      const newMemory = createMockMemory({ id: "mem-2", content: "New memory" });
      const { addMemory } = useMemoryStore.getState();
      addMemory(newMemory, "general");

      const state = useMemoryStore.getState();
      expect(state.generalMemories).toHaveLength(2);
      expect(state.counts.general).toBe(2);
    });

    it("should add to global memories when collection is global", () => {
      useMemoryStore.setState({
        counts: { global: 0, general: 0, flashcards: 0, summaries: 0 },
      });

      const newMemory = createMockMemory({ id: "mem-g1" });
      const { addMemory } = useMemoryStore.getState();
      addMemory(newMemory, "global");

      const state = useMemoryStore.getState();
      expect(state.globalMemories).toHaveLength(1);
      expect(state.counts.global).toBe(1);
      expect(state.selectedCollection).toBe("global");
    });

    it("should select the new memory", () => {
      useMemoryStore.setState({ activeCollection: "general" });
      const newMemory = createMockMemory({ id: "mem-new" });
      const { addMemory } = useMemoryStore.getState();
      addMemory(newMemory, "general");

      const state = useMemoryStore.getState();
      expect(state.selectedMemory).toEqual(newMemory);
      expect(state.selectedCollection).toBe("general");
    });

    it("should clear editing state", () => {
      useMemoryStore.setState({ isEditing: true, isNewMemory: true, editContent: "draft" });
      const { addMemory } = useMemoryStore.getState();
      addMemory(createMockMemory(), "general");

      const state = useMemoryStore.getState();
      expect(state.isEditing).toBe(false);
      expect(state.isNewMemory).toBe(false);
      expect(state.editContent).toBe("");
    });
  });

  describe("editMemory", () => {
    it("should replace memory by old ID", () => {
      const original = createMockMemory({ id: "mem-1", content: "old" });
      useMemoryStore.setState({ activeCollection: "general", generalMemories: [original] });

      const updated = createMockMemory({ id: "mem-1-new", content: "updated" });
      const { editMemory } = useMemoryStore.getState();
      editMemory("mem-1", updated);

      const state = useMemoryStore.getState();
      expect(state.generalMemories).toHaveLength(1);
      expect(state.generalMemories[0].id).toBe("mem-1-new");
      expect(state.generalMemories[0].content).toBe("updated");
    });

    it("should select the updated memory", () => {
      useMemoryStore.setState({
        activeCollection: "general",
        generalMemories: [createMockMemory({ id: "mem-1" })],
      });

      const updated = createMockMemory({ id: "mem-1-new" });
      const { editMemory } = useMemoryStore.getState();
      editMemory("mem-1", updated);

      expect(useMemoryStore.getState().selectedMemory).toEqual(updated);
    });

    it("should clear editing state", () => {
      useMemoryStore.setState({
        activeCollection: "general",
        generalMemories: [createMockMemory()],
        isEditing: true,
        editContent: "content",
      });

      const { editMemory } = useMemoryStore.getState();
      editMemory("mem-1", createMockMemory({ id: "new-id" }));

      const state = useMemoryStore.getState();
      expect(state.isEditing).toBe(false);
      expect(state.editContent).toBe("");
    });
  });

  describe("removeMemory", () => {
    it("should remove from general and decrement count", () => {
      useMemoryStore.setState({
        generalMemories: [
          createMockMemory({ id: "mem-1" }),
          createMockMemory({ id: "mem-2" }),
        ],
        counts: { global: 0, general: 2, flashcards: 0, summaries: 0 },
      });

      const { removeMemory } = useMemoryStore.getState();
      removeMemory("general", "mem-1");

      const state = useMemoryStore.getState();
      expect(state.generalMemories).toHaveLength(1);
      expect(state.generalMemories[0].id).toBe("mem-2");
      expect(state.counts.general).toBe(1);
    });

    it("should remove from flashcards and decrement count", () => {
      useMemoryStore.setState({
        flashcardMemories: [createMockMemory({ id: "fc-1" })],
        counts: { global: 0, general: 0, flashcards: 1, summaries: 0 },
      });

      const { removeMemory } = useMemoryStore.getState();
      removeMemory("flashcards", "fc-1");

      const state = useMemoryStore.getState();
      expect(state.flashcardMemories).toHaveLength(0);
      expect(state.counts.flashcards).toBe(0);
    });

    it("should remove from summaries and decrement count", () => {
      useMemoryStore.setState({
        summaryMemories: [createMockMemory({ id: "sum-1" })],
        counts: { global: 0, general: 0, flashcards: 0, summaries: 1 },
      });

      const { removeMemory } = useMemoryStore.getState();
      removeMemory("summaries", "sum-1");

      expect(useMemoryStore.getState().summaryMemories).toHaveLength(0);
      expect(useMemoryStore.getState().counts.summaries).toBe(0);
    });

    it("should deselect if deleted memory was selected", () => {
      const memory = createMockMemory({ id: "mem-1" });
      useMemoryStore.setState({
        generalMemories: [memory],
        counts: { global: 0, general: 1, flashcards: 0, summaries: 0 },
        selectedMemory: memory,
        selectedCollection: "general",
      });

      const { removeMemory } = useMemoryStore.getState();
      removeMemory("general", "mem-1");

      const state = useMemoryStore.getState();
      expect(state.selectedMemory).toBeNull();
      expect(state.selectedCollection).toBeNull();
    });

    it("should not deselect if a different memory was deleted", () => {
      const selected = createMockMemory({ id: "mem-1" });
      useMemoryStore.setState({
        generalMemories: [selected, createMockMemory({ id: "mem-2" })],
        counts: { global: 0, general: 2, flashcards: 0, summaries: 0 },
        selectedMemory: selected,
        selectedCollection: "general",
      });

      const { removeMemory } = useMemoryStore.getState();
      removeMemory("general", "mem-2");

      expect(useMemoryStore.getState().selectedMemory).toEqual(selected);
    });

    it("should not go below zero for count", () => {
      useMemoryStore.setState({
        generalMemories: [],
        counts: { general: 0, flashcards: 0, summaries: 0 },
      });

      const { removeMemory } = useMemoryStore.getState();
      removeMemory("general", "nonexistent");

      expect(useMemoryStore.getState().counts.general).toBe(0);
    });
  });

  describe("clearAllFlashcards", () => {
    it("should clear all flashcard memories and zero count", () => {
      useMemoryStore.setState({
        flashcardMemories: [
          createMockMemory({ id: "fc-1" }),
          createMockMemory({ id: "fc-2" }),
        ],
        counts: { global: 0, general: 3, flashcards: 2, summaries: 1 },
      });

      const { clearAllFlashcards } = useMemoryStore.getState();
      clearAllFlashcards();

      const state = useMemoryStore.getState();
      expect(state.flashcardMemories).toEqual([]);
      expect(state.counts.flashcards).toBe(0);
      // Other counts should be unchanged
      expect(state.counts.general).toBe(3);
      expect(state.counts.summaries).toBe(1);
    });

    it("should deselect if viewing a flashcard", () => {
      const fc = createMockMemory({ id: "fc-1" });
      useMemoryStore.setState({
        flashcardMemories: [fc],
        counts: { global: 0, general: 0, flashcards: 1, summaries: 0 },
        selectedMemory: fc,
        selectedCollection: "flashcards",
      });

      const { clearAllFlashcards } = useMemoryStore.getState();
      clearAllFlashcards();

      const state = useMemoryStore.getState();
      expect(state.selectedMemory).toBeNull();
      expect(state.selectedCollection).toBeNull();
    });

    it("should not deselect if viewing a general memory", () => {
      const general = createMockMemory({ id: "mem-1" });
      useMemoryStore.setState({
        flashcardMemories: [createMockMemory({ id: "fc-1" })],
        counts: { global: 0, general: 1, flashcards: 1, summaries: 0 },
        selectedMemory: general,
        selectedCollection: "general",
      });

      const { clearAllFlashcards } = useMemoryStore.getState();
      clearAllFlashcards();

      expect(useMemoryStore.getState().selectedMemory).toEqual(general);
    });
  });

  // ===========================================================================
  // Action feedback
  // ===========================================================================

  describe("action feedback", () => {
    it("setActionResult should set last action", () => {
      const { setActionResult } = useMemoryStore.getState();
      setActionResult("add", true);

      const state = useMemoryStore.getState();
      expect(state.lastAction.type).toBe("add");
      expect(state.lastAction.success).toBe(true);
      expect(state.lastAction.error).toBeNull();
    });

    it("setActionResult should include error message", () => {
      const { setActionResult } = useMemoryStore.getState();
      setActionResult("delete", false, "Not found");

      const state = useMemoryStore.getState();
      expect(state.lastAction.success).toBe(false);
      expect(state.lastAction.error).toBe("Not found");
    });

    it("clearActionResult should reset last action", () => {
      useMemoryStore.setState({
        lastAction: { type: "add", success: true, error: null },
      });

      const { clearActionResult } = useMemoryStore.getState();
      clearActionResult();

      const state = useMemoryStore.getState();
      expect(state.lastAction.type).toBeNull();
      expect(state.lastAction.success).toBeNull();
      expect(state.lastAction.error).toBeNull();
    });
  });
});
