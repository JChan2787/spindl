import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import MemoriesPage from "./page";
import { useMemoryStore } from "@/lib/stores";

// Mock getSocket to return a fake socket
vi.mock("@/lib/socket", () => ({
  getSocket: vi.fn(() => ({
    emit: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    connected: false,
  })),
}));

describe("MemoriesPage", () => {
  beforeEach(() => {
    // Reset memory store to default state
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

  describe("rendering", () => {
    it("should render page title", () => {
      render(<MemoriesPage />);
      expect(screen.getByText("Memories")).toBeInTheDocument();
    });

    it("should render search input", () => {
      render(<MemoriesPage />);
      expect(screen.getByPlaceholderText("Semantic search...")).toBeInTheDocument();
    });

    it("should disable search when memory is disabled", () => {
      render(<MemoriesPage />);
      const input = screen.getByPlaceholderText("Semantic search...");
      expect(input).toBeDisabled();
    });

    it("should enable search when memory is enabled", () => {
      useMemoryStore.setState({ memoryEnabled: true });
      render(<MemoriesPage />);
      const input = screen.getByPlaceholderText("Semantic search...");
      expect(input).not.toBeDisabled();
    });
  });

  describe("collection tabs", () => {
    it("should render Global tab", () => {
      render(<MemoriesPage />);
      expect(screen.getByText("Global")).toBeInTheDocument();
    });

    it("should render Sessions tab", () => {
      render(<MemoriesPage />);
      expect(screen.getByText("Sessions")).toBeInTheDocument();
    });

    it("should render General tab", () => {
      render(<MemoriesPage />);
      expect(screen.getByText("General")).toBeInTheDocument();
    });
  });

  describe("collection stats", () => {
    it("should show disabled warning when memory is not enabled", () => {
      render(<MemoriesPage />);
      expect(screen.getByText("Memory system unavailable — start services to view memories")).toBeInTheDocument();
    });

    it("should show counts when memory is enabled", () => {
      useMemoryStore.setState({
        memoryEnabled: true,
        counts: { global: 2, general: 5, flashcards: 3, summaries: 1 },
      });
      render(<MemoriesPage />);
      expect(screen.getByText("2")).toBeInTheDocument(); // global
      expect(screen.getByText("4")).toBeInTheDocument(); // sessions (3+1)
      expect(screen.getByText("5")).toBeInTheDocument(); // general
    });
  });

  describe("empty states", () => {
    it("should show empty state for global collection", () => {
      render(<MemoriesPage />);
      expect(
        screen.getByText(/No global memories yet/)
      ).toBeInTheDocument();
    });
  });

  describe("action feedback toast", () => {
    it("should show success toast", () => {
      useMemoryStore.setState({
        lastAction: { type: "add", success: true, error: null },
      });
      render(<MemoriesPage />);
      expect(screen.getByText(/added successfully/i)).toBeInTheDocument();
    });

    it("should show error toast", () => {
      useMemoryStore.setState({
        lastAction: { type: "delete", success: false, error: "Not found" },
      });
      render(<MemoriesPage />);
      expect(screen.getByText(/Not found/)).toBeInTheDocument();
    });

    it("should not show toast when no action", () => {
      render(<MemoriesPage />);
      expect(screen.queryByText(/successfully/i)).not.toBeInTheDocument();
    });
  });

  describe("memory list with data", () => {
    it("should render global memory cards when data exists", () => {
      useMemoryStore.setState({
        memoryEnabled: true,
        globalMemories: [
          { id: "m1", content: "User likes coffee in the morning", metadata: { timestamp: "2026-02-07T12:00:00Z" } },
          { id: "m2", content: "User prefers dark mode", metadata: { timestamp: "2026-02-07T12:01:00Z" } },
        ],
      });
      render(<MemoriesPage />);
      expect(screen.getByText(/User likes coffee/)).toBeInTheDocument();
      expect(screen.getByText(/User prefers dark mode/)).toBeInTheDocument();
    });
  });

  describe("detail panel", () => {
    it("should show placeholder when nothing is selected", () => {
      render(<MemoriesPage />);
      expect(screen.getByText("Select a memory to view details")).toBeInTheDocument();
    });

    it("should show memory content when a memory is selected", () => {
      const memory = {
        id: "m1",
        content: "User loves espresso with oat milk",
        metadata: { timestamp: "2026-02-07T12:00:00Z", source: "gui_manual" },
      };
      useMemoryStore.setState({
        selectedMemory: memory,
        selectedCollection: "global",
      });
      render(<MemoriesPage />);
      expect(screen.getByText("User loves espresso with oat milk")).toBeInTheDocument();
    });
  });

  describe("search mode", () => {
    it("should render search results header in search mode", () => {
      useMemoryStore.setState({
        isSearchMode: true,
        searchQuery: "coffee",
        searchResults: [],
      });
      render(<MemoriesPage />);
      expect(screen.getByText("Search Results")).toBeInTheDocument();
    });

    it("should show no results message", () => {
      useMemoryStore.setState({
        isSearchMode: true,
        searchQuery: "xyz",
        searchResults: [],
        isSearching: false,
      });
      render(<MemoriesPage />);
      expect(screen.getByText("No results found")).toBeInTheDocument();
    });

    it("should show searching state", () => {
      useMemoryStore.setState({
        isSearchMode: true,
        searchQuery: "test",
        isSearching: true,
      });
      render(<MemoriesPage />);
      expect(screen.getByText("Searching...")).toBeInTheDocument();
    });

    it("should render search results", () => {
      useMemoryStore.setState({
        isSearchMode: true,
        searchQuery: "coffee",
        isSearching: false,
        searchResults: [
          {
            id: "sr-1",
            content: "User drinks coffee every day",
            metadata: {},
            collection: "global",
            distance: 0.3,
          },
        ],
      });
      render(<MemoriesPage />);
      expect(screen.getByText(/User drinks coffee/)).toBeInTheDocument();
    });
  });
});
