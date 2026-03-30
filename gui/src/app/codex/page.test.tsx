import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import CodexPage from "./page";
import { useCodexStore } from "@/lib/stores";
import type { CharacterBookEntry } from "@/types/events";

// Mock getSocket to return a fake socket
vi.mock("@/lib/socket", () => ({
  getSocket: vi.fn(() => ({
    emit: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    connected: false,
  })),
}));

// Mock fetchGlobalCodex to prevent actual fetch on mount
vi.mock("@/lib/stores", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/stores")>();
  return {
    ...actual,
    fetchGlobalCodex: vi.fn(() =>
      Promise.resolve({ entries: [], name: "Global Codex" })
    ),
    createCodexEntryApi: vi.fn(),
    updateCodexEntryApi: vi.fn(),
    deleteCodexEntryApi: vi.fn(),
  };
});

const sampleEntry: CharacterBookEntry = {
  id: 1,
  keys: ["dragon", "fire"],
  content: "Dragons breathe fire and hoard gold.",
  extensions: {},
  enabled: true,
  insertion_order: 0,
  case_sensitive: false,
  name: "Dragon Lore",
  priority: 10,
  comment: "",
  selective: false,
  secondary_keys: [],
  constant: false,
  position: "after_char",
};

const secondEntry: CharacterBookEntry = {
  id: 2,
  keys: ["elf"],
  content: "Elves live in forests and are immortal.",
  extensions: {},
  enabled: false,
  insertion_order: 1,
  case_sensitive: false,
  name: "Elf Knowledge",
  priority: 5,
  comment: "",
  selective: false,
  secondary_keys: [],
  constant: false,
  position: "after_char",
};

describe("CodexPage", () => {
  beforeEach(() => {
    // Reset codex store to default state
    useCodexStore.setState({
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
    });
  });

  describe("rendering", () => {
    it("should render codex name as heading", () => {
      render(<CodexPage />);
      expect(screen.getByText("Global Codex")).toBeInTheDocument();
    });

    it("should render description text", () => {
      render(<CodexPage />);
      expect(
        screen.getByText(/Global entries are active across all characters/)
      ).toBeInTheDocument();
    });

    it("should render custom codex name from store", () => {
      useCodexStore.setState({ globalCodexName: "World Lore" });
      render(<CodexPage />);
      expect(screen.getByText("World Lore")).toBeInTheDocument();
    });
  });

  describe("empty state", () => {
    it("should show empty entry list message", () => {
      render(<CodexPage />);
      expect(screen.getByText("No codex entries yet")).toBeInTheDocument();
    });

    it("should show entry count as 0", () => {
      render(<CodexPage />);
      expect(screen.getByText("0 Entries")).toBeInTheDocument();
    });

    it("should show selection placeholder when not editing", () => {
      render(<CodexPage />);
      expect(
        screen.getByText("Select an entry to edit or create a new one")
      ).toBeInTheDocument();
    });
  });

  describe("entry list with data", () => {
    it("should render entry names", () => {
      useCodexStore.setState({ globalEntries: [sampleEntry, secondEntry] });
      render(<CodexPage />);
      expect(screen.getByText("Dragon Lore")).toBeInTheDocument();
      expect(screen.getByText("Elf Knowledge")).toBeInTheDocument();
    });

    it("should show correct entry count", () => {
      useCodexStore.setState({ globalEntries: [sampleEntry, secondEntry] });
      render(<CodexPage />);
      expect(screen.getByText("2 Entries")).toBeInTheDocument();
    });

    it("should show singular entry count", () => {
      useCodexStore.setState({ globalEntries: [sampleEntry] });
      render(<CodexPage />);
      expect(screen.getByText("1 Entry")).toBeInTheDocument();
    });

    it("should show content preview", () => {
      useCodexStore.setState({ globalEntries: [sampleEntry] });
      render(<CodexPage />);
      expect(
        screen.getByText(/Dragons breathe fire/)
      ).toBeInTheDocument();
    });

    it("should show keyword badges", () => {
      useCodexStore.setState({ globalEntries: [sampleEntry] });
      render(<CodexPage />);
      expect(screen.getByText("dragon")).toBeInTheDocument();
      expect(screen.getByText("fire")).toBeInTheDocument();
    });
  });

  describe("loading state", () => {
    it("should show Add Entry button", () => {
      render(<CodexPage />);
      expect(
        screen.getByRole("button", { name: /add entry/i })
      ).toBeInTheDocument();
    });
  });

  describe("editing state", () => {
    it("should show entry form when editing a global entry", () => {
      useCodexStore.setState({
        globalEntries: [sampleEntry],
        editingEntry: sampleEntry,
        editingEntryCharacterId: null,
        isNewEntry: false,
      });
      render(<CodexPage />);
      // "Dragon Lore" appears in both entry list and form header
      expect(screen.getAllByText("Dragon Lore").length).toBeGreaterThanOrEqual(2);
      expect(screen.getByRole("button", { name: /save/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
    });

    it("should show New Codex Entry title for new entries", () => {
      useCodexStore.setState({
        editingEntry: {
          ...sampleEntry,
          id: undefined,
          name: "",
          content: "",
          keys: [],
        },
        editingEntryCharacterId: null,
        isNewEntry: true,
      });
      render(<CodexPage />);
      expect(screen.getByText("New Codex Entry")).toBeInTheDocument();
    });

    it("should not show form when editing a character entry (not global)", () => {
      useCodexStore.setState({
        editingEntry: sampleEntry,
        editingEntryCharacterId: "some-character-id",
        isNewEntry: false,
      });
      render(<CodexPage />);
      expect(
        screen.getByText("Select an entry to edit or create a new one")
      ).toBeInTheDocument();
    });

    it("should show delete button for existing entries", () => {
      useCodexStore.setState({
        globalEntries: [sampleEntry],
        editingEntry: sampleEntry,
        editingEntryCharacterId: null,
        isNewEntry: false,
      });
      render(<CodexPage />);
      expect(screen.getByRole("button", { name: /delete/i })).toBeInTheDocument();
    });

    it("should not show delete button for new entries", () => {
      useCodexStore.setState({
        editingEntry: {
          ...sampleEntry,
          id: undefined,
          name: "",
          content: "",
          keys: [],
        },
        editingEntryCharacterId: null,
        isNewEntry: true,
      });
      render(<CodexPage />);
      expect(screen.queryByRole("button", { name: /delete/i })).not.toBeInTheDocument();
    });
  });

  describe("entry form tabs", () => {
    it("should show Basic, Keywords, Timing, and Advanced tabs", () => {
      useCodexStore.setState({
        editingEntry: sampleEntry,
        editingEntryCharacterId: null,
        isNewEntry: false,
      });
      render(<CodexPage />);
      expect(screen.getByText("Basic")).toBeInTheDocument();
      expect(screen.getByText("Keywords")).toBeInTheDocument();
      expect(screen.getByText("Timing")).toBeInTheDocument();
      expect(screen.getByText("Advanced")).toBeInTheDocument();
    });
  });
});
