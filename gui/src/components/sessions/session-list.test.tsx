import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SessionList } from "./session-list";
import { useSessionStore } from "@/lib/stores/session-store";
import { useAgentStore } from "@/lib/stores/agent-store";
import type { SessionInfo } from "@/types/events";

// Test fixture factory
function createMockSession(overrides?: Partial<SessionInfo>): SessionInfo {
  return {
    filepath: "/conversations/spindle_20260127_120000.jsonl",
    persona: "spindle",
    timestamp: "20260127_120000",
    turn_count: 10,
    visible_count: 8,
    file_size: 4096,
    ...overrides,
  };
}

describe("SessionList", () => {
  const mockOnRefresh = vi.fn();
  const mockOnCreateSession = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset stores
    useSessionStore.setState({
      sessions: [],
      isLoading: false,
      selectedSession: null,
      selectedSessionTurns: [],
      isLoadingDetail: false,
      personaFilter: null,
      lastAction: { type: null, filepath: null, success: null, error: null },
    });
    useAgentStore.setState({ health: null });
  });

  describe("rendering", () => {
    it("should render session list card", () => {
      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("Sessions")).toBeInTheDocument();
    });

    it("should show empty state when no sessions", () => {
      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("No sessions found")).toBeInTheDocument();
      expect(screen.getByText("Start a conversation to create one")).toBeInTheDocument();
    });

    it("should display session count badge", () => {
      useSessionStore.setState({
        sessions: [
          createMockSession({ filepath: "/path/1.jsonl" }),
          createMockSession({ filepath: "/path/2.jsonl" }),
        ],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("2")).toBeInTheDocument();
    });
  });

  describe("session cards", () => {
    it("should render session card for each session", () => {
      useSessionStore.setState({
        sessions: [
          createMockSession({ persona: "spindle" }),
          createMockSession({ filepath: "/path/2.jsonl", persona: "echo" }),
        ],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("spindle")).toBeInTheDocument();
      expect(screen.getByText("echo")).toBeInTheDocument();
    });

    it("should display formatted timestamp", () => {
      useSessionStore.setState({
        sessions: [createMockSession({ timestamp: "20260127_143025" })],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("2026-01-27 14:30:25")).toBeInTheDocument();
    });

    it("should display turn counts (visible / total)", () => {
      useSessionStore.setState({
        sessions: [createMockSession({ turn_count: 10, visible_count: 8 })],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("8 / 10")).toBeInTheDocument();
    });

    it("should display file size in KB", () => {
      useSessionStore.setState({
        sessions: [createMockSession({ file_size: 4096 })],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("4.0 KB")).toBeInTheDocument();
    });

    it("should display file size in MB for large files", () => {
      useSessionStore.setState({
        sessions: [createMockSession({ file_size: 1048576 })],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("1.0 MB")).toBeInTheDocument();
    });

    it("should display file size in bytes for small files", () => {
      useSessionStore.setState({
        sessions: [createMockSession({ file_size: 512 })],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("512 B")).toBeInTheDocument();
    });
  });

  describe("selection", () => {
    it("should call selectSession when card is clicked", () => {
      const session = createMockSession();
      useSessionStore.setState({ sessions: [session] });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      fireEvent.click(screen.getByText("spindle"));

      expect(useSessionStore.getState().selectedSession).toEqual(session);
    });

    it("should highlight selected session", () => {
      const session = createMockSession();
      useSessionStore.setState({
        sessions: [session],
        selectedSession: session,
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      // Get the session card (the one with tabIndex="0")
      const cards = screen.getAllByRole("button");
      const sessionCard = cards.find((c) => c.getAttribute("tabindex") === "0");
      expect(sessionCard).toHaveClass("border-primary");
    });

    it("should support keyboard selection with Enter", () => {
      const session = createMockSession();
      useSessionStore.setState({ sessions: [session] });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      // Get the session card (not the refresh button)
      const cards = screen.getAllByRole("button");
      const sessionCard = cards.find((c) => c.getAttribute("tabindex") === "0");
      fireEvent.keyDown(sessionCard!, { key: "Enter" });

      expect(useSessionStore.getState().selectedSession).toEqual(session);
    });

    it("should support keyboard selection with Space", () => {
      const session = createMockSession();
      useSessionStore.setState({ sessions: [session] });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      // Get the session card (not the refresh button)
      const cards = screen.getAllByRole("button");
      const sessionCard = cards.find((c) => c.getAttribute("tabindex") === "0");
      fireEvent.keyDown(sessionCard!, { key: " " });

      expect(useSessionStore.getState().selectedSession).toEqual(session);
    });
  });

  describe("refresh", () => {
    it("should call onRefresh when refresh button clicked", () => {
      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      const refreshButton = screen.getByTitle("Refresh");
      fireEvent.click(refreshButton);

      expect(mockOnRefresh).toHaveBeenCalledTimes(1);
    });

    it("should disable refresh button when loading", () => {
      useSessionStore.setState({ isLoading: true });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      const refreshButton = screen.getByTitle("Refresh");
      expect(refreshButton).toBeDisabled();
    });
  });

  describe("new session button", () => {
    it("should disable new session button when services not running", () => {
      useAgentStore.setState({ health: null });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      const newButton = screen.getByTitle("Services not running");
      expect(newButton).toBeDisabled();
    });

    it("should enable new session button when services running", () => {
      useAgentStore.setState({ health: { status: "healthy" } as any });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      const newButton = screen.getByTitle("New Session");
      expect(newButton).not.toBeDisabled();
    });

    it("should call onCreateSession when new session button clicked", () => {
      useAgentStore.setState({ health: { status: "healthy" } as any });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      const newButton = screen.getByTitle("New Session");
      fireEvent.click(newButton);

      expect(mockOnCreateSession).toHaveBeenCalledTimes(1);
    });
  });

  describe("NANO-077: persona filter", () => {
    it("should filter sessions by persona when personaFilter is set", () => {
      useSessionStore.setState({
        sessions: [
          createMockSession({ filepath: "/path/1.jsonl", persona: "spindle" }),
          createMockSession({ filepath: "/path/2.jsonl", persona: "mryummers" }),
          createMockSession({ filepath: "/path/3.jsonl", persona: "spindle" }),
        ],
        personaFilter: "spindle",
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      // Should show spindle sessions
      const personaBadges = screen.getAllByText("spindle");
      expect(personaBadges.length).toBeGreaterThan(0);
      // Should NOT show mryummers
      expect(screen.queryByText("mryummers")).not.toBeInTheDocument();
    });

    it("should show all sessions when personaFilter is null", () => {
      useSessionStore.setState({
        sessions: [
          createMockSession({ filepath: "/path/1.jsonl", persona: "spindle" }),
          createMockSession({ filepath: "/path/2.jsonl", persona: "mryummers" }),
        ],
        personaFilter: null,
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("spindle")).toBeInTheDocument();
      expect(screen.getByText("mryummers")).toBeInTheDocument();
    });

    it("should show filtered/total count badge when filtering", () => {
      useSessionStore.setState({
        sessions: [
          createMockSession({ filepath: "/path/1.jsonl", persona: "spindle" }),
          createMockSession({ filepath: "/path/2.jsonl", persona: "mryummers" }),
          createMockSession({ filepath: "/path/3.jsonl", persona: "spindle" }),
        ],
        personaFilter: "spindle",
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText("2/3")).toBeInTheDocument();
    });

    it("should show 'show all characters' link when filtered with no results", () => {
      useSessionStore.setState({
        sessions: [
          createMockSession({ filepath: "/path/1.jsonl", persona: "mryummers" }),
        ],
        personaFilter: "spindle",
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      expect(screen.getByText(/Show all characters/)).toBeInTheDocument();
    });

    it("should clear filter when 'show all characters' is clicked", () => {
      useSessionStore.setState({
        sessions: [
          createMockSession({ filepath: "/path/1.jsonl", persona: "mryummers" }),
        ],
        personaFilter: "spindle",
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      fireEvent.click(screen.getByText(/Show all characters/));
      expect(useSessionStore.getState().personaFilter).toBeNull();
    });
  });

  describe("edge cases", () => {
    it("should handle session with short timestamp gracefully", () => {
      useSessionStore.setState({
        sessions: [createMockSession({ timestamp: "short" })],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      // Should display the raw timestamp if it can't be parsed
      expect(screen.getByText("short")).toBeInTheDocument();
    });

    it("should handle empty persona gracefully", () => {
      useSessionStore.setState({
        sessions: [createMockSession({ persona: "" })],
      });

      render(<SessionList onRefresh={mockOnRefresh} onCreateSession={mockOnCreateSession} />);

      // Should still render without crashing
      expect(screen.getByText("Sessions")).toBeInTheDocument();
    });
  });
});
