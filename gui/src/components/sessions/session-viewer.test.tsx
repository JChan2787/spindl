import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SessionViewer } from "./session-viewer";
import { useSessionStore } from "@/lib/stores/session-store";
import type { SessionInfo, Turn } from "@/types/events";

// Test fixture factories
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

function createMockTurn(overrides?: Partial<Turn>): Turn {
  return {
    turn_id: 1,
    uuid: "test-uuid-1234",
    role: "user",
    content: "Hello, assistant!",
    timestamp: "2026-01-27T12:00:00.000000+00:00",
    hidden: false,
    ...overrides,
  };
}

describe("SessionViewer", () => {
  const mockOnResume = vi.fn();
  const mockOnDelete = vi.fn();
  const mockOnExport = vi.fn();
  const mockOnGenerateSummary = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store
    useSessionStore.setState({
      sessions: [],
      isLoading: false,
      selectedSession: null,
      selectedSessionTurns: [],
      isLoadingDetail: false,
      personaFilter: null,
      lastAction: { type: null, filepath: null, success: null, error: null },
    });
  });

  describe("empty state", () => {
    it("should show placeholder when no session selected", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("Select a session to view")).toBeInTheDocument();
    });
  });

  describe("session header", () => {
    it("should display persona badge", () => {
      useSessionStore.setState({
        selectedSession: createMockSession({ persona: "spindle" }),
        selectedSessionTurns: [],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("spindle")).toBeInTheDocument();
    });

    it("should display turn count", () => {
      useSessionStore.setState({
        selectedSession: createMockSession({ turn_count: 42 }),
        selectedSessionTurns: [],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("42 turns")).toBeInTheDocument();
    });
  });

  describe("action buttons", () => {
    beforeEach(() => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [createMockTurn()],
      });
    });

    it("should render resume button", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("Resume")).toBeInTheDocument();
    });

    it("should call onResume when resume clicked", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      fireEvent.click(screen.getByText("Resume"));

      expect(mockOnResume).toHaveBeenCalledWith("/conversations/spindle_20260127_120000.jsonl");
    });

    it("should render export button", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("Export")).toBeInTheDocument();
    });

    it("should call onExport when export clicked", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      fireEvent.click(screen.getByText("Export"));

      expect(mockOnExport).toHaveBeenCalledWith("/conversations/spindle_20260127_120000.jsonl");
    });

    it("should show delete confirmation on first click", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      // Find trash button (icon only)
      const trashButtons = screen.getAllByRole("button").filter(
        (b) => b.classList.contains("text-destructive")
      );
      fireEvent.click(trashButtons[0]);

      expect(screen.getByText("Confirm")).toBeInTheDocument();
      expect(screen.getByText("Cancel")).toBeInTheDocument();
    });

    it("should call onDelete when confirm clicked", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      // Click trash first
      const trashButtons = screen.getAllByRole("button").filter(
        (b) => b.classList.contains("text-destructive")
      );
      fireEvent.click(trashButtons[0]);

      // Then confirm
      fireEvent.click(screen.getByText("Confirm"));

      expect(mockOnDelete).toHaveBeenCalledWith("/conversations/spindle_20260127_120000.jsonl");
    });

    it("should cancel delete on cancel click", () => {
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      // Click trash first
      const trashButtons = screen.getAllByRole("button").filter(
        (b) => b.classList.contains("text-destructive")
      );
      fireEvent.click(trashButtons[0]);

      // Then cancel
      fireEvent.click(screen.getByText("Cancel"));

      // Confirm should be gone
      expect(screen.queryByText("Confirm")).not.toBeInTheDocument();
      expect(mockOnDelete).not.toHaveBeenCalled();
    });
  });

  describe("turn rendering", () => {
    it("should render user turn with correct styling", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [
          createMockTurn({ role: "user", content: "Hello there" }),
        ],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("User")).toBeInTheDocument();
    });

    it("should render assistant turn with correct styling", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [
          createMockTurn({ role: "assistant", content: "Hi!" }),
        ],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("Assistant")).toBeInTheDocument();
    });

    it("should render summary turn with correct styling", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [
          createMockTurn({ role: "summary", content: "Summary of conversation" }),
        ],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("Summary")).toBeInTheDocument();
    });

    it("should show turn ID", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [createMockTurn({ turn_id: 42 })],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("#42")).toBeInTheDocument();
    });

    it("should show hidden badge for hidden turns", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [createMockTurn({ hidden: true })],
      });

      // Need to show hidden turns first
      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      // Click show hidden button
      const showHiddenButton = screen.getByText(/Show Hidden/);
      fireEvent.click(showHiddenButton);

      expect(screen.getByText("Hidden")).toBeInTheDocument();
    });
  });

  describe("hidden turns toggle", () => {
    it("should hide hidden turns by default", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [
          createMockTurn({ turn_id: 1, hidden: false, content: "Visible turn" }),
          createMockTurn({ turn_id: 2, hidden: true, content: "Hidden turn" }),
        ],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("#1")).toBeInTheDocument();
      expect(screen.queryByText("#2")).not.toBeInTheDocument();
    });

    it("should show hidden count in toggle button", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [
          createMockTurn({ uuid: "uuid-1", hidden: false }),
          createMockTurn({ uuid: "uuid-2", turn_id: 2, hidden: true }),
          createMockTurn({ uuid: "uuid-3", turn_id: 3, hidden: true }),
        ],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText(/Show Hidden \(2\)/)).toBeInTheDocument();
    });

    it("should show hidden turns when toggled", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [
          createMockTurn({ uuid: "uuid-1", turn_id: 1, hidden: false }),
          createMockTurn({ uuid: "uuid-2", turn_id: 2, hidden: true }),
        ],
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      fireEvent.click(screen.getByText(/Show Hidden/));

      expect(screen.getByText("#1")).toBeInTheDocument();
      expect(screen.getByText("#2")).toBeInTheDocument();
    });
  });

  describe("loading state", () => {
    it("should show loading spinner when detail is loading", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [],
        isLoadingDetail: true,
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      // Loading spinner should be visible (animate-spin class)
      const spinner = document.querySelector(".animate-spin");
      expect(spinner).toBeInTheDocument();
    });
  });

  describe("action feedback", () => {
    it("should show success message for resume action", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [],
        lastAction: {
          type: "resume",
          filepath: "/conversations/spindle_20260127_120000.jsonl",
          success: true,
          error: null,
        },
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("Session resumed successfully")).toBeInTheDocument();
    });

    it("should show error message for failed action", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [],
        lastAction: {
          type: "delete",
          filepath: "/conversations/spindle_20260127_120000.jsonl",
          success: false,
          error: "Cannot delete active session",
        },
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("Cannot delete active session")).toBeInTheDocument();
    });

    it("should not show feedback for different session", () => {
      useSessionStore.setState({
        selectedSession: createMockSession({ filepath: "/different/path.jsonl" }),
        selectedSessionTurns: [],
        lastAction: {
          type: "resume",
          filepath: "/conversations/spindle_20260127_120000.jsonl",
          success: true,
          error: null,
        },
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.queryByText("Session resumed successfully")).not.toBeInTheDocument();
    });
  });

  describe("empty session", () => {
    it("should show empty message when session has no turns", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [],
        isLoadingDetail: false,
      });

      render(
        <SessionViewer
          onResume={mockOnResume}
          onDelete={mockOnDelete}
          onExport={mockOnExport}
          onGenerateSummary={mockOnGenerateSummary}
        />
      );

      expect(screen.getByText("No turns in this session")).toBeInTheDocument();
    });
  });
});
