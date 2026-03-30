import { describe, it, expect, beforeEach } from "vitest";
import { useSessionStore } from "./session-store";
import type { SessionInfo, SessionDetailEvent, Turn } from "@/types/events";

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

function createMockSessionDetail(overrides?: Partial<SessionDetailEvent>): SessionDetailEvent {
  return {
    filepath: "/conversations/spindle_20260127_120000.jsonl",
    turns: [
      createMockTurn({ turn_id: 1, role: "user", content: "Hello" }),
      createMockTurn({ turn_id: 2, role: "assistant", content: "Hi there!" }),
    ],
    ...overrides,
  };
}

describe("useSessionStore", () => {
  beforeEach(() => {
    // Reset store between tests
    useSessionStore.setState({
      sessions: [],
      isLoading: false,
      selectedSession: null,
      selectedSessionTurns: [],
      isLoadingDetail: false,
      personaFilter: null,
      lastAction: {
        type: null,
        filepath: null,
        success: null,
        error: null,
      },
    });
  });

  describe("initial state", () => {
    it("should initialize with empty sessions array", () => {
      const state = useSessionStore.getState();
      expect(state.sessions).toEqual([]);
    });

    it("should initialize with isLoading false", () => {
      const state = useSessionStore.getState();
      expect(state.isLoading).toBe(false);
    });

    it("should initialize with null selected session", () => {
      const state = useSessionStore.getState();
      expect(state.selectedSession).toBeNull();
    });

    it("should initialize with empty turns array", () => {
      const state = useSessionStore.getState();
      expect(state.selectedSessionTurns).toEqual([]);
    });

    it("should initialize with null action state", () => {
      const state = useSessionStore.getState();
      expect(state.lastAction.type).toBeNull();
      expect(state.lastAction.success).toBeNull();
    });
  });

  describe("setSessions", () => {
    it("should set sessions array", () => {
      const sessions = [
        createMockSession({ filepath: "/path/1.jsonl" }),
        createMockSession({ filepath: "/path/2.jsonl" }),
      ];

      useSessionStore.getState().setSessions(sessions);

      expect(useSessionStore.getState().sessions).toHaveLength(2);
    });

    it("should set isLoading to false", () => {
      useSessionStore.setState({ isLoading: true });
      useSessionStore.getState().setSessions([]);

      expect(useSessionStore.getState().isLoading).toBe(false);
    });

    it("should replace existing sessions", () => {
      useSessionStore.getState().setSessions([createMockSession({ filepath: "/old.jsonl" })]);
      useSessionStore.getState().setSessions([createMockSession({ filepath: "/new.jsonl" })]);

      const state = useSessionStore.getState();
      expect(state.sessions).toHaveLength(1);
      expect(state.sessions[0].filepath).toBe("/new.jsonl");
    });
  });

  describe("setLoading", () => {
    it("should set isLoading to true", () => {
      useSessionStore.getState().setLoading(true);
      expect(useSessionStore.getState().isLoading).toBe(true);
    });

    it("should set isLoading to false", () => {
      useSessionStore.setState({ isLoading: true });
      useSessionStore.getState().setLoading(false);
      expect(useSessionStore.getState().isLoading).toBe(false);
    });
  });

  describe("selectSession", () => {
    it("should set selected session", () => {
      const session = createMockSession();
      useSessionStore.getState().selectSession(session);

      expect(useSessionStore.getState().selectedSession).toEqual(session);
    });

    it("should set isLoadingDetail to true when selecting", () => {
      const session = createMockSession();
      useSessionStore.getState().selectSession(session);

      expect(useSessionStore.getState().isLoadingDetail).toBe(true);
    });

    it("should clear turns when selecting new session", () => {
      useSessionStore.setState({
        selectedSessionTurns: [createMockTurn()],
      });

      useSessionStore.getState().selectSession(createMockSession());

      expect(useSessionStore.getState().selectedSessionTurns).toEqual([]);
    });

    it("should clear selection when passing null", () => {
      useSessionStore.setState({
        selectedSession: createMockSession(),
        selectedSessionTurns: [createMockTurn()],
      });

      useSessionStore.getState().selectSession(null);

      expect(useSessionStore.getState().selectedSession).toBeNull();
    });

    it("should not set isLoadingDetail when deselecting", () => {
      useSessionStore.setState({ isLoadingDetail: true });
      useSessionStore.getState().selectSession(null);

      expect(useSessionStore.getState().isLoadingDetail).toBe(false);
    });
  });

  describe("setSessionDetail", () => {
    it("should set session turns", () => {
      const detail = createMockSessionDetail();
      useSessionStore.getState().setSessionDetail(detail);

      expect(useSessionStore.getState().selectedSessionTurns).toHaveLength(2);
    });

    it("should set isLoadingDetail to false", () => {
      useSessionStore.setState({ isLoadingDetail: true });
      useSessionStore.getState().setSessionDetail(createMockSessionDetail());

      expect(useSessionStore.getState().isLoadingDetail).toBe(false);
    });

    it("should preserve turn data integrity", () => {
      const detail = createMockSessionDetail({
        turns: [
          createMockTurn({ turn_id: 1, role: "user", content: "Test content", hidden: false }),
          createMockTurn({ turn_id: 2, role: "summary", content: "Summary", hidden: false }),
          createMockTurn({ turn_id: 3, role: "assistant", content: "Response", hidden: true }),
        ],
      });

      useSessionStore.getState().setSessionDetail(detail);

      const turns = useSessionStore.getState().selectedSessionTurns;
      expect(turns[0].role).toBe("user");
      expect(turns[1].role).toBe("summary");
      expect(turns[2].hidden).toBe(true);
    });
  });

  describe("setPersonaFilter", () => {
    it("should set persona filter", () => {
      useSessionStore.getState().setPersonaFilter("spindle");
      expect(useSessionStore.getState().personaFilter).toBe("spindle");
    });

    it("should clear filter when passing null", () => {
      useSessionStore.setState({ personaFilter: "spindle" });
      useSessionStore.getState().setPersonaFilter(null);
      expect(useSessionStore.getState().personaFilter).toBeNull();
    });
  });

  describe("setActionResult", () => {
    it("should set resume action result", () => {
      useSessionStore.getState().setActionResult("resume", "/path/file.jsonl", true);

      const state = useSessionStore.getState();
      expect(state.lastAction.type).toBe("resume");
      expect(state.lastAction.filepath).toBe("/path/file.jsonl");
      expect(state.lastAction.success).toBe(true);
      expect(state.lastAction.error).toBeNull();
    });

    it("should set delete action result with error", () => {
      useSessionStore.getState().setActionResult("delete", "/path/file.jsonl", false, "Cannot delete active session");

      const state = useSessionStore.getState();
      expect(state.lastAction.type).toBe("delete");
      expect(state.lastAction.success).toBe(false);
      expect(state.lastAction.error).toBe("Cannot delete active session");
    });
  });

  describe("clearActionResult", () => {
    it("should clear all action state", () => {
      useSessionStore.getState().setActionResult("resume", "/path/file.jsonl", true);
      useSessionStore.getState().clearActionResult();

      const state = useSessionStore.getState();
      expect(state.lastAction.type).toBeNull();
      expect(state.lastAction.filepath).toBeNull();
      expect(state.lastAction.success).toBeNull();
      expect(state.lastAction.error).toBeNull();
    });
  });

  describe("removeSession", () => {
    it("should remove session from list", () => {
      const sessions = [
        createMockSession({ filepath: "/path/1.jsonl" }),
        createMockSession({ filepath: "/path/2.jsonl" }),
        createMockSession({ filepath: "/path/3.jsonl" }),
      ];
      useSessionStore.setState({ sessions });

      useSessionStore.getState().removeSession("/path/2.jsonl");

      const state = useSessionStore.getState();
      expect(state.sessions).toHaveLength(2);
      expect(state.sessions.find((s) => s.filepath === "/path/2.jsonl")).toBeUndefined();
    });

    it("should clear selection if removed session was selected", () => {
      const session = createMockSession({ filepath: "/path/1.jsonl" });
      useSessionStore.setState({
        sessions: [session],
        selectedSession: session,
        selectedSessionTurns: [createMockTurn()],
      });

      useSessionStore.getState().removeSession("/path/1.jsonl");

      const state = useSessionStore.getState();
      expect(state.selectedSession).toBeNull();
      expect(state.selectedSessionTurns).toEqual([]);
    });

    it("should not affect selection if different session removed", () => {
      const session1 = createMockSession({ filepath: "/path/1.jsonl" });
      const session2 = createMockSession({ filepath: "/path/2.jsonl" });
      const turns = [createMockTurn()];

      useSessionStore.setState({
        sessions: [session1, session2],
        selectedSession: session1,
        selectedSessionTurns: turns,
      });

      useSessionStore.getState().removeSession("/path/2.jsonl");

      const state = useSessionStore.getState();
      expect(state.selectedSession).toEqual(session1);
      expect(state.selectedSessionTurns).toEqual(turns);
    });

    it("should handle removing non-existent session gracefully", () => {
      const sessions = [createMockSession({ filepath: "/path/1.jsonl" })];
      useSessionStore.setState({ sessions });

      useSessionStore.getState().removeSession("/path/nonexistent.jsonl");

      expect(useSessionStore.getState().sessions).toHaveLength(1);
    });
  });

  describe("session data integrity", () => {
    it("should preserve all session metadata fields", () => {
      const session: SessionInfo = {
        filepath: "/conversations/test_20260127_120000.jsonl",
        persona: "test-persona",
        timestamp: "20260127_120000",
        turn_count: 42,
        visible_count: 38,
        file_size: 12345,
      };

      useSessionStore.getState().setSessions([session]);

      const stored = useSessionStore.getState().sessions[0];
      expect(stored.filepath).toBe(session.filepath);
      expect(stored.persona).toBe(session.persona);
      expect(stored.timestamp).toBe(session.timestamp);
      expect(stored.turn_count).toBe(42);
      expect(stored.visible_count).toBe(38);
      expect(stored.file_size).toBe(12345);
    });

    it("should preserve turn uuid and timestamp", () => {
      const detail = createMockSessionDetail({
        turns: [
          createMockTurn({
            uuid: "specific-uuid-12345",
            timestamp: "2026-01-27T15:30:00.123456+00:00",
          }),
        ],
      });

      useSessionStore.getState().setSessionDetail(detail);

      const turn = useSessionStore.getState().selectedSessionTurns[0];
      expect(turn.uuid).toBe("specific-uuid-12345");
      expect(turn.timestamp).toBe("2026-01-27T15:30:00.123456+00:00");
    });
  });
});
