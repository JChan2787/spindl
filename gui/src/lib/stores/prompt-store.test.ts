import { describe, it, expect, beforeEach } from "vitest";
import { usePromptStore } from "./prompt-store";
import type { PromptSnapshotEvent } from "@/types/events";

// Test fixture factory
function createMockSnapshot(overrides?: Partial<PromptSnapshotEvent>): PromptSnapshotEvent {
  return {
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
    ],
    token_breakdown: {
      total: 100,
      prompt: 80,
      completion: 20,
      system: 60,
      user: 20,
      sections: {
        agent: 30,
        context: 10,
        rules: 15,
        conversation: 5,
      },
    },
    input_modality: "VOICE",
    state_trigger: "activation",
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

describe("usePromptStore", () => {
  beforeEach(() => {
    // Reset store between tests
    usePromptStore.setState({
      currentSnapshot: null,
      snapshotHistory: [],
      selectedMessageIndex: null,
    });
  });

  describe("initial state", () => {
    it("should initialize with null snapshot", () => {
      const state = usePromptStore.getState();
      expect(state.currentSnapshot).toBeNull();
    });

    it("should initialize with empty history", () => {
      const state = usePromptStore.getState();
      expect(state.snapshotHistory).toEqual([]);
    });

    it("should initialize with null selected message", () => {
      const state = usePromptStore.getState();
      expect(state.selectedMessageIndex).toBeNull();
    });
  });

  describe("setSnapshot", () => {
    it("should set current snapshot", () => {
      const snapshot = createMockSnapshot();
      usePromptStore.getState().setSnapshot(snapshot);

      expect(usePromptStore.getState().currentSnapshot).toEqual(snapshot);
    });

    it("should add snapshot to history", () => {
      const snapshot = createMockSnapshot();
      usePromptStore.getState().setSnapshot(snapshot);

      const state = usePromptStore.getState();
      expect(state.snapshotHistory).toHaveLength(1);
      expect(state.snapshotHistory[0]).toEqual(snapshot);
    });

    it("should prepend new snapshots to history (newest first)", () => {
      const snapshot1 = createMockSnapshot({ timestamp: "2026-01-27T10:00:00Z" });
      const snapshot2 = createMockSnapshot({ timestamp: "2026-01-27T11:00:00Z" });

      usePromptStore.getState().setSnapshot(snapshot1);
      usePromptStore.getState().setSnapshot(snapshot2);

      const state = usePromptStore.getState();
      expect(state.snapshotHistory[0].timestamp).toBe("2026-01-27T11:00:00Z");
      expect(state.snapshotHistory[1].timestamp).toBe("2026-01-27T10:00:00Z");
    });

    it("should limit history to 10 items", () => {
      const setSnapshot = usePromptStore.getState().setSnapshot;

      // Add 15 snapshots
      for (let i = 0; i < 15; i++) {
        setSnapshot(createMockSnapshot({ timestamp: `2026-01-27T${i.toString().padStart(2, "0")}:00:00Z` }));
      }

      expect(usePromptStore.getState().snapshotHistory).toHaveLength(10);
    });

    it("should keep most recent 10 when history exceeds limit", () => {
      const setSnapshot = usePromptStore.getState().setSnapshot;

      // Add 15 snapshots (0-14)
      for (let i = 0; i < 15; i++) {
        setSnapshot(createMockSnapshot({ timestamp: `2026-01-27T${i.toString().padStart(2, "0")}:00:00Z` }));
      }

      const history = usePromptStore.getState().snapshotHistory;
      // Most recent (14) should be first
      expect(history[0].timestamp).toBe("2026-01-27T14:00:00Z");
      // Oldest kept (5) should be last
      expect(history[9].timestamp).toBe("2026-01-27T05:00:00Z");
    });
  });

  describe("clearSnapshot", () => {
    it("should clear current snapshot", () => {
      usePromptStore.getState().setSnapshot(createMockSnapshot());
      usePromptStore.getState().clearSnapshot();

      expect(usePromptStore.getState().currentSnapshot).toBeNull();
    });

    it("should clear selected message index", () => {
      usePromptStore.getState().setSnapshot(createMockSnapshot());
      usePromptStore.getState().selectMessage(1);
      usePromptStore.getState().clearSnapshot();

      expect(usePromptStore.getState().selectedMessageIndex).toBeNull();
    });

    it("should NOT clear history", () => {
      usePromptStore.getState().setSnapshot(createMockSnapshot());
      usePromptStore.getState().clearSnapshot();

      // History should still have the snapshot
      expect(usePromptStore.getState().snapshotHistory).toHaveLength(1);
    });
  });

  describe("selectMessage", () => {
    it("should set selected message index", () => {
      usePromptStore.getState().selectMessage(2);

      expect(usePromptStore.getState().selectedMessageIndex).toBe(2);
    });

    it("should allow setting to null", () => {
      usePromptStore.getState().selectMessage(2);
      usePromptStore.getState().selectMessage(null);

      expect(usePromptStore.getState().selectedMessageIndex).toBeNull();
    });
  });

  describe("snapshot data integrity", () => {
    it("should preserve all message roles", () => {
      const snapshot = createMockSnapshot({
        messages: [
          { role: "system", content: "System content" },
          { role: "user", content: "User content" },
          { role: "assistant", content: "Assistant content" },
        ],
      });

      usePromptStore.getState().setSnapshot(snapshot);

      const stored = usePromptStore.getState().currentSnapshot!;
      expect(stored.messages[0].role).toBe("system");
      expect(stored.messages[1].role).toBe("user");
      expect(stored.messages[2].role).toBe("assistant");
    });

    it("should preserve token breakdown sections", () => {
      const snapshot = createMockSnapshot({
        token_breakdown: {
          total: 500,
          prompt: 400,
          completion: 100,
          system: 350,
          user: 50,
          sections: {
            agent: 100,
            context: 75,
            rules: 125,
            conversation: 50,
          },
        },
      });

      usePromptStore.getState().setSnapshot(snapshot);

      const stored = usePromptStore.getState().currentSnapshot!;
      expect(stored.token_breakdown.sections.agent).toBe(100);
      expect(stored.token_breakdown.sections.context).toBe(75);
      expect(stored.token_breakdown.sections.rules).toBe(125);
      expect(stored.token_breakdown.sections.conversation).toBe(50);
    });

    it("should preserve input modality", () => {
      const voiceSnapshot = createMockSnapshot({ input_modality: "VOICE" });
      const textSnapshot = createMockSnapshot({ input_modality: "TEXT" });

      usePromptStore.getState().setSnapshot(voiceSnapshot);
      expect(usePromptStore.getState().currentSnapshot!.input_modality).toBe("VOICE");

      usePromptStore.getState().setSnapshot(textSnapshot);
      expect(usePromptStore.getState().currentSnapshot!.input_modality).toBe("TEXT");
    });

    it("should handle null state_trigger", () => {
      const snapshot = createMockSnapshot({ state_trigger: null });
      usePromptStore.getState().setSnapshot(snapshot);

      expect(usePromptStore.getState().currentSnapshot!.state_trigger).toBeNull();
    });
  });
});
