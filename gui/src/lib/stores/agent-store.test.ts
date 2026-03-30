import { describe, it, expect, beforeEach } from "vitest";
import { useAgentStore } from "./agent-store";
import type { StateChangedEvent, TokenUsageEvent, ToolInvokedEvent, ToolResultEvent } from "@/types/events";

describe("useAgentStore", () => {
  beforeEach(() => {
    // Reset store between tests
    useAgentStore.setState({
      state: "idle",
      stateTransitions: [],
      currentTranscription: "",
      currentResponse: "",
      isTranscriptionFinal: false,
      isResponseFinal: false,
      tokenUsage: null,
      health: null,
      config: null,
      toolActivities: [],
    });
  });

  describe("state management", () => {
    it("should initialize with idle state", () => {
      const state = useAgentStore.getState();
      expect(state.state).toBe("idle");
    });

    it("should update state", () => {
      useAgentStore.getState().setState("listening");
      expect(useAgentStore.getState().state).toBe("listening");
    });

    it("should add state transitions and update current state", () => {
      const event: StateChangedEvent = {
        from: "idle",
        to: "listening",
        trigger: "activation",
        timestamp: new Date().toISOString(),
      };

      useAgentStore.getState().addTransition(event);

      const state = useAgentStore.getState();
      expect(state.state).toBe("listening");
      expect(state.stateTransitions).toHaveLength(1);
      expect(state.stateTransitions[0].from).toBe("idle");
      expect(state.stateTransitions[0].to).toBe("listening");
    });

    it("should keep only last 20 transitions", () => {
      const addTransition = useAgentStore.getState().addTransition;

      // Add 25 transitions
      for (let i = 0; i < 25; i++) {
        addTransition({
          from: "idle",
          to: "listening",
          trigger: `trigger-${i}`,
          timestamp: new Date().toISOString(),
        });
      }

      expect(useAgentStore.getState().stateTransitions).toHaveLength(20);
    });
  });

  describe("transcription management", () => {
    it("should set transcription", () => {
      useAgentStore.getState().setTranscription("Hello world", false);

      const state = useAgentStore.getState();
      expect(state.currentTranscription).toBe("Hello world");
      expect(state.isTranscriptionFinal).toBe(false);
    });

    it("should clear transcription", () => {
      useAgentStore.getState().setTranscription("Hello", true);
      useAgentStore.getState().clearTranscription();

      const state = useAgentStore.getState();
      expect(state.currentTranscription).toBe("");
      expect(state.isTranscriptionFinal).toBe(false);
    });
  });

  describe("response management", () => {
    it("should set response", () => {
      useAgentStore.getState().setResponse("Hi there!", true);

      const state = useAgentStore.getState();
      expect(state.currentResponse).toBe("Hi there!");
      expect(state.isResponseFinal).toBe(true);
    });

    it("should append to response", () => {
      useAgentStore.getState().setResponse("Hello", false);
      useAgentStore.getState().appendResponse(" world");

      expect(useAgentStore.getState().currentResponse).toBe("Hello world");
    });

    it("should clear response", () => {
      useAgentStore.getState().setResponse("Test", true);
      useAgentStore.getState().clearResponse();

      const state = useAgentStore.getState();
      expect(state.currentResponse).toBe("");
      expect(state.isResponseFinal).toBe(false);
    });
  });

  describe("token usage", () => {
    it("should set token usage", () => {
      const usage: TokenUsageEvent = {
        prompt: 100,
        completion: 50,
        total: 150,
        max: 8192,
        percent: 1.83,
      };

      useAgentStore.getState().setTokenUsage(usage);

      expect(useAgentStore.getState().tokenUsage).toEqual(usage);
    });
  });

  describe("health status", () => {
    it("should set health status", () => {
      const health = {
        stt: true,
        tts: true,
        llm: true,
        vlm: false,
      };

      useAgentStore.getState().setHealth(health);

      expect(useAgentStore.getState().health).toEqual(health);
    });
  });

  // NANO-025 Phase 7: Tool Activity tests
  describe("tool activity management", () => {
    it("should initialize with empty tool activities", () => {
      expect(useAgentStore.getState().toolActivities).toEqual([]);
    });

    it("should add tool invoked event", () => {
      const event: ToolInvokedEvent = {
        tool_name: "screen_vision",
        arguments: { query: "what do you see?" },
        iteration: 1,
        tool_call_id: "call_123",
        timestamp: "2026-01-28T05:30:00.000Z",
      };

      useAgentStore.getState().addToolInvoked(event);

      const activities = useAgentStore.getState().toolActivities;
      expect(activities).toHaveLength(1);
      expect(activities[0].id).toBe("call_123");
      expect(activities[0].tool_name).toBe("screen_vision");
      expect(activities[0].status).toBe("running");
      expect(activities[0].arguments).toEqual({ query: "what do you see?" });
      expect(activities[0].iteration).toBe(1);
    });

    it("should update tool with result - success", () => {
      // First add the invoked event
      const invokedEvent: ToolInvokedEvent = {
        tool_name: "screen_vision",
        arguments: {},
        iteration: 1,
        tool_call_id: "call_456",
        timestamp: "2026-01-28T05:30:00.000Z",
      };
      useAgentStore.getState().addToolInvoked(invokedEvent);

      // Then update with result
      const resultEvent: ToolResultEvent = {
        tool_name: "screen_vision",
        success: true,
        result_summary: "I see a code editor with TypeScript",
        duration_ms: 1234,
        iteration: 1,
        tool_call_id: "call_456",
      };
      useAgentStore.getState().updateToolResult(resultEvent);

      const activities = useAgentStore.getState().toolActivities;
      expect(activities).toHaveLength(1);
      expect(activities[0].status).toBe("complete");
      expect(activities[0].success).toBe(true);
      expect(activities[0].result_summary).toBe("I see a code editor with TypeScript");
      expect(activities[0].duration_ms).toBe(1234);
    });

    it("should update tool with result - error", () => {
      // First add the invoked event
      const invokedEvent: ToolInvokedEvent = {
        tool_name: "screen_vision",
        arguments: {},
        iteration: 1,
        tool_call_id: "call_789",
        timestamp: "2026-01-28T05:30:00.000Z",
      };
      useAgentStore.getState().addToolInvoked(invokedEvent);

      // Then update with error result
      const resultEvent: ToolResultEvent = {
        tool_name: "screen_vision",
        success: false,
        result_summary: "Error: VLM provider not configured",
        duration_ms: 50,
        iteration: 1,
        tool_call_id: "call_789",
      };
      useAgentStore.getState().updateToolResult(resultEvent);

      const activities = useAgentStore.getState().toolActivities;
      expect(activities).toHaveLength(1);
      expect(activities[0].status).toBe("error");
      expect(activities[0].success).toBe(false);
    });

    it("should handle multiple tools in sequence", () => {
      // Add first tool
      useAgentStore.getState().addToolInvoked({
        tool_name: "screen_vision",
        arguments: {},
        iteration: 1,
        tool_call_id: "call_1",
        timestamp: "2026-01-28T05:30:00.000Z",
      });

      // Add second tool
      useAgentStore.getState().addToolInvoked({
        tool_name: "web_search",
        arguments: { query: "test" },
        iteration: 1,
        tool_call_id: "call_2",
        timestamp: "2026-01-28T05:30:01.000Z",
      });

      const activities = useAgentStore.getState().toolActivities;
      expect(activities).toHaveLength(2);
      expect(activities[0].tool_name).toBe("screen_vision");
      expect(activities[1].tool_name).toBe("web_search");
    });

    it("should only update matching tool_call_id", () => {
      // Add two tools
      useAgentStore.getState().addToolInvoked({
        tool_name: "tool_a",
        arguments: {},
        iteration: 1,
        tool_call_id: "call_a",
        timestamp: "2026-01-28T05:30:00.000Z",
      });
      useAgentStore.getState().addToolInvoked({
        tool_name: "tool_b",
        arguments: {},
        iteration: 1,
        tool_call_id: "call_b",
        timestamp: "2026-01-28T05:30:01.000Z",
      });

      // Update only tool_a
      useAgentStore.getState().updateToolResult({
        tool_name: "tool_a",
        success: true,
        result_summary: "done",
        duration_ms: 100,
        iteration: 1,
        tool_call_id: "call_a",
      });

      const activities = useAgentStore.getState().toolActivities;
      expect(activities[0].status).toBe("complete");
      expect(activities[1].status).toBe("running"); // Should still be running
    });

    it("should clear tool activities", () => {
      // Add some activities
      useAgentStore.getState().addToolInvoked({
        tool_name: "test_tool",
        arguments: {},
        iteration: 1,
        tool_call_id: "call_test",
        timestamp: "2026-01-28T05:30:00.000Z",
      });

      expect(useAgentStore.getState().toolActivities).toHaveLength(1);

      // Clear
      useAgentStore.getState().clearToolActivities();

      expect(useAgentStore.getState().toolActivities).toEqual([]);
    });
  });
});
