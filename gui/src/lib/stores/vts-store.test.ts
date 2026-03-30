import { describe, it, expect, beforeEach } from "vitest";
import { useVTSStore } from "./vts-store";

describe("useVTSStore", () => {
  beforeEach(() => {
    useVTSStore.setState({
      connected: false,
      authenticated: false,
      enabled: false,
      modelName: null,
      hotkeys: [],
      expressions: [],
    });
  });

  describe("initial state", () => {
    it("should initialize as disconnected and disabled", () => {
      const state = useVTSStore.getState();
      expect(state.connected).toBe(false);
      expect(state.authenticated).toBe(false);
      expect(state.enabled).toBe(false);
      expect(state.modelName).toBeNull();
      expect(state.hotkeys).toEqual([]);
      expect(state.expressions).toEqual([]);
    });
  });

  describe("setStatus", () => {
    it("should update all status fields", () => {
      useVTSStore.getState().setStatus({
        connected: true,
        authenticated: true,
        enabled: true,
        model_name: "TestModel",
        hotkeys: ["wave", "dance"],
        expressions: [{ file: "happy.exp3.json", name: "happy", active: false }],
      });

      const state = useVTSStore.getState();
      expect(state.connected).toBe(true);
      expect(state.authenticated).toBe(true);
      expect(state.enabled).toBe(true);
      expect(state.modelName).toBe("TestModel");
      expect(state.hotkeys).toEqual(["wave", "dance"]);
      expect(state.expressions).toHaveLength(1);
      expect(state.expressions[0].name).toBe("happy");
    });

    it("should handle null model name", () => {
      useVTSStore.getState().setStatus({
        connected: false,
        authenticated: false,
        enabled: true,
        model_name: null,
        hotkeys: [],
        expressions: [],
      });

      expect(useVTSStore.getState().modelName).toBeNull();
    });
  });

  describe("setEnabled", () => {
    it("should update enabled state", () => {
      useVTSStore.getState().setEnabled(true);
      expect(useVTSStore.getState().enabled).toBe(true);

      useVTSStore.getState().setEnabled(false);
      expect(useVTSStore.getState().enabled).toBe(false);
    });
  });

  describe("setHotkeys", () => {
    it("should replace hotkey list", () => {
      useVTSStore.getState().setHotkeys(["wave", "dance", "think"]);
      expect(useVTSStore.getState().hotkeys).toEqual(["wave", "dance", "think"]);

      useVTSStore.getState().setHotkeys([]);
      expect(useVTSStore.getState().hotkeys).toEqual([]);
    });
  });

  describe("setExpressions", () => {
    it("should replace expression list", () => {
      useVTSStore.getState().setExpressions([
        { file: "happy.exp3.json", name: "happy", active: true },
        { file: "sad.exp3.json", name: "sad", active: false },
      ]);

      const exprs = useVTSStore.getState().expressions;
      expect(exprs).toHaveLength(2);
      expect(exprs[0].active).toBe(true);
      expect(exprs[1].active).toBe(false);
    });
  });
});
