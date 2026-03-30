import { describe, it, expect, beforeEach } from "vitest";
import { useConnectionStore } from "./connection-store";

describe("useConnectionStore", () => {
  beforeEach(() => {
    // Reset store between tests
    useConnectionStore.setState({
      connected: false,
      connecting: false,
      error: null,
      lastConnected: null,
    });
  });

  describe("connection state", () => {
    it("should initialize as disconnected", () => {
      const state = useConnectionStore.getState();
      expect(state.connected).toBe(false);
      expect(state.connecting).toBe(false);
    });

    it("should set connected state and record timestamp", () => {
      const before = new Date();
      useConnectionStore.getState().setConnected(true);
      const after = new Date();

      const state = useConnectionStore.getState();
      expect(state.connected).toBe(true);
      expect(state.connecting).toBe(false);
      expect(state.error).toBeNull();
      expect(state.lastConnected).not.toBeNull();
      expect(state.lastConnected!.getTime()).toBeGreaterThanOrEqual(before.getTime());
      expect(state.lastConnected!.getTime()).toBeLessThanOrEqual(after.getTime());
    });

    it("should clear lastConnected when disconnected", () => {
      useConnectionStore.getState().setConnected(true);
      useConnectionStore.getState().setConnected(false);

      expect(useConnectionStore.getState().lastConnected).toBeNull();
    });

    it("should set connecting state", () => {
      useConnectionStore.getState().setConnecting(true);

      const state = useConnectionStore.getState();
      expect(state.connecting).toBe(true);
      expect(state.error).toBeNull();
    });
  });

  describe("error handling", () => {
    it("should set error and clear connection states", () => {
      useConnectionStore.getState().setConnecting(true);
      useConnectionStore.getState().setError("Connection refused");

      const state = useConnectionStore.getState();
      expect(state.error).toBe("Connection refused");
      expect(state.connected).toBe(false);
      expect(state.connecting).toBe(false);
    });

    it("should clear error when connecting", () => {
      useConnectionStore.getState().setError("Previous error");
      useConnectionStore.getState().setConnecting(true);

      expect(useConnectionStore.getState().error).toBeNull();
    });

    it("should clear error when connected", () => {
      useConnectionStore.getState().setError("Previous error");
      useConnectionStore.getState().setConnected(true);

      expect(useConnectionStore.getState().error).toBeNull();
    });
  });
});
