import { vi } from "vitest";
import type { TypedSocket } from "@/lib/socket";

/**
 * Create a mock Socket.IO client for testing
 */
export function createMockSocket(): TypedSocket {
  const listeners: Map<string, Set<(...args: unknown[]) => void>> = new Map();

  const mockSocket = {
    connected: false,

    on: vi.fn((event: string, callback: (...args: unknown[]) => void) => {
      if (!listeners.has(event)) {
        listeners.set(event, new Set());
      }
      listeners.get(event)!.add(callback);
      return mockSocket;
    }),

    off: vi.fn((event: string, callback?: (...args: unknown[]) => void) => {
      if (callback && listeners.has(event)) {
        listeners.get(event)!.delete(callback);
      } else {
        listeners.delete(event);
      }
      return mockSocket;
    }),

    emit: vi.fn(),

    connect: vi.fn(() => {
      mockSocket.connected = true;
      // Trigger connect event
      listeners.get("connect")?.forEach((cb) => cb());
      return mockSocket;
    }),

    disconnect: vi.fn(() => {
      mockSocket.connected = false;
      listeners.get("disconnect")?.forEach((cb) => cb("io client disconnect"));
      return mockSocket;
    }),

    // Test helper: simulate receiving an event from server
    _simulateEvent: (event: string, ...args: unknown[]) => {
      listeners.get(event)?.forEach((cb) => cb(...args));
    },

    // Test helper: get all registered listeners for an event
    _getListeners: (event: string) => listeners.get(event) || new Set(),
  };

  return mockSocket as unknown as TypedSocket;
}
