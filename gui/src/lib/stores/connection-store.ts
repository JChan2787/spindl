import { create } from "zustand";

interface ConnectionState {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  lastConnected: Date | null;

  // Actions
  setConnected: (connected: boolean) => void;
  setConnecting: (connecting: boolean) => void;
  setError: (error: string | null) => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  connected: false,
  connecting: false,
  error: null,
  lastConnected: null,

  setConnected: (connected) =>
    set({
      connected,
      connecting: false,
      error: null,
      lastConnected: connected ? new Date() : null,
    }),

  setConnecting: (connecting) =>
    set({ connecting, error: null }),

  setError: (error) =>
    set({ error, connecting: false, connected: false }),
}));
