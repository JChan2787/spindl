import { create } from "zustand";
import type {
  SessionInfo,
  SessionDetailEvent,
  Turn,
} from "@/types/events";

interface SessionStoreState {
  // Session list
  sessions: SessionInfo[];
  isLoading: boolean;

  // Active session (the one the orchestrator is currently using)
  activeSessionFilepath: string | null;

  // Selected session
  selectedSession: SessionInfo | null;
  selectedSessionTurns: Turn[];
  isLoadingDetail: boolean;

  // Filter
  personaFilter: string | null;

  // Action feedback
  lastAction: {
    type: "resume" | "delete" | "summarize" | "create" | null;
    filepath: string | null;
    success: boolean | null;
    error: string | null;
  };

  // Actions
  setSessions: (sessions: SessionInfo[], activeSession?: string | null) => void;
  setLoading: (loading: boolean) => void;
  selectSession: (session: SessionInfo | null) => void;
  setSessionDetail: (detail: SessionDetailEvent) => void;
  setLoadingDetail: (loading: boolean) => void;
  setPersonaFilter: (persona: string | null) => void;
  setActionResult: (type: "resume" | "delete" | "summarize" | "create", filepath: string, success: boolean, error?: string) => void;
  clearActionResult: () => void;
  removeSession: (filepath: string) => void;
}

export const useSessionStore = create<SessionStoreState>((set) => ({
  sessions: [],
  isLoading: false,

  activeSessionFilepath: null,

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

  setSessions: (sessions, activeSession) => set((state) => {
    const update: Partial<SessionStoreState> = {
      sessions,
      isLoading: false,
      activeSessionFilepath: activeSession !== undefined ? activeSession ?? null : state.activeSessionFilepath,
    };

    // Auto-select active session on first load
    if (!state.selectedSession && sessions.length > 0) {
      // Pick the session the backend marked as active, or fall back to first
      const active = activeSession
        ? sessions.find((s) => s.filepath === activeSession)
        : undefined;
      const target = active ?? sessions[0];
      update.selectedSession = target;
      update.selectedSessionTurns = [];
      update.isLoadingDetail = true;
    }

    return update;
  }),

  setLoading: (isLoading) => set({ isLoading }),

  selectSession: (session) => set({
    selectedSession: session,
    selectedSessionTurns: session ? [] : [],
    isLoadingDetail: session !== null,
  }),

  setSessionDetail: (detail) => set((state) => ({
    selectedSessionTurns: detail.turns,
    isLoadingDetail: false,
    // Update the selected session if it matches
    selectedSession: state.selectedSession?.filepath === detail.filepath
      ? state.selectedSession
      : state.selectedSession,
  })),

  setLoadingDetail: (isLoadingDetail) => set({ isLoadingDetail }),

  setPersonaFilter: (personaFilter) => set({ personaFilter }),

  setActionResult: (type, filepath, success, error) => set({
    lastAction: { type, filepath, success, error: error ?? null },
  }),

  clearActionResult: () => set({
    lastAction: { type: null, filepath: null, success: null, error: null },
  }),

  removeSession: (filepath) => set((state) => ({
    sessions: state.sessions.filter((s) => s.filepath !== filepath),
    selectedSession: state.selectedSession?.filepath === filepath ? null : state.selectedSession,
    selectedSessionTurns: state.selectedSession?.filepath === filepath ? [] : state.selectedSessionTurns,
  })),
}));
