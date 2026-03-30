import { create } from "zustand";
import type {
  AgentState,
  StateChangedEvent,
  TokenUsageEvent,
  HealthStatusEvent,
  ConfigLoadedEvent,
  ToolInvokedEvent,
  ToolResultEvent,
  ActivatedCodexEntry,
  RetrievedMemory,
} from "@/types/events";

interface StateTransition extends StateChangedEvent {
  id: number;
}

// NANO-025 Phase 7: Tool activity tracking
export type ToolStatus = "pending" | "running" | "complete" | "error";

// NANO-028: Shutdown status tracking
export type ShutdownStatus = "idle" | "shutting_down" | "complete" | "error";

export interface ToolActivity {
  id: string; // tool_call_id
  tool_name: string;
  arguments: Record<string, unknown>;
  status: ToolStatus;
  iteration: number;
  timestamp: string;
  result_summary?: string;
  duration_ms?: number;
  success?: boolean;
}

interface AgentStoreState {
  // Current state
  state: AgentState;
  stateTransitions: StateTransition[];

  // Listening control
  isListeningPaused: boolean;

  // Live activity
  currentTranscription: string;
  currentResponse: string;
  isTranscriptionFinal: boolean;
  isResponseFinal: boolean;

  // NANO-042: Reasoning/thinking content from LLM
  currentReasoning: string;

  // NANO-037 Phase 2: Activated codex entries for current response
  activatedCodexEntries: ActivatedCodexEntry[];

  // NANO-044: Retrieved memories for current response
  retrievedMemories: RetrievedMemory[];

  // NANO-056: Stimulus source for current response
  currentStimulusSource: string;

  // NANO-073c: Whether a stimulus is currently in-flight (fired but not yet responded)
  stimulusActive: boolean;

  // NANO-069: Real-time audio output level (0.0-1.0) for portrait
  audioLevel: number;

  // NANO-073b: Real-time mic input level (0.0-1.0) for voice overlay
  micLevel: number;

  // Token usage
  tokenUsage: TokenUsageEvent | null;

  // Health
  health: HealthStatusEvent | null;

  // Config
  config: ConfigLoadedEvent | null;

  // Tool activity (NANO-025 Phase 7)
  toolActivities: ToolActivity[];

  // Shutdown state (NANO-028)
  shutdownStatus: ShutdownStatus;
  shutdownMessage: string | null;

  // Actions
  setState: (state: AgentState) => void;
  addTransition: (event: StateChangedEvent) => void;
  setListeningPaused: (paused: boolean) => void;
  setTranscription: (text: string, isFinal: boolean) => void;
  setResponse: (text: string, isFinal: boolean, codexEntries?: ActivatedCodexEntry[], reasoning?: string, memories?: RetrievedMemory[], stimulusSource?: string) => void;
  appendResponse: (chunk: string) => void;
  clearTranscription: () => void;
  clearResponse: () => void;
  setTokenUsage: (usage: TokenUsageEvent) => void;
  setHealth: (health: HealthStatusEvent | null) => void;
  setConfig: (config: ConfigLoadedEvent | null) => void;
  addToolInvoked: (event: ToolInvokedEvent) => void;
  updateToolResult: (event: ToolResultEvent) => void;
  clearToolActivities: () => void;
  // NANO-073c: Stimulus active flag
  setStimulusActive: (active: boolean) => void;
  // NANO-069: Audio level
  setAudioLevel: (level: number) => void;
  setMicLevel: (level: number) => void;
  // NANO-028: Shutdown actions
  setShutdownStatus: (status: ShutdownStatus) => void;
  setShutdownMessage: (message: string | null) => void;
  setShutdownProgress: (message: string) => void;
  setShutdownComplete: () => void;
  setShutdownError: (error: string) => void;
  resetShutdown: () => void;
}

let transitionCounter = 0;

export const useAgentStore = create<AgentStoreState>((set) => ({
  state: "idle",
  stateTransitions: [],

  isListeningPaused: false,

  currentTranscription: "",
  currentResponse: "",
  isTranscriptionFinal: false,
  isResponseFinal: false,
  currentReasoning: "",
  activatedCodexEntries: [],
  retrievedMemories: [],
  currentStimulusSource: "",
  stimulusActive: false,
  audioLevel: 0,
  micLevel: 0,

  tokenUsage: null,
  health: null,
  config: null,
  toolActivities: [],
  shutdownStatus: "idle",
  shutdownMessage: null,

  setState: (state) => set({ state }),

  addTransition: (event) =>
    set((s) => ({
      state: event.to,
      stateTransitions: [
        { ...event, id: ++transitionCounter },
        ...s.stateTransitions.slice(0, 19), // Keep last 20
      ],
    })),

  setListeningPaused: (paused) => set({ isListeningPaused: paused }),

  setTranscription: (text, isFinal) =>
    set({ currentTranscription: text, isTranscriptionFinal: isFinal }),

  // NANO-037: codex entries, NANO-042: reasoning, NANO-044: memories, NANO-056: stimulus source
  setResponse: (text, isFinal, codexEntries, reasoning, memories, stimulusSource) =>
    set({
      currentResponse: text,
      isResponseFinal: isFinal,
      activatedCodexEntries: codexEntries || [],
      currentReasoning: reasoning || "",
      retrievedMemories: memories || [],
      currentStimulusSource: stimulusSource || "",
    }),

  appendResponse: (chunk) =>
    set((s) => ({ currentResponse: s.currentResponse + chunk })),

  clearTranscription: () =>
    set({ currentTranscription: "", isTranscriptionFinal: false }),

  clearResponse: () =>
    set({ currentResponse: "", isResponseFinal: false, activatedCodexEntries: [], currentReasoning: "", retrievedMemories: [], currentStimulusSource: "", stimulusActive: false }),

  setTokenUsage: (usage) => set({ tokenUsage: usage }),

  setHealth: (health) => set({ health }),

  setConfig: (config) => set({ config }),

  // NANO-025 Phase 7: Tool activity actions
  addToolInvoked: (event) =>
    set((s) => ({
      toolActivities: [
        ...s.toolActivities,
        {
          id: event.tool_call_id,
          tool_name: event.tool_name,
          arguments: event.arguments,
          status: "running" as ToolStatus,
          iteration: event.iteration,
          timestamp: event.timestamp,
        },
      ],
    })),

  updateToolResult: (event) =>
    set((s) => ({
      toolActivities: s.toolActivities.map((activity) =>
        activity.id === event.tool_call_id
          ? {
              ...activity,
              status: (event.success ? "complete" : "error") as ToolStatus,
              result_summary: event.result_summary,
              duration_ms: event.duration_ms,
              success: event.success,
            }
          : activity
      ),
    })),

  clearToolActivities: () => set({ toolActivities: [] }),

  // NANO-073c: Stimulus active flag
  setStimulusActive: (active) => set({ stimulusActive: active }),

  // NANO-069: Audio level
  setAudioLevel: (level) => set({ audioLevel: level }),

  // NANO-073b: Mic input level
  setMicLevel: (level) => set({ micLevel: level }),

  // NANO-028: Shutdown actions
  setShutdownStatus: (status) => set({ shutdownStatus: status }),
  setShutdownMessage: (message) => set({ shutdownMessage: message }),
  setShutdownProgress: (message) =>
    set({ shutdownStatus: "shutting_down", shutdownMessage: message }),
  setShutdownComplete: () =>
    set({
      shutdownStatus: "complete",
      shutdownMessage: "Shutdown complete",
      // Clear stale data since orchestrator is gone
      health: null,
      config: null,
      state: "idle",
      toolActivities: [],
    }),
  setShutdownError: (error) =>
    set({ shutdownStatus: "error", shutdownMessage: error }),
  resetShutdown: () =>
    set({ shutdownStatus: "idle", shutdownMessage: null }),
}));
