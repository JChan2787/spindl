import { create } from "zustand";
import type { ActivatedCodexEntry, RetrievedMemory } from "@/types/events";

export interface SentenceChunk {
  text: string;
  emotion?: string;
  emotionConfidence?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: string;
  isFinal: boolean;
  // Assistant-only metadata
  reasoning?: string;
  activatedCodexEntries?: ActivatedCodexEntry[];
  retrievedMemories?: RetrievedMemory[];
  stimulusSource?: string;
  // Emotion classifier metadata (NANO-094) — display-only, never sent to LLM
  emotion?: string;
  emotionConfidence?: number;
  // Per-sentence sub-bubbles (NANO-111 Session 606) — streaming responses only
  chunks?: SentenceChunk[];
}

interface ChatStoreState {
  messages: ChatMessage[];

  // Actions
  addUserMessage: (text: string) => void;
  addAssistantMessage: (msg: Partial<ChatMessage>) => string;
  updateAssistantMessage: (id: string, update: Partial<ChatMessage>) => void;
  hydrateHistory: (turns: ChatMessage[]) => void;
  clearHistory: () => void;
}

let messageCounter = 0;

function generateId(): string {
  return `msg-${Date.now()}-${++messageCounter}`;
}

export const useChatStore = create<ChatStoreState>((set) => ({
  messages: [],

  addUserMessage: (text) => {
    if (!text.trim()) return;
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id: generateId(),
          role: "user",
          text,
          timestamp: new Date().toISOString(),
          isFinal: true,
        },
      ],
    }));
  },

  addAssistantMessage: (msg) => {
    const id = generateId();
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id,
          role: "assistant",
          text: msg.text || "",
          timestamp: msg.timestamp || new Date().toISOString(),
          isFinal: msg.isFinal ?? false,
          reasoning: msg.reasoning,
          activatedCodexEntries: msg.activatedCodexEntries,
          retrievedMemories: msg.retrievedMemories,
          stimulusSource: msg.stimulusSource,
          emotion: msg.emotion,
          emotionConfidence: msg.emotionConfidence,
          chunks: msg.chunks,
        },
      ],
    }));
    return id;
  },

  updateAssistantMessage: (id, update) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, ...update } : m
      ),
    })),

  hydrateHistory: (turns) =>
    set({ messages: turns }),

  clearHistory: () =>
    set({ messages: [] }),
}));
