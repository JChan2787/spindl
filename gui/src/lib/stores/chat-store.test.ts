import { describe, it, expect, beforeEach } from "vitest";
import { useChatStore } from "./chat-store";

describe("useChatStore", () => {
  beforeEach(() => {
    useChatStore.setState({ messages: [] });
  });

  describe("addUserMessage", () => {
    it("should add a user message", () => {
      useChatStore.getState().addUserMessage("Hello world");

      const { messages } = useChatStore.getState();
      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe("user");
      expect(messages[0].text).toBe("Hello world");
      expect(messages[0].isFinal).toBe(true);
      expect(messages[0].id).toBeTruthy();
      expect(messages[0].timestamp).toBeTruthy();
    });

    it("should not add empty messages", () => {
      useChatStore.getState().addUserMessage("");
      useChatStore.getState().addUserMessage("   ");

      expect(useChatStore.getState().messages).toHaveLength(0);
    });

    it("should maintain message order", () => {
      useChatStore.getState().addUserMessage("First");
      useChatStore.getState().addUserMessage("Second");

      const { messages } = useChatStore.getState();
      expect(messages).toHaveLength(2);
      expect(messages[0].text).toBe("First");
      expect(messages[1].text).toBe("Second");
    });
  });

  describe("addAssistantMessage", () => {
    it("should add an assistant message and return its ID", () => {
      const id = useChatStore.getState().addAssistantMessage({
        text: "Hi there!",
        isFinal: false,
      });

      expect(id).toBeTruthy();
      const { messages } = useChatStore.getState();
      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe("assistant");
      expect(messages[0].text).toBe("Hi there!");
      expect(messages[0].isFinal).toBe(false);
    });

    it("should include metadata fields", () => {
      useChatStore.getState().addAssistantMessage({
        text: "Response",
        isFinal: true,
        reasoning: "Let me think...",
        stimulusSource: "patience",
        activatedCodexEntries: [{ name: "test", keys: ["key"], activation_method: "keyword" }],
        retrievedMemories: [{ content_preview: "mem", collection: "general", distance: 0.5 }],
      });

      const msg = useChatStore.getState().messages[0];
      expect(msg.reasoning).toBe("Let me think...");
      expect(msg.stimulusSource).toBe("patience");
      expect(msg.activatedCodexEntries).toHaveLength(1);
      expect(msg.retrievedMemories).toHaveLength(1);
    });
  });

  describe("updateAssistantMessage", () => {
    it("should update an existing message by ID", () => {
      const id = useChatStore.getState().addAssistantMessage({
        text: "Partial...",
        isFinal: false,
      });

      useChatStore.getState().updateAssistantMessage(id, {
        text: "Full response here.",
        isFinal: true,
      });

      const msg = useChatStore.getState().messages[0];
      expect(msg.text).toBe("Full response here.");
      expect(msg.isFinal).toBe(true);
    });

    it("should not affect other messages", () => {
      useChatStore.getState().addUserMessage("User says hi");
      const id = useChatStore.getState().addAssistantMessage({
        text: "Streaming...",
        isFinal: false,
      });

      useChatStore.getState().updateAssistantMessage(id, { text: "Done." });

      const { messages } = useChatStore.getState();
      expect(messages[0].text).toBe("User says hi");
      expect(messages[1].text).toBe("Done.");
    });
  });

  describe("hydrateHistory", () => {
    it("should replace all messages with hydrated turns", () => {
      useChatStore.getState().addUserMessage("Existing message");

      useChatStore.getState().hydrateHistory([
        { id: "h-1", role: "user", text: "Old user msg", timestamp: "2026-01-01T00:00:00Z", isFinal: true },
        { id: "h-2", role: "assistant", text: "Old assistant msg", timestamp: "2026-01-01T00:00:01Z", isFinal: true },
      ]);

      const { messages } = useChatStore.getState();
      expect(messages).toHaveLength(2);
      expect(messages[0].text).toBe("Old user msg");
      expect(messages[1].text).toBe("Old assistant msg");
    });
  });

  describe("clearHistory", () => {
    it("should clear all messages", () => {
      useChatStore.getState().addUserMessage("A");
      useChatStore.getState().addUserMessage("B");
      expect(useChatStore.getState().messages).toHaveLength(2);

      useChatStore.getState().clearHistory();
      expect(useChatStore.getState().messages).toHaveLength(0);
    });
  });

  describe("interleaved conversation", () => {
    it("should maintain correct order in a multi-turn exchange", () => {
      const store = useChatStore.getState();
      store.addUserMessage("Hello");
      const id1 = store.addAssistantMessage({ text: "Hi!", isFinal: true });
      store.addUserMessage("How are you?");
      const id2 = store.addAssistantMessage({ text: "I'm well.", isFinal: true });

      const { messages } = useChatStore.getState();
      expect(messages).toHaveLength(4);
      expect(messages.map(m => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
      expect(messages.map(m => m.text)).toEqual(["Hello", "Hi!", "How are you?", "I'm well."]);
    });
  });
});
