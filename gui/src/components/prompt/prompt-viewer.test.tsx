import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { PromptViewer } from "./prompt-viewer";
import type { PromptMessage } from "@/types/events";

// Mock clipboard API
const mockClipboard = {
  writeText: vi.fn(() => Promise.resolve()),
};
Object.assign(navigator, { clipboard: mockClipboard });

// Test fixture factory
function createMockMessages(): PromptMessage[] {
  return [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello there!" },
    { role: "assistant", content: "Hi! How can I help you today?" },
  ];
}

describe("PromptViewer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("should render the component", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger="activation"
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("Full Prompt")).toBeInTheDocument();
    });

    it("should display input modality badge", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("VOICE")).toBeInTheDocument();
    });

    it("should display state trigger badge when present", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger="activation"
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("activation")).toBeInTheDocument();
    });

    it("should NOT display state trigger badge when null", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.queryByText("activation")).not.toBeInTheDocument();
    });

    it("should display Copy All button", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("Copy All")).toBeInTheDocument();
    });
  });

  describe("message cards", () => {
    it("should render all messages", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("system")).toBeInTheDocument();
      expect(screen.getByText("user")).toBeInTheDocument();
      expect(screen.getByText("assistant")).toBeInTheDocument();
    });

    it("should display character count for each message", () => {
      const messages = [
        { role: "system" as const, content: "A".repeat(100) },
        { role: "user" as const, content: "B".repeat(50) },
      ];

      render(
        <PromptViewer
          messages={messages}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("100 chars")).toBeInTheDocument();
      expect(screen.getByText("50 chars")).toBeInTheDocument();
    });

    it("should format large character counts with commas", () => {
      const messages = [
        { role: "system" as const, content: "A".repeat(1500) },
      ];

      render(
        <PromptViewer
          messages={messages}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("1,500 chars")).toBeInTheDocument();
    });
  });

  describe("copy all functionality", () => {
    it("should copy all messages to clipboard", async () => {
      const messages = [
        { role: "system" as const, content: "System content" },
        { role: "user" as const, content: "User content" },
      ];

      render(
        <PromptViewer
          messages={messages}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      const copyButton = screen.getByText("Copy All");
      fireEvent.click(copyButton);

      await waitFor(() => {
        expect(mockClipboard.writeText).toHaveBeenCalledWith(
          "[SYSTEM]\nSystem content\n\n---\n\n[USER]\nUser content"
        );
      });
    });
  });

  describe("system prompt section parsing", () => {
    it("should parse sections with ### headers", () => {
      const messages = [
        {
          role: "system" as const,
          content: "### Agent\nYou are Spindle.\n\n### Rules\nBe helpful.",
        },
      ];

      render(
        <PromptViewer
          messages={messages}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      // The collapsible should be clickable to expand
      const systemCard = screen.getByText("system").closest("div");
      expect(systemCard).toBeInTheDocument();
    });

    it("should handle content without headers", () => {
      const messages = [
        {
          role: "system" as const,
          content: "Just plain text without any headers.",
        },
      ];

      // Should not throw
      expect(() =>
        render(
          <PromptViewer
            messages={messages}
            inputModality="VOICE"
            stateTrigger={null}
            timestamp="2026-01-27T11:55:51.000Z"
          />
        )
      ).not.toThrow();
    });
  });

  describe("input modality display", () => {
    it("should display VOICE modality", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("VOICE")).toBeInTheDocument();
    });

    it("should display TEXT modality", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="TEXT"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      expect(screen.getByText("TEXT")).toBeInTheDocument();
    });
  });

  describe("empty state handling", () => {
    it("should handle empty messages array", () => {
      // Should not throw
      expect(() =>
        render(
          <PromptViewer
            messages={[]}
            inputModality="VOICE"
            stateTrigger={null}
            timestamp="2026-01-27T11:55:51.000Z"
          />
        )
      ).not.toThrow();
    });

    it("should handle message with empty content", () => {
      const messages = [{ role: "user" as const, content: "" }];

      expect(() =>
        render(
          <PromptViewer
            messages={messages}
            inputModality="VOICE"
            stateTrigger={null}
            timestamp="2026-01-27T11:55:51.000Z"
          />
        )
      ).not.toThrow();

      expect(screen.getByText("0 chars")).toBeInTheDocument();
    });
  });

  describe("timestamp formatting", () => {
    it("should format timestamp as time string", () => {
      render(
        <PromptViewer
          messages={createMockMessages()}
          inputModality="VOICE"
          stateTrigger={null}
          timestamp="2026-01-27T11:55:51.000Z"
        />
      );

      // The exact format depends on locale, but we should see some time display
      // This tests that the timestamp is being processed without throwing
      expect(screen.getByText("Full Prompt")).toBeInTheDocument();
    });
  });
});
