import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TokenBreakdown } from "./token-breakdown";
import type { TokenBreakdown as TokenBreakdownType } from "@/types/events";

// Test fixture factory
function createMockBreakdown(overrides?: Partial<TokenBreakdownType>): TokenBreakdownType {
  return {
    total: 2101,
    prompt: 2074,
    completion: 27,
    system: 2022,
    user: 52,
    sections: {
      agent: 898,
      context: 145,
      rules: 910,
      conversation: 75,
    },
    ...overrides,
  };
}

describe("TokenBreakdown", () => {
  describe("rendering", () => {
    it("should render token breakdown card", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("Token Breakdown")).toBeInTheDocument();
    });

    it("should display total tokens", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown({ total: 2101 })} />);

      expect(screen.getByText("2,101")).toBeInTheDocument();
      expect(screen.getByText("Total")).toBeInTheDocument();
    });

    it("should display prompt tokens", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown({ prompt: 2074 })} />);

      expect(screen.getByText("2,074")).toBeInTheDocument();
      expect(screen.getByText("Prompt")).toBeInTheDocument();
    });

    it("should display completion tokens", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown({ completion: 27 })} />);

      expect(screen.getByText("27")).toBeInTheDocument();
      expect(screen.getByText("Completion")).toBeInTheDocument();
    });
  });

  describe("prompt distribution section", () => {
    it("should display system prompt bar", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("System Prompt")).toBeInTheDocument();
    });

    it("should display user input bar", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("User Input")).toBeInTheDocument();
    });

    it("should show section header", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("Prompt Distribution")).toBeInTheDocument();
    });
  });

  describe("system prompt sections", () => {
    it("should display agent section", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("Agent (Name, Appearance, Personality)")).toBeInTheDocument();
    });

    it("should display context section", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("Context (Modality, State)")).toBeInTheDocument();
    });

    it("should display rules section", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("Rules")).toBeInTheDocument();
    });

    it("should display conversation section", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("Conversation (Summary, History)")).toBeInTheDocument();
    });

    it("should show section header", () => {
      render(<TokenBreakdown breakdown={createMockBreakdown()} />);

      expect(screen.getByText("System Prompt Sections")).toBeInTheDocument();
    });
  });

  describe("percentage calculations", () => {
    it("should calculate and display percentages", () => {
      const breakdown = createMockBreakdown({
        prompt: 1000,
        system: 800,
        user: 200,
      });
      render(<TokenBreakdown breakdown={breakdown} />);

      // System is 80% of prompt
      expect(screen.getByText(/80\.0%/)).toBeInTheDocument();
      // User is 20% of prompt
      expect(screen.getByText(/20\.0%/)).toBeInTheDocument();
    });

    it("should handle zero total gracefully", () => {
      const breakdown = createMockBreakdown({
        total: 0,
        prompt: 0,
        completion: 0,
        system: 0,
        user: 0,
        sections: { agent: 0, context: 0, rules: 0, conversation: 0 },
      });

      // Should not throw
      expect(() => render(<TokenBreakdown breakdown={breakdown} />)).not.toThrow();
    });
  });

  describe("number formatting", () => {
    it("should format large numbers with commas", () => {
      const breakdown = createMockBreakdown({
        total: 123456,
        prompt: 100000,
        completion: 23456,
      });
      render(<TokenBreakdown breakdown={breakdown} />);

      expect(screen.getByText("123,456")).toBeInTheDocument();
      expect(screen.getByText("100,000")).toBeInTheDocument();
      expect(screen.getByText("23,456")).toBeInTheDocument();
    });

    it("should handle small numbers without commas", () => {
      const breakdown = createMockBreakdown({
        total: 100,
        prompt: 80,
        completion: 20,
      });
      render(<TokenBreakdown breakdown={breakdown} />);

      expect(screen.getByText("100")).toBeInTheDocument();
      expect(screen.getByText("80")).toBeInTheDocument();
      expect(screen.getByText("20")).toBeInTheDocument();
    });
  });
});
