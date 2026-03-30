import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { BlockDetail } from "./block-detail";
import type { BlockInfo } from "@/types/events";

const mockProviderBlock: BlockInfo = {
  id: "persona_name",
  label: "Agent Name",
  order: 0,
  enabled: true,
  is_static: false,
  section_header: "Agent",
  has_override: false,
  content_wrapper: "You are {content}.",
};

const mockStaticBlock: BlockInfo = {
  id: "closing_instruction",
  label: "Closing Instruction",
  order: 11,
  enabled: true,
  is_static: true,
  section_header: "Input",
  has_override: false,
  content_wrapper: null,
};

const mockDeferredBlock: BlockInfo = {
  id: "rag_context",
  label: "Memories",
  order: 6,
  enabled: true,
  is_static: false,
  section_header: null,
  has_override: false,
  content_wrapper: null,
};

describe("BlockDetail", () => {
  const defaultProps = {
    block: mockProviderBlock,
    currentOverride: null as string | null,
    tokenCount: 24,
    currentContent: "### Agent\nYou are TestAgent." as string | null,
    onSetOverride: vi.fn(),
  };

  describe("rendering", () => {
    it("should render block label", () => {
      render(<BlockDetail {...defaultProps} />);
      expect(screen.getByText("Agent Name")).toBeInTheDocument();
    });

    it("should render section header badge", () => {
      render(<BlockDetail {...defaultProps} />);
      expect(screen.getByText("Agent")).toBeInTheDocument();
    });

    it("should render Provider badge for provider blocks", () => {
      render(<BlockDetail {...defaultProps} />);
      expect(screen.getByText("Provider")).toBeInTheDocument();
    });

    it("should render Static badge for static blocks", () => {
      render(<BlockDetail {...defaultProps} block={mockStaticBlock} />);
      expect(screen.getByText("Static")).toBeInTheDocument();
    });

    it("should render Deferred badge for deferred blocks", () => {
      render(<BlockDetail {...defaultProps} block={mockDeferredBlock} />);
      expect(screen.getByText("Deferred")).toBeInTheDocument();
    });

    it("should display token count", () => {
      render(<BlockDetail {...defaultProps} tokenCount={156} />);
      expect(screen.getByText("156 tokens in latest snapshot")).toBeInTheDocument();
    });

    it("should not display token count when null", () => {
      render(<BlockDetail {...defaultProps} tokenCount={null} />);
      expect(screen.queryByText(/tokens in latest snapshot/)).not.toBeInTheDocument();
    });

    it("should show content wrapper when present", () => {
      render(<BlockDetail {...defaultProps} />);
      expect(screen.getByText("You are {content}.")).toBeInTheDocument();
    });

    it("should display current content from snapshot", () => {
      render(<BlockDetail {...defaultProps} currentContent="You are TestAgent." />);
      expect(screen.getByText("Current Content")).toBeInTheDocument();
      expect(screen.getByText("You are TestAgent.")).toBeInTheDocument();
    });

    it("should not display current content when null", () => {
      render(<BlockDetail {...defaultProps} currentContent={null} />);
      expect(screen.queryByText("Current Content")).not.toBeInTheDocument();
    });

    it("should not display current content when empty", () => {
      render(<BlockDetail {...defaultProps} currentContent="" />);
      expect(screen.queryByText("Current Content")).not.toBeInTheDocument();
    });
  });

  describe("override editor", () => {
    it("should show override switch", () => {
      render(<BlockDetail {...defaultProps} />);
      expect(screen.getByText("User Override")).toBeInTheDocument();
      expect(screen.getByRole("switch")).toBeInTheDocument();
    });

    it("should not show textarea when override is off", () => {
      render(<BlockDetail {...defaultProps} />);
      expect(screen.queryByPlaceholderText(/override content/)).not.toBeInTheDocument();
    });

    it("should show textarea when override switch is turned on", () => {
      render(<BlockDetail {...defaultProps} />);
      fireEvent.click(screen.getByRole("switch"));
      expect(screen.getByPlaceholderText("Enter override content for this block...")).toBeInTheDocument();
    });

    it("should pre-populate textarea when override exists", () => {
      render(
        <BlockDetail {...defaultProps} currentOverride="Custom persona name" />
      );
      const textarea = screen.getByPlaceholderText("Enter override content for this block...");
      expect(textarea).toHaveValue("Custom persona name");
    });

    it("should show character count", () => {
      render(
        <BlockDetail {...defaultProps} currentOverride="Hello" />
      );
      expect(screen.getByText("5 characters")).toBeInTheDocument();
    });

    it("should update character count on input", () => {
      render(
        <BlockDetail {...defaultProps} currentOverride="Hi" />
      );
      const textarea = screen.getByPlaceholderText("Enter override content for this block...");
      fireEvent.change(textarea, { target: { value: "Hello World" } });
      expect(screen.getByText("11 characters")).toBeInTheDocument();
    });

    it("should call onSetOverride with content on Apply", () => {
      const onSetOverride = vi.fn();
      render(
        <BlockDetail
          {...defaultProps}
          currentOverride="Old"
          onSetOverride={onSetOverride}
        />
      );
      const textarea = screen.getByPlaceholderText("Enter override content for this block...");
      fireEvent.change(textarea, { target: { value: "New content" } });
      fireEvent.click(screen.getByText("Apply"));
      expect(onSetOverride).toHaveBeenCalledWith("persona_name", "New content");
    });

    it("should call onSetOverride with null when clearing override", () => {
      const onSetOverride = vi.fn();
      render(
        <BlockDetail
          {...defaultProps}
          currentOverride="Some override"
          onSetOverride={onSetOverride}
        />
      );
      // Toggle override off
      fireEvent.click(screen.getByRole("switch"));
      expect(onSetOverride).toHaveBeenCalledWith("persona_name", null);
    });

    it("should disable Apply when no changes", () => {
      render(
        <BlockDetail {...defaultProps} currentOverride="Existing" />
      );
      const applyBtn = screen.getByText("Apply");
      expect(applyBtn).toBeDisabled();
    });

    it("should disable Revert when no changes", () => {
      render(
        <BlockDetail {...defaultProps} currentOverride="Existing" />
      );
      const revertBtn = screen.getByText("Revert");
      expect(revertBtn).toBeDisabled();
    });

    it("should revert textarea to original on Revert click", () => {
      render(
        <BlockDetail {...defaultProps} currentOverride="Original" />
      );
      const textarea = screen.getByPlaceholderText("Enter override content for this block...");
      fireEvent.change(textarea, { target: { value: "Modified" } });
      fireEvent.click(screen.getByText("Revert"));
      expect(textarea).toHaveValue("Original");
    });
  });

  describe("deferred block override suppression", () => {
    const deferredProps = {
      ...defaultProps,
      block: mockDeferredBlock,
      currentOverride: null as string | null,
    };

    it("should disable the override switch for deferred blocks", () => {
      render(<BlockDetail {...deferredProps} />);
      const toggle = screen.getByRole("switch");
      expect(toggle).toBeDisabled();
    });

    it("should show explanatory text for deferred blocks", () => {
      render(<BlockDetail {...deferredProps} />);
      expect(
        screen.getByText("Deferred blocks are populated at runtime and cannot be overridden.")
      ).toBeInTheDocument();
    });

    it("should not show textarea for deferred blocks even when toggled", () => {
      render(<BlockDetail {...deferredProps} />);
      // Try clicking the disabled switch — should not open textarea
      fireEvent.click(screen.getByRole("switch"));
      expect(
        screen.queryByPlaceholderText("Enter override content for this block...")
      ).not.toBeInTheDocument();
    });

    it("should not disable the override switch for non-deferred blocks", () => {
      render(<BlockDetail {...defaultProps} />);
      const toggle = screen.getByRole("switch");
      expect(toggle).not.toBeDisabled();
    });
  });
});
