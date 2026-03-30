import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { BlockList } from "./block-list";
import type { BlockInfo, BlockTokenData } from "@/types/events";

const mockBlocks: BlockInfo[] = [
  {
    id: "persona_name",
    label: "Agent Name",
    order: 0,
    enabled: true,
    is_static: false,
    section_header: "Agent",
    has_override: false,
    content_wrapper: "You are {content}.",
  },
  {
    id: "persona_appearance",
    label: "Appearance",
    order: 1,
    enabled: true,
    is_static: false,
    section_header: null,
    has_override: true,
    content_wrapper: null,
  },
  {
    id: "codex_context",
    label: "Codex",
    order: 5,
    enabled: true,
    is_static: false,
    section_header: null,
    has_override: false,
    content_wrapper: null,
  },
  {
    id: "closing_instruction",
    label: "Closing Instruction",
    order: 11,
    enabled: true,
    is_static: true,
    section_header: "Input",
    has_override: false,
    content_wrapper: null,
  },
];

const mockTokenData: BlockTokenData[] = [
  { id: "persona_name", label: "Agent Name", section: "Agent", tokens: 24 },
  { id: "persona_appearance", label: "Appearance", section: null, tokens: 156 },
  { id: "codex_context", label: "Codex", section: null, tokens: 0 },
  { id: "closing_instruction", label: "Closing Instruction", section: "Input", tokens: 8 },
];

describe("BlockList", () => {
  const defaultProps = {
    blocks: mockBlocks,
    tokenData: mockTokenData,
    disabledBlocks: [] as string[],
    selectedBlockId: null as string | null,
    editMode: false,
    pendingOverrides: {} as Record<string, string | null>,
    onSelectBlock: vi.fn(),
    onToggleBlock: vi.fn(),
  };

  describe("rendering", () => {
    it("should render all block labels", () => {
      render(<BlockList {...defaultProps} />);
      expect(screen.getByText("Agent Name")).toBeInTheDocument();
      expect(screen.getByText("Appearance")).toBeInTheDocument();
      expect(screen.getByText("Codex")).toBeInTheDocument();
      expect(screen.getByText("Closing Instruction")).toBeInTheDocument();
    });

    it("should display token counts from snapshot", () => {
      render(<BlockList {...defaultProps} />);
      expect(screen.getByText("24")).toBeInTheDocument();
      expect(screen.getByText("156")).toBeInTheDocument();
      expect(screen.getByText("8")).toBeInTheDocument();
    });

    it("should show section headers", () => {
      render(<BlockList {...defaultProps} />);
      expect(screen.getByText("Agent")).toBeInTheDocument();
      expect(screen.getByText("Input")).toBeInTheDocument();
    });

    it("should show active count badge", () => {
      render(<BlockList {...defaultProps} />);
      expect(screen.getByText("4/4 active")).toBeInTheDocument();
    });

    it("should show override indicator for blocks with overrides", () => {
      render(<BlockList {...defaultProps} />);
      const dots = screen.getAllByTitle("Has override");
      expect(dots).toHaveLength(1); // persona_appearance has has_override: true
    });

    it("should show dash for tokens when no snapshot data", () => {
      render(<BlockList {...defaultProps} tokenData={null} />);
      const dashes = screen.getAllByText("\u2014");
      expect(dashes.length).toBeGreaterThanOrEqual(4);
    });
  });

  describe("disabled blocks", () => {
    it("should show dash for disabled block tokens", () => {
      render(
        <BlockList {...defaultProps} disabledBlocks={["codex_context"]} />
      );
      // Codex would normally show 0, but when disabled shows dash
      const dashes = screen.getAllByText("\u2014");
      expect(dashes.length).toBeGreaterThanOrEqual(1);
    });

    it("should show correct active count with disabled blocks", () => {
      render(
        <BlockList {...defaultProps} disabledBlocks={["codex_context"]} />
      );
      expect(screen.getByText("3/4 active")).toBeInTheDocument();
    });
  });

  describe("edit mode", () => {
    it("should not show toggle switches in view mode", () => {
      render(<BlockList {...defaultProps} editMode={false} />);
      const switches = screen.queryAllByRole("switch");
      expect(switches).toHaveLength(0);
    });

    it("should show toggle switches in edit mode", () => {
      render(<BlockList {...defaultProps} editMode={true} />);
      const switches = screen.getAllByRole("switch");
      expect(switches).toHaveLength(4);
    });

    it("should call onSelectBlock when clicking a block in edit mode", () => {
      const onSelectBlock = vi.fn();
      render(
        <BlockList {...defaultProps} editMode={true} onSelectBlock={onSelectBlock} />
      );
      fireEvent.click(screen.getByText("Agent Name"));
      expect(onSelectBlock).toHaveBeenCalledWith("persona_name");
    });

    it("should not call onSelectBlock when clicking in view mode", () => {
      const onSelectBlock = vi.fn();
      render(
        <BlockList {...defaultProps} editMode={false} onSelectBlock={onSelectBlock} />
      );
      fireEvent.click(screen.getByText("Agent Name"));
      expect(onSelectBlock).not.toHaveBeenCalled();
    });

    it("should call onToggleBlock when toggling a switch", () => {
      const onToggleBlock = vi.fn();
      render(
        <BlockList {...defaultProps} editMode={true} onToggleBlock={onToggleBlock} />
      );
      const switches = screen.getAllByRole("switch");
      fireEvent.click(switches[0]);
      expect(onToggleBlock).toHaveBeenCalledWith("persona_name");
    });
  });

  describe("pending overrides", () => {
    it("should show override indicator for pending overrides", () => {
      render(
        <BlockList
          {...defaultProps}
          pendingOverrides={{ codex_context: "custom content" }}
        />
      );
      const dots = screen.getAllByTitle("Has override");
      // persona_appearance (has_override: true) + codex_context (pending)
      expect(dots).toHaveLength(2);
    });
  });

  describe("drag handles", () => {
    it("should not show drag handles in view mode", () => {
      render(<BlockList {...defaultProps} editMode={false} onReorder={vi.fn()} />);
      const handles = screen.queryAllByLabelText("Drag to reorder");
      expect(handles).toHaveLength(0);
    });

    it("should show drag handles in edit mode with onReorder", () => {
      render(<BlockList {...defaultProps} editMode={true} onReorder={vi.fn()} />);
      const handles = screen.getAllByLabelText("Drag to reorder");
      expect(handles).toHaveLength(4);
    });

    it("should not show drag handles in edit mode without onReorder", () => {
      render(<BlockList {...defaultProps} editMode={true} />);
      const handles = screen.queryAllByLabelText("Drag to reorder");
      expect(handles).toHaveLength(0);
    });
  });
});
