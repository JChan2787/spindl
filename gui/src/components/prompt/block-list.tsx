"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getBlockBorderColor } from "@/lib/constants/block-colors";
import type { BlockInfo, BlockTokenData } from "@/types/events";
import { Layers, GripVertical } from "lucide-react";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
  arrayMove,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

interface BlockListProps {
  /** Full block metadata from config (already ordered) */
  blocks: BlockInfo[];
  /** Per-block token data from latest snapshot (null if no snapshot) */
  tokenData: BlockTokenData[] | null;
  /** List of currently disabled block IDs (effective: pending or server) */
  disabledBlocks: string[];
  /** Currently selected block ID */
  selectedBlockId: string | null;
  /** Whether edit mode is active */
  editMode: boolean;
  /** Pending overrides record (to show override indicator) */
  pendingOverrides: Record<string, string | null>;
  /** Callback when a block is clicked */
  onSelectBlock: (blockId: string) => void;
  /** Callback when a block's enable/disable toggle changes */
  onToggleBlock: (blockId: string) => void;
  /** Callback when blocks are reordered via drag-and-drop */
  onReorder?: (orderedIds: string[]) => void;
}

// --- Sortable block row (extracted for useSortable hook) ---

interface SortableBlockRowProps {
  block: BlockInfo;
  isDisabled: boolean;
  isSelected: boolean;
  tokens: number | undefined;
  hasOverride: boolean;
  editMode: boolean;
  isDraggable: boolean;
  onSelect: () => void;
  onToggle: () => void;
}

function SortableBlockRow({
  block,
  isDisabled,
  isSelected,
  tokens,
  hasOverride,
  editMode,
  isDraggable,
  onSelect,
  onToggle,
}: SortableBlockRowProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: block.id, disabled: !isDraggable });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 10 : undefined,
    opacity: isDragging ? 0.8 : undefined,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`
        flex items-center gap-3 px-3 py-2.5 rounded-md border-l-4 transition-colors
        ${getBlockBorderColor(block.id)}
        ${isSelected ? "bg-accent" : "hover:bg-accent/30"}
        ${isDisabled ? "opacity-50" : ""}
        ${editMode ? "cursor-pointer" : "cursor-default"}
        ${isDragging ? "shadow-lg ring-1 ring-primary/20" : ""}
      `}
      onClick={() => editMode && onSelect()}
    >
      {/* Drag handle (edit mode + draggable only) */}
      {editMode && isDraggable && (
        <button
          className="shrink-0 cursor-grab active:cursor-grabbing text-muted-foreground hover:text-foreground touch-none"
          aria-label="Drag to reorder"
          {...attributes}
          {...listeners}
          onClick={(e) => e.stopPropagation()}
        >
          <GripVertical className="h-4 w-4" />
        </button>
      )}

      {/* Block info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium truncate">
            {block.label}
          </span>
          {hasOverride && (
            <div
              className="h-2 w-2 rounded-full bg-purple-500 shrink-0"
              title="Has override"
            />
          )}
        </div>
        {block.section_header && (
          <span className="text-xs text-muted-foreground">
            {block.section_header}
          </span>
        )}
      </div>

      {/* Token count */}
      <Badge
        variant="secondary"
        className="text-xs font-mono shrink-0"
      >
        {isDisabled
          ? "\u2014"
          : tokens !== undefined
            ? tokens.toLocaleString()
            : "\u2014"}
      </Badge>

      {/* Toggle (edit mode only) */}
      {editMode && (
        <Switch
          checked={!isDisabled}
          onCheckedChange={() => onToggle()}
          onClick={(e) => e.stopPropagation()}
          className="shrink-0"
        />
      )}
    </div>
  );
}

// --- Main BlockList component ---

export function BlockList({
  blocks,
  tokenData,
  disabledBlocks,
  selectedBlockId,
  editMode,
  pendingOverrides,
  onSelectBlock,
  onToggleBlock,
  onReorder,
}: BlockListProps) {
  // Build a lookup for token data by block ID
  const tokenMap = new Map<string, number>();
  if (tokenData) {
    for (const td of tokenData) {
      tokenMap.set(td.id, td.tokens);
    }
  }

  const blockIds = blocks.map((b) => b.id);
  const isDraggable = editMode && !!onReorder;

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor),
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const oldIndex = blockIds.indexOf(active.id as string);
      const newIndex = blockIds.indexOf(over.id as string);
      const newOrder = arrayMove(blockIds, oldIndex, newIndex);
      onReorder?.(newOrder);
    }
  };

  const blockRows = blocks.map((block) => {
    const isDisabled = disabledBlocks.includes(block.id);
    const isSelected = selectedBlockId === block.id;
    const tokens = tokenMap.get(block.id);
    const hasOverride =
      block.has_override || pendingOverrides[block.id] !== undefined;

    return (
      <SortableBlockRow
        key={block.id}
        block={block}
        isDisabled={isDisabled}
        isSelected={isSelected}
        tokens={tokens}
        hasOverride={hasOverride}
        editMode={editMode}
        isDraggable={isDraggable}
        onSelect={() => onSelectBlock(block.id)}
        onToggle={() => onToggleBlock(block.id)}
      />
    );
  });

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Layers className="h-4 w-4" />
          Prompt Blocks
        </CardTitle>
        <Badge variant="outline" className="text-xs">
          {blocks.filter((b) => !disabledBlocks.includes(b.id)).length}/{blocks.length} active
        </Badge>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[calc(100vh-220px)]">
          <div className="space-y-0.5 p-3">
            <DndContext
              sensors={isDraggable ? sensors : undefined}
              collisionDetection={isDraggable ? closestCenter : undefined}
              onDragEnd={handleDragEnd}
            >
              <SortableContext
                items={blockIds}
                strategy={verticalListSortingStrategy}
              >
                {blockRows}
              </SortableContext>
            </DndContext>
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
