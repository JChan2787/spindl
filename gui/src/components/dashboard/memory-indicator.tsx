"use client";

/**
 * MemoryIndicator - Shows when RAG memories were retrieved for a response.
 * NANO-044: GUI memory retrieval indicator.
 *
 * Features:
 * - Icon appears only when memories are retrieved
 * - Hover shows count tooltip
 * - Click expands to show memory details (preview, collection, distance)
 * - Scale animation on hover
 *
 * Mirrors CodexIndicator structure (NANO-037 Phase 2).
 */

import { useState } from "react";
import { Brain, ChevronDown, ChevronUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { RetrievedMemory } from "@/types/events";

interface MemoryIndicatorProps {
  memories: RetrievedMemory[];
  className?: string;
}

const collectionColors: Record<string, string> = {
  global: "bg-purple-500/20 text-purple-400",
  general: "bg-emerald-500/20 text-emerald-400",
  flashcards: "bg-amber-500/20 text-amber-400",
  summaries: "bg-blue-500/20 text-blue-400",
};

export function MemoryIndicator({ memories, className }: MemoryIndicatorProps) {
  const [expanded, setExpanded] = useState(false);

  if (!memories || memories.length === 0) {
    return null;
  }

  const count = memories.length;

  return (
    <div className={cn("", className)}>
      {/* Indicator button */}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => setExpanded(!expanded)}
              className={cn(
                "inline-flex items-center gap-1 px-2 py-1 rounded-md",
                "text-xs text-muted-foreground",
                "hover:text-foreground hover:bg-muted/50",
                "transition-all duration-150",
                "hover:scale-105",
                expanded && "bg-muted/50 text-foreground"
              )}
            >
              <Brain className="h-3.5 w-3.5" />
              <span>{count}</span>
              {expanded ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="top">
            <p>
              {count} {count === 1 ? "memory" : "memories"} retrieved
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Expanded details */}
      {expanded && (
        <div className="mt-2 p-2 rounded-md bg-muted/30 border border-border/50 text-xs space-y-2">
          {memories.map((mem, idx) => (
            <div
              key={idx}
              className="flex flex-col gap-1 pb-2 last:pb-0 border-b border-border/30 last:border-0"
            >
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className={cn(
                    "text-[10px] px-1 py-0",
                    collectionColors[mem.collection] || ""
                  )}
                >
                  {mem.collection}
                </Badge>
                <span className="text-muted-foreground">
                  d={mem.distance.toFixed(3)}
                  {mem.score !== undefined && ` s=${mem.score.toFixed(3)}`}
                </span>
              </div>
              <p className="text-muted-foreground leading-snug">
                {mem.content_preview}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
