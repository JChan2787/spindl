"use client";

/**
 * CodexIndicator - Shows when codex entries were activated for a response.
 * NANO-037 Phase 2: GUI activation indicator.
 *
 * Features:
 * - Icon appears only when entries are activated
 * - Hover shows count tooltip
 * - Click expands to show entry details
 * - Scale animation on hover
 */

import { useState } from "react";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { ActivatedCodexEntry } from "@/types/events";

interface CodexIndicatorProps {
  entries: ActivatedCodexEntry[];
  className?: string;
}

export function CodexIndicator({ entries, className }: CodexIndicatorProps) {
  const [expanded, setExpanded] = useState(false);

  // Don't render if no entries
  if (!entries || entries.length === 0) {
    return null;
  }

  const count = entries.length;

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
              <BookOpen className="h-3.5 w-3.5" />
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
              {count} codex {count === 1 ? "entry" : "entries"} activated
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Expanded details */}
      {expanded && (
        <div className="mt-2 p-2 rounded-md bg-muted/30 border border-border/50 text-xs space-y-2">
          {entries.map((entry, idx) => (
            <div
              key={idx}
              className="flex flex-col gap-1 pb-2 last:pb-0 border-b border-border/30 last:border-0"
            >
              <div className="flex items-center gap-2">
                <span className="font-medium">{entry.name}</span>
                <Badge
                  variant="outline"
                  className="text-[10px] px-1 py-0"
                >
                  {entry.activation_method}
                </Badge>
              </div>
              <div className="flex flex-wrap gap-1">
                {entry.keys.slice(0, 5).map((key, keyIdx) => (
                  <span
                    key={keyIdx}
                    className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
                  >
                    {key}
                  </span>
                ))}
                {entry.keys.length > 5 && (
                  <span className="text-muted-foreground">
                    +{entry.keys.length - 5} more
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
