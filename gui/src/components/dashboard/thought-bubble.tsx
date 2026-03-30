"use client";

/**
 * ThoughtBubble - Collapsible display for LLM reasoning/thinking content.
 * NANO-042: Shows reasoning from models like Qwen3, DeepSeek R1.
 *
 * Features:
 * - Hidden when no reasoning content is present
 * - Collapsed by default, click to expand
 * - Brain icon with "Thinking..." label
 * - Purple accent border to distinguish from response
 * - Scrollable for long reasoning chains
 */

import { useState } from "react";
import { Brain, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThoughtBubbleProps {
  reasoning: string;
  className?: string;
}

export function ThoughtBubble({ reasoning, className }: ThoughtBubbleProps) {
  const [expanded, setExpanded] = useState(false);

  if (!reasoning) {
    return null;
  }

  return (
    <div
      className={cn(
        "rounded-lg border border-purple-500/30 bg-purple-500/5",
        className
      )}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-2 text-sm",
          "text-purple-400 hover:text-purple-300",
          "transition-colors duration-150"
        )}
      >
        <Brain className="h-4 w-4 shrink-0" />
        <span className="font-medium">Thinking...</span>
        <div className="flex-1" />
        {expanded ? (
          <ChevronUp className="h-4 w-4 shrink-0" />
        ) : (
          <ChevronDown className="h-4 w-4 shrink-0" />
        )}
      </button>

      {expanded && (
        <div className="px-3 pb-3">
          <div className="max-h-60 overflow-y-auto rounded bg-muted/30 p-2">
            <p className="text-xs text-muted-foreground whitespace-pre-wrap">
              {reasoning}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
