"use client";

import { Badge } from "@/components/ui/badge";
import type { MemoryDocument, MemoryCollectionType } from "@/types/events";

interface MemoryCardProps {
  memory: MemoryDocument;
  collection: MemoryCollectionType;
  isSelected: boolean;
  onSelect: () => void;
  distance?: number;
}

function formatTimestamp(ts?: string): string {
  if (!ts) return "";
  try {
    const date = new Date(ts);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

function getSourceBadge(source?: string): { label: string; variant: "default" | "secondary" | "outline" | "destructive" } | null {
  if (!source) return null;
  if (source === "gui_manual") return { label: "Manual", variant: "secondary" };
  if (source === "reflection") return { label: "Reflection", variant: "outline" };
  if (source === "session_summary_generator") return { label: "Summary", variant: "outline" };
  if (source.startsWith("promoted_from_")) {
    const from = source.replace("promoted_from_", "");
    return { label: `From ${from}`, variant: "default" };
  }
  return { label: source, variant: "outline" };
}

export function MemoryCard({ memory, collection, isSelected, onSelect, distance }: MemoryCardProps) {
  const preview = memory.content.length > 120
    ? memory.content.substring(0, 120) + "..."
    : memory.content;

  const timestamp = formatTimestamp(memory.metadata?.timestamp);
  const sourceBadge = getSourceBadge(memory.metadata?.source);
  const sessionId = memory.metadata?.session_id;

  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full text-left p-3 rounded-lg border transition-colors hover:bg-accent/50 ${
        isSelected
          ? "border-primary bg-accent"
          : "border-transparent"
      }`}
    >
      <p className="text-sm leading-relaxed break-words">{preview}</p>

      <div className="flex items-center gap-2 mt-2 flex-wrap">
        {timestamp && (
          <span className="text-xs text-muted-foreground">{timestamp}</span>
        )}
        {sourceBadge && (
          <Badge variant={sourceBadge.variant} className="text-xs px-1.5 py-0">
            {sourceBadge.label}
          </Badge>
        )}
        {sessionId && collection !== "general" && (
          <Badge variant="outline" className="text-xs px-1.5 py-0 font-mono">
            {sessionId.length > 20 ? sessionId.substring(0, 20) + "..." : sessionId}
          </Badge>
        )}
        {distance !== undefined && (
          <Badge variant="secondary" className="text-xs px-1.5 py-0">
            d={distance.toFixed(3)}
          </Badge>
        )}
      </div>
    </button>
  );
}
