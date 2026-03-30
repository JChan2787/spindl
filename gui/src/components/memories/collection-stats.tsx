"use client";

import { Badge } from "@/components/ui/badge";
import { Globe, Database, Clock, AlertTriangle } from "lucide-react";

interface CollectionStatsProps {
  counts: { global: number; general: number; flashcards: number; summaries: number };
  enabled: boolean;
}

export function CollectionStats({ counts, enabled }: CollectionStatsProps) {
  if (!enabled) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <AlertTriangle className="h-4 w-4" />
        <span>Memory system unavailable — start services to view memories</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Badge variant="outline" className="gap-1">
        <Globe className="h-3 w-3" />
        {counts.global}
      </Badge>
      <Badge variant="outline" className="gap-1">
        <Clock className="h-3 w-3" />
        {counts.flashcards + counts.summaries}
      </Badge>
      <Badge variant="outline" className="gap-1">
        <Database className="h-3 w-3" />
        {counts.general}
      </Badge>
    </div>
  );
}
