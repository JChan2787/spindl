"use client";

import { Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface StimulusSourceBadgeProps {
  source?: string;
}

const DISPLAY_NAMES: Record<string, string> = {
  patience: "IDLE",
};

export function StimulusSourceBadge({ source }: StimulusSourceBadgeProps) {
  if (!source) return null;

  const label = DISPLAY_NAMES[source] ?? source.toUpperCase();

  return (
    <Badge variant="outline" className="text-xs gap-1 border-orange-500/50 text-orange-500">
      <Zap className="h-3 w-3" />
      {label}
    </Badge>
  );
}
