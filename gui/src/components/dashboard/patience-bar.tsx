"use client";

import { useEffect, useRef } from "react";
import { Timer, Pause, Keyboard } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useSettingsStore, useConnectionStore, selectEffectiveStimuliConfig } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

function getProgressColor(progress: number, blocked: boolean, reason?: string | null): string {
  if (blocked && reason === "typing") return "bg-purple-500";
  if (blocked) return "bg-blue-500";
  if (progress < 0.5) return "bg-green-500";
  if (progress < 0.8) return "bg-yellow-500";
  return "bg-red-500";
}

function getBlockedLabel(reason?: string | null): string {
  if (reason === "typing") return "TYPING";
  return "PAUSED";
}

export function PatienceBar() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectiveStimuliConfig(store);
  const stimuliProgress = store.stimuliProgress;
  const { connected } = useConnectionStore();
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Poll for progress every 1s when stimuli enabled and connected
  useEffect(() => {
    if (!connected || !effectiveConfig.enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const socket = getSocket();

    // Initial request
    socket.emit("request_patience_progress", {});

    // Poll every 1s
    intervalRef.current = setInterval(() => {
      socket.emit("request_patience_progress", {});
    }, 1000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [connected, effectiveConfig.enabled]);

  // Don't render if stimuli disabled or no connection
  if (!effectiveConfig.enabled || !connected) {
    return null;
  }

  const elapsed = stimuliProgress?.elapsed ?? 0;
  const total = stimuliProgress?.total ?? effectiveConfig.patience_seconds;
  const progress = stimuliProgress?.progress ?? 0;
  const blocked = stimuliProgress?.blocked ?? false;
  const blockedReason = stimuliProgress?.blocked_reason ?? null;
  const progressPercent = Math.min(progress * 100, 100);
  const isTyping = blocked && blockedReason === "typing";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-sm font-medium">
          <div className="flex items-center gap-2">
            {isTyping ? (
              <Keyboard className="h-4 w-4 text-purple-500" />
            ) : blocked ? (
              <Pause className="h-4 w-4 text-blue-500" />
            ) : (
              <Timer className="h-4 w-4" />
            )}
            Idle Timer
            {blocked && (
              <span className={`text-xs font-normal ${isTyping ? "text-purple-500" : "text-blue-500"}`}>
                {getBlockedLabel(blockedReason)}
              </span>
            )}
          </div>
          <span className="text-xs font-mono text-muted-foreground">
            {elapsed.toFixed(0)}s / {total.toFixed(0)}s
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${getProgressColor(progress, blocked, blockedReason)} ${blocked ? "" : "transition-all duration-1000 ease-linear"}`}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
