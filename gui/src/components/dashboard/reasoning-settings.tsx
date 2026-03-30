"use client";

/**
 * ReasoningSettings - Dashboard toggle for enabling/disabling LLM thinking.
 * NANO-042: Controls reasoning_budget (0 = disabled, -1 = unlimited).
 *
 * Note: This persists to YAML config. The change takes effect on the next
 * server launch (reasoning_budget is a launch flag for llama.cpp).
 */

import { useState } from "react";
import { Brain } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { getSocket } from "@/lib/socket";
import { useAgentStore } from "@/lib/stores";

export function ReasoningSettings() {
  // DISABLED: Reasoning UI force-hidden until runtime toggle supports cloud LLMs.
  // Remove this early return to re-enable.
  return null;

  const config = useAgentStore((s) => s.config);

  // Derive initial state from config: reasoning_format being set means the
  // model supports thinking. Budget -1 = enabled, 0 = disabled.
  const llmConfig = config?.providers?.llm?.config as Record<string, unknown> | undefined;
  const hasReasoningSupport = !!llmConfig?.reasoning_format;
  const initialBudget = (llmConfig?.reasoning_budget as number) ?? -1;

  const [enabled, setEnabled] = useState(initialBudget !== 0);
  const [persisted, setPersisted] = useState<boolean | null>(null);

  // Only show if the model is configured for reasoning
  if (!hasReasoningSupport) {
    return null;
  }

  const handleToggle = (checked: boolean) => {
    setEnabled(checked);
    setPersisted(null);

    const socket = getSocket();
    socket.emit("set_reasoning_config", {
      reasoning_budget: checked ? -1 : 0,
    });

    // Listen for confirmation (one-shot)
    socket.once("reasoning_config_updated", (event) => {
      setPersisted(event.persisted);
      setTimeout(() => setPersisted(null), 2000);
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Brain className="h-4 w-4 text-purple-400" />
          Reasoning / Thinking
          {persisted === true && (
            <span className="text-xs text-muted-foreground">(saved)</span>
          )}
          {persisted === false && (
            <span className="text-xs text-destructive">(save failed)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <Label htmlFor="reasoning-toggle" className="text-sm">
            Enable Thinking
          </Label>
          <Switch
            id="reasoning-toggle"
            checked={enabled}
            onCheckedChange={handleToggle}
          />
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          {enabled
            ? "Model will think before responding. Reasoning is visible in the thought bubble but never spoken by TTS."
            : "Model will respond directly without a thinking step."}
        </p>
        <p className="text-xs text-muted-foreground mt-1 italic">
          Takes effect on next server restart.
        </p>
      </CardContent>
    </Card>
  );
}
