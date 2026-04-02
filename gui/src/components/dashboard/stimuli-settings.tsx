"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Zap, Timer, MessageSquare, Users, Plus, Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsStore, selectEffectiveStimuliConfig, type AddressingContextEntry } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  icon: React.ReactNode;
  onChange: (value: number) => void;
  disabled?: boolean;
  unit?: string;
}

function Slider({ label, value, min, max, step, icon, onChange, disabled, unit = "s" }: SliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          {icon}
          {label}
        </Label>
        <span className="text-sm font-mono text-muted-foreground">
          {value.toFixed(step < 1 ? 1 : 0)}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

export function StimuliSettings() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectiveStimuliConfig(store);
  const { updatePendingStimuli, isSavingStimuli, setSavingStimuli } = store;
  const socket = getSocket();

  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const emitChanges = useCallback(
    (changes: Partial<import("@/types/events").SetStimuliConfigPayload>) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        setSavingStimuli(true);
        socket.emit("set_stimuli_config", changes);
      }, 300);
    },
    [socket, setSavingStimuli]
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleEnabledChange = useCallback(
    (checked: boolean) => {
      updatePendingStimuli({ enabled: checked });
      emitChanges({ enabled: checked });
    },
    [updatePendingStimuli, emitChanges]
  );

  const handlePatienceEnabledChange = useCallback(
    (checked: boolean) => {
      updatePendingStimuli({ patience_enabled: checked });
      emitChanges({ patience_enabled: checked });
    },
    [updatePendingStimuli, emitChanges]
  );

  const handlePatienceSecondsChange = useCallback(
    (value: number) => {
      updatePendingStimuli({ patience_seconds: value });
      emitChanges({ patience_seconds: value });
    },
    [updatePendingStimuli, emitChanges]
  );

  // Local state for prompt textarea — only emits on blur, not per-keystroke
  const [localPrompt, setLocalPrompt] = useState(effectiveConfig.patience_prompt);
  const promptSyncedRef = useRef(effectiveConfig.patience_prompt);

  // Sync local state when backend config changes (e.g. character reload)
  useEffect(() => {
    if (effectiveConfig.patience_prompt !== promptSyncedRef.current) {
      setLocalPrompt(effectiveConfig.patience_prompt);
      promptSyncedRef.current = effectiveConfig.patience_prompt;
    }
  }, [effectiveConfig.patience_prompt]);

  const handlePatiencePromptBlur = useCallback(() => {
    if (localPrompt !== promptSyncedRef.current) {
      updatePendingStimuli({ patience_prompt: localPrompt });
      emitChanges({ patience_prompt: localPrompt });
      promptSyncedRef.current = localPrompt;
    }
  }, [localPrompt, updatePendingStimuli, emitChanges]);

  // NANO-110: Addressing-others contexts
  const contexts = effectiveConfig.addressing_others_contexts ?? [
    { id: "ctx_0", label: "Others", prompt: "" },
  ];

  // Local state for context editing (emit on blur, same pattern as patience prompt)
  const [localContexts, setLocalContexts] = useState<AddressingContextEntry[]>(contexts);
  const contextsSyncedRef = useRef(JSON.stringify(contexts));

  // Sync local state when backend config changes
  useEffect(() => {
    const serialized = JSON.stringify(contexts);
    if (serialized !== contextsSyncedRef.current) {
      setLocalContexts(contexts);
      contextsSyncedRef.current = serialized;
    }
  }, [contexts]);

  const emitContexts = useCallback(
    (updated: AddressingContextEntry[]) => {
      updatePendingStimuli({ addressing_others_contexts: updated });
      emitChanges({ addressing_others_contexts: updated });
      contextsSyncedRef.current = JSON.stringify(updated);
    },
    [updatePendingStimuli, emitChanges]
  );

  const handleAddContext = useCallback(() => {
    const newCtx: AddressingContextEntry = {
      id: `ctx_${Date.now()}`,
      label: "",
      prompt: "",
    };
    const updated = [...localContexts, newCtx];
    setLocalContexts(updated);
    emitContexts(updated);
  }, [localContexts, emitContexts]);

  const handleRemoveContext = useCallback(
    (index: number) => {
      if (index === 0) return; // First context is permanent
      const updated = localContexts.filter((_, i) => i !== index);
      setLocalContexts(updated);
      emitContexts(updated);
    },
    [localContexts, emitContexts]
  );

  const handleContextLabelChange = useCallback(
    (index: number, label: string) => {
      const updated = localContexts.map((ctx, i) =>
        i === index ? { ...ctx, label } : ctx
      );
      setLocalContexts(updated);
    },
    [localContexts]
  );

  const handleContextPromptChange = useCallback(
    (index: number, prompt: string) => {
      const updated = localContexts.map((ctx, i) =>
        i === index ? { ...ctx, prompt } : ctx
      );
      setLocalContexts(updated);
    },
    [localContexts]
  );

  const handleContextBlur = useCallback(() => {
    const serialized = JSON.stringify(localContexts);
    if (serialized !== contextsSyncedRef.current) {
      emitContexts(localContexts);
    }
  }, [localContexts, emitContexts]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Zap className="h-4 w-4" />
          Stimuli System
          {isSavingStimuli && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Master enable/disable */}
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-sm">
            Enable Stimuli
          </Label>
          <button
            onClick={() => handleEnabledChange(!effectiveConfig.enabled)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
              effectiveConfig.enabled ? "bg-primary" : "bg-muted"
            }`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                effectiveConfig.enabled ? "translate-x-4" : "translate-x-0.5"
              }`}
            />
          </button>
        </div>

        {effectiveConfig.enabled && (
          <>
            {/* Idle timer section */}
            <div className="border-t border-border pt-4 space-y-4">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-2 text-sm font-medium">
                  <Timer className="h-3 w-3" />
                  Idle Timer
                </Label>
                <button
                  onClick={() => handlePatienceEnabledChange(!effectiveConfig.patience_enabled)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    effectiveConfig.patience_enabled ? "bg-primary" : "bg-muted"
                  }`}
                >
                  <span
                    className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                      effectiveConfig.patience_enabled ? "translate-x-4" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>

              <Slider
                label="Timeout"
                value={effectiveConfig.patience_seconds}
                min={10}
                max={300}
                step={5}
                icon={<Timer className="h-3 w-3" />}
                onChange={handlePatienceSecondsChange}
                disabled={!effectiveConfig.patience_enabled}
              />

              <div className={`space-y-2${!effectiveConfig.patience_enabled ? " opacity-50 pointer-events-none" : ""}`}>
                <Label className="flex items-center gap-2 text-sm">
                  <MessageSquare className="h-3 w-3" />
                  Idle Prompt
                </Label>
                <Textarea
                  value={localPrompt}
                  onChange={(e) => setLocalPrompt(e.target.value)}
                  onBlur={handlePatiencePromptBlur}
                  rows={3}
                  className="text-xs resize-none"
                  disabled={!effectiveConfig.patience_enabled}
                  placeholder="Prompt sent when idle timer fires..."
                />
              </div>
            </div>

            {/* NANO-110: Addressing Others section */}
            <div className="border-t border-border pt-4 space-y-4">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-2 text-sm font-medium">
                  <Users className="h-3 w-3" />
                  Addressing Others
                </Label>
                <button
                  onClick={handleAddContext}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                  title="Add context"
                >
                  <Plus className="h-3 w-3" />
                  Add
                </button>
              </div>

              <p className="text-xs text-muted-foreground">
                Each context maps to a Stream Deck button. Hold to suppress voice input; on release, the prompt is injected into the next response.
              </p>

              <div className="space-y-3">
                {localContexts.map((ctx, index) => (
                  <div
                    key={ctx.id}
                    className="space-y-2 rounded-md border border-border p-3"
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={ctx.label}
                        onChange={(e) =>
                          handleContextLabelChange(index, e.target.value)
                        }
                        onBlur={handleContextBlur}
                        placeholder={index === 0 ? "Others" : "Label (e.g. Chat, Discord)"}
                        className="flex-1 h-7 rounded-md border border-input bg-background px-2 text-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      />
                      {index > 0 && (
                        <button
                          onClick={() => handleRemoveContext(index)}
                          className="flex items-center justify-center h-7 w-7 rounded-md text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                          title="Remove context"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                    <Textarea
                      value={ctx.prompt}
                      onChange={(e) =>
                        handleContextPromptChange(index, e.target.value)
                      }
                      onBlur={handleContextBlur}
                      rows={2}
                      className="text-xs resize-none"
                      placeholder="The User was just speaking to someone else — not you. The preceding input may reference a conversation you were not part of."
                    />
                  </div>
                ))}
              </div>
            </div>

            <p className="text-xs text-muted-foreground">
              The stimuli system enables autonomous behavior. The idle timer fires after the agent has been idle for the configured timeout.</p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
