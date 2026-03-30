"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Zap, Timer, MessageSquare } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsStore, selectEffectiveStimuliConfig } from "@/lib/stores";
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

              <div className="space-y-2">
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
                  placeholder="Prompt sent when idle timer fires..."
                />
              </div>
            </div>

            <p className="text-xs text-muted-foreground">
              The stimuli system enables autonomous behavior. The idle timer fires after the agent has been idle for the configured timeout.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
