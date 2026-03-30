"use client";

import { useCallback, useEffect, useRef } from "react";
import { Settings2, Percent } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { useSettingsStore, selectEffectivePipelineConfig } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  icon: React.ReactNode;
  onChange: (value: number) => void;
}

function Slider({ label, value, min, max, step, unit, icon, onChange }: SliderProps) {
  const displayValue = unit === "%" ? (value * 100).toFixed(0) : value.toLocaleString();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          {icon}
          {label}
        </Label>
        <span className="text-sm font-mono text-muted-foreground">
          {displayValue}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{unit === "%" ? `${(min * 100).toFixed(0)}%` : min.toLocaleString()}</span>
        <span>{unit === "%" ? `${(max * 100).toFixed(0)}%` : max.toLocaleString()}</span>
      </div>
    </div>
  );
}

export function PipelineSettings() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectivePipelineConfig(store);
  const { updatePendingPipeline, isSavingPipeline, setSavingPipeline } = store;
  const socket = getSocket();

  // Debounce timer ref
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // Debounced emit to backend
  const emitChanges = useCallback(
    (changes: Partial<typeof effectiveConfig>) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        setSavingPipeline(true);
        socket.emit("set_pipeline_config", changes);
      }, 300);
    },
    [socket, setSavingPipeline]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleSummarizationThresholdChange = useCallback(
    (value: number) => {
      updatePendingPipeline({ summarization_threshold: value });
      emitChanges({ summarization_threshold: value });
    },
    [updatePendingPipeline, emitChanges]
  );

  return (
    <CollapsibleCard
      id="pipeline"
      title="Pipeline Settings"
      icon={<Settings2 className="h-4 w-4" />}
      headerExtra={
        isSavingPipeline ? (
          <span className="text-xs text-muted-foreground">(saving...)</span>
        ) : null
      }
    >
      <Slider
        label="Summarization Threshold"
        value={effectiveConfig.summarization_threshold}
        min={0.5}
        max={0.95}
        step={0.05}
        unit="%"
        icon={<Percent className="h-3 w-3" />}
        onChange={handleSummarizationThresholdChange}
      />

      <div className="flex items-center justify-between p-2 bg-muted/50 rounded-md">
        <span className="text-sm">Budget Strategy</span>
        <Badge variant="outline">{effectiveConfig.budget_strategy}</Badge>
      </div>

      <p className="text-xs text-muted-foreground">
        Summarization threshold determines when conversation history is summarized (as a percentage of the model&apos;s context window). Context size is set per-provider in the LLM configuration.
      </p>
    </CollapsibleCard>
  );
}
