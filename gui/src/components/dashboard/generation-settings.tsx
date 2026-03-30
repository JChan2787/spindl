"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Flame, Hash, Percent } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useSettingsStore } from "@/lib/stores";
import type { GenerationParamsConfig } from "@/lib/stores/settings-store";
import { getSocket } from "@/lib/socket";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  icon: React.ReactNode;
  formatValue?: (v: number) => string;
  parseInput?: (v: string) => number;
  minLabel?: string;
  maxLabel?: string;
  onChange: (value: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  icon,
  formatValue,
  parseInput,
  minLabel,
  maxLabel,
  onChange,
}: SliderProps) {
  const decimals = step < 1 ? 2 : 0;
  const display = formatValue ? formatValue(value) : value.toFixed(decimals);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(display);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.select();
    }
  }, [editing]);

  const commitEdit = () => {
    setEditing(false);
    const parsed = parseInput ? parseInput(editText) : parseFloat(editText);
    if (!isNaN(parsed)) {
      const clamped = Math.min(max, Math.max(min, parsed));
      onChange(clamped);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          {icon}
          {label}
        </Label>
        {editing ? (
          <input
            ref={inputRef}
            type="text"
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitEdit();
              if (e.key === "Escape") setEditing(false);
            }}
            className="w-20 text-right text-sm font-mono bg-muted border border-border rounded px-1 py-0 text-foreground outline-none focus:ring-1 focus:ring-primary"
          />
        ) : (
          <span
            className="text-sm font-mono text-muted-foreground cursor-pointer hover:text-foreground hover:underline"
            onClick={() => {
              setEditText(display);
              setEditing(true);
            }}
            title="Click to edit"
          >
            {display}
          </span>
        )}
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
        <span>{minLabel ?? min}</span>
        <span>{maxLabel ?? max}</span>
      </div>
    </div>
  );
}

export function GenerationSettings() {
  const store = useSettingsStore();
  const config = store.generationConfig;
  const { setGenerationConfig, isSavingGeneration, setSavingGeneration } = store;
  const socket = getSocket();

  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const handleChange = useCallback(
    (changes: Partial<GenerationParamsConfig>) => {
      // Update local state immediately for responsive UI
      setGenerationConfig({ ...config, ...changes });

      // Debounce the server emit
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        setSavingGeneration(true);
        socket.emit("set_generation_params", changes);
      }, 300);
    },
    [config, socket, setGenerationConfig, setSavingGeneration]
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleTemperatureChange = useCallback(
    (value: number) => handleChange({ temperature: value }),
    [handleChange]
  );

  const handleMaxTokensChange = useCallback(
    (value: number) => handleChange({ max_tokens: value }),
    [handleChange]
  );

  const handleTopPChange = useCallback(
    (value: number) => handleChange({ top_p: value }),
    [handleChange]
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Flame className="h-4 w-4" />
          Generation Parameters
          {isSavingGeneration && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <Slider
          label="Temperature"
          value={config.temperature}
          min={0}
          max={2}
          step={0.01}
          icon={<Flame className="h-3 w-3" />}
          minLabel="0.0 (deterministic)"
          maxLabel="2.0 (creative)"
          onChange={handleTemperatureChange}
        />

        <Slider
          label="Max Tokens"
          value={config.max_tokens}
          min={64}
          max={8192}
          step={64}
          icon={<Hash className="h-3 w-3" />}
          formatValue={(v) => v.toString()}
          parseInput={(v) => parseInt(v, 10)}
          minLabel="64"
          maxLabel="8192"
          onChange={handleMaxTokensChange}
        />

        <Slider
          label="Top P"
          value={config.top_p}
          min={0}
          max={1}
          step={0.01}
          icon={<Percent className="h-3 w-3" />}
          minLabel="0.0 (narrow)"
          maxLabel="1.0 (full)"
          onChange={handleTopPChange}
        />

        <p className="text-xs text-muted-foreground">
          Temperature controls randomness. Max Tokens limits response length. Top P controls nucleus sampling breadth. Changes apply to the next LLM call.
        </p>
      </CardContent>
    </Card>
  );
}
