"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Filter, Flame, GitBranch, Hash, ListFilter, Percent, Repeat, Rewind, TrendingDown, UserMinus } from "lucide-react";
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

  const handleTopKChange = useCallback(
    (value: number) => handleChange({ top_k: value }),
    [handleChange]
  );

  const handleMinPChange = useCallback(
    (value: number) => handleChange({ min_p: value }),
    [handleChange]
  );

  const handleRepeatPenaltyChange = useCallback(
    (value: number) => handleChange({ repeat_penalty: value }),
    [handleChange]
  );

  const handleRepeatLastNChange = useCallback(
    (value: number) => handleChange({ repeat_last_n: value }),
    [handleChange]
  );

  const handleFrequencyPenaltyChange = useCallback(
    (value: number) => handleChange({ frequency_penalty: value }),
    [handleChange]
  );

  const handlePresencePenaltyChange = useCallback(
    (value: number) => handleChange({ presence_penalty: value }),
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

        <div className="border-t border-border pt-4">
          <p className="text-xs font-medium text-muted-foreground mb-4">Tail Sampling (local-only)</p>

          <div className="space-y-6">
            <Slider
              label="Top K"
              value={config.top_k}
              min={0}
              max={200}
              step={1}
              icon={<ListFilter className="h-3 w-3" />}
              formatValue={(v) => v.toString()}
              parseInput={(v) => parseInt(v, 10)}
              minLabel="0 (disabled)"
              maxLabel="200"
              onChange={handleTopKChange}
            />

            <Slider
              label="Min P"
              value={config.min_p}
              min={0}
              max={1}
              step={0.005}
              icon={<Filter className="h-3 w-3" />}
              minLabel="0.0 (off)"
              maxLabel="1.0"
              onChange={handleMinPChange}
            />

            <p className="text-xs text-muted-foreground">
              Top K caps the candidate pool to the K highest-probability tokens. Min P drops any token below min_p * top_token_prob. Both clip the distribution tail where training-data leaks and EOS drift live. Defaults: top_k=40, min_p=0.05.
            </p>
          </div>
        </div>

        <div className="border-t border-border pt-4">
          <p className="text-xs font-medium text-muted-foreground mb-4">Repetition Control</p>

          <div className="space-y-6">
            <Slider
              label="Repeat Penalty"
              value={config.repeat_penalty}
              min={0}
              max={2}
              step={0.01}
              icon={<Repeat className="h-3 w-3" />}
              minLabel="0.0 (off)"
              maxLabel="2.0 (heavy)"
              onChange={handleRepeatPenaltyChange}
            />

            <Slider
              label="Repeat Last N"
              value={config.repeat_last_n}
              min={0}
              max={2048}
              step={1}
              icon={<Rewind className="h-3 w-3" />}
              formatValue={(v) => v.toString()}
              parseInput={(v) => parseInt(v, 10)}
              minLabel="0 (disabled)"
              maxLabel="2048"
              onChange={handleRepeatLastNChange}
            />

            <Slider
              label="Frequency Penalty"
              value={config.frequency_penalty}
              min={-2}
              max={2}
              step={0.01}
              icon={<TrendingDown className="h-3 w-3" />}
              minLabel="-2.0"
              maxLabel="2.0"
              onChange={handleFrequencyPenaltyChange}
            />

            <Slider
              label="Presence Penalty"
              value={config.presence_penalty}
              min={-2}
              max={2}
              step={0.01}
              icon={<UserMinus className="h-3 w-3" />}
              minLabel="-2.0"
              maxLabel="2.0"
              onChange={handlePresencePenaltyChange}
            />

            <p className="text-xs text-muted-foreground">
              Repeat Penalty and Repeat Last N are local-only (llama.cpp). Frequency and Presence Penalty apply to all providers.
            </p>
          </div>
        </div>

        <div className="border-t border-border pt-4">
          <p className="text-xs font-medium text-muted-foreground mb-3 flex items-center gap-2">
            <GitBranch className="h-3 w-3" />
            History Mode
          </p>
          <div className="flex gap-1 rounded-md bg-muted p-1">
            {(["auto", "splice", "flatten"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => handleChange({ force_role_history: mode })}
                className={`flex-1 text-xs py-1.5 px-2 rounded transition-colors ${
                  config.force_role_history === mode
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {mode === "auto" ? "Auto" : mode === "splice" ? "Splice" : "Flatten"}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Auto defers to provider capability. Splice sends real role-array turns (required for Gemma). Flatten embeds history as text in the system prompt.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
