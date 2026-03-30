"use client";

import { useCallback, useEffect, useRef } from "react";
import { Mic, Volume2, Clock } from "lucide-react";
import { Label } from "@/components/ui/label";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { useSettingsStore, selectEffectiveVadConfig } from "@/lib/stores";
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
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          {icon}
          {label}
        </Label>
        <span className="text-sm font-mono text-muted-foreground">
          {value.toFixed(step < 1 ? 2 : 0)}{unit}
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
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

export function VADSettings() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectiveVadConfig(store);
  const { updatePendingVad, isSavingVad, setSavingVad } = store;
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
        setSavingVad(true);
        socket.emit("set_vad_config", changes);
      }, 300);
    },
    [socket, setSavingVad]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleThresholdChange = useCallback(
    (value: number) => {
      updatePendingVad({ threshold: value });
      emitChanges({ threshold: value });
    },
    [updatePendingVad, emitChanges]
  );

  const handleMinSpeechChange = useCallback(
    (value: number) => {
      updatePendingVad({ min_speech_ms: value });
      emitChanges({ min_speech_ms: value });
    },
    [updatePendingVad, emitChanges]
  );

  const handleMinSilenceChange = useCallback(
    (value: number) => {
      updatePendingVad({ min_silence_ms: value });
      emitChanges({ min_silence_ms: value });
    },
    [updatePendingVad, emitChanges]
  );

  const handleSpeechPadChange = useCallback(
    (value: number) => {
      updatePendingVad({ speech_pad_ms: value });
      emitChanges({ speech_pad_ms: value });
    },
    [updatePendingVad, emitChanges]
  );

  return (
    <CollapsibleCard
      id="vad"
      title="Voice Activity Detection"
      icon={<Mic className="h-4 w-4" />}
      headerExtra={
        isSavingVad ? (
          <span className="text-xs text-muted-foreground">(saving...)</span>
        ) : null
      }
    >
      <Slider
        label="Threshold"
        value={effectiveConfig.threshold}
        min={0}
        max={1}
        step={0.05}
        unit=""
        icon={<Volume2 className="h-3 w-3" />}
        onChange={handleThresholdChange}
      />

      <Slider
        label="Min Speech Duration"
        value={effectiveConfig.min_speech_ms}
        min={100}
        max={1000}
        step={50}
        unit="ms"
        icon={<Clock className="h-3 w-3" />}
        onChange={handleMinSpeechChange}
      />

      <Slider
        label="Min Silence Duration"
        value={effectiveConfig.min_silence_ms}
        min={200}
        max={3000}
        step={100}
        unit="ms"
        icon={<Clock className="h-3 w-3" />}
        onChange={handleMinSilenceChange}
      />

      <Slider
        label="Speech Padding"
        value={effectiveConfig.speech_pad_ms}
        min={0}
        max={500}
        step={25}
        unit="ms"
        icon={<Clock className="h-3 w-3" />}
        onChange={handleSpeechPadChange}
      />

      <p className="text-xs text-muted-foreground">
        Adjust these settings to control when speech is detected and when pauses are interpreted as end of utterance.
      </p>
    </CollapsibleCard>
  );
}
