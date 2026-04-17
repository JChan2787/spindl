"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Tv, Wifi, WifiOff, Layers, MessageSquare, History, Scissors } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsStore, selectEffectiveStimuliConfig } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import type { TwitchStatus } from "@/lib/stores/settings-store";

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

function Slider({ label, value, min, max, step, icon, onChange, disabled, unit = "" }: SliderProps) {
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

export function TwitchChatCard() {
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

  // Credential gate — backend resolves env var fallback and reports this flag
  const hasCredentials =
    effectiveConfig.twitch_has_credentials ||
    (effectiveConfig.twitch_channel.trim() !== "" &&
     effectiveConfig.twitch_app_id.trim() !== "" &&
     effectiveConfig.twitch_app_secret.trim() !== "");

  // Twitch status polling
  const twitchStatus = useSettingsStore((s) => s.twitchStatus);
  const setTwitchStatus = useSettingsStore((s) => s.setTwitchStatus);

  const handleTwitchEnabledChange = useCallback(
    (checked: boolean) => {
      updatePendingStimuli({ twitch_enabled: checked });
      emitChanges({ twitch_enabled: checked });
    },
    [updatePendingStimuli, emitChanges]
  );

  // Local state for prompt template — emit on blur
  const [localTwitchPrompt, setLocalTwitchPrompt] = useState(effectiveConfig.twitch_prompt_template);
  const twitchPromptSyncedRef = useRef(effectiveConfig.twitch_prompt_template);

  useEffect(() => {
    if (effectiveConfig.twitch_prompt_template !== twitchPromptSyncedRef.current) {
      setLocalTwitchPrompt(effectiveConfig.twitch_prompt_template);
      twitchPromptSyncedRef.current = effectiveConfig.twitch_prompt_template;
    }
  }, [effectiveConfig.twitch_prompt_template]);

  const twitchPromptMissingPlaceholder =
    localTwitchPrompt.trim().length > 0 && !localTwitchPrompt.includes("{messages}");

  const handleTwitchPromptBlur = useCallback(() => {
    if (twitchPromptMissingPlaceholder) {
      return;
    }
    if (localTwitchPrompt !== twitchPromptSyncedRef.current) {
      updatePendingStimuli({ twitch_prompt_template: localTwitchPrompt });
      emitChanges({ twitch_prompt_template: localTwitchPrompt });
      twitchPromptSyncedRef.current = localTwitchPrompt;
    }
  }, [localTwitchPrompt, twitchPromptMissingPlaceholder, updatePendingStimuli, emitChanges]);

  const handleBufferSizeChange = useCallback(
    (value: number) => {
      updatePendingStimuli({ twitch_buffer_size: value });
      emitChanges({ twitch_buffer_size: value });
    },
    [updatePendingStimuli, emitChanges]
  );

  const handleAudienceWindowChange = useCallback(
    (value: number) => {
      updatePendingStimuli({ twitch_audience_window: value });
      emitChanges({ twitch_audience_window: value });
    },
    [updatePendingStimuli, emitChanges]
  );

  const handleAudienceCharCapChange = useCallback(
    (value: number) => {
      updatePendingStimuli({ twitch_audience_char_cap: value });
      emitChanges({ twitch_audience_char_cap: value });
    },
    [updatePendingStimuli, emitChanges]
  );

  // Poll Twitch status every 2 seconds when enabled
  useEffect(() => {
    if (!effectiveConfig.twitch_enabled) {
      setTwitchStatus(null);
      return;
    }
    const poll = () => socket.emit("request_twitch_status", {});
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [effectiveConfig.twitch_enabled, socket, setTwitchStatus]);

  useEffect(() => {
    const handler = (data: TwitchStatus) => setTwitchStatus(data);
    socket.on("twitch_status", handler);
    return () => { socket.off("twitch_status", handler); };
  }, [socket, setTwitchStatus]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Tv className="h-4 w-4" />
          Twitch Chat
          {twitchStatus && (
            <span className="flex items-center gap-1 text-xs">
              {twitchStatus.connected ? (
                <Wifi className="h-3 w-3 text-green-500" />
              ) : (
                <WifiOff className="h-3 w-3 text-red-500" />
              )}
            </span>
          )}
          {isSavingStimuli && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Enable/disable — gated on credentials */}
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-sm">
            Enable Twitch Chat
          </Label>
          <button
            onClick={() => hasCredentials && handleTwitchEnabledChange(!effectiveConfig.twitch_enabled)}
            disabled={!hasCredentials}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
              !hasCredentials
                ? "bg-muted opacity-50 cursor-not-allowed"
                : effectiveConfig.twitch_enabled
                  ? "bg-primary"
                  : "bg-muted"
            }`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                effectiveConfig.twitch_enabled && hasCredentials ? "translate-x-4" : "translate-x-0.5"
              }`}
            />
          </button>
        </div>

        {!hasCredentials && (
          <p className="text-xs text-muted-foreground">
            Configure Twitch credentials in Settings to enable.
          </p>
        )}

        {effectiveConfig.twitch_enabled && hasCredentials && (
          <div className="space-y-3">
            <Slider
              label="Buffer Size"
              value={effectiveConfig.twitch_buffer_size}
              min={1}
              max={50}
              step={1}
              icon={<Layers className="h-3 w-3" />}
              onChange={handleBufferSizeChange}
            />

            <Slider
              label="Audience Window"
              value={effectiveConfig.twitch_audience_window}
              min={25}
              max={300}
              step={5}
              icon={<History className="h-3 w-3" />}
              onChange={handleAudienceWindowChange}
              unit=" msgs"
            />

            <Slider
              label="Message Length Cap"
              value={effectiveConfig.twitch_audience_char_cap}
              min={50}
              max={500}
              step={10}
              icon={<Scissors className="h-3 w-3" />}
              onChange={handleAudienceCharCapChange}
              unit=" chars"
            />

            <div className="space-y-1">
              <Label className="flex items-center gap-2 text-xs">
                <MessageSquare className="h-3 w-3" />
                Prompt Template
              </Label>
              <Textarea
                value={localTwitchPrompt}
                onChange={(e) => setLocalTwitchPrompt(e.target.value)}
                onBlur={handleTwitchPromptBlur}
                rows={3}
                className={`text-xs resize-none ${twitchPromptMissingPlaceholder ? "border-red-500" : ""}`}
                placeholder="Template for Twitch chat injection..."
                aria-invalid={twitchPromptMissingPlaceholder}
              />
              {twitchPromptMissingPlaceholder && (
                <p className="text-xs text-red-500">
                  Template must contain {"{messages}"}. Without it, buffered Twitch messages have nowhere to render and the model receives only the directive text. Changes will not save until this is fixed.
                </p>
              )}
            </div>

            {/* Live message preview */}
            {twitchStatus && twitchStatus.recent_messages.length > 0 && (
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  Buffered ({twitchStatus.buffer_count})
                </Label>
                <div className="bg-muted/50 rounded p-2 text-xs font-mono max-h-24 overflow-y-auto">
                  {twitchStatus.recent_messages.map((msg, i) => (
                    <div key={i} className="truncate">{msg}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
