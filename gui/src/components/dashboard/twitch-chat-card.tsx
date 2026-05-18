"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Tv, Wifi, WifiOff, Layers, MessageSquare, History, Scissors,
  Timer, Filter, Volume2, Type, Hash, Gauge, ChevronDown, ChevronRight,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsStore, selectEffectiveStimuliConfig } from "@/lib/stores";
import { useLauncherStore } from "@/lib/stores/launcher-store";
import { getSocket } from "@/lib/socket";
import { ModelCombobox, type ModelOption } from "@/components/ui/model-combobox";
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

  const handleEventsEnabledChange = useCallback(
    (checked: boolean) => {
      updatePendingStimuli({ twitch_events_enabled: checked });
      emitChanges({ twitch_events_enabled: checked });
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

  // ── NANO-130 Phase 3: Selection Pass + Chat-TTS controls ──

  const [selectionOpen, setSelectionOpen] = useState(false);
  const [chatTtsOpen, setChatTtsOpen] = useState(false);

  // Selection pass model combobox
  const [selectionModels, setSelectionModels] = useState<ModelOption[]>([]);
  const [isLoadingSelModels, setIsLoadingSelModels] = useState(false);
  const [selModelsError, setSelModelsError] = useState<string | null>(null);
  const hasFetchedSelModelsRef = useRef(false);

  // Chat-TTS format string local state
  const [localChatTtsFormat, setLocalChatTtsFormat] = useState(effectiveConfig.twitch_chat_tts_format);
  const chatTtsFormatSyncedRef = useRef(effectiveConfig.twitch_chat_tts_format);

  useEffect(() => {
    if (effectiveConfig.twitch_chat_tts_format !== chatTtsFormatSyncedRef.current) {
      setLocalChatTtsFormat(effectiveConfig.twitch_chat_tts_format);
      chatTtsFormatSyncedRef.current = effectiveConfig.twitch_chat_tts_format;
    }
  }, [effectiveConfig.twitch_chat_tts_format]);

  const handleChatTtsFormatBlur = useCallback(() => {
    if (localChatTtsFormat !== chatTtsFormatSyncedRef.current) {
      updatePendingStimuli({ twitch_chat_tts_format: localChatTtsFormat });
      emitChanges({ twitch_chat_tts_format: localChatTtsFormat });
      chatTtsFormatSyncedRef.current = localChatTtsFormat;
    }
  }, [localChatTtsFormat, updatePendingStimuli, emitChanges]);

  // Chat-TTS server status
  const [chatTtsStatus, setChatTtsStatus] = useState<{
    running: boolean;
    process_alive: boolean;
    reachable: boolean;
  } | null>(null);
  const [chatTtsLaunching, setChatTtsLaunching] = useState(false);
  const [chatTtsError, setChatTtsError] = useState<string | null>(null);

  const handleLaunchChatTts = useCallback(() => {
    setChatTtsLaunching(true);
    setChatTtsError(null);
    socket.emit("launch_chat_tts", {
      host: effectiveConfig.twitch_chat_tts_host,
      port: effectiveConfig.twitch_chat_tts_port,
      device: effectiveConfig.twitch_chat_tts_device,
    });
  }, [socket, effectiveConfig.twitch_chat_tts_host, effectiveConfig.twitch_chat_tts_port, effectiveConfig.twitch_chat_tts_device]);

  const handleStopChatTts = useCallback(() => {
    socket.emit("stop_chat_tts", {});
  }, [socket]);

  useEffect(() => {
    const onLaunched = (data: { success: boolean; error?: string; already_running?: boolean }) => {
      setChatTtsLaunching(false);
      if (data.success) {
        setChatTtsError(null);
      } else {
        setChatTtsError(data.error || "Launch failed");
      }
    };
    const onStopped = () => {
      setChatTtsStatus(null);
    };
    const onStatus = (data: { running: boolean; process_alive: boolean; reachable: boolean }) => {
      setChatTtsStatus(data);
    };

    socket.on("chat_tts_launched", onLaunched);
    socket.on("chat_tts_stopped", onStopped);
    socket.on("chat_tts_status", onStatus);
    return () => {
      socket.off("chat_tts_launched", onLaunched);
      socket.off("chat_tts_stopped", onStopped);
      socket.off("chat_tts_status", onStatus);
    };
  }, [socket]);

  // Poll chat-TTS status when section is open and enabled
  useEffect(() => {
    if (!chatTtsOpen || !effectiveConfig.twitch_chat_tts_enabled) return;
    const poll = () => socket.emit("request_chat_tts_status", {});
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, [chatTtsOpen, effectiveConfig.twitch_chat_tts_enabled, socket]);

  // Selection pass API key — resolve from launcher fallback
  const launcherCloud = useLauncherStore((s) => s.llmCloud);
  const fallbackKey = launcherCloud.provider === "openrouter" && launcherCloud.apiKey
    ? launcherCloud.apiKey
    : null;
  const effectiveSelApiKey = effectiveConfig.twitch_selection_pass_api_key || fallbackKey;

  const fetchSelectionModels = useCallback(async () => {
    const key = effectiveSelApiKey;
    if (!key) return;

    setIsLoadingSelModels(true);
    setSelModelsError(null);

    try {
      const response = await fetch("/api/launcher/fetch-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: "openrouter", apiKey: key }),
      });

      const data = await response.json();

      if (!data.success) {
        setSelModelsError(data.error || "Failed to fetch models");
        setSelectionModels([]);
      } else {
        setSelectionModels(data.models);
      }
    } catch (err) {
      setSelModelsError(err instanceof Error ? err.message : "Network error");
      setSelectionModels([]);
    } finally {
      setIsLoadingSelModels(false);
    }
  }, [effectiveSelApiKey]);

  useEffect(() => {
    if (selectionOpen && !hasFetchedSelModelsRef.current && effectiveSelApiKey) {
      hasFetchedSelModelsRef.current = true;
      fetchSelectionModels();
    }
  }, [selectionOpen, effectiveSelApiKey, fetchSelectionModels]);

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
            {/* NANO-132: EventSub follow callouts toggle */}
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2 text-sm">
                Follow Callouts
                {twitchStatus?.events_connected && (
                  <Wifi className="h-3 w-3 text-green-500" />
                )}
              </Label>
              <button
                onClick={() => handleEventsEnabledChange(!effectiveConfig.twitch_events_enabled)}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                  effectiveConfig.twitch_events_enabled
                    ? "bg-primary"
                    : "bg-muted"
                }`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                    effectiveConfig.twitch_events_enabled ? "translate-x-4" : "translate-x-0.5"
                  }`}
                />
              </button>
            </div>

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

            {/* ── Selection Pass Section ── */}
            <div className="border rounded-md">
              <button
                onClick={() => setSelectionOpen(!selectionOpen)}
                className="flex items-center gap-2 w-full px-3 py-2 text-xs font-medium hover:bg-muted/50 transition-colors"
              >
                {selectionOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                <Filter className="h-3 w-3" />
                Selection Pass
                <span className="ml-auto text-muted-foreground font-mono">
                  {effectiveConfig.twitch_selection_mode}
                </span>
              </button>
              {selectionOpen && (
                <div className="px-3 pb-3 space-y-3 border-t">
                  {/* Selection mode toggle */}
                  <div className="flex items-center justify-between pt-2">
                    <Label className="text-xs">Mode</Label>
                    <div className="flex gap-1">
                      {(["llm", "heuristic"] as const).map((mode) => (
                        <button
                          key={mode}
                          onClick={() => {
                            updatePendingStimuli({ twitch_selection_mode: mode });
                            emitChanges({ twitch_selection_mode: mode });
                          }}
                          className={`px-2 py-0.5 text-xs rounded transition-colors ${
                            effectiveConfig.twitch_selection_mode === mode
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted hover:bg-muted/80"
                          }`}
                        >
                          {mode.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Staleness threshold */}
                  <Slider
                    label="Staleness Threshold"
                    value={effectiveConfig.twitch_max_message_age_seconds}
                    min={1}
                    max={120}
                    step={1}
                    icon={<Timer className="h-3 w-3" />}
                    onChange={(value) => {
                      updatePendingStimuli({ twitch_max_message_age_seconds: value });
                      emitChanges({ twitch_max_message_age_seconds: value });
                    }}
                    unit="s"
                  />

                  {/* Selection model (LLM mode only) */}
                  {effectiveConfig.twitch_selection_mode === "llm" && (
                    <div className="space-y-1">
                      <Label className="text-xs">Selection Model (OpenRouter)</Label>
                      <ModelCombobox
                        value={effectiveConfig.twitch_selection_pass_model}
                        onValueChange={(model) => {
                          updatePendingStimuli({ twitch_selection_pass_model: model });
                          emitChanges({ twitch_selection_pass_model: model });
                        }}
                        models={selectionModels}
                        isLoading={isLoadingSelModels}
                        error={selModelsError}
                        onRefresh={() => {
                          hasFetchedSelModelsRef.current = false;
                          fetchSelectionModels();
                        }}
                        placeholder="Select a model..."
                      />
                      {!effectiveSelApiKey && (
                        <p className="text-xs text-muted-foreground">
                          No API key available. Configure in launcher or provide a dedicated key.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Chat-TTS Section ── */}
            <div className="border rounded-md">
              <button
                onClick={() => setChatTtsOpen(!chatTtsOpen)}
                className="flex items-center gap-2 w-full px-3 py-2 text-xs font-medium hover:bg-muted/50 transition-colors"
              >
                {chatTtsOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                <Volume2 className="h-3 w-3" />
                Chat TTS
                <span className={`ml-auto text-xs ${effectiveConfig.twitch_chat_tts_enabled ? "text-green-500" : "text-muted-foreground"}`}>
                  {effectiveConfig.twitch_chat_tts_enabled ? "ON" : "OFF"}
                </span>
              </button>
              {chatTtsOpen && (
                <div className="px-3 pb-3 space-y-3 border-t">
                  {/* Enable toggle */}
                  <div className="flex items-center justify-between pt-2">
                    <Label className="text-xs">Enable Chat TTS</Label>
                    <button
                      onClick={() => {
                        const next = !effectiveConfig.twitch_chat_tts_enabled;
                        updatePendingStimuli({ twitch_chat_tts_enabled: next });
                        emitChanges({ twitch_chat_tts_enabled: next });
                      }}
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                        effectiveConfig.twitch_chat_tts_enabled ? "bg-primary" : "bg-muted"
                      }`}
                    >
                      <span
                        className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                          effectiveConfig.twitch_chat_tts_enabled ? "translate-x-4" : "translate-x-0.5"
                        }`}
                      />
                    </button>
                  </div>

                  {effectiveConfig.twitch_chat_tts_enabled && (
                    <div className="space-y-3">
                      {/* Host + Port */}
                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                          <Label className="text-xs flex items-center gap-1">
                            <Hash className="h-3 w-3" />
                            Host
                          </Label>
                          <Input
                            value={effectiveConfig.twitch_chat_tts_host}
                            onChange={(e) => {
                              updatePendingStimuli({ twitch_chat_tts_host: e.target.value });
                              emitChanges({ twitch_chat_tts_host: e.target.value });
                            }}
                            className="text-xs h-7"
                            placeholder="127.0.0.1"
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs flex items-center gap-1">
                            <Hash className="h-3 w-3" />
                            Port
                          </Label>
                          <Input
                            type="number"
                            value={effectiveConfig.twitch_chat_tts_port}
                            onChange={(e) => {
                              const port = parseInt(e.target.value) || 5560;
                              updatePendingStimuli({ twitch_chat_tts_port: port });
                              emitChanges({ twitch_chat_tts_port: port });
                            }}
                            className="text-xs h-7"
                            min={1}
                            max={65535}
                          />
                        </div>
                      </div>

                      {/* Launch / Stop button + status */}
                      <div className="flex items-center gap-2">
                        {chatTtsStatus?.running ? (
                          <button
                            onClick={handleStopChatTts}
                            className="px-3 py-1 text-xs rounded bg-red-600 hover:bg-red-700 text-white transition-colors"
                          >
                            Stop Server
                          </button>
                        ) : (
                          <button
                            onClick={handleLaunchChatTts}
                            disabled={chatTtsLaunching}
                            className="px-3 py-1 text-xs rounded bg-primary hover:bg-primary/90 text-primary-foreground transition-colors disabled:opacity-50"
                          >
                            {chatTtsLaunching ? "Starting..." : "Launch Server"}
                          </button>
                        )}
                        <span className="flex items-center gap-1 text-xs">
                          {chatTtsStatus?.running ? (
                            <>
                              <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                              <span className="text-green-500">Running</span>
                            </>
                          ) : chatTtsLaunching ? (
                            <>
                              <span className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse" />
                              <span className="text-yellow-500">Starting</span>
                            </>
                          ) : (
                            <>
                              <span className="h-2 w-2 rounded-full bg-muted-foreground" />
                              <span className="text-muted-foreground">Idle</span>
                            </>
                          )}
                        </span>
                      </div>
                      {chatTtsError && (
                        <p className="text-xs text-red-500">{chatTtsError}</p>
                      )}

                      {/* Device */}
                      <div className="space-y-1">
                        <Label className="text-xs">Device</Label>
                        <div className="flex gap-1">
                          {["cpu", "cuda:0", "cuda:1"].map((dev) => (
                            <button
                              key={dev}
                              onClick={() => {
                                updatePendingStimuli({ twitch_chat_tts_device: dev });
                                emitChanges({ twitch_chat_tts_device: dev });
                              }}
                              className={`px-2 py-0.5 text-xs rounded transition-colors ${
                                effectiveConfig.twitch_chat_tts_device === dev
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted hover:bg-muted/80"
                              }`}
                            >
                              {dev}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Voice */}
                      <div className="space-y-1">
                        <Label className="text-xs">Voice ID</Label>
                        <Input
                          value={effectiveConfig.twitch_chat_tts_voice}
                          onChange={(e) => {
                            updatePendingStimuli({ twitch_chat_tts_voice: e.target.value });
                            emitChanges({ twitch_chat_tts_voice: e.target.value });
                          }}
                          className="text-xs h-7"
                          placeholder="af_sarah"
                        />
                      </div>

                      {/* Speed */}
                      <Slider
                        label="Speed"
                        value={effectiveConfig.twitch_chat_tts_speed}
                        min={0.5}
                        max={2.0}
                        step={0.1}
                        icon={<Gauge className="h-3 w-3" />}
                        onChange={(value) => {
                          updatePendingStimuli({ twitch_chat_tts_speed: value });
                          emitChanges({ twitch_chat_tts_speed: value });
                        }}
                        unit="x"
                      />

                      {/* Format string */}
                      <div className="space-y-1">
                        <Label className="text-xs flex items-center gap-1">
                          <Type className="h-3 w-3" />
                          TTS Format
                        </Label>
                        <Input
                          value={localChatTtsFormat}
                          onChange={(e) => setLocalChatTtsFormat(e.target.value)}
                          onBlur={handleChatTtsFormatBlur}
                          className="text-xs h-7"
                          placeholder="{username} says: {message}"
                        />
                        <p className="text-xs text-muted-foreground">
                          Use {"{username}"} and {"{message}"} placeholders
                        </p>
                      </div>

                      {/* Max length */}
                      <Slider
                        label="Max TTS Length"
                        value={effectiveConfig.twitch_chat_tts_max_length}
                        min={20}
                        max={500}
                        step={10}
                        icon={<Scissors className="h-3 w-3" />}
                        onChange={(value) => {
                          updatePendingStimuli({ twitch_chat_tts_max_length: value });
                          emitChanges({ twitch_chat_tts_max_length: value });
                        }}
                        unit=" chars"
                      />
                    </div>
                  )}
                </div>
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
