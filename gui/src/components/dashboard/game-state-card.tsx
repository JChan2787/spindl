"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Gamepad2, Wifi, WifiOff, Layers, MessageSquare, Key, Brain, BookOpen } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsStore, selectEffectiveStimuliConfig } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import type { GameStateStatus } from "@/lib/stores/settings-store";

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

export function GameStateCard() {
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

  // Connection gate — bridge must be reachable
  const gameStateStatus = useSettingsStore((s) => s.gameStateStatus);
  const setGameStateStatus = useSettingsStore((s) => s.setGameStateStatus);
  const bridgeConnected = gameStateStatus?.connected ?? false;

  // Toggle handlers
  const handleDialogueEnabledChange = useCallback(
    (checked: boolean) => {
      updatePendingStimuli({ game_state_dialogue_enabled: checked });
      emitChanges({ game_state_dialogue_enabled: checked });
    },
    [updatePendingStimuli, emitChanges]
  );

  // Local state for textareas — emit on blur
  const [localDialoguePrompt, setLocalDialoguePrompt] = useState(effectiveConfig.game_state_dialogue_prompt_template);
  const dialoguePromptSyncedRef = useRef(effectiveConfig.game_state_dialogue_prompt_template);

  const [localSummarizerPersona, setLocalSummarizerPersona] = useState(effectiveConfig.game_state_dialogue_summarizer_persona);
  const summarizerPersonaSyncedRef = useRef(effectiveConfig.game_state_dialogue_summarizer_persona);

  const [localSummarizerModel, setLocalSummarizerModel] = useState(effectiveConfig.game_state_dialogue_summarizer_model);
  const summarizerModelSyncedRef = useRef(effectiveConfig.game_state_dialogue_summarizer_model);

  const [localSummarizerApiKey, setLocalSummarizerApiKey] = useState(effectiveConfig.game_state_dialogue_summarizer_api_key);
  const summarizerApiKeySyncedRef = useRef(effectiveConfig.game_state_dialogue_summarizer_api_key);

  // Sync local state when backend config changes
  useEffect(() => {
    if (effectiveConfig.game_state_dialogue_prompt_template !== dialoguePromptSyncedRef.current) {
      setLocalDialoguePrompt(effectiveConfig.game_state_dialogue_prompt_template);
      dialoguePromptSyncedRef.current = effectiveConfig.game_state_dialogue_prompt_template;
    }
    if (effectiveConfig.game_state_dialogue_summarizer_persona !== summarizerPersonaSyncedRef.current) {
      setLocalSummarizerPersona(effectiveConfig.game_state_dialogue_summarizer_persona);
      summarizerPersonaSyncedRef.current = effectiveConfig.game_state_dialogue_summarizer_persona;
    }
    if (effectiveConfig.game_state_dialogue_summarizer_model !== summarizerModelSyncedRef.current) {
      setLocalSummarizerModel(effectiveConfig.game_state_dialogue_summarizer_model);
      summarizerModelSyncedRef.current = effectiveConfig.game_state_dialogue_summarizer_model;
    }
    if (effectiveConfig.game_state_dialogue_summarizer_api_key !== summarizerApiKeySyncedRef.current) {
      setLocalSummarizerApiKey(effectiveConfig.game_state_dialogue_summarizer_api_key);
      summarizerApiKeySyncedRef.current = effectiveConfig.game_state_dialogue_summarizer_api_key;
    }
  }, [
    effectiveConfig.game_state_dialogue_prompt_template,
    effectiveConfig.game_state_dialogue_summarizer_persona,
    effectiveConfig.game_state_dialogue_summarizer_model,
    effectiveConfig.game_state_dialogue_summarizer_api_key,
  ]);

  const dialoguePromptMissingPlaceholder =
    localDialoguePrompt.trim().length > 0 && !localDialoguePrompt.includes("{dialogue}");

  const handleDialoguePromptBlur = useCallback(() => {
    if (dialoguePromptMissingPlaceholder) return;
    if (localDialoguePrompt !== dialoguePromptSyncedRef.current) {
      updatePendingStimuli({ game_state_dialogue_prompt_template: localDialoguePrompt });
      emitChanges({ game_state_dialogue_prompt_template: localDialoguePrompt });
      dialoguePromptSyncedRef.current = localDialoguePrompt;
    }
  }, [localDialoguePrompt, dialoguePromptMissingPlaceholder, updatePendingStimuli, emitChanges]);

  const handleSummarizerPersonaBlur = useCallback(() => {
    if (localSummarizerPersona !== summarizerPersonaSyncedRef.current) {
      updatePendingStimuli({ game_state_dialogue_summarizer_persona: localSummarizerPersona });
      emitChanges({ game_state_dialogue_summarizer_persona: localSummarizerPersona });
      summarizerPersonaSyncedRef.current = localSummarizerPersona;
    }
  }, [localSummarizerPersona, updatePendingStimuli, emitChanges]);

  const handleSummarizerModelBlur = useCallback(() => {
    if (localSummarizerModel !== summarizerModelSyncedRef.current) {
      updatePendingStimuli({ game_state_dialogue_summarizer_model: localSummarizerModel });
      emitChanges({ game_state_dialogue_summarizer_model: localSummarizerModel });
      summarizerModelSyncedRef.current = localSummarizerModel;
    }
  }, [localSummarizerModel, updatePendingStimuli, emitChanges]);

  const handleSummarizerApiKeyBlur = useCallback(() => {
    if (localSummarizerApiKey !== summarizerApiKeySyncedRef.current) {
      updatePendingStimuli({ game_state_dialogue_summarizer_api_key: localSummarizerApiKey });
      emitChanges({ game_state_dialogue_summarizer_api_key: localSummarizerApiKey });
      summarizerApiKeySyncedRef.current = localSummarizerApiKey;
    }
  }, [localSummarizerApiKey, updatePendingStimuli, emitChanges]);

  const handleDialogueBufferSizeChange = useCallback(
    (value: number) => {
      updatePendingStimuli({ game_state_dialogue_buffer_size: value });
      emitChanges({ game_state_dialogue_buffer_size: value });
    },
    [updatePendingStimuli, emitChanges]
  );

  const handleTokenBudgetChange = useCallback(
    (value: number) => {
      updatePendingStimuli({ game_state_dialogue_token_budget: value });
      emitChanges({ game_state_dialogue_token_budget: value });
    },
    [updatePendingStimuli, emitChanges]
  );

  // Poll game-state status every 2 seconds when dialogue enabled
  useEffect(() => {
    if (!effectiveConfig.game_state_dialogue_enabled) {
      setGameStateStatus(null);
      return;
    }
    const poll = () => socket.emit("request_game_state_status", {});
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [effectiveConfig.game_state_dialogue_enabled, socket, setGameStateStatus]);

  useEffect(() => {
    const handler = (data: GameStateStatus) => setGameStateStatus(data);
    socket.on("game_state_status", handler);
    return () => { socket.off("game_state_status", handler); };
  }, [socket, setGameStateStatus]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Gamepad2 className="h-4 w-4" />
          Game Bridge
          {gameStateStatus && (
            <span className="flex items-center gap-1 text-xs">
              {gameStateStatus.connected ? (
                <Wifi className="h-3 w-3 text-green-500" />
              ) : (
                <WifiOff className="h-3 w-3 text-red-500" />
              )}
            </span>
          )}
          {gameStateStatus?.protocol_version && (
            <span className="text-xs text-muted-foreground font-mono">
              v{gameStateStatus.protocol_version}
            </span>
          )}
          {isSavingStimuli && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Dialogue Enable/Disable — gated on bridge connection */}
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-sm">
            Enable Dialogue Commentary
          </Label>
          <button
            onClick={() => bridgeConnected && handleDialogueEnabledChange(!effectiveConfig.game_state_dialogue_enabled)}
            disabled={!bridgeConnected}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
              !bridgeConnected
                ? "bg-muted opacity-50 cursor-not-allowed"
                : effectiveConfig.game_state_dialogue_enabled
                  ? "bg-primary"
                  : "bg-muted"
            }`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                effectiveConfig.game_state_dialogue_enabled && bridgeConnected ? "translate-x-4" : "translate-x-0.5"
              }`}
            />
          </button>
        </div>

        {!bridgeConnected && (
          <p className="text-xs text-muted-foreground">
            Configure Game Bridge connection in Settings to enable.
          </p>
        )}

        {effectiveConfig.game_state_dialogue_enabled && bridgeConnected && (
          <div className="space-y-3">
            {/* Dialogue stimulus template */}
            <div className="space-y-1">
              <Label className="flex items-center gap-2 text-xs">
                <MessageSquare className="h-3 w-3" />
                Dialogue Stimulus Template
              </Label>
              <Textarea
                value={localDialoguePrompt}
                onChange={(e) => setLocalDialoguePrompt(e.target.value)}
                onBlur={handleDialoguePromptBlur}
                rows={3}
                className={`text-xs resize-none ${dialoguePromptMissingPlaceholder ? "border-red-500" : ""}`}
                placeholder="Template for game dialogue injection..."
                aria-invalid={dialoguePromptMissingPlaceholder}
              />
              {dialoguePromptMissingPlaceholder && (
                <p className="text-xs text-red-500">
                  Template must contain {"{dialogue}"}. Without it, buffered dialogue lines have nowhere to render.
                </p>
              )}
            </div>

            {/* Summarizer persona */}
            <div className="space-y-1">
              <Label className="flex items-center gap-2 text-xs">
                <Brain className="h-3 w-3" />
                Summarizer Persona
              </Label>
              <Textarea
                value={localSummarizerPersona}
                onChange={(e) => setLocalSummarizerPersona(e.target.value)}
                onBlur={handleSummarizerPersonaBlur}
                rows={3}
                className="text-xs resize-none"
                placeholder="Persona prompt for the rolling summarizer (leave blank for default)..."
              />
            </div>

            {/* Summarizer model */}
            <div className="space-y-1">
              <Label className="flex items-center gap-2 text-xs">
                <BookOpen className="h-3 w-3" />
                Summarizer Model (OpenRouter)
              </Label>
              <Input
                value={localSummarizerModel}
                onChange={(e) => setLocalSummarizerModel(e.target.value)}
                onBlur={handleSummarizerModelBlur}
                placeholder="anthropic/claude-sonnet-4-20250514"
                className="text-xs h-8"
              />
            </div>

            {/* OpenRouter API Key */}
            <div className="space-y-1">
              <Label className="flex items-center gap-2 text-xs">
                <Key className="h-3 w-3" />
                OpenRouter API Key
              </Label>
              <Input
                type="password"
                value={localSummarizerApiKey}
                onChange={(e) => setLocalSummarizerApiKey(e.target.value)}
                onBlur={handleSummarizerApiKeyBlur}
                placeholder="sk-or-... or ${OPENROUTER_API_KEY}"
                className="text-xs h-8"
              />
            </div>

            {/* Sliders */}
            <Slider
              label="Dialogue Buffer Size"
              value={effectiveConfig.game_state_dialogue_buffer_size}
              min={1}
              max={200}
              step={1}
              icon={<Layers className="h-3 w-3" />}
              onChange={handleDialogueBufferSizeChange}
            />

            <Slider
              label="Token Budget"
              value={effectiveConfig.game_state_dialogue_token_budget}
              min={500}
              max={10000}
              step={100}
              icon={<BookOpen className="h-3 w-3" />}
              onChange={handleTokenBudgetChange}
            />

            {/* Live dialogue feed preview */}
            {gameStateStatus && gameStateStatus.recent_lines.length > 0 && (
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  Recent Dialogue ({gameStateStatus.buffer_count} buffered)
                </Label>
                <div className="bg-muted/50 rounded p-2 text-xs font-mono max-h-24 overflow-y-auto">
                  {gameStateStatus.recent_lines.map((line, i) => (
                    <div key={i} className="truncate">{line}</div>
                  ))}
                </div>
              </div>
            )}

            {/* Current summary blob — read-only view of summarizer output */}
            {gameStateStatus && gameStateStatus.current_summary && (
              <div className="space-y-1">
                <Label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Brain className="h-3 w-3" />
                  Current Summary
                </Label>
                <div className="bg-muted/50 rounded p-2 text-xs max-h-40 overflow-y-auto whitespace-pre-wrap">
                  {gameStateStatus.current_summary}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
