"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Brain, Target, Layers, Shield, RotateCcw, Filter, RefreshCw, ChevronDown, ChevronRight, FileText } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useSettingsStore, selectEffectiveMemoryConfig } from "@/lib/stores";
import { useLauncherStore } from "@/lib/stores/launcher-store";
import { getSocket } from "@/lib/socket";
import { ModelCombobox, type ModelOption } from "@/components/ui/model-combobox";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  icon: React.ReactNode;
  onChange: (value: number) => void;
}

function Slider({ label, value, min, max, step, icon, onChange }: SliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          {icon}
          {label}
        </Label>
        <span className="text-sm font-mono text-muted-foreground">
          {value.toFixed(step < 1 ? 2 : 0)}
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
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

interface EditableSliderProps {
  label: string;
  value: number;
  sliderMin: number;
  sliderMax: number;
  step: number;
  icon: React.ReactNode;
  onChange: (value: number) => void;
}

function EditableSlider({ label, value, sliderMin, sliderMax, step, icon, onChange }: EditableSliderProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleStartEdit = () => {
    setEditText(value.toFixed(2));
    setIsEditing(true);
  };

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleCommit = () => {
    const parsed = parseFloat(editText);
    if (!isNaN(parsed) && parsed >= 0) {
      onChange(parsed);
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleCommit();
    } else if (e.key === "Escape") {
      setIsEditing(false);
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    if (raw === "" || raw === "." || /^\d*\.?\d*$/.test(raw)) {
      setEditText(raw);
    }
  };

  const sliderValue = Math.min(value, sliderMax);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          {icon}
          {label}
        </Label>
        {isEditing ? (
          <input
            ref={inputRef}
            type="text"
            value={editText}
            onChange={handleTextChange}
            onBlur={handleCommit}
            onKeyDown={handleKeyDown}
            className="w-16 text-sm font-mono text-right bg-muted border border-border rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary"
          />
        ) : (
          <button
            onClick={handleStartEdit}
            className="text-sm font-mono text-muted-foreground hover:text-foreground hover:bg-muted rounded px-1 py-0.5 transition-colors cursor-text"
            title="Click to edit"
          >
            {value.toFixed(2)}
          </button>
        )}
      </div>
      <input
        type="range"
        min={sliderMin}
        max={sliderMax}
        step={step}
        value={sliderValue}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{sliderMin.toFixed(1)} (loose)</span>
        <span>{sliderMax.toFixed(1)} (strict)</span>
      </div>
    </div>
  );
}

const DEFAULT_CURATION_PROMPT =
  "You are a memory deduplication judge for an AI character's long-term memory store.\n" +
  "Given a NEW memory and EXISTING similar memories, call the memory_decision tool\n" +
  "with the appropriate action. Prefer SKIP over UPDATE when the new memory adds no\n" +
  "meaningful information. Prefer UPDATE over DELETE when facts can be merged.\n" +
  "Only use DELETE when the new memory directly contradicts an existing one.";

const DEFAULT_REFLECTION_PROMPT =
  'Given only the conversation below, what are the 3 most salient high-level ' +
  'questions we can answer about the subjects discussed?\n\n' +
  'Format each as a Question and Answer pair. Separate pairs with "{qa}".\n' +
  'Output only the Q&A pairs, no explanations.\n\n' +
  'Conversation:\n{transcript}';

const DEFAULT_REFLECTION_SYSTEM_MESSAGE =
  "You are a fact extraction assistant. Extract key facts " +
  "from conversations as concise Q&A pairs. Be precise and " +
  "factual. Do not add information that is not in the " +
  "conversation.";

export function MemorySettings() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectiveMemoryConfig(store);
  const { updatePendingMemory, isSavingMemory, setSavingMemory } = store;
  const socket = getSocket();
  const [curationModels, setCurationModels] = useState<ModelOption[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const hasFetchedModelsRef = useRef(false);

  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const curationDebounceRef = useRef<NodeJS.Timeout | null>(null);

  const [reflectionExpanded, setReflectionExpanded] = useState(false);

  const emitChanges = useCallback(
    (changes: Partial<{ top_k: number; relevance_threshold: number | null; dedup_threshold: number | null; reflection_interval: number; reflection_prompt: string | null; reflection_system_message: string | null; reflection_delimiter: string }>) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        setSavingMemory(true);
        socket.emit("set_memory_config", changes);
      }, 300);
    },
    [socket, setSavingMemory]
  );

  const emitCurationChanges = useCallback(
    (changes: Record<string, unknown>) => {
      if (curationDebounceRef.current) {
        clearTimeout(curationDebounceRef.current);
      }
      curationDebounceRef.current = setTimeout(() => {
        setSavingMemory(true);
        socket.emit("set_curation_config", changes);
      }, 300);
    },
    [socket, setSavingMemory]
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (curationDebounceRef.current) clearTimeout(curationDebounceRef.current);
      if (reflectionPromptDebounceRef.current) clearTimeout(reflectionPromptDebounceRef.current);
    };
  }, []);

  const handleRelevanceThresholdChange = useCallback(
    (value: number) => {
      updatePendingMemory({ relevance_threshold: value });
      emitChanges({ relevance_threshold: value });
    },
    [updatePendingMemory, emitChanges]
  );

  const handleTopKChange = useCallback(
    (value: number) => {
      updatePendingMemory({ top_k: value });
      emitChanges({ top_k: value });
    },
    [updatePendingMemory, emitChanges]
  );

  const handleDedupThresholdChange = useCallback(
    (value: number) => {
      updatePendingMemory({ dedup_threshold: value });
      emitChanges({ dedup_threshold: value });
    },
    [updatePendingMemory, emitChanges]
  );

  const handleReflectionIntervalChange = useCallback(
    (value: number) => {
      updatePendingMemory({ reflection_interval: value });
      emitChanges({ reflection_interval: value });
    },
    [updatePendingMemory, emitChanges]
  );

  const reflectionPromptDebounceRef = useRef<NodeJS.Timeout | null>(null);

  const emitReflectionPromptChanges = useCallback(
    (changes: Partial<{ reflection_prompt: string | null; reflection_system_message: string | null; reflection_delimiter: string }>) => {
      if (reflectionPromptDebounceRef.current) {
        clearTimeout(reflectionPromptDebounceRef.current);
      }
      reflectionPromptDebounceRef.current = setTimeout(() => {
        setSavingMemory(true);
        socket.emit("set_memory_config", changes);
      }, 500);
    },
    [socket, setSavingMemory]
  );

  useEffect(() => {
    return () => {
      if (reflectionPromptDebounceRef.current) clearTimeout(reflectionPromptDebounceRef.current);
    };
  }, []);

  const handleReflectionPromptChange = useCallback(
    (value: string) => {
      const val = value || null;
      updatePendingMemory({ reflection_prompt: val });
      emitReflectionPromptChanges({ reflection_prompt: val });
    },
    [updatePendingMemory, emitReflectionPromptChanges]
  );

  const handleReflectionSystemMessageChange = useCallback(
    (value: string) => {
      const val = value || null;
      updatePendingMemory({ reflection_system_message: val });
      emitReflectionPromptChanges({ reflection_system_message: val });
    },
    [updatePendingMemory, emitReflectionPromptChanges]
  );

  const handleReflectionDelimiterChange = useCallback(
    (value: string) => {
      if (!value) return; // Don't accept empty delimiter
      updatePendingMemory({ reflection_delimiter: value });
      emitReflectionPromptChanges({ reflection_delimiter: value });
    },
    [updatePendingMemory, emitReflectionPromptChanges]
  );

  const handleResetReflectionPrompt = useCallback(() => {
    updatePendingMemory({ reflection_prompt: null, reflection_system_message: null, reflection_delimiter: "{qa}" });
    setSavingMemory(true);
    socket.emit("set_memory_config", { reflection_prompt: null, reflection_system_message: null, reflection_delimiter: "{qa}" });
  }, [updatePendingMemory, setSavingMemory, socket]);

  const curation = effectiveConfig.curation;

  // Auto-populate: if main LLM is OpenRouter and curation has no key, use the main key
  const launcherCloud = useLauncherStore((s) => s.llmCloud);
  const fallbackKey = launcherCloud.provider === "openrouter" && launcherCloud.apiKey
    ? launcherCloud.apiKey
    : null;
  const effectiveApiKey = curation.api_key || fallbackKey;

  const handleCurationToggle = useCallback(() => {
    const next = !curation.enabled;
    updatePendingMemory({ curation: { ...curation, enabled: next } });
    emitCurationChanges({ enabled: next });
  }, [curation, updatePendingMemory, emitCurationChanges]);

  const handleCurationApiKey = useCallback(
    (value: string) => {
      updatePendingMemory({ curation: { ...curation, api_key: value || null } });
      emitCurationChanges({ api_key: value || null });
    },
    [curation, updatePendingMemory, emitCurationChanges]
  );

  const handleCurationModel = useCallback(
    (modelId: string) => {
      updatePendingMemory({ curation: { ...curation, model: modelId } });
      emitCurationChanges({ model: modelId });
    },
    [curation, updatePendingMemory, emitCurationChanges]
  );

  const fetchCurationModels = useCallback(async () => {
    const key = effectiveApiKey;
    if (!key) return;

    setIsLoadingModels(true);
    setModelsError(null);

    try {
      const response = await fetch("/api/launcher/fetch-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "openrouter",
          apiKey: key,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        setModelsError(data.error || "Failed to fetch models");
        setCurationModels([]);
      } else {
        setCurationModels(data.models);
      }
    } catch (err) {
      setModelsError(err instanceof Error ? err.message : "Network error");
      setCurationModels([]);
    } finally {
      setIsLoadingModels(false);
    }
  }, [effectiveApiKey]);

  useEffect(() => {
    if (curation.enabled && effectiveApiKey && !hasFetchedModelsRef.current) {
      hasFetchedModelsRef.current = true;
      fetchCurationModels();
    }
  }, [curation.enabled, effectiveApiKey, fetchCurationModels]);

  const handleCurationPrompt = useCallback(
    (value: string) => {
      updatePendingMemory({ curation: { ...curation, prompt: value || null } });
      emitCurationChanges({ prompt: value || null });
    },
    [curation, updatePendingMemory, emitCurationChanges]
  );

  const handleResetPrompt = useCallback(() => {
    updatePendingMemory({ curation: { ...curation, prompt: null } });
    emitCurationChanges({ prompt: null });
  }, [curation, updatePendingMemory, emitCurationChanges]);

  if (!effectiveConfig.enabled) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Brain className="h-4 w-4" />
          Memory Settings
          {isSavingMemory && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <EditableSlider
          label="Relevance Threshold"
          value={effectiveConfig.relevance_threshold ?? 0.75}
          sliderMin={0.0}
          sliderMax={1.0}
          step={0.01}
          icon={<Target className="h-3 w-3" />}
          onChange={handleRelevanceThresholdChange}
        />

        <Slider
          label="Top K Results"
          value={effectiveConfig.top_k}
          min={1}
          max={20}
          step={1}
          icon={<Layers className="h-3 w-3" />}
          onChange={handleTopKChange}
        />

        <Slider
          label="Reflection Interval"
          value={effectiveConfig.reflection_interval}
          min={5}
          max={100}
          step={5}
          icon={<RefreshCw className="h-3 w-3" />}
          onChange={handleReflectionIntervalChange}
        />

        <EditableSlider
          label="Dedup Threshold"
          value={effectiveConfig.dedup_threshold ?? 0.30}
          sliderMin={0.0}
          sliderMax={2.0}
          step={0.05}
          icon={<Filter className="h-3 w-3" />}
          onChange={handleDedupThresholdChange}
        />

        <p className="text-xs text-muted-foreground">
          Relevance threshold controls memory matching strictness. Top K limits memories per prompt. Reflection interval sets how many new messages trigger a memory extraction cycle. Dedup threshold sets the L2 distance below which new memories are considered duplicates (higher = more aggressive dedup).
        </p>

        {/* NANO-104: Reflection Prompt */}
        <div className="border-t border-border pt-4 space-y-4">
          <button
            onClick={() => setReflectionExpanded(!reflectionExpanded)}
            className="flex items-center gap-2 text-sm font-medium hover:text-foreground transition-colors w-full text-left"
          >
            {reflectionExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            <FileText className="h-3 w-3" />
            Reflection Prompt
          </button>

          {reflectionExpanded && (
            <div className="space-y-3 pl-5">
              {/* Reflection Prompt Textarea */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Extraction Prompt</Label>
                  <button
                    onClick={handleResetReflectionPrompt}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    title="Reset all to defaults"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset
                  </button>
                </div>
                <textarea
                  value={effectiveConfig.reflection_prompt ?? ""}
                  onChange={(e) => handleReflectionPromptChange(e.target.value)}
                  placeholder={DEFAULT_REFLECTION_PROMPT}
                  rows={6}
                  className="w-full text-xs font-mono bg-muted border border-border rounded px-2 py-1.5 resize-y focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-muted-foreground/50"
                />
                <p className="text-xs text-muted-foreground">
                  Controls what the reflection system extracts from conversations. Must contain {"{transcript}"}. Empty = use built-in default.
                </p>
              </div>

              {/* System Message Textarea */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">System Message</Label>
                <textarea
                  value={effectiveConfig.reflection_system_message ?? ""}
                  onChange={(e) => handleReflectionSystemMessageChange(e.target.value)}
                  placeholder={DEFAULT_REFLECTION_SYSTEM_MESSAGE}
                  rows={2}
                  className="w-full text-xs font-mono bg-muted border border-border rounded px-2 py-1.5 resize-y focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-muted-foreground/50"
                />
              </div>

              {/* Delimiter Input */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Entry Delimiter</Label>
                <input
                  type="text"
                  value={effectiveConfig.reflection_delimiter}
                  onChange={(e) => handleReflectionDelimiterChange(e.target.value)}
                  className="w-full text-xs font-mono bg-muted border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
                />
                <p className="text-xs text-muted-foreground">
                  Separator between entries in LLM output. Use something unlikely to appear in natural text.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* NANO-102: Memory Curation */}
        <div className="border-t border-border pt-4 space-y-4">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2 text-sm">
              <Shield className="h-3 w-3" />
              LLM-Assisted Curation
            </Label>
            <button
              onClick={handleCurationToggle}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                curation.enabled ? "bg-primary" : "bg-muted"
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                  curation.enabled ? "translate-x-4" : "translate-x-0.5"
                }`}
              />
            </button>
          </div>

          {curation.enabled && (
            <div className="space-y-3 pl-5">
              {/* API Key */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">OpenRouter API Key</Label>
                <input
                  type="password"
                  value={curation.api_key ?? ""}
                  onChange={(e) => handleCurationApiKey(e.target.value)}
                  placeholder={fallbackKey ? "Using LLM provider key (override optional)" : "sk-or-v1-..."}
                  className="w-full text-xs font-mono bg-muted border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>

              {/* Model Selector */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Curation Model</Label>
                <ModelCombobox
                  value={curation.model}
                  onValueChange={handleCurationModel}
                  models={curationModels}
                  isLoading={isLoadingModels}
                  error={modelsError}
                  onRefresh={fetchCurationModels}
                  placeholder={effectiveApiKey ? "Select a model..." : "Enter API key first"}
                  disabled={!effectiveApiKey}
                />
              </div>

              {/* Prompt Textarea */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Curation Prompt</Label>
                  <button
                    onClick={handleResetPrompt}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    title="Reset to default"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset
                  </button>
                </div>
                <textarea
                  value={curation.prompt ?? DEFAULT_CURATION_PROMPT}
                  onChange={(e) => handleCurationPrompt(e.target.value)}
                  rows={4}
                  className="w-full text-xs font-mono bg-muted border border-border rounded px-2 py-1.5 resize-y focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>

              <p className="text-xs text-muted-foreground">
                When enabled, ambiguous near-duplicate memories are sent to a frontier model for
                ADD/SKIP/UPDATE/DELETE classification. Requires an OpenRouter API key. Falls back
                to distance-based heuristic if unavailable.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
