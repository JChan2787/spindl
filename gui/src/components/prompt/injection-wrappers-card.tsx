"use client";

import { useCallback, useRef } from "react";
import { MessageSquareText, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useSettingsStore } from "@/lib/stores";
import { selectEffectivePromptConfig } from "@/lib/stores/settings-store";
import type { PromptConfig } from "@/lib/stores/settings-store";
import { getSocket } from "@/lib/socket";

const DEFAULT_VALUES: PromptConfig = {
  rag_prefix:
    "The following are relevant memories about the user and past conversations. Use them to inform your response:",
  rag_suffix: "End of memories.",
  codex_prefix: "The following facts are always true in this context:",
  codex_suffix: "",
  example_dialogue_prefix:
    "The following are example dialogues demonstrating this character's voice, tone, and response style. Use them as style reference only — do not repeat or quote them directly:",
  example_dialogue_suffix: "End of style examples.",
  voice_state_barge_in: "The User interrupted you mid-sentence.",
  voice_state_empty_transcription: "The User made a sound but no words were detected.",
  voice_state_error: "An error occurred. Acknowledge briefly and continue.",
};

interface WrapperFieldProps {
  label: string;
  description: string;
  value: string;
  onChange: (value: string) => void;
}

function WrapperField({ label, description, value, onChange }: WrapperFieldProps) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs font-medium text-zinc-400">{label}</Label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={description}
        rows={2}
        className="w-full rounded-md border border-zinc-700 bg-zinc-800/50 px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:border-purple-500 focus:outline-none focus:ring-1 focus:ring-purple-500 resize-y"
      />
    </div>
  );
}

export function InjectionWrappersCard() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectivePromptConfig(store);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const emitChanges = useCallback(
    (changes: Partial<PromptConfig>) => {
      store.setSavingPrompt(true);

      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }

      debounceRef.current = setTimeout(() => {
        const socket = getSocket();
        if (socket) {
          socket.emit("set_prompt_config", changes);
        }
        debounceRef.current = null;
      }, 500);
    },
    [store]
  );

  const handleChange = useCallback(
    (field: keyof PromptConfig, value: string) => {
      const changes = { [field]: value };
      store.updatePendingPrompt(changes);
      emitChanges(changes);
    },
    [store, emitChanges]
  );

  const handleReset = useCallback(() => {
    store.updatePendingPrompt(DEFAULT_VALUES);
    emitChanges(DEFAULT_VALUES);
  }, [store, emitChanges]);

  return (
    <Card className="border-zinc-800 bg-zinc-900/50">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm font-medium text-zinc-300">
          <MessageSquareText className="h-4 w-4 text-purple-400" />
          Injection Wrappers
          {store.isSavingPrompt && (
            <span className="text-xs text-zinc-500">(saving...)</span>
          )}
          <button
            onClick={handleReset}
            className="ml-auto flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            title="Reset to defaults"
          >
            <RotateCcw className="h-3 w-3" />
            Reset
          </button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <p className="text-xs text-zinc-500">
            Prefix/suffix strings wrapped around RAG memories, codex entries,
            and example dialogue when injected into the system prompt.
          </p>
          <WrapperField
            label="RAG Prefix"
            description="Header before memory list"
            value={effectiveConfig.rag_prefix}
            onChange={(v) => handleChange("rag_prefix", v)}
          />
          <WrapperField
            label="RAG Suffix"
            description="Footer after memory list"
            value={effectiveConfig.rag_suffix}
            onChange={(v) => handleChange("rag_suffix", v)}
          />
          <WrapperField
            label="Codex Prefix"
            description="Header before codex entries"
            value={effectiveConfig.codex_prefix}
            onChange={(v) => handleChange("codex_prefix", v)}
          />
          <WrapperField
            label="Codex Suffix"
            description="Footer after codex entries"
            value={effectiveConfig.codex_suffix}
            onChange={(v) => handleChange("codex_suffix", v)}
          />
          <WrapperField
            label="Example Dialogue Prefix"
            description="Header before example dialogue"
            value={effectiveConfig.example_dialogue_prefix}
            onChange={(v) => handleChange("example_dialogue_prefix", v)}
          />
          <WrapperField
            label="Example Dialogue Suffix"
            description="Footer after example dialogue"
            value={effectiveConfig.example_dialogue_suffix}
            onChange={(v) => handleChange("example_dialogue_suffix", v)}
          />
          <div className="border-t border-zinc-700 pt-3 mt-3">
            <p className="text-xs text-zinc-500 mb-3">
              Voice state strings injected into the user message when a voice
              event triggers. Only applies to voice input.
            </p>
            <div className="space-y-3">
              <WrapperField
                label="Barge-In (User Interrupts)"
                description="Injected when user interrupts mid-response"
                value={effectiveConfig.voice_state_barge_in}
                onChange={(v) => handleChange("voice_state_barge_in", v)}
              />
              <WrapperField
                label="Empty Transcription (Sound, No Words)"
                description="Injected when VAD triggers but STT returns nothing"
                value={effectiveConfig.voice_state_empty_transcription}
                onChange={(v) => handleChange("voice_state_empty_transcription", v)}
              />
              <WrapperField
                label="Error"
                description="Injected when the voice pipeline errors"
                value={effectiveConfig.voice_state_error}
                onChange={(v) => handleChange("voice_state_error", v)}
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
