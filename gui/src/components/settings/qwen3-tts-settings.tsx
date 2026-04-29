"use client";

import { Volume2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useLauncherStore } from "@/lib/stores";
import { useAgentStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

export function Qwen3TTSSettings() {
  const ttsLocal = useLauncherStore((s) => s.ttsLocal);
  const updateTTSLocal = useLauncherStore((s) => s.updateTTSLocal);
  const health = useAgentStore((s) => s.health);
  const socket = getSocket();

  const isQwen3 = ttsLocal.provider === "qwen3";
  const isTTSHealthy = health?.tts === true;

  if (!isQwen3 || !isTTSHealthy) {
    return null;
  }

  const pushToBackend = (changes: Record<string, unknown>) => {
    if (socket.connected) {
      socket.emit("set_tts_config", changes);
    }
  };

  const hasEmotionPlaceholder = ttsLocal.instructTemplate.includes("{emotion}");
  const showTemplateWarning =
    ttsLocal.instructTemplate.length > 0 && !hasEmotionPlaceholder;

  return (
    <CollapsibleCard
      id="qwen3-tts"
      title="Qwen3-TTS"
      icon={<Volume2 className="h-4 w-4" />}
    >
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {ttsLocal.host}:{ttsLocal.port}
          </span>
          <Badge variant="default">Connected</Badge>
        </div>

        <div className="space-y-2">
          <Label htmlFor="qwen3-speaker">Speaker</Label>
          <Input
            id="qwen3-speaker"
            value={ttsLocal.speaker}
            onChange={(e) => {
              updateTTSLocal({ speaker: e.target.value });
              pushToBackend({ speaker: e.target.value });
            }}
            placeholder="danny"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="qwen3-temperature">Temperature</Label>
          <Input
            id="qwen3-temperature"
            type="number"
            step={0.1}
            min={0}
            max={2}
            value={ttsLocal.temperature}
            onChange={(e) => {
              const val = parseFloat(e.target.value) || 0.6;
              updateTTSLocal({ temperature: val });
              pushToBackend({ temperature: val });
            }}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="qwen3-instruct-template">
            Emotion Instruct Template
          </Label>
          <Textarea
            id="qwen3-instruct-template"
            value={ttsLocal.instructTemplate}
            onChange={(e) => {
              updateTTSLocal({ instructTemplate: e.target.value });
              pushToBackend({ instruct_template: e.target.value });
            }}
            placeholder="Express the following with a {emotion} tone."
            rows={2}
          />
          {showTemplateWarning && (
            <p className="text-xs text-yellow-500">
              Template is set but missing {"{emotion}"} placeholder.
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            Leave empty to disable per-sentence emotion instruct.
          </p>
        </div>
      </div>
    </CollapsibleCard>
  );
}
