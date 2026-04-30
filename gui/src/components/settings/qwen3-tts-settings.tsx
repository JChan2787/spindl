"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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

  // Local state for text controls — emit on blur, not on every keystroke
  const [localSpeaker, setLocalSpeaker] = useState(ttsLocal.speaker);
  const speakerSyncedRef = useRef(ttsLocal.speaker);

  const [localTemperature, setLocalTemperature] = useState(ttsLocal.temperature);
  const temperatureSyncedRef = useRef(ttsLocal.temperature);

  const [localInstructTemplate, setLocalInstructTemplate] = useState(ttsLocal.instructTemplate);
  const instructTemplateSyncedRef = useRef(ttsLocal.instructTemplate);

  // Sync local state when backend/store config changes
  useEffect(() => {
    if (ttsLocal.speaker !== speakerSyncedRef.current) {
      setLocalSpeaker(ttsLocal.speaker);
      speakerSyncedRef.current = ttsLocal.speaker;
    }
    if (ttsLocal.temperature !== temperatureSyncedRef.current) {
      setLocalTemperature(ttsLocal.temperature);
      temperatureSyncedRef.current = ttsLocal.temperature;
    }
    if (ttsLocal.instructTemplate !== instructTemplateSyncedRef.current) {
      setLocalInstructTemplate(ttsLocal.instructTemplate);
      instructTemplateSyncedRef.current = ttsLocal.instructTemplate;
    }
  }, [ttsLocal.speaker, ttsLocal.temperature, ttsLocal.instructTemplate]);

  const pushToBackend = useCallback(
    (changes: Record<string, unknown>) => {
      if (socket.connected) {
        socket.emit("set_tts_config", changes);
      }
    },
    [socket]
  );

  const handleSpeakerBlur = useCallback(() => {
    if (localSpeaker !== speakerSyncedRef.current) {
      updateTTSLocal({ speaker: localSpeaker });
      pushToBackend({ speaker: localSpeaker });
      speakerSyncedRef.current = localSpeaker;
    }
  }, [localSpeaker, updateTTSLocal, pushToBackend]);

  const handleTemperatureBlur = useCallback(() => {
    const val = localTemperature;
    if (val !== temperatureSyncedRef.current) {
      updateTTSLocal({ temperature: val });
      pushToBackend({ temperature: val });
      temperatureSyncedRef.current = val;
    }
  }, [localTemperature, updateTTSLocal, pushToBackend]);

  const handleInstructTemplateBlur = useCallback(() => {
    if (localInstructTemplate !== instructTemplateSyncedRef.current) {
      updateTTSLocal({ instructTemplate: localInstructTemplate });
      pushToBackend({ instruct_template: localInstructTemplate });
      instructTemplateSyncedRef.current = localInstructTemplate;
    }
  }, [localInstructTemplate, updateTTSLocal, pushToBackend]);

  if (!isQwen3) {
    return null;
  }

  const hasEmotionPlaceholder = localInstructTemplate.includes("{emotion}");
  const showTemplateWarning =
    localInstructTemplate.length > 0 && !hasEmotionPlaceholder;

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
          <Badge variant={isTTSHealthy ? "default" : "destructive"}>
            {isTTSHealthy ? "Connected" : "Disconnected"}
          </Badge>
        </div>

        <div className="space-y-2">
          <Label htmlFor="qwen3-speaker">Speaker</Label>
          <Input
            id="qwen3-speaker"
            value={localSpeaker}
            onChange={(e) => setLocalSpeaker(e.target.value)}
            onBlur={handleSpeakerBlur}
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
            value={localTemperature}
            onChange={(e) => setLocalTemperature(parseFloat(e.target.value) || 0.6)}
            onBlur={handleTemperatureBlur}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="qwen3-instruct-template">
            Emotion Instruct Template
          </Label>
          <Textarea
            id="qwen3-instruct-template"
            value={localInstructTemplate}
            onChange={(e) => setLocalInstructTemplate(e.target.value)}
            onBlur={handleInstructTemplateBlur}
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
