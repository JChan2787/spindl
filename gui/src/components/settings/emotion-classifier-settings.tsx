"use client";

import { useCallback } from "react";
import { Brain } from "lucide-react";
import { Label } from "@/components/ui/label";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { getSocket } from "@/lib/socket";
import { useSettingsStore } from "@/lib/stores";

const CLASSIFIER_OPTIONS = [
  { value: "off", label: "Off" },
  { value: "classifier", label: "ONNX Classifier" },
] as const;

export function EmotionClassifierSettings() {
  const avatarConfig = useSettingsStore((s) => s.avatarConfig);

  const handleClassifierChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const socket = getSocket();
      socket.emit("set_avatar_config", {
        emotion_classifier: e.target.value as "classifier" | "off",
      });
    },
    [],
  );

  const handleShowInChatToggle = useCallback(() => {
    const socket = getSocket();
    socket.emit("set_avatar_config", {
      show_emotion_in_chat: !avatarConfig.show_emotion_in_chat,
    });
  }, [avatarConfig.show_emotion_in_chat]);

  return (
    <CollapsibleCard
      id="emotion-classifier"
      title="Emotion Classifier"
      icon={<Brain className="h-4 w-4" />}
    >
      {/* Emotion classifier dropdown */}
      <div className="space-y-1.5">
        <Label className="text-sm">Classification Mode</Label>
        <select
          value={avatarConfig.emotion_classifier}
          onChange={handleClassifierChange}
          disabled={!avatarConfig.enabled}
          className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {CLASSIFIER_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-muted-foreground">
          {avatarConfig.emotion_classifier === "classifier"
            ? "DistilBERT/ONNX model \u2014 requires first-time download (~67MB)"
            : "No mood events emitted to avatar"}
        </p>
      </div>

      {/* Show in chat toggle */}
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          Show in Chat
        </Label>
        <button
          onClick={handleShowInChatToggle}
          disabled={avatarConfig.emotion_classifier === "off"}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            avatarConfig.show_emotion_in_chat && avatarConfig.emotion_classifier !== "off"
              ? "bg-primary"
              : "bg-muted"
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              avatarConfig.show_emotion_in_chat && avatarConfig.emotion_classifier !== "off"
                ? "translate-x-4"
                : "translate-x-0.5"
            }`}
          />
        </button>
      </div>
      <p className="text-xs text-muted-foreground -mt-1">
        Display classified emotion below assistant messages in chat history
      </p>
    </CollapsibleCard>
  );
}
