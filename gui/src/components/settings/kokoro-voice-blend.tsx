"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Volume2 } from "lucide-react";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { useLauncherStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

function VoiceSlider({
  voiceId,
  weight,
  onChange,
  onCommit,
  disabled,
}: {
  voiceId: string;
  weight: number;
  onChange: (value: number) => void;
  onCommit: () => void;
  disabled: boolean;
}) {
  return (
    <div
      className={`flex flex-col items-center gap-1 min-w-[52px] ${
        disabled ? "opacity-40 pointer-events-none" : ""
      }`}
    >
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(weight * 100)}
        onChange={(e) => onChange(parseInt(e.target.value) / 100)}
        onMouseUp={onCommit}
        onTouchEnd={onCommit}
        className="h-[100px] cursor-pointer accent-primary"
        style={{
          writingMode: "vertical-lr",
          direction: "rtl",
          WebkitAppearance: "none",
          width: "20px",
        }}
      />
      <span className="text-[10px] text-muted-foreground tabular-nums">
        {weight.toFixed(2)}
      </span>
      <span className="text-[10px] text-muted-foreground truncate max-w-[48px] text-center">
        {voiceId}
      </span>
    </div>
  );
}

export function KokoroVoiceBlend() {
  const ttsLocal = useLauncherStore((s) => s.ttsLocal);
  const updateTTSLocal = useLauncherStore((s) => s.updateTTSLocal);
  const socket = getSocket();

  const isKokoro = ttsLocal.provider === "kokoro";

  const [voices, setVoices] = useState<string[]>([]);
  const [localWeights, setLocalWeights] = useState<Record<string, number>>({});
  const [localName, setLocalName] = useState(ttsLocal.voiceBlend?.name ?? "");
  const [blendEnabled, setBlendEnabled] = useState(
    ttsLocal.voiceBlend?.enabled ?? false
  );
  const [missingVoices, setMissingVoices] = useState<string[]>([]);

  const nameSyncedRef = useRef(localName);
  const weightsSyncedRef = useRef(localWeights);

  // Sync from store when config changes externally
  useEffect(() => {
    if (ttsLocal.voiceBlend) {
      setLocalName(ttsLocal.voiceBlend.name);
      nameSyncedRef.current = ttsLocal.voiceBlend.name;
      setBlendEnabled(ttsLocal.voiceBlend.enabled);
      setLocalWeights(ttsLocal.voiceBlend.weights);
      weightsSyncedRef.current = ttsLocal.voiceBlend.weights;
    }
  }, [ttsLocal.voiceBlend]);

  // Request voice list on mount
  useEffect(() => {
    if (!isKokoro) return;

    const handleVoiceList = (data: { voices: string[] }) => {
      setVoices(data.voices);
    };
    const handleBlendStatus = (data: {
      missing: string[];
      enabled: boolean;
    }) => {
      setMissingVoices(data.missing);
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const s = socket as any;
    s.on("voice_list", handleVoiceList);
    s.on("voice_blend_status", handleBlendStatus);

    const requestVoiceList = () => s.emit("get_voice_list", {});

    if (socket.connected) {
      requestVoiceList();
    }
    s.on("connect", requestVoiceList);

    return () => {
      s.off("voice_list", handleVoiceList);
      s.off("voice_blend_status", handleBlendStatus);
      s.off("connect", requestVoiceList);
    };
  }, [isKokoro, socket]);

  const pushBlendToBackend = useCallback(
    (
      overrideEnabled?: boolean,
      overrideWeights?: Record<string, number>,
      overrideName?: string
    ) => {
      const en = overrideEnabled ?? blendEnabled;
      const wts = overrideWeights ?? localWeights;
      const nm = overrideName ?? localName;

      const activeWeights: Record<string, number> = {};
      for (const [id, w] of Object.entries(wts)) {
        if (w > 0) activeWeights[id] = w;
      }
      const blend = { name: nm, enabled: en, weights: activeWeights };
      updateTTSLocal({ voiceBlend: blend });
      if (socket.connected) {
        socket.emit("set_tts_config", { voice_blend: blend });
      }
    },
    [blendEnabled, localWeights, localName, updateTTSLocal, socket]
  );

  const handleWeightChange = useCallback((voiceId: string, value: number) => {
    setLocalWeights((prev) => ({ ...prev, [voiceId]: value }));
  }, []);

  const handleWeightCommit = useCallback(() => {
    pushBlendToBackend();
  }, [pushBlendToBackend]);

  const handleNameBlur = useCallback(() => {
    if (localName !== nameSyncedRef.current) {
      pushBlendToBackend(undefined, undefined, localName);
      nameSyncedRef.current = localName;
    }
  }, [localName, pushBlendToBackend]);

  const handleToggle = useCallback(
    (checked: boolean) => {
      setBlendEnabled(checked);
      pushBlendToBackend(checked);
    },
    [pushBlendToBackend]
  );

  const handleResetAll = useCallback(() => {
    const cleared: Record<string, number> = {};
    for (const v of voices) {
      cleared[v] = 0;
    }
    setLocalWeights(cleared);
    pushBlendToBackend(undefined, cleared);
  }, [voices, pushBlendToBackend]);

  if (!isKokoro) {
    return null;
  }

  const activeVoices = Object.entries(localWeights)
    .filter(([, w]) => w > 0)
    .sort(([, a], [, b]) => b - a);

  // Show full voice list when services are running, fall back to saved weight keys when not
  const displayVoices = voices.length > 0
    ? voices
    : Object.keys(localWeights).filter((k) => localWeights[k] > 0).sort();

  return (
    <CollapsibleCard
      id="kokoro-voice-blend"
      title="Kokoro Voice Blend"
      icon={<Volume2 className="h-4 w-4" />}
    >
      <div className="space-y-4 max-w-full">
        <div className="flex items-center justify-between">
          <Label htmlFor="blend-toggle" className="text-sm">
            Use Blend
          </Label>
          <Switch
            id="blend-toggle"
            checked={blendEnabled}
            onCheckedChange={handleToggle}
          />
        </div>

        {!blendEnabled && (
          <p className="text-xs text-muted-foreground">
            Using voice from Launcher page: <strong>{ttsLocal.voice || "af_bella"}</strong>
          </p>
        )}

        <div className={!blendEnabled ? "opacity-50 pointer-events-none" : ""}>
          <div className="space-y-2">
            <Label htmlFor="blend-name">Blend Name</Label>
            <Input
              id="blend-name"
              value={localName}
              onChange={(e) => setLocalName(e.target.value)}
              onBlur={handleNameBlur}
              placeholder="My Custom Voice"
            />
          </div>
        </div>

        {missingVoices.length > 0 && (
          <div className="rounded-md bg-yellow-500/10 border border-yellow-500/30 px-3 py-2">
            <p className="text-xs text-yellow-500">
              Missing voices: {missingVoices.map((v) => `${v}.pt`).join(", ")}{" "}
              — skipped from blend
            </p>
          </div>
        )}

        {displayVoices.length > 0 ? (
          <div className={`w-full overflow-hidden ${!blendEnabled ? "opacity-50 pointer-events-none" : ""}`}>
            <div
              className="overflow-x-auto flex gap-1.5 pb-2 pt-1 w-0 min-w-full"
              style={{ scrollbarWidth: "thin" }}
            >
              {displayVoices.map((voiceId) => (
                <VoiceSlider
                  key={voiceId}
                  voiceId={voiceId}
                  weight={localWeights[voiceId] ?? 0}
                  onChange={(val) => handleWeightChange(voiceId, val)}
                  onCommit={handleWeightCommit}
                  disabled={!blendEnabled}
                />
              ))}
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Launch services to load available voices.
          </p>
        )}

        {activeVoices.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Active:{" "}
            {activeVoices
              .map(([id, w]) => `${id} (${w.toFixed(2)})`)
              .join(", ")}
          </p>
        )}

        <Button
          variant="outline"
          size="sm"
          onClick={handleResetAll}
          disabled={!blendEnabled || activeVoices.length === 0}
        >
          Reset All
        </Button>
      </div>
    </CollapsibleCard>
  );
}
