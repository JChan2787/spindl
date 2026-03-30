"use client";

import { useEffect, useCallback } from "react";
import { useAgentStore, useSettingsStore } from "@/lib/stores";
import { useCharacterStore, fetchAvatar } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import { User, ChevronDown, Loader2, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

/**
 * State -> border color mapping.
 * Complete string literals — no Tailwind interpolation (NANO-041 lesson).
 */
const stateColors: Record<string, { border: string; glow: string }> = {
  idle: { border: "border-gray-500", glow: "107, 114, 128" },
  listening: { border: "border-blue-500", glow: "59, 130, 246" },
  user_speaking: { border: "border-green-500", glow: "34, 197, 94" },
  processing: { border: "border-yellow-500", glow: "234, 179, 8" },
  system_speaking: { border: "border-purple-500", glow: "168, 85, 247" },
};

const stateLabels: Record<string, string> = {
  idle: "Idle",
  listening: "Listening",
  user_speaking: "User Speaking",
  processing: "Processing",
  system_speaking: "Speaking",
};

export function CharacterPortrait() {
  const { state, config, audioLevel } = useAgentStore();
  const avatarConnected = useSettingsStore((s) => s.avatarConfig.avatar_connected);
  const {
    characters,
    avatarCache,
    setAvatar,
    isSwitchingCharacter,
    setSwitchingCharacter,
  } = useCharacterStore();

  const personaId = config?.persona?.id;
  const personaName = config?.persona?.name;

  // Fetch avatar on mount / persona change
  useEffect(() => {
    if (!personaId) return;
    if (personaId in avatarCache) return;

    fetchAvatar(personaId)
      .then((data) => setAvatar(personaId, data.image_data))
      .catch(() => setAvatar(personaId, null));
  }, [personaId, avatarCache, setAvatar]);

  // NANO-077: Switch character from dashboard dropdown
  const handleSwitchCharacter = useCallback(
    (characterId: string) => {
      if (characterId === personaId || isSwitchingCharacter) return;
      setSwitchingCharacter(true);
      const socket = getSocket();
      socket.emit("set_persona", { persona_id: characterId });
    },
    [personaId, isSwitchingCharacter, setSwitchingCharacter]
  );

  const avatarUrl = personaId ? avatarCache[personaId] : null;
  // NANO-097: Suppress portrait glow when avatar window is connected
  const isSpeaking = audioLevel > 0 && !avatarConnected;
  // Latch system_speaking color when audio is playing — state machine
  // re-enters LISTENING for barge-in support, but audio level is the truth signal.
  const effectiveState = isSpeaking ? "system_speaking" : state;
  const colors = stateColors[effectiveState] || stateColors.idle;
  const label = isSwitchingCharacter ? "Switching..." : (stateLabels[effectiveState] || "Idle");
  const isProcessing = effectiveState === "processing";

  // Audio-reactive glow: raw RMS on float32 speech is typically 0.02–0.15.
  // Amplify by 5x and clamp so speech fills the visual range.
  // Scale: 8px base -> 32px at full, opacity 0.5 -> 1.0.
  const amplified = Math.min(1, audioLevel * 5);
  const glowRadius = isSpeaking ? 8 + amplified * 24 : 0;
  const glowSpread = isSpeaking ? 2 + amplified * 8 : 0;
  const glowOpacity = isSpeaking ? 0.5 + amplified * 0.5 : 0;

  const audioGlowStyle = isSpeaking
    ? {
        boxShadow: `0 0 ${glowRadius}px ${glowSpread}px rgba(${colors.glow}, ${glowOpacity})`,
        transition: "box-shadow 0.05s ease-out",
      }
    : undefined;

  // CSS animation class — only applied when NOT speaking (audio takes over)
  const animationClass = !isSpeaking && !isProcessing
    ? `portrait-state-${state}`
    : undefined;

  // For CSS animations, set the glow color as rgba with fixed opacity
  const cssGlowColor = `rgba(${colors.glow}, 0.6)`;

  const hasMultipleCharacters = characters.length > 1;

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Portrait ring */}
      <div className="relative">
        {/* Processing: spinning conic gradient ring behind the avatar */}
        {isProcessing && !isSpeaking && (
          <div
            className="portrait-state-processing absolute -inset-1 rounded-full"
            style={
              {
                "--portrait-glow-color": cssGlowColor,
              } as React.CSSProperties
            }
          />
        )}

        {/* NANO-077: Switching overlay */}
        {isSwitchingCharacter && (
          <div className="absolute inset-0 z-10 flex items-center justify-center rounded-full bg-background/60 backdrop-blur-sm">
            <Loader2 className="size-8 animate-spin text-primary" />
          </div>
        )}

        {/* Avatar circle */}
        <div
          className={cn(
            "relative size-32 lg:size-40 rounded-full border-3 overflow-hidden bg-muted",
            colors.border,
            animationClass
          )}
          style={
            {
              ...audioGlowStyle,
              "--portrait-glow-color": cssGlowColor,
            } as React.CSSProperties
          }
        >
          {avatarUrl ? (
            <img
              src={avatarUrl}
              alt={personaName || "Character"}
              className="size-full object-cover"
            />
          ) : (
            <div className="size-full flex items-center justify-center">
              <User className="size-12 lg:size-16 text-muted-foreground" />
            </div>
          )}
        </div>
      </div>

      {/* NANO-077: Name + state label with character dropdown */}
      <div className="text-center">
        {hasMultipleCharacters ? (
          <DropdownMenu>
            <DropdownMenuTrigger
              className="flex items-center gap-1 text-sm font-medium hover:text-primary transition-colors cursor-pointer outline-none"
              disabled={isSwitchingCharacter}
            >
              {personaName || "No Character"}
              <ChevronDown className="h-3 w-3 text-muted-foreground" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="center">
              {characters.map((char) => (
                <DropdownMenuItem
                  key={char.id}
                  onClick={() => handleSwitchCharacter(char.id)}
                  disabled={char.id === personaId}
                  className="flex items-center gap-2"
                >
                  <Avatar className="h-6 w-6">
                    {avatarCache[char.id] ? (
                      <AvatarImage src={avatarCache[char.id]!} alt={char.name} />
                    ) : null}
                    <AvatarFallback className="text-xs">
                      <User className="h-3 w-3" />
                    </AvatarFallback>
                  </Avatar>
                  <span className="flex-1">{char.name}</span>
                  {char.id === personaId && (
                    <Check className="h-4 w-4 text-primary" />
                  )}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <p className="text-sm font-medium">
            {personaName || "No Character"}
          </p>
        )}
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}
