"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useAgentStore, useConnectionStore, useCharacterStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import { useChatStore } from "@/lib/stores/chat-store";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Send, Zap } from "lucide-react";

// --- Voice Overlay (NANO-073c) ---
// Animated bars driven by micLevel, renders during user_speaking state.

const BAR_COUNT = 24;

function VoiceOverlay({ micLevel }: { micLevel: number }) {
  // Generate bar heights from micLevel with pseudo-random variation per bar.
  // Each bar gets a slightly different multiplier so it doesn't look uniform.
  const bars = Array.from({ length: BAR_COUNT }, (_, i) => {
    // Deterministic variation per bar index
    const variation = 0.5 + 0.5 * Math.sin(i * 1.7 + i * i * 0.3);
    const height = Math.max(4, micLevel * 100 * variation);
    return height;
  });

  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center gap-[3px] rounded-md bg-background/90 backdrop-blur-sm transition-opacity duration-200">
      {bars.map((h, i) => (
        <div
          key={i}
          className="w-[3px] rounded-full bg-purple-500 transition-[height] duration-75"
          style={{ height: `${Math.min(h, 80)}%` }}
        />
      ))}
    </div>
  );
}

// --- Stimulus Indicator (NANO-073c) ---
// Subtle overlay when the agent initiated the exchange autonomously.

function StimulusIndicator({ characterName }: { characterName: string }) {
  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center gap-2 rounded-md bg-background/80 backdrop-blur-sm transition-opacity duration-300">
      <Zap className="h-4 w-4 text-orange-500 animate-pulse" />
      <span className="text-sm text-orange-500/90 font-medium">
        {characterName} is thinking...
      </span>
    </div>
  );
}

// --- ChatInput (NANO-073c) ---
// Unified input bar with voice overlay and stimulus indicator.

export function ChatInput() {
  const { connected } = useConnectionStore();
  const { state, micLevel, stimulusActive, config } = useAgentStore();
  const { addUserMessage } = useChatStore();
  const isSwitchingCharacter = useCharacterStore((s) => s.isSwitchingCharacter);

  const [textInput, setTextInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const characterName = config?.persona?.name || "Assistant";
  const isUserSpeaking = state === "user_speaking";

  // Determine which overlay to show (voice takes priority over stimulus)
  const showVoiceOverlay = isUserSpeaking;
  const showStimulusIndicator = !isUserSpeaking && stimulusActive;

  const handleSendMessage = useCallback(() => {
    const trimmed = textInput.trim();
    if (!trimmed || isSending || !connected || isSwitchingCharacter) return;

    setIsSending(true);
    const socket = getSocket();
    socket.emit("typing_active", { active: false });
    socket.emit("send_message", { text: trimmed });
    addUserMessage(trimmed);
    setTextInput("");
    setTimeout(() => setIsSending(false), 500);
  }, [textInput, isSending, connected, isSwitchingCharacter, addUserMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    },
    [handleSendMessage]
  );

  const handleFocus = useCallback(() => {
    const socket = getSocket();
    socket.emit("typing_active", { active: true });
  }, []);

  const handleBlur = useCallback(() => {
    if (!textInput.trim()) {
      const socket = getSocket();
      socket.emit("typing_active", { active: false });
    }
  }, [textInput]);

  // Restore focus to textarea when overlays dismiss
  useEffect(() => {
    if (!showVoiceOverlay && !showStimulusIndicator && textareaRef.current) {
      // Only refocus if textarea had content (user was mid-compose)
      if (textInput.trim()) {
        textareaRef.current.focus();
      }
    }
  }, [showVoiceOverlay, showStimulusIndicator, textInput]);

  return (
    <Card>
      <CardContent className="pt-4">
        <div className="relative">
          {/* Overlays — voice takes priority */}
          {showVoiceOverlay && <VoiceOverlay micLevel={micLevel} />}
          {showStimulusIndicator && (
            <StimulusIndicator characterName={characterName} />
          )}

          {/* Text input — always in DOM, covered by overlay when active */}
          <div className="flex gap-2">
            <Textarea
              ref={textareaRef}
              placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={handleFocus}
              onBlur={handleBlur}
              disabled={!connected || isSending || isSwitchingCharacter}
              className="min-h-[48px] max-h-[120px] resize-none"
              tabIndex={showVoiceOverlay || showStimulusIndicator ? -1 : 0}
            />
            <Button
              onClick={handleSendMessage}
              disabled={
                !connected ||
                isSending ||
                !textInput.trim() ||
                showVoiceOverlay ||
                showStimulusIndicator ||
                isSwitchingCharacter
              }
              size="icon"
              className="shrink-0 self-end h-10 w-10"
            >
              {isSending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
