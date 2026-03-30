"use client";

import { cn } from "@/lib/utils";
import { ThoughtBubble } from "./thought-bubble";
import { CodexIndicator } from "./codex-indicator";
import { MemoryIndicator } from "./memory-indicator";
import { StimulusSourceBadge } from "./stimulus-source-badge";
import { useSettingsStore } from "@/lib/stores";
import type { ChatMessage } from "@/lib/stores/chat-store";

interface ChatMessageBubbleProps {
  message: ChatMessage;
  characterName?: string;
}

export function ChatMessageBubble({ message, characterName }: ChatMessageBubbleProps) {
  const isUser = message.role === "user";
  const showEmotion = useSettingsStore((s) => s.avatarConfig.show_emotion_in_chat);
  const timestamp = message.timestamp
    ? new Date(message.timestamp).toLocaleTimeString()
    : "";

  return (
    <div
      className={cn(
        "flex flex-col gap-1",
        isUser ? "items-end" : "items-start"
      )}
    >
      {/* Reasoning bubble above assistant message */}
      {!isUser && message.reasoning && (
        <div className="w-full max-w-[85%]">
          <ThoughtBubble reasoning={message.reasoning} />
        </div>
      )}

      <div
        className={cn(
          "max-w-[85%] rounded-lg px-3 py-2",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted/50"
        )}
      >
        {/* Role label */}
        <div className="flex items-center justify-between gap-2 mb-0.5">
          <span className="text-xs font-medium opacity-70">
            {isUser ? "You" : characterName || "Assistant"}
          </span>
          {/* Assistant metadata badges */}
          {!isUser && (
            <div className="flex items-center gap-1">
              <StimulusSourceBadge source={message.stimulusSource} />
              <MemoryIndicator memories={message.retrievedMemories || []} />
              <CodexIndicator entries={message.activatedCodexEntries || []} />
            </div>
          )}
        </div>

        {/* Message text */}
        <p className={cn(
          "text-sm whitespace-pre-wrap break-words",
          !message.isFinal && !isUser && "animate-pulse"
        )}>
          {message.text}
          {!message.isFinal && !isUser && (
            <span className="inline-block ml-1 opacity-50">▊</span>
          )}
        </p>

        {/* Emotion classifier tag (NANO-094) — display-only */}
        {!isUser && showEmotion && message.emotion && (
          <span className="text-xs text-muted-foreground mt-0.5 block">
            {message.emotion}
            {message.emotionConfidence != null && ` \u2014 ${Math.round(message.emotionConfidence * 100)}%`}
          </span>
        )}

        {/* Timestamp */}
        {timestamp && (
          <span className="text-[10px] opacity-50 mt-0.5 block">
            {timestamp}
          </span>
        )}
      </div>
    </div>
  );
}
