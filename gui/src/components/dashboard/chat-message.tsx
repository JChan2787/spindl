"use client";

import { cn } from "@/lib/utils";
import { ThoughtBubble } from "./thought-bubble";
import { CodexIndicator } from "./codex-indicator";
import { MemoryIndicator } from "./memory-indicator";
import { StimulusSourceBadge } from "./stimulus-source-badge";
import { useSettingsStore } from "@/lib/stores";
import type { ChatMessage, SentenceChunk } from "@/lib/stores/chat-store";

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

        {/* Message content — sub-bubbles or single text */}
        {!isUser && message.chunks && message.chunks.length > 0 ? (
          <div className="flex flex-col gap-1">
            {message.chunks.map((chunk, i) => (
              <SentenceBubble key={i} chunk={chunk} />
            ))}
            {!message.isFinal && (
              <span className="inline-block ml-1 opacity-50 animate-pulse">▊</span>
            )}
          </div>
        ) : (
          <p className={cn(
            "text-sm whitespace-pre-wrap break-words",
            !message.isFinal && !isUser && "animate-pulse"
          )}>
            {message.text}
            {!message.isFinal && !isUser && (
              <span className="inline-block ml-1 opacity-50">▊</span>
            )}
          </p>
        )}

        {/* Single emotion label for entire response — bottom-right */}
        {!isUser && showEmotion && message.emotion && (
          <span className="text-xs text-muted-foreground mt-1 block text-right">
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

function SentenceBubble({ chunk }: { chunk: SentenceChunk }) {
  return (
    <div className="rounded border border-border/30 bg-background/20 px-2 py-1.5">
      <p className="text-sm whitespace-pre-wrap break-words">
        {chunk.text}
      </p>
    </div>
  );
}
