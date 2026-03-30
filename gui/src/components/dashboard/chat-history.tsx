"use client";

import { useEffect, useRef, useCallback } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatStore } from "@/lib/stores/chat-store";
import { useAgentStore } from "@/lib/stores/agent-store";
import { ChatMessageBubble } from "./chat-message";

interface ChatHistoryProps {
  characterName?: string;
}

export function ChatHistory({ characterName }: ChatHistoryProps) {
  const messages = useChatStore((s) => s.messages);
  const config = useAgentStore((s) => s.config);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const userScrolledUp = useRef(false);

  const name = characterName || config?.persona?.name || "Assistant";

  // NANO-074c: Auto-scroll within the ScrollArea viewport, not the page.
  // scrollIntoView() bubbles to the nearest scrollable ancestor (often the page).
  useEffect(() => {
    if (!userScrolledUp.current) {
      const container = scrollContainerRef.current;
      const viewport = container?.querySelector("[data-slot='scroll-area-viewport']");
      if (viewport) {
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: "smooth" });
      }
    }
  }, [messages]);

  // Detect if user has scrolled up (disengage auto-scroll)
  const handleScroll = useCallback((e: Event) => {
    const viewport = e.target as HTMLElement;
    if (!viewport) return;

    const { scrollTop, scrollHeight, clientHeight } = viewport;
    // Consider "at bottom" if within 60px of the bottom
    const atBottom = scrollHeight - scrollTop - clientHeight < 60;
    userScrolledUp.current = !atBottom;
  }, []);

  // Attach scroll listener to the ScrollArea viewport
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    // The ScrollArea viewport is the first child with data-slot="scroll-area-viewport"
    const viewport = container.querySelector("[data-slot='scroll-area-viewport']");
    if (viewport) {
      viewport.addEventListener("scroll", handleScroll);
      return () => viewport.removeEventListener("scroll", handleScroll);
    }
  }, [handleScroll]);

  if (messages.length === 0) {
    return (
      <div className="flex items-center justify-center h-[300px] text-sm text-muted-foreground italic">
        No messages yet. Send a message or speak to begin.
      </div>
    );
  }

  return (
    <div ref={scrollContainerRef}>
      <ScrollArea className="h-[400px]">
        <div className="space-y-3 p-3">
          {messages.map((message) => (
            <ChatMessageBubble
              key={message.id}
              message={message}
              characterName={name}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
