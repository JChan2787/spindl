"use client";

/**
 * NANO-114: Message Array Preview
 *
 * Renders the role-array structure sent to strict-chat-template providers
 * (Gemma-3 / Gemma-4 via llama.cpp --jinja). When the active provider's
 * `supports_role_history` capability is true, history turns are spliced
 * between the main system prompt and the current user turn rather than
 * flattened into a [RECENT_HISTORY] block. This viewer makes that
 * architecture visible in the Workshop.
 *
 * Layout: [system] → [summary (if present)] → [user/assistant turns] → [current user]
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { PromptMessage } from "@/types/events";
import { MessageSquare, User, Bot, Cog } from "lucide-react";

interface MessageArrayPreviewProps {
  messages: PromptMessage[];
}

const ROLE_META: Record<
  PromptMessage["role"],
  { label: string; icon: typeof User; badgeClass: string; borderClass: string }
> = {
  system: {
    label: "system",
    icon: Cog,
    badgeClass: "bg-slate-500/15 text-slate-400 border-slate-500/40",
    borderClass: "border-l-slate-500",
  },
  user: {
    label: "user",
    icon: User,
    badgeClass: "bg-blue-500/15 text-blue-400 border-blue-500/40",
    borderClass: "border-l-blue-500",
  },
  assistant: {
    label: "assistant",
    icon: Bot,
    badgeClass: "bg-purple-500/15 text-purple-400 border-purple-500/40",
    borderClass: "border-l-purple-500",
  },
};

function truncate(text: string, max: number = 400): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\u2026`;
}

export function MessageArrayPreview({ messages }: MessageArrayPreviewProps) {
  const roleCounts = messages.reduce<Record<string, number>>((acc, m) => {
    acc[m.role] = (acc[m.role] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <MessageSquare className="h-4 w-4" />
          Message Array
        </CardTitle>
        <div className="flex items-center gap-2 mt-1 flex-wrap">
          <Badge variant="outline" className="text-xs">
            {messages.length} total
          </Badge>
          {Object.entries(roleCounts).map(([role, count]) => (
            <Badge
              key={role}
              variant="secondary"
              className="text-xs capitalize"
            >
              {role}: {count}
            </Badge>
          ))}
          <span className="text-xs text-muted-foreground ml-auto">
            history spliced — position anchored by chat template
          </span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[400px]">
          <div className="space-y-2 p-3">
            {messages.map((msg, idx) => {
              const meta = ROLE_META[msg.role];
              const Icon = meta.icon;
              const isCurrentUser =
                msg.role === "user" && idx === messages.length - 1;

              return (
                <div
                  key={idx}
                  className={`
                    rounded-md border-l-4 ${meta.borderClass} bg-accent/20
                    px-3 py-2 space-y-1
                  `}
                >
                  <div className="flex items-center gap-2">
                    <Icon className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    <Badge
                      variant="outline"
                      className={`text-[10px] font-normal ${meta.badgeClass}`}
                    >
                      {meta.label}
                    </Badge>
                    <span className="text-[10px] text-muted-foreground font-mono">
                      [{idx}]
                    </span>
                    {isCurrentUser && (
                      <Badge
                        variant="outline"
                        className="text-[10px] font-normal border-amber-500/40 text-amber-500"
                      >
                        current turn
                      </Badge>
                    )}
                  </div>
                  <pre className="text-xs font-mono whitespace-pre-wrap break-words text-foreground/90 leading-relaxed">
                    {truncate(msg.content)}
                  </pre>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
