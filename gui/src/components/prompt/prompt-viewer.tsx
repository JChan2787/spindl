"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardAction } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { PromptMessage } from "@/types/events";
import { FileText, Copy, Check, ChevronDown, ChevronRight } from "lucide-react";

interface PromptViewerProps {
  messages: PromptMessage[];
  inputModality: string;
  stateTrigger: string | null;
  timestamp: string;
}

interface MessageCardProps {
  message: PromptMessage;
  index: number;
}

function MessageCard({ message, index }: MessageCardProps) {
  const [isOpen, setIsOpen] = useState(message.role === "user");
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const roleColors: Record<string, string> = {
    system: "bg-purple-500/10 border-purple-500/20 text-purple-400",
    user: "bg-blue-500/10 border-blue-500/20 text-blue-400",
    assistant: "bg-green-500/10 border-green-500/20 text-green-400",
  };

  const roleColor = roleColors[message.role] || roleColors.system;

  // For system prompts, parse sections
  const sections =
    message.role === "system" ? parseSystemPromptSections(message.content) : null;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className={`rounded-lg border ${roleColor}`}>
        <CollapsibleTrigger asChild>
          <div className="flex items-center justify-between p-3 cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center gap-2">
              {isOpen ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              <Badge variant="outline" className="capitalize">
                {message.role}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {message.content.length.toLocaleString()} chars
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={(e) => {
                e.stopPropagation();
                handleCopy();
              }}
            >
              {copied ? (
                <Check className="h-3 w-3 text-green-500" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </Button>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="border-t border-inherit">
            {sections ? (
              <div className="divide-y divide-border">
                {sections.map((section, i) => (
                  <div key={i} className="p-3">
                    {section.header && (
                      <div className="text-xs font-semibold uppercase text-muted-foreground mb-2">
                        {section.header}
                      </div>
                    )}
                    <pre className="text-sm whitespace-pre-wrap font-mono bg-muted/30 rounded p-2 overflow-x-auto">
                      {section.content}
                    </pre>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-3">
                <pre className="text-sm whitespace-pre-wrap font-mono bg-muted/30 rounded p-2 overflow-x-auto">
                  {message.content}
                </pre>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

interface Section {
  header: string | null;
  content: string;
}

function parseSystemPromptSections(content: string): Section[] {
  const sections: Section[] = [];
  const headerRegex = /^###\s+(.+)$/gm;
  let lastIndex = 0;
  let match;

  while ((match = headerRegex.exec(content)) !== null) {
    // Content before this header (if any, and if we've seen a header before)
    if (lastIndex > 0 && match.index > lastIndex) {
      const prevContent = content.slice(lastIndex, match.index).trim();
      if (prevContent && sections.length > 0) {
        sections[sections.length - 1].content = prevContent;
      }
    } else if (match.index > 0 && sections.length === 0) {
      // Content before first header
      const preamble = content.slice(0, match.index).trim();
      if (preamble) {
        sections.push({ header: null, content: preamble });
      }
    }

    sections.push({ header: match[1], content: "" });
    lastIndex = match.index + match[0].length;
  }

  // Content after last header
  if (lastIndex < content.length && sections.length > 0) {
    sections[sections.length - 1].content = content.slice(lastIndex).trim();
  }

  // If no headers found, return whole content
  if (sections.length === 0) {
    sections.push({ header: null, content: content });
  }

  return sections;
}

export function PromptViewer({
  messages,
  inputModality,
  stateTrigger,
  timestamp,
}: PromptViewerProps) {
  const [allCopied, setAllCopied] = useState(false);

  const handleCopyAll = async () => {
    const fullPrompt = messages
      .map((m) => `[${m.role.toUpperCase()}]\n${m.content}`)
      .join("\n\n---\n\n");
    await navigator.clipboard.writeText(fullPrompt);
    setAllCopied(true);
    setTimeout(() => setAllCopied(false), 2000);
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Full Prompt
          </CardTitle>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="outline" className="text-xs capitalize">
              {inputModality}
            </Badge>
            {stateTrigger && (
              <Badge variant="secondary" className="text-xs">
                {stateTrigger}
              </Badge>
            )}
            <span className="text-xs text-muted-foreground">
              {new Date(timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
        <CardAction>
          <Button variant="outline" size="sm" onClick={handleCopyAll}>
            {allCopied ? (
              <>
                <Check className="h-4 w-4 text-green-500" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-4 w-4" />
                Copy All
              </>
            )}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[500px] pr-4">
          <div className="space-y-3">
            {messages.map((message, index) => (
              <MessageCard key={index} message={message} index={index} />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
