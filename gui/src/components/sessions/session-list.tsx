"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  MessageSquare,
  Calendar,
  FileText,
  User,
  RefreshCw,
  Loader2,
  Plus,
  Radio,
  Filter,
} from "lucide-react";
import { useSessionStore, useAgentStore } from "@/lib/stores";
import type { SessionInfo } from "@/types/events";
import { cn } from "@/lib/utils";

interface SessionListProps {
  onRefresh: () => void;
  onCreateSession: () => void;
}

function formatTimestamp(timestamp: string): string {
  // Format: YYYYMMDD_HHMMSS -> readable date
  if (!timestamp || timestamp.length < 15) return timestamp;
  const year = timestamp.slice(0, 4);
  const month = timestamp.slice(4, 6);
  const day = timestamp.slice(6, 8);
  const hour = timestamp.slice(9, 11);
  const min = timestamp.slice(11, 13);
  const sec = timestamp.slice(13, 15);
  return `${year}-${month}-${day} ${hour}:${min}:${sec}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function SessionCard({
  session,
  isSelected,
  isActive,
  onSelect,
}: {
  session: SessionInfo;
  isSelected: boolean;
  isActive: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border p-3 cursor-pointer transition-colors",
        "hover:bg-muted/50",
        isActive && "border-green-500/50",
        isSelected && "border-primary bg-muted/30"
      )}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Badge variant="outline" className="text-xs">
              <User className="h-3 w-3 mr-1" />
              {session.persona}
            </Badge>
            {isActive && (
              <Badge variant="outline" className="text-xs border-green-500/50 text-green-500">
                <Radio className="h-3 w-3 mr-1" />
                Active
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Calendar className="h-3 w-3" />
            <span>{formatTimestamp(session.timestamp)}</span>
          </div>
        </div>
        <div className="text-right text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <MessageSquare className="h-3 w-3" />
            <span>{session.visible_count} / {session.turn_count}</span>
          </div>
          <div className="flex items-center gap-1 mt-1">
            <FileText className="h-3 w-3" />
            <span>{formatFileSize(session.file_size)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function SessionList({ onRefresh, onCreateSession }: SessionListProps) {
  const {
    sessions,
    isLoading,
    selectedSession,
    selectSession,
    activeSessionFilepath,
    personaFilter,
    setPersonaFilter,
  } = useSessionStore();
  const servicesRunning = useAgentStore((s) => s.health) !== null;
  const currentPersonaId = useAgentStore((s) => s.config?.persona?.id) ?? null;

  // NANO-077: Filter sessions by active persona when filter is set
  const isFiltering = personaFilter !== null;
  const filteredSessions = useMemo(() => {
    if (!personaFilter) return sessions;
    return sessions.filter((s) => s.persona === personaFilter);
  }, [sessions, personaFilter]);

  const displayCount = filteredSessions.length;
  const totalCount = sessions.length;
  const hasOtherPersonaSessions = isFiltering && totalCount > displayCount;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <MessageSquare className="h-4 w-4" />
            Sessions
            {displayCount > 0 && (
              <Badge variant="secondary" className="ml-1">
                {isFiltering && totalCount !== displayCount
                  ? `${displayCount}/${totalCount}`
                  : displayCount}
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-1">
            {/* NANO-077: Persona filter toggle */}
            {totalCount > 0 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setPersonaFilter(isFiltering ? null : currentPersonaId)}
                className={cn("h-8 w-8", isFiltering && "text-primary")}
                title={isFiltering ? "Show all characters" : "Filter by current character"}
              >
                <Filter className="h-4 w-4" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={onCreateSession}
              disabled={!servicesRunning}
              className="h-8 w-8"
              title={servicesRunning ? "New Session" : "Services not running"}
            >
              <Plus className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={onRefresh}
              disabled={isLoading}
              className="h-8 w-8"
              title="Refresh"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 pt-0">
        <ScrollArea className="h-[calc(100vh-320px)]">
          {filteredSessions.length === 0 && !isLoading ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <MessageSquare className="h-8 w-8 mb-2 opacity-50" />
              <p className="text-sm">No sessions found</p>
              {hasOtherPersonaSessions ? (
                <button
                  className="text-xs mt-1 text-primary hover:underline cursor-pointer"
                  onClick={() => setPersonaFilter(null)}
                >
                  Show all characters ({totalCount} sessions)
                </button>
              ) : (
                <p className="text-xs mt-1">Start a conversation to create one</p>
              )}
            </div>
          ) : (
            <div className="space-y-2 pr-4">
              {filteredSessions.map((session) => (
                <SessionCard
                  key={session.filepath}
                  session={session}
                  isSelected={selectedSession?.filepath === session.filepath}
                  isActive={session.filepath === activeSessionFilepath}
                  onSelect={() => selectSession(session)}
                />
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
