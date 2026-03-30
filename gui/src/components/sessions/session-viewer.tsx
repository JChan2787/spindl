"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Eye,
  EyeOff,
  Play,
  Trash2,
  Download,
  ChevronDown,
  User,
  Bot,
  FileText,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { useSessionStore } from "@/lib/stores";
import type { Turn } from "@/types/events";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";

interface SessionViewerProps {
  onResume: (filepath: string) => void;
  onDelete: (filepath: string) => void;
  onExport: (filepath: string) => void;
  onGenerateSummary: (filepath: string) => void;
}

function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString();
  } catch {
    return iso;
  }
}

function TurnCard({ turn, defaultOpen }: { turn: Turn; defaultOpen: boolean }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const roleConfig = {
    user: {
      icon: User,
      label: "User",
      className: "border-l-blue-500",
      badgeVariant: "default" as const,
    },
    assistant: {
      icon: Bot,
      label: "Assistant",
      className: "border-l-green-500",
      badgeVariant: "secondary" as const,
    },
    summary: {
      icon: FileText,
      label: "Summary",
      className: "border-l-yellow-500",
      badgeVariant: "outline" as const,
    },
  };

  const config = roleConfig[turn.role];
  const Icon = config.icon;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div
        className={cn(
          "rounded-lg border border-l-4 bg-card",
          config.className,
          turn.hidden && "opacity-50"
        )}
      >
        <CollapsibleTrigger asChild>
          <div className="flex items-center justify-between p-3 cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center gap-2">
              <Icon className="h-4 w-4" />
              <Badge variant={config.badgeVariant} className="text-xs">
                {config.label}
              </Badge>
              <span className="text-xs text-muted-foreground">
                #{turn.turn_id}
              </span>
              {turn.hidden && (
                <Badge variant="outline" className="text-xs">
                  <EyeOff className="h-3 w-3 mr-1" />
                  Hidden
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">
                {formatTimestamp(turn.timestamp)}
              </span>
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform",
                  isOpen && "rotate-180"
                )}
              />
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="px-3 pb-3 border-t">
            <pre className="mt-2 text-sm whitespace-pre-wrap font-mono bg-muted/30 p-3 rounded-md overflow-x-auto">
              {turn.content}
            </pre>
            <div className="mt-2 text-xs text-muted-foreground">
              UUID: {turn.uuid}
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

export function SessionViewer({
  onResume,
  onDelete,
  onExport,
  onGenerateSummary,
}: SessionViewerProps) {
  const {
    selectedSession,
    selectedSessionTurns,
    isLoadingDetail,
    lastAction,
  } = useSessionStore();

  const [showHidden, setShowHidden] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);

  // Reset summary spinner when action result arrives
  useEffect(() => {
    if (lastAction.type === "summarize") {
      setIsGeneratingSummary(false);
    }
  }, [lastAction]);

  if (!selectedSession) {
    return (
      <Card className="h-full flex items-center justify-center">
        <div className="text-center text-muted-foreground">
          <Eye className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Select a session to view</p>
        </div>
      </Card>
    );
  }

  const visibleTurns = showHidden
    ? selectedSessionTurns
    : selectedSessionTurns.filter((t) => !t.hidden);

  const hiddenCount = selectedSessionTurns.filter((t) => t.hidden).length;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Badge variant="outline">{selectedSession.persona}</Badge>
              <span className="text-muted-foreground text-sm font-normal">
                {selectedSession.turn_count} turns
              </span>
            </CardTitle>
          </div>
          <div className="flex items-center gap-2">
            {hiddenCount > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowHidden(!showHidden)}
                className="h-8"
              >
                {showHidden ? (
                  <>
                    <EyeOff className="h-4 w-4 mr-1" />
                    Hide ({hiddenCount})
                  </>
                ) : (
                  <>
                    <Eye className="h-4 w-4 mr-1" />
                    Show Hidden ({hiddenCount})
                  </>
                )}
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setIsGeneratingSummary(true);
                onGenerateSummary(selectedSession.filepath);
              }}
              disabled={isGeneratingSummary}
              className="h-8"
            >
              {isGeneratingSummary ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <FileText className="h-4 w-4 mr-1" />
              )}
              Summarize
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onExport(selectedSession.filepath)}
              className="h-8"
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <Button
              variant="default"
              size="sm"
              onClick={() => onResume(selectedSession.filepath)}
              className="h-8"
            >
              <Play className="h-4 w-4 mr-1" />
              Resume
            </Button>
            {confirmDelete ? (
              <div className="flex items-center gap-1">
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => {
                    onDelete(selectedSession.filepath);
                    setConfirmDelete(false);
                  }}
                  className="h-8"
                >
                  Confirm
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setConfirmDelete(false)}
                  className="h-8"
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setConfirmDelete(true)}
                className="h-8 text-destructive hover:text-destructive"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Action feedback */}
        {lastAction.type && lastAction.filepath === selectedSession.filepath && (
          <div
            className={cn(
              "mt-2 p-2 rounded-md text-sm",
              lastAction.success
                ? "bg-green-500/10 text-green-500"
                : "bg-destructive/10 text-destructive"
            )}
          >
            {lastAction.success ? (
              <span>
                {lastAction.type === "resume"
                  ? "Session resumed successfully"
                  : lastAction.type === "summarize"
                  ? "Session summary generated"
                  : "Session deleted"}
              </span>
            ) : (
              <span className="flex items-center gap-1">
                <AlertCircle className="h-4 w-4" />
                {lastAction.error || "Action failed"}
              </span>
            )}
          </div>
        )}
      </CardHeader>
      <CardContent className="flex-1 pt-0">
        {isLoadingDetail ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <ScrollArea className="h-[calc(100vh-320px)]">
            <div className="space-y-2 pr-4">
              {visibleTurns.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <p className="text-sm">No turns in this session</p>
                </div>
              ) : (
                visibleTurns.map((turn, index) => (
                  <TurnCard
                    key={turn.uuid ?? `turn-${index}`}
                    turn={turn}
                    defaultOpen={index >= visibleTurns.length - 4}
                  />
                ))
              )}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
