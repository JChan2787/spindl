"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plus, RefreshCw, BookOpen, Loader2 } from "lucide-react";
import type { CharacterBookEntry } from "@/types/events";

interface CodexEntryListProps {
  entries: CharacterBookEntry[];
  isLoading: boolean;
  selectedEntryId: number | null;
  onSelect: (entry: CharacterBookEntry) => void;
  onCreate: () => void;
  onRefresh: () => void;
  onToggleEnabled?: (entry: CharacterBookEntry, enabled: boolean) => void;
}

export function CodexEntryList({
  entries,
  isLoading,
  selectedEntryId,
  onSelect,
  onCreate,
  onRefresh,
  onToggleEnabled,
}: CodexEntryListProps) {
  // Sort entries by insertion_order
  const sortedEntries = [...entries].sort(
    (a, b) => a.insertion_order - b.insertion_order
  );

  return (
    <Card>
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          <span className="font-medium">
            {entries.length} {entries.length === 1 ? "Entry" : "Entries"}
          </span>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
          <Button size="sm" onClick={onCreate}>
            <Plus className="h-4 w-4 mr-1" />
            Add Entry
          </Button>
        </div>
      </div>

      <ScrollArea className="h-[calc(100vh-320px)]">
        <CardContent className="p-2">
          {sortedEntries.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <BookOpen className="h-12 w-12 mb-2 opacity-50" />
              <p>No codex entries yet</p>
              <p className="text-sm">Create one to add dynamic context</p>
            </div>
          ) : (
            <div className="space-y-2">
              {sortedEntries.map((entry) => (
                <CodexEntryCard
                  key={entry.id}
                  entry={entry}
                  isSelected={entry.id === selectedEntryId}
                  onSelect={() => onSelect(entry)}
                  onToggleEnabled={onToggleEnabled}
                />
              ))}
            </div>
          )}
        </CardContent>
      </ScrollArea>
    </Card>
  );
}

interface CodexEntryCardProps {
  entry: CharacterBookEntry;
  isSelected: boolean;
  onSelect: () => void;
  onToggleEnabled?: (entry: CharacterBookEntry, enabled: boolean) => void;
}

function CodexEntryCard({
  entry,
  isSelected,
  onSelect,
  onToggleEnabled,
}: CodexEntryCardProps) {
  return (
    <div
      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
        isSelected
          ? "bg-accent border-accent-foreground/20"
          : "hover:bg-muted/50"
      } ${!entry.enabled ? "opacity-60" : ""}`}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium truncate">
              {entry.name || `Entry #${entry.id}`}
            </span>
            {entry.constant && (
              <Badge variant="outline" className="text-xs">
                Constant
              </Badge>
            )}
          </div>

          {/* Keywords preview */}
          {(entry.keys || []).length > 0 && (
            <div className="flex flex-wrap gap-1 mb-1">
              {entry.keys.slice(0, 3).map((key) => (
                <Badge
                  key={key}
                  variant="secondary"
                  className="text-xs font-mono"
                >
                  {key.length > 15 ? key.slice(0, 15) + "..." : key}
                </Badge>
              ))}
              {entry.keys.length > 3 && (
                <Badge variant="secondary" className="text-xs">
                  +{entry.keys.length - 3}
                </Badge>
              )}
            </div>
          )}

          {/* Content preview */}
          <p className="text-xs text-muted-foreground truncate">
            {entry.content.slice(0, 80)}
            {entry.content.length > 80 ? "..." : ""}
          </p>

          {/* Timing indicators */}
          <div className="flex gap-2 mt-1">
            {entry.sticky !== undefined && entry.sticky > 0 && (
              <Badge variant="outline" className="text-xs">
                Sticky: {entry.sticky}
              </Badge>
            )}
            {entry.cooldown !== undefined && entry.cooldown > 0 && (
              <Badge variant="outline" className="text-xs">
                CD: {entry.cooldown}
              </Badge>
            )}
            {entry.delay !== undefined && entry.delay > 0 && (
              <Badge variant="outline" className="text-xs">
                Delay: {entry.delay}
              </Badge>
            )}
          </div>
        </div>

        {/* Enabled toggle */}
        {onToggleEnabled && (
          <Switch
            checked={entry.enabled}
            onCheckedChange={(checked) => onToggleEnabled(entry, checked)}
            onClick={(e) => e.stopPropagation()}
          />
        )}
      </div>
    </div>
  );
}
