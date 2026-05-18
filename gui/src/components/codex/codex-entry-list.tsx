"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Plus, RefreshCw, BookOpen, Loader2, ChevronRight } from "lucide-react";
import type { CharacterBookEntry, CodexVolume } from "@/types/events";

const VOLUME_ACCENT_COLORS = [
  "border-l-violet-500",
  "border-l-emerald-500",
  "border-l-sky-500",
  "border-l-amber-500",
  "border-l-rose-500",
  "border-l-teal-500",
  "border-l-indigo-500",
  "border-l-orange-500",
];

interface CodexEntryListProps {
  entries: CharacterBookEntry[];
  volumes: CodexVolume[];
  isLoading: boolean;
  selectedEntryId: number | null;
  onSelect: (entry: CharacterBookEntry) => void;
  onCreate: () => void;
  onRefresh: () => void;
  onToggleEnabled?: (entry: CharacterBookEntry, enabled: boolean) => void;
  onToggleVolume?: (volumeId: string, enabled: boolean) => void;
}

export function CodexEntryList({
  entries,
  volumes,
  isLoading,
  selectedEntryId,
  onSelect,
  onCreate,
  onRefresh,
  onToggleEnabled,
  onToggleVolume,
}: CodexEntryListProps) {
  const sortedVolumes = [...volumes].sort(
    (a, b) => a.insertion_order - b.insertion_order
  );

  const entriesByVolume = (volumeId: string) =>
    entries
      .filter((e) => (e.volume_id || "vol_default") === volumeId)
      .sort((a, b) => a.insertion_order - b.insertion_order);

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
          {entries.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <BookOpen className="h-12 w-12 mb-2 opacity-50" />
              <p>No codex entries yet</p>
              <p className="text-sm">Create one to add dynamic context</p>
            </div>
          ) : (
            <div className="space-y-1">
              {sortedVolumes.map((vol, idx) => {
                const volEntries = entriesByVolume(vol.id);
                return (
                  <VolumeSection
                    key={vol.id}
                    volume={vol}
                    entries={volEntries}
                    accentColor={VOLUME_ACCENT_COLORS[idx % VOLUME_ACCENT_COLORS.length]}
                    selectedEntryId={selectedEntryId}
                    onSelect={onSelect}
                    onToggleEnabled={onToggleEnabled}
                    onToggleVolume={onToggleVolume}
                  />
                );
              })}
            </div>
          )}
        </CardContent>
      </ScrollArea>
    </Card>
  );
}

interface VolumeSectionProps {
  volume: CodexVolume;
  entries: CharacterBookEntry[];
  accentColor: string;
  selectedEntryId: number | null;
  onSelect: (entry: CharacterBookEntry) => void;
  onToggleEnabled?: (entry: CharacterBookEntry, enabled: boolean) => void;
  onToggleVolume?: (volumeId: string, enabled: boolean) => void;
}

function VolumeSection({
  volume,
  entries,
  accentColor,
  selectedEntryId,
  onSelect,
  onToggleEnabled,
  onToggleVolume,
}: VolumeSectionProps) {
  const [open, setOpen] = useState(true);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div
        className={`flex items-center gap-2 px-3 py-2 rounded-md bg-muted/30 border-l-3 ${accentColor} ${
          !volume.enabled ? "opacity-60" : ""
        }`}
      >
        <CollapsibleTrigger asChild>
          <button className="flex items-center gap-2 flex-1 min-w-0 text-left">
            <ChevronRight
              className={`h-4 w-4 shrink-0 transition-transform duration-200 ${
                open ? "rotate-90" : ""
              }`}
            />
            <span className="font-medium text-sm truncate">{volume.name}</span>
            <Badge variant="secondary" className="text-xs shrink-0">
              {entries.length}
            </Badge>
          </button>
        </CollapsibleTrigger>
        {onToggleVolume && (
          <Switch
            checked={volume.enabled}
            onCheckedChange={(checked) => onToggleVolume(volume.id, checked)}
            onClick={(e) => e.stopPropagation()}
          />
        )}
      </div>

      <CollapsibleContent className="overflow-hidden">
        {entries.length === 0 ? (
          <div className="flex items-center gap-2 py-4 px-6 text-muted-foreground">
            <BookOpen className="h-4 w-4 opacity-50" />
            <span className="text-sm">No entries in this volume</span>
          </div>
        ) : (
          <div className="space-y-1 py-1 pl-2">
            {entries.map((entry) => (
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
      </CollapsibleContent>
    </Collapsible>
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

          <p className="text-xs text-muted-foreground truncate">
            {entry.content.slice(0, 80)}
            {entry.content.length > 80 ? "..." : ""}
          </p>

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
