"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Globe, Database, Clock, Plus, Trash2, RefreshCw, Loader2 } from "lucide-react";
import { MemoryCard } from "./memory-card";
import type { MemoryDocument, MemoryCollectionType } from "@/types/events";

interface MemoryListProps {
  memories: MemoryDocument[];
  isLoading: boolean;
  collection: MemoryCollectionType;
  selectedMemoryId: string | null;
  onSelect: (memory: MemoryDocument) => void;
  onRefresh: () => void;
  onCreateNew?: () => void;
  onClearAll?: () => void;
}

const collectionIcons: Record<MemoryCollectionType, React.ReactNode> = {
  global: <Globe className="h-5 w-5" />,
  general: <Database className="h-5 w-5" />,
  flashcards: <Clock className="h-5 w-5" />,
  summaries: <Clock className="h-5 w-5" />,
};

const collectionLabels: Record<MemoryCollectionType, string> = {
  global: "Global Memories",
  general: "Character Memories",
  flashcards: "Flash Cards",
  summaries: "Session Summaries",
};

const emptyMessages: Record<MemoryCollectionType, string> = {
  global: "No global memories yet. Add cross-character facts here.",
  general: "No character memories yet. Escalate entries from Sessions or add manually.",
  flashcards: "No flash cards generated yet. They appear automatically after conversations.",
  summaries: "No session summaries yet. Generate one from the Sessions page.",
};

export function MemoryList({
  memories,
  isLoading,
  collection,
  selectedMemoryId,
  onSelect,
  onRefresh,
  onCreateNew,
  onClearAll,
}: MemoryListProps) {
  // Sort by timestamp descending (newest first)
  const sorted = [...memories].sort((a, b) => {
    const ta = a.metadata?.timestamp || "";
    const tb = b.metadata?.timestamp || "";
    return tb.localeCompare(ta);
  });

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between py-3 px-4">
        <CardTitle className="flex items-center gap-2 text-base">
          {collectionIcons[collection]}
          {collectionLabels[collection]}
          <span className="text-muted-foreground font-normal text-sm">
            ({memories.length})
          </span>
        </CardTitle>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            disabled={isLoading}
            className="h-8 w-8"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
          {collection === "flashcards" && onClearAll && memories.length > 0 && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="h-8">
                  <Trash2 className="h-4 w-4 mr-1" />
                  Clear All
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Clear all flash cards?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will permanently delete all {memories.length} flash cards.
                    This action cannot be undone. Consider promoting important ones to General first.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={onClearAll}>
                    Clear All
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
          {(collection === "global" || collection === "general") && onCreateNew && (
            <Button size="sm" onClick={onCreateNew} className="h-8">
              <Plus className="h-4 w-4 mr-1" />
              Add Memory
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="px-2 pb-2">
        <ScrollArea className="h-[calc(100vh-320px)]">
          {sorted.length === 0 && !isLoading ? (
            <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
              {collectionIcons[collection]}
              <p className="text-sm text-muted-foreground mt-3 max-w-xs">
                {emptyMessages[collection]}
              </p>
            </div>
          ) : (
            <div className="space-y-1 px-2">
              {sorted.map((memory) => (
                <MemoryCard
                  key={memory.id}
                  memory={memory}
                  collection={collection}
                  isSelected={memory.id === selectedMemoryId}
                  onSelect={() => onSelect(memory)}
                />
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
