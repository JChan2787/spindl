"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Eye, Pencil, Trash2, ArrowUpCircle, X, Save, Loader2 } from "lucide-react";
import { useState } from "react";
import type { MemoryDocument, MemoryCollectionType } from "@/types/events";

interface MemoryDetailPanelProps {
  memory: MemoryDocument | null;
  collection: MemoryCollectionType | null;
  isEditing: boolean;
  isNewMemory: boolean;
  editContent: string;
  onEditContent: (content: string) => void;
  onSave: () => void;
  onStartEdit: () => void;
  onCancel: () => void;
  onDelete: () => void;
  onPromote: (deleteSource: boolean) => void;
  isSaving: boolean;
}

function formatFullTimestamp(ts?: string): string {
  if (!ts) return "Unknown";
  try {
    return new Date(ts).toLocaleString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

export function MemoryDetailPanel({
  memory,
  collection,
  isEditing,
  editContent,
  onEditContent,
  onSave,
  onStartEdit,
  onCancel,
  onDelete,
  onPromote,
  isSaving,
}: MemoryDetailPanelProps) {
  const [promoteDialogOpen, setPromoteDialogOpen] = useState(false);

  // Empty state
  if (!memory && !isEditing) {
    return (
      <Card className="h-full">
        <CardContent className="flex flex-col items-center justify-center h-full py-12">
          <Eye className="h-10 w-10 text-muted-foreground mb-3" />
          <p className="text-sm text-muted-foreground">
            Select a memory to view details
          </p>
        </CardContent>
      </Card>
    );
  }

  const canPromote = collection === "flashcards" || collection === "summaries";
  const canEdit = collection === "global" || collection === "general";
  const canSave = editContent.trim().length > 0 && !isSaving;

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between py-3 px-4">
        <CardTitle className="text-base">
          {isEditing ? "Edit Memory" : "Memory Detail"}
        </CardTitle>
        <div className="flex items-center gap-1">
          {isEditing ? (
            <>
              <Button
                size="sm"
                onClick={onSave}
                disabled={!canSave}
                className="h-8"
              >
                {isSaving ? (
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-1" />
                )}
                Save
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onCancel}
                className="h-8 w-8"
              >
                <X className="h-4 w-4" />
              </Button>
            </>
          ) : (
            <>
              {canEdit && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onStartEdit}
                  className="h-8 w-8"
                >
                  <Pencil className="h-4 w-4" />
                </Button>
              )}
              {canPromote && (
                <Dialog open={promoteDialogOpen} onOpenChange={setPromoteDialogOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" className="h-8">
                      <ArrowUpCircle className="h-4 w-4 mr-1" />
                      Promote
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Promote to General Memory</DialogTitle>
                      <DialogDescription>
                        This will add the memory to your permanent General collection.
                        How would you like to handle the source?
                      </DialogDescription>
                    </DialogHeader>
                    <DialogFooter className="flex gap-2 sm:gap-0">
                      <Button
                        variant="outline"
                        onClick={() => {
                          onPromote(false);
                          setPromoteDialogOpen(false);
                        }}
                      >
                        Copy to General
                      </Button>
                      <Button
                        onClick={() => {
                          onPromote(true);
                          setPromoteDialogOpen(false);
                        }}
                      >
                        Move to General
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              )}
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-destructive hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete this memory?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will permanently remove this memory from the {collection} collection.
                      This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={onDelete}>
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Content */}
        {isEditing ? (
          <Textarea
            value={editContent}
            onChange={(e) => onEditContent(e.target.value)}
            className="min-h-[200px] resize-y"
            autoFocus
          />
        ) : (
          <div className="text-sm leading-relaxed whitespace-pre-wrap break-words bg-muted/30 rounded-lg p-3">
            {memory?.content}
          </div>
        )}

        {/* Metadata */}
        {memory && !isEditing && (
          <>
            <Separator />
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Created</span>
                <span>{formatFullTimestamp(memory.metadata?.timestamp)}</span>
              </div>
              {memory.metadata?.edited_at && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Edited</span>
                  <span>{formatFullTimestamp(memory.metadata.edited_at)}</span>
                </div>
              )}
              {memory.metadata?.promoted_at && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Promoted</span>
                  <span>{formatFullTimestamp(memory.metadata.promoted_at)}</span>
                </div>
              )}
              {memory.metadata?.source && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Source</span>
                  <Badge variant="outline" className="text-xs px-1.5 py-0">
                    {memory.metadata.source}
                  </Badge>
                </div>
              )}
              {memory.metadata?.session_id && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Session</span>
                  <span className="font-mono text-xs truncate max-w-[200px]">
                    {memory.metadata.session_id}
                  </span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-muted-foreground">ID</span>
                <span className="font-mono text-xs truncate max-w-[200px]">
                  {memory.id}
                </span>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
