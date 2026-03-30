"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
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
} from "@/components/ui/alert-dialog";
import { Plus, BookOpen, Pencil, Trash2, ArrowLeft } from "lucide-react";
import { CodexEntryForm } from "./codex-entry-form";
import { createEmptyCodexEntry } from "@/lib/stores";
import type { CharacterBookEntry, CharacterBook } from "@/types/events";

interface CharacterCodexTabProps {
  characterBook: CharacterBook | undefined;
  onChange: (book: CharacterBook) => void;
  onCharacterSave?: (updatedBook: CharacterBook) => void;
}

export function CharacterCodexTab({
  characterBook,
  onChange,
  onCharacterSave,
}: CharacterCodexTabProps) {
  const entries = characterBook?.entries || [];

  const [editingEntry, setEditingEntry] = useState<CharacterBookEntry | null>(null);
  const [isNewEntry, setIsNewEntry] = useState(false);
  const [deleteEntry, setDeleteEntry] = useState<CharacterBookEntry | null>(null);

  // Create empty book if needed
  const ensureBook = useCallback((): CharacterBook => {
    return characterBook || {
      entries: [],
      name: undefined,
      description: undefined,
      scan_depth: undefined,
      token_budget: undefined,
      recursive_scanning: undefined,
      extensions: {},
    };
  }, [characterBook]);

  // Add entry
  const handleCreate = useCallback(() => {
    const newEntry = createEmptyCodexEntry();
    // Auto-assign next ID
    const existingIds = entries.map((e) => e.id).filter((id): id is number => id !== undefined);
    newEntry.id = Math.max(...existingIds, -1) + 1;
    setEditingEntry(newEntry);
    setIsNewEntry(true);
  }, [entries]);

  // Edit entry
  const handleEdit = useCallback((entry: CharacterBookEntry) => {
    setEditingEntry({ ...entry });
    setIsNewEntry(false);
  }, []);

  // Save entry (create or update)
  const handleSave = useCallback(() => {
    if (!editingEntry) return;

    const book = ensureBook();
    let newEntries: CharacterBookEntry[];

    if (isNewEntry) {
      newEntries = [...book.entries, editingEntry];
    } else {
      newEntries = book.entries.map((e) =>
        e.id === editingEntry.id ? editingEntry : e
      );
    }

    const updatedBook = { ...book, entries: newEntries };
    onChange(updatedBook);
    setEditingEntry(null);
    setIsNewEntry(false);

    // NANO-039: Cascade to character save for disk persistence
    onCharacterSave?.(updatedBook);
  }, [editingEntry, isNewEntry, ensureBook, onChange, onCharacterSave]);

  // Delete entry
  const handleDelete = useCallback(() => {
    if (!deleteEntry) return;

    const book = ensureBook();
    const newEntries = book.entries.filter((e) => e.id !== deleteEntry.id);
    const updatedBook = { ...book, entries: newEntries };
    onChange(updatedBook);
    setDeleteEntry(null);

    // NANO-039: Cascade to character save for disk persistence
    onCharacterSave?.(updatedBook);
  }, [deleteEntry, ensureBook, onChange, onCharacterSave]);

  // Toggle enabled
  const handleToggleEnabled = useCallback(
    (entry: CharacterBookEntry, enabled: boolean) => {
      const book = ensureBook();
      const newEntries = book.entries.map((e) =>
        e.id === entry.id ? { ...e, enabled } : e
      );
      const updatedBook = { ...book, entries: newEntries };
      onChange(updatedBook);

      // NANO-039: Cascade to character save for disk persistence
      onCharacterSave?.(updatedBook);
    },
    [ensureBook, onChange, onCharacterSave]
  );

  // Sort entries by insertion_order
  const sortedEntries = [...entries].sort(
    (a, b) => a.insertion_order - b.insertion_order
  );

  // If editing, show the form
  if (editingEntry !== null) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setEditingEntry(null);
              setIsNewEntry(false);
            }}
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to List
          </Button>
          <span className="text-sm text-muted-foreground">
            {isNewEntry ? "New Codex Entry" : "Edit Codex Entry"}
          </span>
        </div>
        <CodexEntryForm
          entry={editingEntry}
          isNew={isNewEntry}
          isSaving={false}
          error={null}
          onChange={setEditingEntry}
          onSave={handleSave}
          onCancel={() => {
            setEditingEntry(null);
            setIsNewEntry(false);
          }}
        />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          <span className="font-medium">
            {entries.length} {entries.length === 1 ? "Entry" : "Entries"}
          </span>
        </div>
        <Button size="sm" onClick={handleCreate}>
          <Plus className="h-4 w-4 mr-1" />
          Add Entry
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        Character-specific codex entries that activate based on keywords in
        conversation. These are embedded in the character card.
      </p>

      {/* Entry List */}
      <ScrollArea className="h-[calc(100vh-520px)]">
        {sortedEntries.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <BookOpen className="h-12 w-12 mb-2 opacity-50" />
            <p>No codex entries</p>
            <p className="text-sm">Add entries to inject context based on keywords</p>
          </div>
        ) : (
          <div className="space-y-2">
            {sortedEntries.map((entry) => (
              <div
                key={entry.id}
                className={`p-3 rounded-lg border ${
                  !entry.enabled ? "opacity-60" : ""
                }`}
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
                      {entry.content.slice(0, 60)}
                      {entry.content.length > 60 ? "..." : ""}
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={entry.enabled}
                      onCheckedChange={(checked) =>
                        handleToggleEnabled(entry, checked)
                      }
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleEdit(entry)}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setDeleteEntry(entry)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Delete Confirmation */}
      <AlertDialog
        open={deleteEntry !== null}
        onOpenChange={(open: boolean) => {
          if (!open) setDeleteEntry(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Codex Entry</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;
              {deleteEntry?.name || `Entry #${deleteEntry?.id}`}&quot;? This
              action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
