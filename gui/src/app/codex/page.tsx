"use client";

import { useEffect, useCallback } from "react";
import {
  useCodexStore,
  fetchGlobalCodex,
  createCodexEntryApi,
  updateCodexEntryApi,
  deleteCodexEntryApi,
} from "@/lib/stores";
import { CodexEntryList, CodexEntryForm } from "@/components/codex";
import type { CharacterBookEntry } from "@/types/events";

export default function CodexPage() {
  const {
    globalEntries,
    globalCodexName,
    isLoadingGlobal,
    editingEntry,
    editingEntryCharacterId,
    isNewEntry,
    lastAction,
    setLoadingGlobal,
    setGlobalCodex,
    startEditEntry,
    startNewEntry,
    updateEditingEntry,
    cancelEditEntry,
    clearActionResult,
    setActionResult,
    setActionError,
    addEntry,
    updateEntry,
    removeEntry,
  } = useCodexStore();

  // Fetch global codex on mount via REST API
  useEffect(() => {
    const loadGlobalCodex = async () => {
      setLoadingGlobal(true);
      try {
        const data = await fetchGlobalCodex();
        setGlobalCodex(data.entries, data.name);
      } catch (error) {
        console.error("Failed to fetch global codex:", error);
        setActionError(error instanceof Error ? error.message : "Failed to fetch global codex");
        setLoadingGlobal(false);
      }
    };
    loadGlobalCodex();
  }, [setLoadingGlobal, setGlobalCodex, setActionError]);

  // Clear action result after delay
  useEffect(() => {
    if (lastAction.success !== null) {
      const timer = setTimeout(clearActionResult, 3000);
      return () => clearTimeout(timer);
    }
  }, [lastAction.success, clearActionResult]);

  // Handlers
  const handleRefresh = useCallback(async () => {
    setLoadingGlobal(true);
    try {
      const data = await fetchGlobalCodex();
      setGlobalCodex(data.entries, data.name);
    } catch (error) {
      console.error("Failed to refresh global codex:", error);
      setActionError(error instanceof Error ? error.message : "Failed to refresh global codex");
      setLoadingGlobal(false);
    }
  }, [setLoadingGlobal, setGlobalCodex, setActionError]);

  const handleCreate = useCallback(() => {
    startNewEntry(null); // null = global codex
  }, [startNewEntry]);

  const handleSave = useCallback(async () => {
    if (!editingEntry) return;

    try {
      if (isNewEntry) {
        const result = await createCodexEntryApi(editingEntry, null);
        const newEntry = { ...editingEntry, id: result.entry_id };
        addEntry(newEntry, null);
        setActionResult("create", result.entry_id, null, true);
      } else if (editingEntry.id !== undefined) {
        await updateCodexEntryApi(editingEntry, editingEntry.id, null);
        updateEntry(editingEntry, null);
        setActionResult("update", editingEntry.id, null, true);
      }
    } catch (error) {
      console.error("Failed to save codex entry:", error);
      setActionError(error instanceof Error ? error.message : "Failed to save codex entry");
    }
  }, [editingEntry, isNewEntry, addEntry, updateEntry, setActionResult, setActionError]);

  const handleDelete = useCallback(async () => {
    if (!editingEntry || editingEntry.id === undefined) return;

    try {
      await deleteCodexEntryApi(editingEntry.id, null);
      removeEntry(editingEntry.id, null);
      setActionResult("delete", editingEntry.id, null, true);
    } catch (error) {
      console.error("Failed to delete codex entry:", error);
      setActionError(error instanceof Error ? error.message : "Failed to delete codex entry");
    }
  }, [editingEntry, removeEntry, setActionResult, setActionError]);

  const handleToggleEnabled = useCallback(
    async (entry: CharacterBookEntry, enabled: boolean) => {
      if (entry.id === undefined) return;

      try {
        const updatedEntry = { ...entry, enabled };
        await updateCodexEntryApi(updatedEntry, entry.id, null);
        updateEntry(updatedEntry, null);
      } catch (error) {
        console.error("Failed to toggle entry:", error);
        setActionError(error instanceof Error ? error.message : "Failed to toggle entry");
      }
    },
    [updateEntry, setActionError]
  );

  // Show form if editing global (null character_id)
  const isEditing = editingEntry !== null && editingEntryCharacterId === null;

  return (
    <div className="space-y-4">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">{globalCodexName}</h1>
        <p className="text-muted-foreground">
          Global entries are active across all characters. Create entries here
          for world facts, setting details, or universal rules.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Entry List */}
        <CodexEntryList
          entries={globalEntries}
          isLoading={isLoadingGlobal}
          selectedEntryId={editingEntry?.id ?? null}
          onSelect={(entry) => startEditEntry(entry, null)}
          onCreate={handleCreate}
          onRefresh={handleRefresh}
          onToggleEnabled={handleToggleEnabled}
        />

        {/* Entry Form or Placeholder */}
        {isEditing ? (
          <CodexEntryForm
            entry={editingEntry}
            isNew={isNewEntry}
            isSaving={lastAction.type !== null && lastAction.success === null}
            error={lastAction.error}
            onChange={updateEditingEntry}
            onSave={handleSave}
            onDelete={isNewEntry ? undefined : handleDelete}
            onCancel={cancelEditEntry}
          />
        ) : (
          <div className="flex items-center justify-center h-64 border rounded-lg text-muted-foreground">
            Select an entry to edit or create a new one
          </div>
        )}
      </div>
    </div>
  );
}
