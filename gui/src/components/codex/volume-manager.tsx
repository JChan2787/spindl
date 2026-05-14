"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Settings2, Plus, Trash2 } from "lucide-react";
import type { CodexVolume, CharacterBookEntry } from "@/types/events";

interface VolumeManagerProps {
  volumes: CodexVolume[];
  entries: CharacterBookEntry[];
  onCreateVolume: (name: string, description?: string) => Promise<void>;
  onUpdateVolume: (volumeId: string, updates: Partial<Omit<CodexVolume, "id">>) => Promise<void>;
  onDeleteVolume: (volumeId: string) => Promise<void>;
  onToggleVolume: (volumeId: string, enabled: boolean) => Promise<void>;
}

export function VolumeManager({
  volumes,
  entries,
  onCreateVolume,
  onUpdateVolume,
  onDeleteVolume,
  onToggleVolume,
}: VolumeManagerProps) {
  const [open, setOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");

  const sortedVolumes = [...volumes].sort((a, b) => a.insertion_order - b.insertion_order);

  const entryCountByVolume = (volumeId: string) =>
    entries.filter((e) => (e.volume_id || "vol_default") === volumeId).length;

  const handleCreate = async () => {
    if (!newName.trim()) return;
    await onCreateVolume(newName.trim(), newDescription.trim() || undefined);
    setNewName("");
    setNewDescription("");
  };

  const handleRename = async (volumeId: string) => {
    if (!editName.trim()) return;
    await onUpdateVolume(volumeId, { name: editName.trim() });
    setEditingId(null);
    setEditName("");
  };

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b bg-muted/20">
      <span className="text-xs text-muted-foreground">
        {volumes.length} {volumes.length === 1 ? "volume" : "volumes"}
      </span>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button variant="ghost" size="sm" className="h-7 text-xs">
            <Settings2 className="h-3.5 w-3.5 mr-1" />
            Manage Volumes
          </Button>
        </DialogTrigger>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Manage Codex Volumes</DialogTitle>
          </DialogHeader>

          <div className="space-y-3 max-h-[400px] overflow-y-auto">
            {sortedVolumes.map((vol) => (
              <div
                key={vol.id}
                className="flex items-center gap-2 p-2 rounded-md border bg-card"
              >
                {editingId === vol.id ? (
                  <div className="flex-1 flex gap-2">
                    <Input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleRename(vol.id)}
                      className="h-7 text-sm"
                      autoFocus
                    />
                    <Button size="sm" className="h-7" onClick={() => handleRename(vol.id)}>
                      Save
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7"
                      onClick={() => setEditingId(null)}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <>
                    <button
                      className="flex-1 text-left text-sm font-medium truncate"
                      onClick={() => {
                        setEditingId(vol.id);
                        setEditName(vol.name);
                      }}
                    >
                      {vol.name}
                    </button>
                    <Badge variant="secondary" className="text-xs shrink-0">
                      {entryCountByVolume(vol.id)}
                    </Badge>
                    <Switch
                      checked={vol.enabled}
                      onCheckedChange={(checked) => onToggleVolume(vol.id, checked)}
                    />
                    {vol.id !== "vol_default" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 w-7 p-0 text-destructive"
                        onClick={() => onDeleteVolume(vol.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>

          <div className="border-t pt-3 space-y-2">
            <Label className="text-xs">New Volume</Label>
            <div className="flex gap-2">
              <Input
                placeholder="Volume name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                className="h-8 text-sm"
              />
              <Button size="sm" className="h-8" onClick={handleCreate} disabled={!newName.trim()}>
                <Plus className="h-3.5 w-3.5 mr-1" />
                Add
              </Button>
            </div>
            <Input
              placeholder="Description (optional)"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              className="h-8 text-sm"
            />
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
